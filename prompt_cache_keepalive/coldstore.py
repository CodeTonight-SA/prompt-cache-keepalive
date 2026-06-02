"""The COLD store — a private, lossless, local context offload.

What this is
------------
A place to put context bytes you have evicted from the live window so they
survive idle gaps and session boundaries at zero PROVIDER-token cost, and page
back BYTE-IDENTICAL on demand. (See :mod:`cold_economics` for why "zero cost"
is only the disk leg — the page-back itself still bills as a cache write.)

Why a FILE, not the system clipboard or a named pasteboard
----------------------------------------------------------
The decision is overdetermined by four independent axes, not aesthetics:

* **0-dependency core** — a named ``NSPasteboard`` needs pyobjc; the file ring is
  pure stdlib, so the package's 0-dep badge stays intact.
* **Durability** — the proven win is OVERNIGHT survival. Every OS clipboard /
  pasteboard is RAM-backed and dies on reboot/logoff. Only a file survives the
  exact failure mode COLD exists for.
* **Privacy** — a ``0600`` file (filesystem ACL gates read) is STRICTLY more
  private than any named pasteboard, which any same-user process that knows the
  name can read with no prompt — and the name lives in OSS source.
* **Testability** — CI is ubuntu-only; the file ring is the only backend the
  matrix can exercise.

The general system clipboard is NEVER used: it cross-device-syncs via Universal
Clipboard and every clipboard manager reads it — a leak vector for context,
secrets, and client data.

Backend shape
-------------
:class:`ColdStore` is a ``Protocol`` so a native pasteboard backend can be
injected later (deliverable B / an optional extra) without touching callers —
the dependency-inversion ethos of the rest of the package. The 0-dep default,
and the only backend shipped in core, is :class:`FileRingStore`.

Safety contract (every backend must honour)
-------------------------------------------
* A ``Redactor`` runs BEFORE every write and FAILS CLOSED (a redaction error
  refuses the write — never writes cleartext).
* Reads are SCOPE-CHECKED before returning bytes (channel-scope isolation holds
  at the cache layer, not just messaging).
* Entries are namespaced ``pck-cold-<scope>-`` so a human (or a clipboard
  manager, for a future pasteboard backend) can recognise them.
* On overflow the ring REFUSES or evicts oldest — it never silently truncates a
  stored blob (a torn blob would break the lossless contract).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol, runtime_checkable

from ._locking import file_lock
from .cold_scope import ColdScopePolicy, deny_unknown
from .redaction import Redactor, RedactionError, default_redactor

_NAMESPACE_PREFIX = "pck-cold"
_ENTRY_SUFFIX = ".cold"


class CacheMiss(Exception):
    """Raised when a COLD ref cannot be paged back (evicted, expired, absent).

    A typed miss, never a silent ``None`` masquerading as content — the caller
    must decide to re-derive, not accidentally proceed on empty context.
    """


@dataclass(frozen=True)
class ColdRef:
    """Opaque handle to a stored COLD entry. Carries its own scope for re-check."""

    key: str
    scope: str
    slot: int


@runtime_checkable
class ColdStore(Protocol):
    """The injectable COLD-store boundary (dependency inversion)."""

    def put(self, key: str, text: str, *, scope: str) -> ColdRef: ...
    def get(self, ref: ColdRef, *, scope: str) -> str: ...
    def evict(self, ref: ColdRef) -> None: ...
    def list(self) -> List[ColdRef]: ...


@dataclass
class RingConfig:
    """Bounds for the file ring. ``overflow`` governs the full-ring policy."""

    max_entries: int = 16
    max_age_s: float = 86_400.0          # 24h: the overnight horizon COLD targets
    overflow: str = "evict_oldest"        # "refuse" | "evict_oldest"
    shred_on_pageback: bool = True        # zero+unlink a slot once paged back

    def __post_init__(self) -> None:
        if self.overflow not in ("refuse", "evict_oldest"):
            raise ValueError(f"overflow must be 'refuse' or 'evict_oldest', got {self.overflow!r}")


def _default_cold_dir() -> Path:
    """Resolve the private COLD dir. Env override -> XDG state -> ~/.local/state."""
    env = os.environ.get("PROMPT_CACHE_COLD_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "prompt-cache-keepalive" / "cold"


class FileRingStore:
    """Stdlib-only, mode-0600 file ring. The 0-dependency default COLD backend.

    Each entry is written to a monotonic slot via temp-file -> ``fsync`` ->
    ``os.replace`` (atomic, crash-safe: a reader sees whole-old or whole-new,
    never a torn write). State is FILESYSTEM-authoritative — slots are derived
    from existing files under a per-OS advisory lock, so concurrent processes and
    fresh instances stay consistent and no write is lost. No non-stdlib import
    appears anywhere in this module's import graph.
    """

    def __init__(
        self,
        config: Optional[RingConfig] = None,
        *,
        cold_dir: Optional[Path] = None,
        redactor: Optional[Redactor] = None,
        scope_policy: Optional[ColdScopePolicy] = None,
        namespace: str = "default",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = config or RingConfig()
        self._dir = cold_dir or _default_cold_dir()
        self._redactor = redactor or default_redactor()
        self._scope = scope_policy or deny_unknown()
        self._ns = namespace
        self._clock = clock
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock_path = self._dir / ".ring.lock"

    # -- internal helpers (filesystem is authoritative, not in-memory) -------
    #
    # State lives ON DISK, not in instance memory: slot numbers are derived from
    # the existing slot files under the lock, and listing scans the directory.
    # This is what makes the ring correct across CONCURRENT processes and FRESH
    # store instances (a new SessionStart process must see a prior Stop's dump).

    def _slot_path(self, slot: int) -> Path:
        return self._dir / f"{_NAMESPACE_PREFIX}-{self._ns}-{slot:06d}{_ENTRY_SUFFIX}"

    def _existing_slots(self) -> List[int]:
        """Slot numbers currently on disk for this namespace, ascending."""
        slots: List[int] = []
        prefix = f"{_NAMESPACE_PREFIX}-{self._ns}-"
        for p in self._dir.glob(f"{prefix}*{_ENTRY_SUFFIX}"):
            stem = p.name[len(prefix):-len(_ENTRY_SUFFIX)]
            if stem.isdigit():
                slots.append(int(stem))
        return sorted(slots)

    def _next_slot(self) -> int:
        """One past the highest slot on disk — monotonic, collision-free under
        the lock even across processes (each reads the shared directory)."""
        existing = self._existing_slots()
        return (existing[-1] + 1) if existing else 0

    def _atomic_write(self, path: Path, data: str) -> None:
        """temp -> fsync -> os.replace, with a Windows sharing-violation retry.

        ``os.replace`` can raise ``PermissionError`` on Windows when a reader has
        the target open; a bounded retry recovers (POSIX never hits this, which
        is exactly why the ubuntu-only CI cannot see the gap). The temp name
        carries the PID so concurrent writers never collide on the temp file."""
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # pragma: no cover - Windows-only path
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def _encode(self, key: str, scope: str, payload: str) -> str:
        """One-line JSON header + payload so a ref is reconstructible cross-process."""
        header = json.dumps({"key": key, "scope": scope, "written_at": self._clock()})
        return header + "\n" + payload

    @staticmethod
    def _decode(raw: str) -> "tuple[dict, str]":
        head, _, body = raw.partition("\n")
        try:
            meta = json.loads(head)
        except json.JSONDecodeError:
            return {}, raw
        return (meta if isinstance(meta, dict) else {}), body

    def _read_entry(self, slot: int) -> Optional["tuple[ColdRef, str, float]"]:
        """Return (ref, payload, written_at) for a slot on disk, or None."""
        path = self._slot_path(slot)
        if not path.exists():
            return None
        meta, body = self._decode(path.read_text(encoding="utf-8"))
        if "key" not in meta or "scope" not in meta:
            return None
        ref = ColdRef(key=str(meta["key"]), scope=str(meta["scope"]), slot=slot)
        return ref, body, float(meta.get("written_at", 0.0))

    def _shred(self, path: Path) -> None:
        """Zero then unlink — no indefinite plaintext-at-rest after page-back."""
        try:
            if path.exists():
                size = path.stat().st_size
                fd = os.open(path, os.O_WRONLY)
                try:
                    os.write(fd, b"\x00" * size)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                path.unlink()
        except FileNotFoundError:
            pass

    def _prune_expired_locked(self) -> None:
        """Shred any on-disk slot older than max_age. Caller holds the lock."""
        now = self._clock()
        for slot in self._existing_slots():
            entry = self._read_entry(slot)
            if entry is not None and now - entry[2] > self._cfg.max_age_s:
                self._shred(self._slot_path(slot))

    def _make_room_locked(self) -> None:
        """Enforce max_entries before a write. Caller holds the lock."""
        slots = self._existing_slots()
        if len(slots) < self._cfg.max_entries:
            return
        if self._cfg.overflow == "refuse":
            raise CacheMiss("COLD ring full and overflow=refuse; not truncating")
        self._shred(self._slot_path(slots[0]))  # evict oldest by slot number

    # -- public API ---------------------------------------------------------

    def put(self, key: str, text: str, *, scope: str) -> ColdRef:
        """Redact, scope-check, then atomically store *text*. Fail-closed.

        Order matters: the scope allowlist (load-bearing for unshaped client
        data) is checked first, then the redactor (secret shapes) runs and any
        redaction error REFUSES the write — cleartext never reaches disk.
        """
        if not self._scope.is_cold_safe(scope):
            raise PermissionError(f"scope {scope!r} denied COLD persistence (fail-closed)")
        try:
            safe = self._redactor(text)
        except RedactionError as exc:
            raise PermissionError(f"redaction failed; refusing COLD write: {exc}") from exc
        with file_lock(self._lock_path):
            self._prune_expired_locked()
            self._make_room_locked()
            slot = self._next_slot()           # derived from disk, not memory
            path = self._slot_path(slot)
            self._atomic_write(path, self._encode(key, scope, safe))
            return ColdRef(key=key, scope=scope, slot=slot)

    def get(self, ref: ColdRef, *, scope: str) -> str:
        """Page back *ref* BYTE-IDENTICAL. Scope-checked; typed miss on absence.

        The scope guard makes channel isolation hold at the cache layer: a ref
        minted under one scope cannot be read under another. Reads come straight
        off disk, so a fresh store instance sees prior processes' writes."""
        if scope != ref.scope or not self._scope.is_cold_safe(scope):
            raise PermissionError(f"scope {scope!r} may not read ref scoped {ref.scope!r}")
        entry = self._read_entry(ref.slot)
        if entry is None or entry[0].key != ref.key:
            raise CacheMiss(f"COLD ref {ref.key!r} slot {ref.slot} not present")
        data = entry[1]
        if self._cfg.shred_on_pageback:
            with file_lock(self._lock_path):
                self._shred(self._slot_path(ref.slot))
        return data

    def evict(self, ref: ColdRef) -> None:
        with file_lock(self._lock_path):
            self._shred(self._slot_path(ref.slot))

    def list(self) -> List[ColdRef]:
        """Refs currently on disk for this namespace (prunes expired first)."""
        with file_lock(self._lock_path):
            self._prune_expired_locked()
        refs: List[ColdRef] = []
        for slot in self._existing_slots():
            entry = self._read_entry(slot)
            if entry is not None:
                refs.append(entry[0])
        return refs
