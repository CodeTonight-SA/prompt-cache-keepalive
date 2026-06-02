"""Session-boundary offload/restore — the plugin's only real logic.

Kept deliberately small and pure so the three hook entry-points are thin
wrappers. ``dump_cold_context`` writes a durable context blob to the COLD store
(lossless, redacted, scope-gated). ``restore_cold_context`` pages the most recent
blob back BYTE-IDENTICAL and returns it for the hook to emit on stdout.

Neither function touches CC's live window or conversation array — they only read
durable context the caller hands them and round-trip it through the store. That
constraint is what keeps the plugin inside its honest scope (see package
docstring).
"""
from __future__ import annotations

from typing import List, Optional

from ..coldstore import CacheMiss, ColdRef, FileRingStore

# A stable, recognisable key so a restore can find the latest boundary dump.
_BOUNDARY_KEY = "cc-session-boundary"


def dump_cold_context(
    store: FileRingStore,
    context_text: str,
    *,
    scope: str = "session",
) -> Optional[ColdRef]:
    """Offload *context_text* to the COLD store at a session boundary.

    Returns the ref, or ``None`` if the scope gate / redactor refused the write
    (fail-closed: a refusal is a non-event, never an exception that crashes the
    hook and breaks the user's session)."""
    if not context_text:
        return None
    try:
        return store.put(_BOUNDARY_KEY, context_text, scope=scope)
    except PermissionError:
        # Scope-denied or redaction-refused: skip silently, do not break Stop.
        return None


def restore_cold_context(
    store: FileRingStore,
    *,
    scope: str = "session",
) -> Optional[str]:
    """Page back the most recent boundary dump BYTE-IDENTICAL, or ``None``.

    A typed :class:`CacheMiss` (evicted / expired / absent) becomes ``None`` so a
    fresh session with no prior COLD context starts cleanly rather than erroring.
    """
    refs: List[ColdRef] = [r for r in store.list() if r.key == _BOUNDARY_KEY]
    if not refs:
        return None
    latest = refs[-1]
    try:
        return store.get(latest, scope=scope)
    except (CacheMiss, PermissionError):
        return None
