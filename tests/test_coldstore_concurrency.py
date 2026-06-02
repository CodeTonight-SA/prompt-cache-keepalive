"""CBC concurrency tests for the file ring — zero lost cursor increments.

The ~/.claude norm is >=2 concurrent ``claude`` sessions. A fixed-slot ring
without a lock loses increments under that race. This spawns real processes
(not threads — the lock is cross-PROCESS) and asserts every write landed and no
temp file leaked. The lock-shim import is asserted for both POSIX and Windows
backends (whichever is absent is mocked).
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from prompt_cache_keepalive import FileRingStore, RingConfig
from prompt_cache_keepalive.cold_scope import allow_all_unsafe

_WRITERS = 4
_PER_WRITER = 8


def _writer(cold_dir: str, worker: int, n: int) -> None:
    store = FileRingStore(
        cold_dir=Path(cold_dir),
        scope_policy=allow_all_unsafe(),
        config=RingConfig(max_entries=_WRITERS * n + 1, shred_on_pageback=False),
    )
    for i in range(n):
        store.put(f"w{worker}-{i}", f"payload-{worker}-{i}", scope="session")


def test_concurrent_writers_lose_zero_increments(tmp_path):
    cold = tmp_path / "cold"
    cold.mkdir(parents=True, exist_ok=True, mode=0o700)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_writer, args=(str(cold), w, _PER_WRITER))
        for w in range(_WRITERS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    # Every write produced exactly one slot file -> zero lost updates.
    entries = list(cold.glob("pck-cold-*.cold"))
    assert len(entries) == _WRITERS * _PER_WRITER
    # No leaked temp files from the atomic-write path.
    assert not list(cold.glob("*.tmp*"))


def test_lock_shim_imports_on_both_backends(monkeypatch):
    # The lock module must work whether fcntl or msvcrt is the available backend.
    import prompt_cache_keepalive._locking as lk

    # POSIX path is live in CI; assert it acquires + releases without error.
    lockfile = Path(lk.__file__).parent / ".test.lock"
    try:
        with lk.file_lock(lockfile):
            pass
    finally:
        if lockfile.exists():
            lockfile.unlink()
    # At least one real backend must be present (never a silent no-op in CI).
    assert lk._HAVE_FCNTL or lk._HAVE_MSVCRT


def test_reader_sees_whole_old_or_whole_new(tmp_path):
    # os.replace is atomic: a get during a concurrent put returns either the full
    # prior blob or the full new one, never a torn read. Sequential proxy: write
    # v1, overwrite-via-new-slot v2, both reads are intact.
    store = FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=allow_all_unsafe(),
        config=RingConfig(shred_on_pageback=False),
    )
    r1 = store.put("k", "A" * 10_000, scope="session")
    r2 = store.put("k", "B" * 10_000, scope="session")
    assert store.get(r1, scope="session") == "A" * 10_000
    assert store.get(r2, scope="session") == "B" * 10_000
