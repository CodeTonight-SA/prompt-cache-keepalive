"""Per-OS advisory file lock — stdlib only, cross-platform.

The COLD ring is written by >=2 concurrent processes in the common case (two
``claude`` sessions sharing a home dir). A fixed-slot cursor without a lock loses
increments under that race. POSIX has ``fcntl.flock``; Windows has
``msvcrt.locking``; the two are a HARD fork (one is always absent). This shim
hides the fork behind one ``file_lock(path)`` context manager.

Caveat (documented, see README "Open risks"): POSIX advisory locks are
documented-unreliable over networked mounts (NFS/SMB) without a working lock
daemon. This shim is sound on local filesystems; it does not claim safety on
networked ``$HOME``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # POSIX
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - platform-dependent
    _HAVE_FCNTL = False

try:  # Windows
    import msvcrt

    _HAVE_MSVCRT = True
except ImportError:  # pragma: no cover - platform-dependent
    _HAVE_MSVCRT = False


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive advisory lock on *path* for the block's duration.

    Uses ``fcntl`` on POSIX, ``msvcrt`` on Windows. If neither is available the
    lock degrades to a no-op (single-process correctness only) rather than
    crashing — explicit, not silent: a no-op lock on an exotic platform is the
    documented fallback, not a guarantee.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if _HAVE_FCNTL:
            fcntl.flock(fd, fcntl.LOCK_EX)
        elif _HAVE_MSVCRT:  # pragma: no cover - exercised on Windows CI only
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if _HAVE_FCNTL:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif _HAVE_MSVCRT:  # pragma: no cover
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)
