#!/usr/bin/env python3
"""SessionStart hook: page back the most recent COLD boundary dump.

Emits the restored context on stdout so a fresh / post-``/clear`` session
re-acquires durable context VERBATIM. Re-injecting it bills as a fresh cache
write (new bottom-of-window tokens) — this hook cannot, and does not claim to,
restore the live window in place. Its value is lossless continuity + avoided
re-derivation.
"""
from __future__ import annotations

from .. import coldstore
from ._io import read_hook_payload
from .boundary import restore_cold_context


def main() -> int:
    payload = read_hook_payload()
    scope = str(payload.get("scope", "session"))
    store = coldstore.FileRingStore()
    restored = restore_cold_context(store, scope=scope)
    if restored:
        print(restored)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
