#!/usr/bin/env python3
"""PreCompact hook: offload durable context BEFORE CC's lossy native compaction.

Same offload path as the Stop hook — the point is to capture the about-to-be-
compacted context VERBATIM into the COLD store first, so it survives losslessly
even though CC's own compaction is lossy. It does NOT steer or replace CC's
compaction (a PreCompact hook's stdout only steers a lossy summary); it preserves
a lossless copy alongside.
"""
from __future__ import annotations

from .. import coldstore
from ._io import durable_context_from, read_hook_payload
from .boundary import dump_cold_context


def main() -> int:
    payload = read_hook_payload()
    scope = str(payload.get("scope", "session"))
    context_text = durable_context_from(payload)
    store = coldstore.FileRingStore()
    dump_cold_context(store, context_text, scope=scope)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
