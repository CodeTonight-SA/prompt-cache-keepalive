#!/usr/bin/env python3
"""Stop hook: offload durable context to the COLD store at session end.

Lossless, redacted, scope-gated. A scope-denied or redaction-refused write is a
silent non-event (never breaks the Stop). This hook only writes disk; it never
touches the conversation array or any provider.
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
