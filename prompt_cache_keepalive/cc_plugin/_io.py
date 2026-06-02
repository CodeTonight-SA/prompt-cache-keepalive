"""Shared stdin/stdout plumbing for the CC hook entry-points (DRY).

Claude Code hooks receive a JSON payload on stdin and may emit a small block on
stdout. These helpers centralise the read/emit so the three hook scripts stay
near-trivial and identical in their I/O contract.
"""
from __future__ import annotations

import json
import sys
from typing import Dict


def read_hook_payload() -> Dict[str, object]:
    """Parse the hook JSON payload from stdin; ``{}`` on empty/malformed input.

    Never raises — a hook that crashes on bad input would break the session it is
    meant to help. A malformed payload degrades to "do nothing"."""
    try:
        raw = sys.stdin.read()
    except Exception:  # noqa: BLE001
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def durable_context_from(payload: Dict[str, object]) -> str:
    """Extract the durable-context string a caller passes for offload.

    Looks at the conventional keys CC-style payloads use. Returns "" when none is
    present (then ``dump_cold_context`` no-ops)."""
    for key in ("cold_context", "durable_context", "transcript_excerpt"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    return ""
