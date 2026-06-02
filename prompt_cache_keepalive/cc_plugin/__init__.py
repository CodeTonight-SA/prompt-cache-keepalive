"""Claude Code plugin: session-boundary COLD offload/restore.

HARD scope boundary (read before believing any claim here)
----------------------------------------------------------
Claude Code hooks (SessionStart / Stop / PreCompact / PostCompact) fire AROUND
the request loop, NEVER inside a turn. The CC harness owns its own request loop
and cache breakpoints, so a plugin CANNOT:

* shrink CC's live context window mid-turn,
* mutate the conversation array, or
* touch CC's KV cache.

A ``SessionStart`` restore re-injects bytes as NEW bottom-of-window tokens — a
fresh cache write, not an in-place restore. Therefore this plugin's honest scope
is exactly: **losslessly OFFLOAD durable context to the private COLD store at a
boundary, and verbatim PAGE IT BACK on the next session** — surviving ``/clear``
and overnight idle at zero disk-leg token cost. Its measured value is
RE-DERIVATION-AVOIDED + lossless continuity vs CC's lossy native compaction. It
is explicitly NOT "lossless restore of the live context window" and NOT
mid-session token saving.

The plugin's only channels are stdout (to surface a restored digest) and disk
(the COLD store). It never calls a provider, never holds a secret (the store's
redactor + scope gate do that).
"""
from __future__ import annotations

from .boundary import dump_cold_context, restore_cold_context

__all__ = ["dump_cold_context", "restore_cold_context"]
