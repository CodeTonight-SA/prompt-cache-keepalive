"""prompt-cache-keepalive — keep an LLM prompt cache warm across idle gaps.

Most LLM prompt caches evict a cached conversation prefix after a short idle
TTL (Anthropic: ~5 minutes). The next turn after the window lapses reprocesses
the ENTIRE prefix at full price instead of the ~0.1x cache-read price. This
package keeps the cache warm with periodic minimal "touch" requests, bounded
by an economic spend cap so a dead session can never bleed tokens.

Provider-agnostic by construction: the core never imports an LLM SDK. The
caller injects a ``touch_fn`` (see ``examples/anthropic_keepalive.py``).

Public API:
    PromptCacheKeepalive, KeepaliveConfig, KeepaliveStats, TouchResult
    CacheEconomics, net_savings, breakeven_resume_probability, touches_for_idle
"""
from __future__ import annotations

from .keepalive import (
    KeepaliveConfig,
    KeepaliveStats,
    PromptCacheKeepalive,
    TouchResult,
)
from .cost_model import (
    CacheEconomics,
    breakeven_resume_probability,
    cost_with_keepalive,
    cost_without_keepalive,
    net_savings,
    touches_for_idle,
)

__version__ = "0.1.0"

__all__ = [
    "PromptCacheKeepalive",
    "KeepaliveConfig",
    "KeepaliveStats",
    "TouchResult",
    "CacheEconomics",
    "net_savings",
    "breakeven_resume_probability",
    "cost_with_keepalive",
    "cost_without_keepalive",
    "touches_for_idle",
]
