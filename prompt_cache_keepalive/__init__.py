"""prompt-cache-keepalive — keep an LLM prompt cache warm across idle gaps.

Most LLM prompt caches evict a cached conversation prefix after a short idle
TTL (Anthropic: ~5 minutes). The next turn after the window lapses reprocesses
the ENTIRE prefix at full price instead of the ~0.1x cache-read price. This
package keeps the cache warm with periodic minimal "touch" requests, bounded
by an economic spend cap so a dead session can never bleed tokens.

Provider-agnostic by construction: the core never imports an LLM SDK. The
caller injects a ``touch_fn`` (see ``examples/anthropic_keepalive.py``).

Three cost tiers (see ``cold_economics`` for the full map):
    HOT   live context window — every turn pays ~0.1x the cached prefix.
    WARM  keepalive-pinged server cache — pay ~0.1x per ping to dodge a re-write.
    COLD  context evicted to a PRIVATE LOCAL store at zero provider-token cost,
          paged back VERBATIM. Lossless where compaction is lossy; for LONG /
          overnight idle and session boundaries where WARM cannot help.

Public API:
    PromptCacheKeepalive, KeepaliveConfig, KeepaliveStats, TouchResult
    CacheEconomics, net_savings, breakeven_resume_probability, touches_for_idle
    ColdTierEconomics, Tier, should_offload_cold, FileRingStore, RingConfig
    ColdStore, ColdRef, CacheMiss, default_redactor, deny_unknown
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
from .compaction import (
    CompactionResult,
    compact_prefix,
    estimate_tokens,
)
from .cold_economics import (
    ColdTierEconomics,
    Tier,
    should_offload_cold,
)
from .coldstore import (
    CacheMiss,
    ColdRef,
    ColdStore,
    FileRingStore,
    RingConfig,
)
from .redaction import (
    RedactionError,
    Redactor,
    default_redactor,
)
from .cold_scope import (
    ColdScopePolicy,
    allow_all_unsafe,
    deny_unknown,
)

__version__ = "0.3.0"

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
    "CompactionResult",
    "compact_prefix",
    "estimate_tokens",
    # COLD tier
    "ColdTierEconomics",
    "Tier",
    "should_offload_cold",
    "ColdStore",
    "FileRingStore",
    "ColdRef",
    "RingConfig",
    "CacheMiss",
    "Redactor",
    "RedactionError",
    "default_redactor",
    "ColdScopePolicy",
    "deny_unknown",
    "allow_all_unsafe",
]
