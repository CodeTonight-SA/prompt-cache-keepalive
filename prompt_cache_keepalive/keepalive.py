"""Provider-agnostic prompt-cache keepalive.

The problem
-----------
Most LLM prompt caches evict a cached conversation prefix after a short idle
TTL (Anthropic's is ~5 minutes, sliding). The next real turn after the window
lapses must reprocess the ENTIRE prefix at full (cache-write) price instead of
the ~0.1x cache-read price. For an 88k-token prefix that is a ~110k-token-
equivalent penalty on every resumed-after-idle turn.

The fix
-------
Before the window lapses, issue a minimal "touch" request that re-uses the
exact cached prefix. A cache *read* refreshes the entry's TTL, so the prefix
stays warm and the next real turn is a cheap cache read.

Design
------
This module is provider-agnostic by construction (dependency inversion): it
never imports any LLM SDK. The caller injects a ``touch_fn`` that performs the
minimal cache-reusing request and returns a :class:`TouchResult`. That makes
the keepalive (a) unit-testable with zero network, (b) usable with any provider
whose cache TTL is refreshed by re-use, (c) free of any secret handling.

Honesty bound
-------------
A touch is not free (~0.1x the prefix). Past a breakeven number of touches it
is cheaper to let the cache expire and eat one re-write. ``KeepaliveConfig.
max_touches`` caps spend near that breakeven, and the keepalive only pays off
if the session actually resumes. See :mod:`prompt_cache_keepalive.cost_model`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class TouchResult:
    """Outcome of one cache-touch request.

    ``cache_read_tokens`` / ``cache_creation_tokens`` mirror the provider's
    usage fields so callers can reconcile real spend against the cost model.
    """

    ok: bool
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


@dataclass
class KeepaliveConfig:
    """Timing and spend bounds for the keepalive loop."""

    ttl_seconds: float = 300.0           # provider cache TTL (Anthropic 5-min default)
    margin_seconds: float = 30.0         # touch early so a slow request still lands in-window
    max_touches: int = 12                # spend cap near the breakeven vs one re-write
    min_interval_seconds: float = 5.0    # floor; never busy-loop

    def interval(self) -> float:
        """Seconds between touches: TTL minus margin, floored to avoid spin."""
        return max(self.min_interval_seconds, self.ttl_seconds - self.margin_seconds)


@dataclass
class KeepaliveStats:
    """Running tally of keepalive activity (folded from each touch)."""

    touches_made: int = 0
    touches_failed: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0


class PromptCacheKeepalive:
    """Keeps a provider prompt-cache warm by periodic minimal touches.

    The injected ``touch_fn`` must issue ONE minimal request that re-uses the
    exact cached prefix (same messages + same ``cache_control`` breakpoint) and
    return a :class:`TouchResult`. This class owns only the *timing* and the
    *spend bound* — it never talks to a provider directly.
    """

    def __init__(
        self,
        touch_fn: Callable[[], TouchResult],
        config: Optional[KeepaliveConfig] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._touch = touch_fn
        self._cfg = config or KeepaliveConfig()
        self._sleep = sleep
        self.stats = KeepaliveStats()

    def touch_once(self) -> TouchResult:
        """Issue a single touch and fold its usage into :attr:`stats`."""
        result = self._touch()
        if result.ok:
            self.stats.touches_made += 1
            self.stats.cache_read_tokens += result.cache_read_tokens
            self.stats.output_tokens += result.output_tokens
        else:
            self.stats.touches_failed += 1
        return result

    def run(self, is_active: Callable[[], bool]) -> KeepaliveStats:
        """Touch every interval while the session may still resume.

        ``is_active()`` is the caller's "the conversation may still resume"
        predicate (user still present, task queue non-empty, ...). The loop
        exits the moment it returns False, when the spend cap is reached, or on
        a touch error — it never spins past its bound.
        """
        interval = self._cfg.interval()
        while self.stats.touches_made < self._cfg.max_touches and is_active():
            self._sleep(interval)
            if not is_active():
                break
            if not self.touch_once().ok:
                break
        return self.stats
