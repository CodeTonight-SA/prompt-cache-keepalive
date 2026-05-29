"""Economics of prompt-cache keepalive — the falsifiable savings model.

All costs are expressed in *base-input-token-equivalents* (multiply by your
provider's input price-per-Mtok for currency). Anthropic multipliers as of
2026-05::

    cache write (5-min):  1.25x base
    cache read:           0.10x base
    normal input:         1.00x base

A cache *read* refreshes the entry's TTL at no extra write cost — that is the
mechanism the keepalive exploits.

This model exists so the keepalive can *prove* it saves tokens rather than
assert it, and it is deliberately honest about the regime where keepalive
LOSES (long idle that exhausts the spend cap, or low resume probability).

Per one idle gap, with ``R = rewrite cost`` and ``c = single cache-read cost``::

    net_savings(p) = p * (R - c)  -  n * c          (cache stayed warm)

where ``p`` is the probability the session resumes and ``n`` is the number of
touches. Breakeven resume probability is ``p* = n*c / (R - c)``. If the idle
exceeds ``max_touches * interval`` the cache expires anyway and every touch is
pure waste on top of the re-write — the model returns negative savings and a
breakeven of 1.0 (never worth it) for that regime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheEconomics:
    """Per-prefix cache economics in base-input-token-equivalents."""

    prefix_tokens: int
    cache_write_mult: float = 1.25
    cache_read_mult: float = 0.10

    def rewrite_cost(self) -> float:
        """Cost to re-establish the cache on a cold resume (one cache write)."""
        return self.prefix_tokens * self.cache_write_mult

    def read_cost(self) -> float:
        """Cost of one cache-read of the full prefix (a touch, or a warm resume)."""
        return self.prefix_tokens * self.cache_read_mult


def touches_for_idle(idle_seconds: float, interval_seconds: float, max_touches: int) -> int:
    """Number of touches a given idle gap incurs, bounded by the spend cap."""
    if idle_seconds <= 0 or interval_seconds <= 0:
        return 0
    return min(max_touches, math.floor(idle_seconds / interval_seconds))


def _cache_expired_despite_touches(idle_seconds: float, interval_seconds: float, max_touches: int) -> bool:
    """True when the idle outlasts the capped touch budget, so the cache
    expires anyway and the resume still pays a full re-write."""
    return idle_seconds > max_touches * interval_seconds


def cost_without_keepalive(econ: CacheEconomics, resumes: bool) -> float:
    """No keepalive: a resume after expiry re-writes the full prefix; a session
    that never resumes costs nothing further."""
    return econ.rewrite_cost() if resumes else 0.0


def cost_with_keepalive(
    econ: CacheEconomics,
    n_touches: int,
    resumes: bool,
    cache_expired_despite_touches: bool = False,
) -> float:
    """Keepalive: ``n_touches`` cache-reads during idle, plus the resume turn.

    If touches kept the prefix warm the resume is a cheap read; if the cap was
    exhausted and the cache expired anyway the resume re-writes."""
    touch_cost = n_touches * econ.read_cost()
    if not resumes:
        return touch_cost
    resume_cost = econ.rewrite_cost() if cache_expired_despite_touches else econ.read_cost()
    return touch_cost + resume_cost


def net_savings(
    econ: CacheEconomics,
    idle_seconds: float,
    interval_seconds: float,
    max_touches: int,
    resume_probability: float,
) -> float:
    """Expected token-equivalents saved by keepalive over one idle gap.

    Positive = keepalive wins. Folds in the cap-exhaustion regime where a very
    long idle expires the cache even with touches, making the touches pure
    waste on top of the unavoidable re-write."""
    n = touches_for_idle(idle_seconds, interval_seconds, max_touches)
    expired = _cache_expired_despite_touches(idle_seconds, interval_seconds, max_touches)
    p = max(0.0, min(1.0, resume_probability))
    exp_without = p * cost_without_keepalive(econ, True)
    exp_with = (
        p * cost_with_keepalive(econ, n, True, cache_expired_despite_touches=expired)
        + (1.0 - p) * cost_with_keepalive(econ, n, False)
    )
    return exp_without - exp_with


def breakeven_resume_probability(
    econ: CacheEconomics,
    idle_seconds: float,
    interval_seconds: float,
    max_touches: int,
) -> float:
    """Minimum resume probability for keepalive to be >= 0 EV over one gap.

    Below this probability the expected touch waste on non-resuming sessions
    exceeds the expected re-write saved on resuming ones — do not keepalive.
    Returns 1.0 when keepalive can never win (cap exhausted: touches cost more
    than the re-write they fail to avoid)."""
    n = touches_for_idle(idle_seconds, interval_seconds, max_touches)
    expired = _cache_expired_despite_touches(idle_seconds, interval_seconds, max_touches)
    touch = n * econ.read_cost()
    save_on_resume = cost_without_keepalive(econ, True) - cost_with_keepalive(
        econ, n, True, cache_expired_despite_touches=expired
    )
    denom = save_on_resume + touch
    if denom <= 0:
        return 1.0
    return max(0.0, min(1.0, touch / denom))
