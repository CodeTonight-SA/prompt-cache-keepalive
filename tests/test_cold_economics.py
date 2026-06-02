"""CBC tests for COLD-tier economics — pinned numbers, Goodhart-resistant.

These assert EXACT token-equivalents and the disjoint tier bands, so a mutation
that (a) makes page-back a 0.10x read instead of a 1.25x write, (b) flips the
net-savings inequality, or (c) lets COLD fire inside WARM's band, FAILS the
suite. Base case: a 100k-token prefix.
"""
from __future__ import annotations

from prompt_cache_keepalive import ColdTierEconomics, Tier, should_offload_cold

# 100k prefix, restore 25% of it, write multiplier 1.25
ECON = ColdTierEconomics(prefix_tokens=100_000, restore_fraction=0.25)


def test_store_is_free_restore_is_a_write_not_a_read():
    # The store leg is genuinely free (local disk).
    assert ECON.store_cost() == 0.0
    # Page-back bills as a cache WRITE on the restored fraction:
    # 100000 * 0.25 * 1.25 = 31250 — NOT a 0.10x read (which would be 2500).
    assert ECON.restore_cost() == 31_250.0


def test_cold_net_is_negative_when_context_is_cheap_to_rebuild():
    # Re-derivation cheaper than the page-back write => COLD LOSES (negative).
    assert ECON.cold_net_vs_rederivation(10_000) == 10_000 - 31_250
    assert ECON.cold_net_vs_rederivation(10_000) < 0


def test_cold_net_is_positive_when_rederivation_is_expensive():
    # Expensive-to-regenerate verbatim context => COLD WINS.
    assert ECON.cold_net_vs_rederivation(200_000) == 200_000 - 31_250
    assert ECON.cold_net_vs_rederivation(200_000) > 0


def test_short_idle_live_session_is_HOT():
    # tau < interval: cache still warm, do nothing.
    tier = should_offload_cold(
        idle_seconds=100, interval_seconds=270, max_touches=12,
        at_session_boundary=False, rederivation_tokens=200_000, econ=ECON,
    )
    assert tier is Tier.HOT


def test_warm_band_is_WARM_even_when_cold_would_otherwise_win():
    # 20-min idle on a live session is WARM's band. COLD must NOT fire here —
    # WARM keeps it warm for ~0.10x; COLD would pay 1.25x to page back. Even with
    # huge rederivation_tokens, the band rule forbids COLD.
    tier = should_offload_cold(
        idle_seconds=1200, interval_seconds=270, max_touches=12,
        at_session_boundary=False, rederivation_tokens=10_000_000, econ=ECON,
    )
    assert tier is Tier.WARM


def test_overnight_live_session_is_COLD_only_when_it_wins():
    # tau past the cap (12*270=3240s): WARM can't help. COLD iff it beats re-write.
    win = should_offload_cold(
        idle_seconds=40_000, interval_seconds=270, max_touches=12,
        at_session_boundary=False, rederivation_tokens=200_000, econ=ECON,
    )
    lose = should_offload_cold(
        idle_seconds=40_000, interval_seconds=270, max_touches=12,
        at_session_boundary=False, rederivation_tokens=10_000, econ=ECON,
    )
    assert win is Tier.COLD
    assert lose is Tier.NONE  # cheap to rebuild -> nothing


def test_session_boundary_is_COLD_when_it_beats_rederivation():
    # Boundary: transcript gone; alternatives are COLD or re-derivation.
    tier = should_offload_cold(
        idle_seconds=0, interval_seconds=270, max_touches=12,
        at_session_boundary=True, rederivation_tokens=200_000, econ=ECON,
    )
    assert tier is Tier.COLD


def test_session_boundary_is_NONE_when_context_is_cheap():
    tier = should_offload_cold(
        idle_seconds=0, interval_seconds=270, max_touches=12,
        at_session_boundary=True, rederivation_tokens=10_000, econ=ECON,
    )
    assert tier is Tier.NONE


def test_restore_cost_can_never_be_below_a_read_for_same_fraction():
    # Disk is not a provider tier: a restored fraction billed as a WRITE (1.25x)
    # is strictly costlier than the same fraction as a READ (0.10x). If this
    # inverts, someone wrongly modelled disk as a cheap provider tier.
    read_equiv = ECON.prefix_tokens * ECON.restore_fraction * 0.10
    assert ECON.restore_cost() > read_equiv
