"""CBC tests for the cost model — pinned economics, Goodhart-resistant.

These assert exact token-equivalent numbers so that breaking a multiplier, the
floor-division, or the honest cap-exhaustion (loss) branch FAILS the suite.
Base case throughout: an 88k-token prefix, Anthropic multipliers.
"""
from __future__ import annotations

import math

import pytest

from prompt_cache_keepalive import (
    CacheEconomics,
    breakeven_resume_probability,
    cost_with_keepalive,
    cost_without_keepalive,
    net_savings,
    touches_for_idle,
)

ECON = CacheEconomics(prefix_tokens=88_000)  # write 1.25x, read 0.10x


def test_unit_costs_are_pinned():
    assert ECON.rewrite_cost() == 110_000.0   # 88000 * 1.25
    assert ECON.read_cost() == 8_800.0        # 88000 * 0.10


def test_touches_for_idle_floor_and_cap():
    assert touches_for_idle(540, 270, 12) == 2      # floor(540/270)
    assert touches_for_idle(4000, 270, 12) == 12     # floor(14) capped at 12
    assert touches_for_idle(0, 270, 12) == 0
    assert touches_for_idle(540, 0, 12) == 0         # guard against div-by-zero


def test_cost_without_keepalive():
    assert cost_without_keepalive(ECON, resumes=True) == 110_000.0
    assert cost_without_keepalive(ECON, resumes=False) == 0.0


def test_cost_with_keepalive_warm_vs_expired():
    # warm resume: 2 touches (2 * 8800) + warm read (8800) = 26400
    assert cost_with_keepalive(ECON, 2, resumes=True, cache_expired_despite_touches=False) == 26_400.0
    # expired resume: 12 touches + full re-write = 105600 + 110000 = 215600
    assert cost_with_keepalive(ECON, 12, resumes=True, cache_expired_despite_touches=True) == 215_600.0
    # never resumes: only the touch spend is wasted
    assert cost_with_keepalive(ECON, 2, resumes=False) == 17_600.0


def test_short_idle_is_strongly_positive():
    # 9-minute idle, 270s interval, certain resume: saves ~83.6k token-equiv
    saved = net_savings(ECON, idle_seconds=540, interval_seconds=270, max_touches=12, resume_probability=1.0)
    assert saved == pytest.approx(83_600.0)
    assert saved > 0  # keepalive wins in the common short-idle-then-resume case


def test_long_idle_is_an_honest_loss():
    # Idle far past the cap: cache expires anyway; touches are pure waste on
    # top of the re-write. This MUST be negative — the honest-loss regime.
    lost = net_savings(ECON, idle_seconds=4000, interval_seconds=270, max_touches=12, resume_probability=1.0)
    assert lost == pytest.approx(-105_600.0)
    assert lost < 0


def test_breakeven_resume_probability_short_idle():
    # p* = n*read / (R - read) = 17600 / (110000 - 8800)
    p = breakeven_resume_probability(ECON, idle_seconds=540, interval_seconds=270, max_touches=12)
    assert p == pytest.approx(17_600.0 / 101_200.0)
    assert 0.17 < p < 0.18  # ~17.4% — realistic coding sessions clear this easily


def test_breakeven_is_one_when_cap_exhausted():
    # Cache expires despite the cap -> keepalive can never win -> p* == 1.0
    p = breakeven_resume_probability(ECON, idle_seconds=4000, interval_seconds=270, max_touches=12)
    assert p == 1.0


def test_net_savings_scales_with_resume_probability():
    # At the breakeven probability, expected net savings is ~0; above it, positive.
    p_star = breakeven_resume_probability(ECON, 540, 270, 12)
    at_breakeven = net_savings(ECON, 540, 270, 12, resume_probability=p_star)
    above = net_savings(ECON, 540, 270, 12, resume_probability=min(1.0, p_star + 0.2))
    assert math.isclose(at_breakeven, 0.0, abs_tol=1e-6)
    assert above > at_breakeven
