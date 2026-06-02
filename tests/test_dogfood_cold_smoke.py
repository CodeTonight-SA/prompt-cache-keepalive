"""CBC smoke tests for the COLD dogfood A/B — network-free, exogenous-correct.

Goodhart-resistant: assert the harness compares TOTAL billed tokens across the
two arms (verbatim-restore vs re-derivation) and reports a win iff X < Y — NOT
the page-back leg in isolation (which would falsify COLD by construction). A
mutation that compares only page-back tokens, or flips the inequality, fails.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dogfood-cold-tier.py"


def _load():
    spec = importlib.util.spec_from_file_location("dogfood_cold_tier", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve its own module's namespace
    # (Python 3.14 looks up cls.__module__ in sys.modules during field parsing).
    sys.modules["dogfood_cold_tier"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_total_billed_sums_input_side_only():
    mod = _load()
    arm = mod.ArmUsage(input_tokens=1000, cache_creation_input_tokens=200, cache_read_input_tokens=50, output_tokens=999)
    # Output tokens are excluded; the discriminating cost is the input side.
    assert arm.total_billed() == 1250


def test_cold_win_iff_restore_cheaper_than_rederivation():
    mod = _load()
    cheap_restore = mod.ArmUsage(cache_creation_input_tokens=25_000)
    expensive_rederive = mod.ArmUsage(input_tokens=180_000)
    assert mod.cold_is_a_win(cheap_restore, expensive_rederive) is True
    # And the loss regime: cheap-to-rebuild context -> COLD must NOT be a win.
    cheap_rederive = mod.ArmUsage(input_tokens=5_000)
    assert mod.cold_is_a_win(cheap_restore, cheap_rederive) is False


def test_self_check_runs_and_reports_win(capsys):
    mod = _load()
    assert mod.main([]) == 0  # network-free self-check exits 0
    out = capsys.readouterr().out
    assert "WIN" in out
