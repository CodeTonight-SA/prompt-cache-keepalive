#!/usr/bin/env python3
"""Flag-gated A/B that PROVES the COLD tier saves tokens — exogenously.

The exogenous anchor
--------------------
The win is decided by Anthropic's own usage fields
(``cache_creation_input_tokens`` / ``cache_read_input_tokens`` /
``input_tokens``) — a process the generating model cannot please. The
pre-registered criterion (H799-COLD) is:

    arm X (evict -> COLD -> verbatim restore -> 1 turn)
      vs
    arm Y (evict -> REGENERATE the same context via the LLM -> 1 turn)

COLD is reported a WIN iff ``X_total_billed_tokens < Y_total_billed_tokens``.

Why this and not "count the page-back tokens"
---------------------------------------------
Measuring page-back wire tokens alone would falsify COLD BY CONSTRUCTION (the
page-back is a 1.25x write). The value is what arm Y has to spend RE-DERIVING the
context that arm X simply paged back verbatim. So the harness compares TOTAL
billed tokens across the two arms, never the page-back leg in isolation. That is
the single instrumentation error this script exists to prevent.

Live run is opt-in behind ``--live`` (needs a real Anthropic client + key),
mirroring the keepalive dogfood. Without it the script self-checks the
comparison logic on synthetic usage and exits.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmUsage:
    """Billed-token usage for one A/B arm (mirrors Anthropic usage fields)."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    def total_billed(self) -> int:
        """Total input-side billed tokens (the figure the criterion compares).

        Output tokens are excluded — both arms emit the same one-turn answer, so
        the discriminating cost is the INPUT side: arm Y's re-derivation input vs
        arm X's page-back write."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


def cold_is_a_win(cold_arm: ArmUsage, rederivation_arm: ArmUsage) -> bool:
    """COLD wins iff the verbatim-restore arm bills strictly fewer input tokens
    than the re-derivation arm. The exogenous, pre-registered criterion."""
    return cold_arm.total_billed() < rederivation_arm.total_billed()


def _self_check() -> int:
    """Network-free sanity: on a heavy prefix, restore should beat re-derivation.

    Arm X pages back a 25k slice (a 1.25x write ~= 31250 equiv-ish, here modelled
    as raw cache_creation tokens). Arm Y re-reads files + re-thinks to regenerate
    the same context (much larger input). COLD should be reported a win."""
    cold = ArmUsage(cache_creation_input_tokens=25_000, output_tokens=200)
    rederive = ArmUsage(input_tokens=180_000, cache_creation_input_tokens=20_000, output_tokens=200)
    assert cold_is_a_win(cold, rederive), "self-check: COLD should win on heavy re-derivation"
    print(f"[self-check] COLD total={cold.total_billed()} < rederive total={rederive.total_billed()} -> WIN")
    return 0


def _live_run() -> int:  # pragma: no cover - requires a real client + key
    import anthropic  # local import: package stays 0-dep

    client = anthropic.Anthropic()
    model = "claude-opus-4-8"
    # The durable context both arms hinge on. In a real run this is your evicted
    # session prefix; here a representative block stands in.
    durable = "…large durable GRIP context that is expensive to regenerate…\n" * 200

    # Arm X: pages the context back verbatim, then asks one question.
    x = client.messages.create(
        model=model, max_tokens=200,
        messages=[{"role": "user", "content": durable + "\nGiven the above, summarise the next step."}],
    )
    # Arm Y: must RE-DERIVE the context (no verbatim paste) before answering.
    y = client.messages.create(
        model=model, max_tokens=200,
        messages=[{"role": "user", "content": "Reconstruct the prior GRIP context from scratch, then summarise the next step."}],
    )

    def _usage(resp) -> ArmUsage:
        u = resp.usage
        return ArmUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
        )

    cold, rederive = _usage(x), _usage(y)
    win = cold_is_a_win(cold, rederive)
    print(f"COLD billed={cold.total_billed()} rederive billed={rederive.total_billed()} -> {'WIN' if win else 'LOSS'}")
    return 0 if win else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COLD-tier A/B dogfood (exogenous via Anthropic usage)")
    parser.add_argument("--live", action="store_true", help="run against a real Anthropic client (needs API key)")
    args = parser.parse_args(argv)
    return _live_run() if args.live else _self_check()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
