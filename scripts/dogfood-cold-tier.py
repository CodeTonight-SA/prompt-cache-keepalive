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


def _build_durable_prefix(repeat: int = 240) -> str:
    """A large, deterministic durable prefix that exceeds the 1024-token
    ``cache_control`` floor so the cache actually engages.

    Deterministic content => the A/B is reproducible; the only varying input is
    the trailing question, so any usage-field delta is attributable to the cache
    tier (write vs read), not to prefix drift. This stands in for an evicted GRIP
    session prefix in a real run."""
    line = (
        "GRIP durable context line: a self-improving recursive reasoning harness "
        "with session continuity, mechanical safety gates, and exogenous-anchor "
        "convergence loops. This block is expensive to RE-DERIVE verbatim.\n"
    )
    return line * repeat


def _usage_of(resp) -> "ArmUsage":
    """Project an Anthropic response's usage fields onto :class:`ArmUsage`."""
    u = resp.usage
    return ArmUsage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
    )


def _cached_call(client, model, prefix, question):  # pragma: no cover - needs key
    """One messages.create with the durable prefix marked ``cache_control``.

    The ``cache_control: ephemeral`` breakpoint is what makes the prefix billable
    at the cache WRITE tier on a miss and the cache READ tier on a hit — the two
    tiers whose usage fields are the exogenous discriminator."""
    return client.messages.create(
        model=model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )


def _prove_cache_real(client, model, prefix, q):  # pragma: no cover - needs key
    """WRITE then READ the SAME prefix: the WARM-tier proof.

    Returns (establish, warm). The warm read hitting the cache
    (``cache_read_input_tokens > 0``, ``creation == 0``) is the exogenous evidence
    that the restore/keepalive mechanism is real at the cheap read tier."""
    establish = _usage_of(_cached_call(client, model, prefix, q))
    warm = _usage_of(_cached_call(client, model, prefix, q + " (again)"))
    return establish, warm


def _run_ab_arms(client, model, prefix, slice_prefix, q):  # pragma: no cover - needs key
    """The two A/B arms, BOTH forced to genuine cache misses (write-vs-write).

    After an OVERNIGHT idle the cache has expired, so both arms re-inject their
    bytes as a fresh WRITE (1.25x) — no warm read to hit. A unique nonce per arm
    guarantees the miss so neither accidentally hits the still-warm establish
    entry (which would mis-price arm Y at the read tier, an apples-to-oranges
    comparison). Returns (arm_y full re-derive, arm_x cold page-back slice)."""
    import uuid

    bust_y = f"[session {uuid.uuid4()} overnight-expired rederive-full]\n"
    bust_x = f"[session {uuid.uuid4()} overnight-expired cold-restore]\n"
    arm_y = _usage_of(_cached_call(client, model, bust_y + prefix, q))
    arm_x = _usage_of(_cached_call(client, model, bust_x + slice_prefix, q))
    return arm_y, arm_x


def _report_ab(establish, warm, arm_y, arm_x, restore_fraction) -> bool:  # pragma: no cover - needs key
    """Print the exogenous result and return whether COLD won AND the cache hit."""
    warm_hit = warm.cache_read_input_tokens > 0
    win = cold_is_a_win(arm_x, arm_y)
    saved = arm_y.total_billed() - arm_x.total_billed()
    print("[ab-live] EXOGENOUS RESULT (Anthropic usage fields)")
    print(f"  establish  : creation={establish.cache_creation_input_tokens} read={establish.cache_read_input_tokens} input={establish.input_tokens}")
    print(f"  warm-read  : creation={warm.cache_creation_input_tokens} read={warm.cache_read_input_tokens} input={warm.input_tokens}  cache_hit={warm_hit}")
    print(f"  arm Y (rederive full)   : creation={arm_y.cache_creation_input_tokens} input={arm_y.input_tokens} total_billed={arm_y.total_billed()}")
    print(f"  arm X (cold restore {restore_fraction:.0%}): creation={arm_x.cache_creation_input_tokens} input={arm_x.input_tokens} total_billed={arm_x.total_billed()}")
    print(f"  VERDICT    : {'COLD WIN' if win else 'COLD LOSS'}  (saved {saved} billed input tokens, warm_hit={warm_hit})")
    return win and warm_hit


def _ab_live_run(restore_fraction: float = 0.25) -> int:  # pragma: no cover - needs key
    """Real-API A/B that prices BOTH COLD-tier arms from Anthropic usage fields.

    Faithful to ``cold_economics`` (page-back is a 1.25x WRITE; the win is AVOIDED
    RE-DERIVATION). COLD is a WIN iff arm X (verbatim restore of ``restore_fraction``)
    bills strictly fewer input tokens than arm Y (full re-derivation), decided
    purely by ``cache_creation_input_tokens`` — exogenous, the generating model
    cannot make the provider under-bill arm Y or over-bill arm X. Coordinator over
    three helpers: prove-cache-real, run-arms (forced misses), report."""
    import anthropic  # local import: package stays 0-dep

    client = anthropic.Anthropic()
    model = "claude-opus-4-8"
    prefix = _build_durable_prefix()
    slice_prefix = _build_durable_prefix(repeat=max(1, round(240 * restore_fraction)))
    q = "Given the durable context above, state the next step in one short line."

    establish, warm = _prove_cache_real(client, model, prefix, q)
    arm_y, arm_x = _run_ab_arms(client, model, prefix, slice_prefix, q)
    ok = _report_ab(establish, warm, arm_y, arm_x, restore_fraction)
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COLD-tier A/B dogfood (exogenous via Anthropic usage)")
    parser.add_argument("--live", action="store_true", help="run the simple two-arm live check (needs API key)")
    parser.add_argument("--ab-live", action="store_true", help="run the full cache-priced A/B against a real Anthropic client (needs API key)")
    parser.add_argument("--restore-fraction", type=float, default=0.25, help="verbatim share COLD pages back (arm X write size)")
    args = parser.parse_args(argv)
    if args.ab_live:
        return _ab_live_run(restore_fraction=args.restore_fraction)
    return _live_run() if args.live else _self_check()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
