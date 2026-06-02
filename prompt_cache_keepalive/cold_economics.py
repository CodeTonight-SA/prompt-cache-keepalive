"""COLD-tier economics — the third tier below HOT (live) and WARM (keepalive).

The honest framing (read this before believing the pitch)
---------------------------------------------------------
A common misframing is "evict context to disk, page it back for free, save
tokens". That is a CATEGORY ERROR and :mod:`prompt_cache_keepalive.cost_model`
already refutes it: a provider has exactly two prefix price tiers — a cache
*write* at ``1.25x`` and a cache *read* at ``0.10x`` — and **disk is not a
provider pricing tier**. Bytes paged back off local disk into a live request
re-enter as fresh tokens and bill as a cache WRITE (``1.25x``), byte-identical
to ``cost_without_keepalive(resumes=True)``. There is no sub-``0.10x`` tier.

So "zero token cost" is TRUE only for the disk write/read leg and is quarantined
to that leg everywhere in this package. The defensible benefit of COLD is **not**
"cheap page-back" — it is **avoided RE-DERIVATION** of context that is

    * expensive to regenerate (re-reading files, re-running tools, re-thinking),
    * required VERBATIM (a lossy summary will not do), and
    * idle long enough (or across a session boundary) that WARM cannot help.

COLD's win is ``rederivation_tokens_avoided - restore_cost``. It INVERTS to a
loss on cheap-to-rebuild context — so :func:`should_offload_cold` gates COLD to
exactly the regime where it net-saves and returns WARM / HOT / nothing
everywhere it would lose.

The tier map over idle duration ``tau`` (one piecewise cost-minimiser)
---------------------------------------------------------------------
Disjoint bands, each tier strictly dominating in its own band (no oscillation)::

    tau < interval                              -> HOT   (do nothing; cache live)
    interval <= tau <= max_touches*interval     -> WARM  (resume is a 0.10x read;
        AND session live                                 COLD here LOSES at 1.25x)
    tau > max_touches*interval (overnight)      -> compare do-nothing (1.25x) vs
        AND session live                                 COLD; pick cheaper
    session boundary (/clear, process death)    -> COLD iff it beats RE-DERIVATION
                                                     (no do-nothing baseline: the
                                                      transcript is already gone)

COLD and compaction stay ORTHOGONAL: COLD is lossless (relocates bytes off-wire);
compaction is lossy (~4x, cheapens a miss you still pay). A mature loop runs
both. They share the verbatim contract — compaction's ``kept_verbatim_count``
tail is exactly the turns COLD pages back intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cost_model import CacheEconomics


class Tier(str, Enum):
    """Which cost tier a block belongs in for a given idle gap."""

    HOT = "hot"          # live window; cache still warm; do nothing
    WARM = "warm"        # keepalive-pinged server cache; resume is a cheap read
    COLD = "cold"        # evict to private local store; page back verbatim
    NONE = "none"        # do nothing (COLD would lose; no action warranted)


@dataclass(frozen=True)
class ColdTierEconomics:
    """COLD-tier costs in base-input-token-equivalents (matches CacheEconomics).

    ``restore_fraction`` is the share of the prefix actually re-injected on a
    page-back (you rarely restore the whole thing — typically the hot slice the
    model needs, mirroring ``compact_prefix``'s ``keep_recent_turns`` tail). It
    is a tuning knob, not a measured constant; the W6 dogfood settles it live.
    """

    prefix_tokens: int
    restore_fraction: float = 0.25
    cache_write_mult: float = 1.25

    def store_cost(self) -> float:
        """Provider-token cost to WRITE context to the COLD store: zero.

        This is the structural advantage and the ONLY thing that is free — it is
        a local disk write, no bytes cross the wire. (Disk I/O latency exists but
        is not a provider-token cost; do not conflate the two.)
        """
        return 0.0

    def restore_cost(self) -> float:
        """Provider-token cost to PAGE BACK the restored slice.

        Page-back re-injects bytes as new tokens => a cache WRITE at
        ``cache_write_mult``. Disk is not a provider tier, so this is NOT a
        ``0.10x`` read — it is ``1.25x`` on the restored fraction. The honest
        cost the naive pitch ignores.
        """
        return self.prefix_tokens * self.restore_fraction * self.cache_write_mult

    def cold_net_vs_rederivation(self, rederivation_tokens: float) -> float:
        """Net token-equivalents saved by COLD vs RE-DERIVING the same context.

        Positive = COLD wins (avoided re-derivation exceeds the page-back write).
        NEGATIVE = COLD loses (context was cheap to rebuild) — returned, not
        hidden, exactly as :func:`net_savings` returns a negative in its loss
        regime. The store leg is free, so the whole cost is the restore.
        """
        return rederivation_tokens - (self.store_cost() + self.restore_cost())


def _warm_band_max_seconds(interval_seconds: float, max_touches: int) -> float:
    """Upper edge of WARM's band: the last touch the spend cap permits."""
    return max(0.0, interval_seconds) * max(0, max_touches)


def should_offload_cold(
    idle_seconds: float,
    interval_seconds: float,
    max_touches: int,
    at_session_boundary: bool,
    rederivation_tokens: float,
    econ: ColdTierEconomics,
) -> Tier:
    """Pick the cost-minimising tier for one block over one idle gap.

    Encodes the disjoint ``tau``-bands above. COLD is returned ONLY where it
    net-saves; WARM/HOT/NONE everywhere it would lose. The single rule that makes
    COLD honest: it fires only when ``cold_net_vs_rederivation`` is strictly
    positive AND no cheaper live-session tier (HOT/WARM) covers the gap.
    """
    cold_wins = econ.cold_net_vs_rederivation(rederivation_tokens) > 0

    # Session boundary: transcript is gone. The real alternatives are
    # RE-DERIVATION or loss — there is no do-nothing baseline. COLD iff it beats
    # re-deriving; otherwise nothing can be done.
    if at_session_boundary:
        return Tier.COLD if cold_wins else Tier.NONE

    # Live session, short gap: cache is still warm. Do nothing.
    if idle_seconds < interval_seconds:
        return Tier.HOT

    # Live session, WARM's band: a resume is a cheap 0.10x read. COLD here would
    # pay 1.25x to page back what WARM keeps warm for ~0.10x — forbidden.
    if idle_seconds <= _warm_band_max_seconds(interval_seconds, max_touches):
        return Tier.WARM

    # Live session, overnight gap past the cap: WARM can no longer help (cache
    # expires anyway -> do-nothing pays a 1.25x re-write). COLD competes against
    # that re-write via avoided re-derivation; take it only if it wins.
    return Tier.COLD if cold_wins else Tier.NONE
