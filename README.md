# prompt-cache-keepalive

**Your LLM prompt cache has a 5-minute memory. Give it a heartbeat.**

[![CI](https://github.com/CodeTonight-SA/prompt-cache-keepalive/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeTonight-SA/prompt-cache-keepalive/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)
[![docs](https://img.shields.io/badge/docs-GitHub%20Pages-5ec8a0.svg)](https://codetonight-sa.github.io/prompt-cache-keepalive/)

Provider-agnostic keepalive for LLM prompt caches — with a cost model that
**proves** the savings instead of asserting them.

> **The core insight.** Autonomous loops should pace **sub-270s wakes** (or a
> keepalive touch) during idle. The Anthropic prompt cache has a **~5-minute
> (300s) TTL**, so a wake under **~270s** keeps the prefix warm and the next
> turn *reads* cache instead of *re-creating* it. Keepalive is worth it whenever
> the session's **resume probability > 17.4%** (see [the breakeven](#it-proves-it-pays--and-tells-you-when-it-doesnt)).

[**Documentation site →**](https://codetonight-sa.github.io/prompt-cache-keepalive/)

```bash
pip install prompt-cache-keepalive
```

---

## The 5-minute problem

Modern LLM APIs let you **cache a long conversation prefix**: pay full price
once, then ~0.1× on every later turn. But the cache has a short **idle TTL** —
Anthropic's is **~5 minutes, sliding**. Step away for a coffee, and the cached
prefix is **evicted on the provider's servers**. Your next turn reprocesses the
*entire* prefix at full price.

```
  without keepalive                 with keepalive
  ─────────────────                 ──────────────
  turn  ─[ cache written ]          turn  ─[ cache written ]
         · idle 6 min                      ♥ touch  (every 270s)
         ✗ evicted                         ♥ touch   prefix stays warm
  resume ████ full reprocess         resume ─[ cache read ]
         ~110,000 tok-equiv                 ~8,800 tok-equiv
```

A touch is a **minimal request that re-uses the cached prefix**. A cache *read*
refreshes the TTL — so the prefix never goes cold and your next real turn is
cheap.

## 60-second start

```python
from prompt_cache_keepalive import PromptCacheKeepalive, KeepaliveConfig, TouchResult

# You inject the touch — the library never imports an SDK (see examples/).
def touch() -> TouchResult:
    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1,
        system=SYSTEM,                                   # last block: cache_control
        messages=PREFIX + [{"role": "user", "content": "."}],
    )
    u = resp.usage
    return TouchResult(ok=True, cache_read_tokens=u.cache_read_input_tokens,
                       output_tokens=u.output_tokens)

keepalive = PromptCacheKeepalive(touch, KeepaliveConfig(ttl_seconds=300, margin_seconds=30))
keepalive.run(is_active=lambda: session_still_open())    # touches every 270s while active
```

Full Anthropic wiring → [`examples/anthropic_keepalive.py`](examples/anthropic_keepalive.py).

## It proves it pays — and tells you when it doesn't

A touch is **not free** (~0.1× the prefix). This package ships a cost model so
you can see exactly when keepalive wins. Costs are in *base-input-token-
equivalents* (Anthropic: cache-write 1.25×, cache-read 0.10×).

```python
from prompt_cache_keepalive import CacheEconomics, net_savings, breakeven_resume_probability

econ = CacheEconomics(prefix_tokens=88_000)
net_savings(econ, idle_seconds=540, interval_seconds=270, max_touches=12, resume_probability=1.0)
# -> 83_600.0   saved on a 9-minute idle that resumes
breakeven_resume_probability(econ, 540, 270, 12)
# -> 0.174      keepalive wins whenever the session is >17% likely to resume
```

| Idle gap | Touches | Resumes? | Net (88k prefix) |
|---|---|---|---|
| 9 min | 2 | yes | **+83,600 tok** |
| 9 min | 2 | no | −17,600 tok (wasted touches) |
| 67 min (past cap) | 12 | yes | **−105,600 tok** (honest loss) |

The library **caps touches** (`max_touches`, default 12) near the breakeven
against a single re-write, so a session that goes quiet for an hour can't bleed
tokens forever — it stops and lets the cache expire. No silent waste.

### "Millions of tokens" — the honest at-scale math

A platform with 10,000 daily coding sessions, ~5 short coffee-break idles each
over an 88k cached prefix, mostly resuming:

```
10,000 × 5 × ~83,600 saved  ≈  4.18 billion token-equivalents / day
```

Even at 1% addressable, ~42M token-equivalents a day. The win is real *because*
it's bounded and conditional — the cost model is the proof, not the marketing.

## Why it's provider-agnostic (and trivially testable)

The core never imports an LLM SDK. You inject a `touch_fn`; the library owns
only the **timing** and the **spend bound**:

- works with any provider whose cache TTL is refreshed by re-use;
- unit-tested with **zero network** (inject a fake `touch_fn` + `sleep`);
- no secret handling inside the library.

## Prefix compaction (complementary)

The keepalive prevents most misses. For the ones that slip through — first run,
keepalive not yet started, TTL tighter than expected — `compact_prefix` cheapens
the miss by shrinking what the provider must reprocess.

It keeps the system prompt and the most-recent K turns **byte-identical** (so
the provider can cache them immediately) and replaces older turns with a compact
deterministic digest. A typical 88k-token prefix compacts to ~20k tokens — a
**4x reduction** on the miss cost.

```python
from prompt_cache_keepalive import compact_prefix, CompactionResult

result: CompactionResult = compact_prefix(
    messages,           # full conversation history
    system=SYSTEM,      # kept verbatim; helps cache re-warm after miss
    keep_recent_turns=6,
)
# result.ratio  => ~4x or better on an 88k prefix
# result.compacted  => send this to the provider on the next request
```

`compact_prefix` is **pure-stdlib and network-free**. The optional
`llm_summariser` callback (default: deterministic head/tail digest) can be
replaced with any callable that returns a summary string — injection makes it
trivially testable.

**What compaction does not do:** it does not prevent the miss (that is the
keepalive's job) and it does not warm the cache by itself. Run both: keepalive
during idle, compaction as the fallback when a miss happens anyway.

## The COLD tier — a private local offload for long idles (new in 0.3)

Keepalive (WARM) is the right tool for **short** idles — a coffee break, where
pinging beats re-writing. But what about an **overnight** gap, or a session you
`/clear` and resume tomorrow? WARM can't help: past its spend cap the cache
expires anyway. That is what the **COLD tier** is for.

**Three tiers, one cost-minimiser:**

| Tier | What it is | Pays | Best for |
|---|---|---|---|
| **HOT** | the live context window | ~0.1× cached prefix per turn | active conversation |
| **WARM** | keepalive-pinged server cache | ~0.1× per ping | short idle (coffee break) |
| **COLD** | context evicted to a **private local store**, paged back **verbatim** | **0 provider tokens to store** | long/overnight idle + session boundaries |

```python
from prompt_cache_keepalive import (
    ColdTierEconomics, should_offload_cold, Tier, FileRingStore, deny_unknown,
)

# Which tier wins for THIS idle gap? (returns COLD only where it net-saves)
tier = should_offload_cold(
    idle_seconds=40_000, interval_seconds=270, max_touches=12,
    at_session_boundary=False, rederivation_tokens=200_000,
    econ=ColdTierEconomics(prefix_tokens=100_000),
)
# -> Tier.COLD   (overnight gap on expensive-to-regenerate context)

# The store is a stdlib, 0-dependency, mode-0600 file ring — never the system clipboard.
store = FileRingStore(scope_policy=deny_unknown())
ref = store.put("session-prefix", durable_context, scope="grip")   # redacted + scope-gated
restored = store.get(ref, scope="grip")                            # byte-identical page-back
```

### The honest framing (read this — it is the whole point)

> **"Zero token cost" is true only for the disk write/read leg.**

A common misframing is "offload context to disk and page it back for free". That
is a **category error**: a provider has exactly two prefix price tiers — a cache
*write* (1.25×) and a cache *read* (0.10×) — and **disk is not a provider pricing
tier**. Bytes paged back off disk into a live request re-enter as fresh tokens
and bill as a cache **write**. There is no sub-0.10× tier.

So COLD's real, defensible win is **avoided re-derivation**, not cheap page-back:
the cost of *regenerating* context that is expensive to rebuild (re-reading
files, re-running tools, re-thinking) and must be **verbatim** (a lossy summary
won't do). `should_offload_cold` returns `COLD` **only** in that regime —
`WARM`/`HOT`/`NONE` everywhere COLD would lose. It inverts to a loss on
cheap-to-rebuild context, and the API says so by returning a negative from
`cold_net_vs_rederivation`.

**COLD vs compaction:** orthogonal. COLD is *lossless* (relocates bytes
off-wire); compaction is *lossy* (~4×, cheapens a miss you still pay). A mature
loop runs both — offload durable-idle blocks to COLD, compact whatever still
rides the live request.

### Claude Code plugin (session-boundary offload)

A thin plugin wires the COLD tier to Claude Code's `SessionStart` / `Stop` /
`PreCompact` hooks (`.claude-plugin/plugin.json`). Its **honest scope is
session-boundary offload only**:

- **It CAN**: losslessly offload durable context to the private store at a
  boundary, and page it back **verbatim** on the next session — surviving
  `/clear` and overnight idle at zero disk-leg token cost. Lossless where Claude
  Code's native compaction is lossy.
- **It CANNOT**: shrink Claude Code's *live* context window mid-turn. The harness
  owns its own request loop and cache breakpoints; hooks fire **around** the loop,
  never inside a turn. A restore re-injects bytes as **new** bottom-of-window
  tokens (a fresh cache write), not an in-place restore.

So the plugin's measured value is **re-derivation-avoided + lossless
continuity**, explicitly **not** "lossless restore of the live window" and **not**
mid-session token saving.

### Privacy & safety (non-negotiable)

The COLD store is a private mode-0600 file under a mode-0700 dir — **never the
general system clipboard** (which cross-device-syncs via Universal Clipboard and
is readable by every clipboard manager: a leak vector for context, secrets, and
client data). Two fail-closed gates run before any write:

1. **Scope allowlist** (load-bearing) — a session tagged client / financial /
   named-org is **denied** COLD persistence entirely. Unshaped client data (a
   bare client name, a day-rate, decision prose) has no secret *shape* a regex
   could catch, so the sound control is **exclusion**, not redaction.
2. **Secret redaction** (defence-in-depth) — canonical secret shapes (`sk-…`,
   `ghp_…`, `AKIA…`, PEM keys) and keychain locators are scrubbed before write;
   a redaction error **refuses** the write (never fail-open).

Reads are scope-checked too (a ref minted under one scope can't be read under
another), entries shred on page-back (no plaintext-at-rest), and a native named
pasteboard backend can be **injected** later via the `ColdStore` protocol without
touching the 0-dependency core.

### Prove it pays — exogenously

`scripts/dogfood-cold-tier.py` is a flag-gated A/B that measures the win via
Anthropic's own usage fields (`cache_creation_input_tokens` etc.) — comparing
**total billed tokens** of *(evict → COLD → verbatim restore)* against *(evict →
re-derive via the LLM)*. It reports COLD a win **iff** the restore arm bills
fewer tokens — never by counting page-back tokens in isolation (which would
falsify COLD by construction).

## Honest limitations

- **It prevents the miss; it doesn't shrink the prefix.** Pair it with
  `compact_prefix` (above) to cheapen the misses that do slip through.
  Different layers; use both.
- **It only helps if the session resumes.** Below the breakeven resume
  probability, don't run it — the model gives you the threshold.
- **It works *within* the TTL, not *around* it.** There's no client-side way to
  lengthen the window itself; if your provider offers a longer-cache tier,
  prefer that.

## For autonomous agents / harnesses

If you control the request loop (an agent, a scheduler), the same idea is a
**sub-TTL wake**: schedule the next iteration `< (ttl − margin)` seconds out so
the loop itself re-uses the cached prefix and never crosses eviction.
`KeepaliveConfig.interval()` gives you that number.

## API at a glance

| Symbol | Purpose |
|---|---|
| `PromptCacheKeepalive(touch_fn, config, sleep)` | the loop: touch on a timer, bounded by the spend cap |
| `KeepaliveConfig(ttl_seconds, margin_seconds, max_touches, ...)` | timing + spend bounds; `.interval()` = when to touch |
| `TouchResult(ok, cache_read_tokens, ...)` | what your `touch_fn` returns |
| `CacheEconomics(prefix_tokens, cache_write_mult, cache_read_mult)` | the price model |
| `net_savings(...)`, `breakeven_resume_probability(...)` | prove (or disprove) the win for your numbers |
| `ColdTierEconomics(...)`, `should_offload_cold(...)`, `Tier` | the 3-tier (HOT/WARM/COLD) cost-minimiser; COLD only where it net-saves |
| `FileRingStore(...)`, `ColdStore`, `RingConfig`, `ColdRef` | the private, lossless, 0-dep local COLD store (mode-0600 file ring) |
| `deny_unknown(...)`, `default_redactor(...)` | fail-closed scope allowlist + secret redaction run before every COLD write |

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 95 tests, network-free
```

### Open risks (COLD tier — tracked, not hidden)

- **Networked `$HOME` (NFS/SMB):** POSIX advisory locks are documented-unreliable
  over networked mounts without a working lock daemon. The file ring's
  concurrency is sound on **local** filesystems; it does not claim safety on a
  networked home.
- **`restore_fraction` is a tuning knob, not yet benchmarked:** the break-even
  inverts on cheap-to-rebuild context; only the live dogfood across real sessions
  settles the right default.
- **Double-pay hazard:** if a restored COLD prefix is injected *and* the model
  then re-reads the source anyway, you pay restore + re-derivation. Treat a
  paged-back block as authoritative (a prompt-framing contract, not a code
  guarantee).

## License

MIT — see [LICENSE](LICENSE).
