# prompt-cache-keepalive

**Your LLM prompt cache has a 5-minute memory. Give it a heartbeat.**

[![CI](https://github.com/CodeTonight-SA/prompt-cache-keepalive/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeTonight-SA/prompt-cache-keepalive/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![dependencies: 0](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)

Provider-agnostic keepalive for LLM prompt caches — with a cost model that
**proves** the savings instead of asserting them.

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

## Honest limitations

- **It prevents the miss; it doesn't shrink the prefix.** A complementary
  technique — *pre-idle prefix compaction* — shrinks what you'd reprocess *if* a
  miss happens. Different layers; use both.
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

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 17 tests, network-free
```

## License

MIT — see [LICENSE](LICENSE).
