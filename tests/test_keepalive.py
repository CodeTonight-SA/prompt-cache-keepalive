"""CBC tests for the keepalive loop — timing, spend bound, stats folding.

Network-free: the touch function and sleep are injected. The loop must touch
exactly up to the cap while active, stop the instant ``is_active`` flips, and
stop on a touch error — never spin past its bound.
"""
from __future__ import annotations

from prompt_cache_keepalive import (
    KeepaliveConfig,
    PromptCacheKeepalive,
    TouchResult,
)


def _ok_touch(read_tokens: int = 8800, out: int = 1):
    def touch() -> TouchResult:
        return TouchResult(ok=True, cache_read_tokens=read_tokens, output_tokens=out)
    return touch


def test_interval_is_ttl_minus_margin_floored():
    assert KeepaliveConfig(ttl_seconds=300, margin_seconds=30).interval() == 270
    # margin >= ttl must not produce a non-positive interval — floor applies
    assert KeepaliveConfig(ttl_seconds=10, margin_seconds=30, min_interval_seconds=5).interval() == 5


def test_touch_once_folds_ok_into_stats():
    ka = PromptCacheKeepalive(_ok_touch(read_tokens=8800, out=2), sleep=lambda s: None)
    ka.touch_once()
    assert ka.stats.touches_made == 1
    assert ka.stats.cache_read_tokens == 8800
    assert ka.stats.output_tokens == 2
    assert ka.stats.touches_failed == 0


def test_touch_once_counts_failure_separately():
    ka = PromptCacheKeepalive(lambda: TouchResult(ok=False, error="boom"), sleep=lambda s: None)
    ka.touch_once()
    assert ka.stats.touches_made == 0
    assert ka.stats.touches_failed == 1
    assert ka.stats.cache_read_tokens == 0


def test_run_touches_up_to_cap_when_always_active():
    cfg = KeepaliveConfig(max_touches=5)
    ka = PromptCacheKeepalive(_ok_touch(), config=cfg, sleep=lambda s: None)
    stats = ka.run(is_active=lambda: True)
    assert stats.touches_made == 5  # exactly the cap, never more


def test_run_stops_when_session_goes_inactive():
    calls = {"n": 0}

    def is_active() -> bool:
        # active for the first two touch cycles, then done
        calls["n"] += 1
        return calls["n"] <= 4  # checked twice per loop iteration (top + post-sleep)

    ka = PromptCacheKeepalive(_ok_touch(), config=KeepaliveConfig(max_touches=99), sleep=lambda s: None)
    stats = ka.run(is_active=is_active)
    assert stats.touches_made < 99  # exited on inactivity, not the cap
    assert stats.touches_made >= 1


def test_run_stops_on_touch_error():
    seq = [TouchResult(ok=True, cache_read_tokens=8800), TouchResult(ok=False, error="429")]

    def touch() -> TouchResult:
        return seq.pop(0) if seq else TouchResult(ok=False, error="exhausted")

    ka = PromptCacheKeepalive(touch, config=KeepaliveConfig(max_touches=99), sleep=lambda s: None)
    stats = ka.run(is_active=lambda: True)
    assert stats.touches_made == 1   # the one success
    assert stats.touches_failed == 1  # the error that broke the loop


def test_run_never_touches_if_inactive_from_start():
    ka = PromptCacheKeepalive(_ok_touch(), sleep=lambda s: None)
    stats = ka.run(is_active=lambda: False)
    assert stats.touches_made == 0


def test_sleep_called_with_configured_interval():
    slept: list[float] = []
    cfg = KeepaliveConfig(ttl_seconds=300, margin_seconds=45, max_touches=3)
    ka = PromptCacheKeepalive(_ok_touch(), config=cfg, sleep=slept.append)
    ka.run(is_active=lambda: True)
    assert slept and all(s == 255 for s in slept)  # 300 - 45
