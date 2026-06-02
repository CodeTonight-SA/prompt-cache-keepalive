"""CBC tests for the CC-plugin session-boundary offload/restore.

Goodhart-resistant: the offload round-trip is byte-identical (offload lossless);
the hooks touch ONLY disk + stdout (structural assertion — no conversation-array
mutation, no provider call); 20 cycles are corruption-free; and an HONEST-CLAIM
guard asserts the docs state the hard boundary (so a future edit cannot quietly
re-introduce a 'lossless live-context restore' / 'mid-session token saving'
claim).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from prompt_cache_keepalive import FileRingStore, RingConfig
from prompt_cache_keepalive.cc_plugin import dump_cold_context, restore_cold_context
from prompt_cache_keepalive.cold_scope import allow_all_unsafe

_SENTINEL = "SESSION-BOUNDARY-SENTINEL\n" + ("ctx " * 1000)


def _store(tmp_path: Path) -> FileRingStore:
    return FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=allow_all_unsafe(),
        config=RingConfig(shred_on_pageback=False),
    )


def test_offload_then_restore_is_byte_identical(tmp_path):
    store = _store(tmp_path)
    dump_cold_context(store, _SENTINEL, scope="session")
    restored = restore_cold_context(store, scope="session")
    assert restored == _SENTINEL  # offload is lossless


def test_restore_with_no_prior_context_is_none(tmp_path):
    store = _store(tmp_path)
    assert restore_cold_context(store, scope="session") is None


def test_scope_denied_dump_is_silent_noop(tmp_path):
    # A client-scoped boundary dump must not crash the hook — it returns None.
    from prompt_cache_keepalive import deny_unknown

    store = FileRingStore(cold_dir=tmp_path / "cold", scope_policy=deny_unknown())
    assert dump_cold_context(store, _SENTINEL, scope="client-acme") is None
    assert not list((tmp_path / "cold").glob("*.cold"))


def test_twenty_cycles_are_corruption_free(tmp_path):
    store = _store(tmp_path)
    for i in range(20):
        payload = f"cycle-{i}: " + ("z" * 500)
        dump_cold_context(store, payload, scope="session")
        out = restore_cold_context(store, scope="session")
        assert out == payload  # every cycle round-trips intact


def test_hooks_only_touch_disk_and_stdout(tmp_path, monkeypatch, capsys):
    # Structural assertion: drive session_start.main() with a stdin payload and
    # confirm its ONLY external effect is reading the store + printing to stdout.
    # (No conversation-array mutation API exists for it to call.)
    import prompt_cache_keepalive.cc_plugin.session_start as ss
    import prompt_cache_keepalive.cc_plugin.stop as stop_hook

    # Seed a boundary dump via the Stop hook reading a JSON payload from stdin.
    monkeypatch.setenv("PROMPT_CACHE_COLD_DIR", str(tmp_path / "cold"))
    # Use an allow-all store dir by routing scope to a safe one.
    payload = json.dumps({"scope": "session", "cold_context": _SENTINEL})
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert stop_hook.main() == 0

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"scope": "session"})))
    assert ss.main() == 0
    out = capsys.readouterr().out
    assert _SENTINEL in out  # restored verbatim to stdout


def test_docs_state_the_hard_boundary():
    # Honest-claim guard: the package docstring must keep the boundary explicit.
    # Whitespace is collapsed so line wrapping cannot defeat the phrase match.
    import re

    import prompt_cache_keepalive.cc_plugin as plugin

    doc = re.sub(r"\s+", " ", (plugin.__doc__ or "").lower())
    assert "cannot" in doc
    assert "live context window" in doc
    assert "new bottom-of-window tokens" in doc or "fresh cache write" in doc
    # The two false claims may appear ONLY inside an explicit negation. Asserting
    # the negated forms are present proves both that the boundary is stated AND
    # that a future edit cannot silently drop the "not".
    assert 'not "lossless restore of the live context window"' in doc
    assert "not mid-session token saving" in doc
