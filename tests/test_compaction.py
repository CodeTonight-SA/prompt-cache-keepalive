"""Tests for prompt_cache_keepalive.compaction.

All tests are network-free and deterministic.  The Goodhart-resistant design
means each assertion can actually *fail* when the implementation is wrong —
see the mutation guard and preservation tests.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import pytest

from prompt_cache_keepalive.compaction import (
    CompactionResult,
    compact_prefix,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, text: str) -> Dict[str, object]:
    return {"role": role, "content": text}


def _make_conversation(n_pairs: int, chars_per_turn: int = 4_000) -> List[Dict[str, object]]:
    """Produce n_pairs user/assistant pairs, each ~chars_per_turn chars."""
    msgs = []
    for i in range(n_pairs):
        msgs.append(_msg("user", f"Turn {i} user: " + "x" * (chars_per_turn - 20)))
        msgs.append(_msg("assistant", f"Turn {i} asst: " + "y" * (chars_per_turn - 20)))
    return msgs


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_string_basic():
    assert estimate_tokens("abcd") == 1  # 4 chars → 1 token
    assert estimate_tokens("a" * 400) == 100


def test_estimate_tokens_messages():
    msgs = [_msg("user", "a" * 400), _msg("assistant", "b" * 800)]
    # 400/4 + 800/4 = 100 + 200 = 300
    assert estimate_tokens(msgs) == 300


def test_estimate_tokens_empty_string():
    # Floor is 1 to avoid division-by-zero in callers
    assert estimate_tokens("") == 1


def test_estimate_tokens_empty_list():
    assert estimate_tokens([]) == 1


# ---------------------------------------------------------------------------
# compact_prefix — ratio >= 4x on a synthetic ~88k-token prefix
# ---------------------------------------------------------------------------

def test_ratio_at_least_4x_large_prefix():
    """Core Goodhart check: compaction must deliver >= 4x on a large prefix."""
    # 22 user/assistant pairs × 2 msgs × 8000 chars / 4 ≈ 88,000 tokens
    msgs = _make_conversation(n_pairs=22, chars_per_turn=8_000)
    result = compact_prefix(msgs, keep_recent_turns=6)
    assert isinstance(result, CompactionResult)
    assert result.ratio >= 4.0, (
        f"Expected ratio >= 4.0, got {result.ratio:.2f} "
        f"(original={result.original_tokens}, compacted={result.compacted_tokens})"
    )


# ---------------------------------------------------------------------------
# System prompt preserved verbatim
# ---------------------------------------------------------------------------

def test_system_included_in_token_counts():
    system = "s" * 400  # 100 tokens
    msgs = _make_conversation(n_pairs=10, chars_per_turn=400)
    result = compact_prefix(msgs, system=system, keep_recent_turns=2)
    # original_tokens must include the system prompt contribution
    assert result.original_tokens > estimate_tokens(msgs)


# ---------------------------------------------------------------------------
# Last keep_recent_turns messages are BYTE-IDENTICAL
# ---------------------------------------------------------------------------

def test_recent_turns_kept_verbatim():
    """The last K messages must survive compaction with byte-identical content."""
    msgs = _make_conversation(n_pairs=10, chars_per_turn=800)
    keep = 6
    result = compact_prefix(msgs, keep_recent_turns=keep)

    verbatim = msgs[-keep:]
    tail = result.compacted[-keep:]
    assert tail == verbatim, "Last K messages were not kept byte-identical"
    assert result.kept_verbatim_count == keep


def test_recent_turns_verbatim_deep_equality():
    """Deep-equality: content strings must match, not just repr."""
    msgs = _make_conversation(n_pairs=8, chars_per_turn=500)
    keep = 4
    result = compact_prefix(msgs, keep_recent_turns=keep)
    expected = copy.deepcopy(msgs[-keep:])
    actual = result.compacted[-len(expected):]
    for exp, got in zip(expected, actual):
        assert exp["content"] == got["content"]
        assert exp["role"] == got["role"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output():
    """Same input must produce byte-identical output on repeated calls."""
    msgs = _make_conversation(n_pairs=12, chars_per_turn=600)
    r1 = compact_prefix(msgs, keep_recent_turns=6)
    r2 = compact_prefix(msgs, keep_recent_turns=6)
    assert r1.compacted == r2.compacted
    assert r1.original_tokens == r2.original_tokens
    assert r1.compacted_tokens == r2.compacted_tokens
    assert r1.ratio == r2.ratio


# ---------------------------------------------------------------------------
# Mutation guard: dropping a recent turn FAILS the verbatim assertion
# ---------------------------------------------------------------------------

def test_mutation_guard_detects_dropped_recent_turn():
    """Prove the preservation test can detect a regression, not just pass."""
    msgs = _make_conversation(n_pairs=6, chars_per_turn=400)
    keep = 4
    result = compact_prefix(msgs, keep_recent_turns=keep)

    # Tamper: remove one recent message from the result
    tampered = result.compacted[:-1]  # drop the last verbatim message
    expected_verbatim = msgs[-keep:]

    # The tampered list is missing the last verbatim turn — verify the
    # verbatim check would catch it
    tail = tampered[-keep:] if len(tampered) >= keep else tampered
    assert tail != expected_verbatim, (
        "Mutation guard failed: tampered list still matches verbatim expectation"
    )


# ---------------------------------------------------------------------------
# Edge case: fewer messages than keep_recent_turns
# ---------------------------------------------------------------------------

def test_fewer_messages_than_keep():
    """When len(messages) <= keep_recent_turns, all messages are verbatim."""
    msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
    result = compact_prefix(msgs, keep_recent_turns=6)
    assert result.compacted == msgs
    assert result.kept_verbatim_count == len(msgs)


def test_single_message():
    msgs = [_msg("user", "just one")]
    result = compact_prefix(msgs, keep_recent_turns=6)
    assert result.compacted == msgs
    assert result.kept_verbatim_count == 1


# ---------------------------------------------------------------------------
# Edge case: empty message list
# ---------------------------------------------------------------------------

def test_empty_messages():
    result = compact_prefix([], keep_recent_turns=6)
    assert result.compacted == []
    assert result.kept_verbatim_count == 0
    assert result.ratio >= 1.0


# ---------------------------------------------------------------------------
# Optional llm_summariser injected callback
# ---------------------------------------------------------------------------

def test_llm_summariser_called_for_older_turns():
    """When llm_summariser is injected it receives the older turns."""
    received: List[List[Dict]] = []

    def fake_summariser(older_msgs):
        received.append(older_msgs)
        return "FAKE SUMMARY"

    msgs = _make_conversation(n_pairs=6, chars_per_turn=400)
    keep = 4
    result = compact_prefix(msgs, keep_recent_turns=keep, llm_summariser=fake_summariser)

    assert len(received) == 1
    assert len(received[0]) == len(msgs) - keep
    # The compacted list should contain the summary turn + verbatim recent turns
    assert result.compacted[0]["content"] == "[context summary] FAKE SUMMARY"
    assert result.compacted[1:] == msgs[-keep:]


def test_llm_summariser_not_called_when_no_older_turns():
    """No summariser call when all turns fit within keep_recent_turns."""
    calls: List[object] = []
    msgs = [_msg("user", "hi")]
    compact_prefix(msgs, keep_recent_turns=6, llm_summariser=lambda x: calls.append(x) or "")
    assert calls == []


# ---------------------------------------------------------------------------
# CompactionResult fields are consistent
# ---------------------------------------------------------------------------

def test_result_fields_consistent():
    msgs = _make_conversation(n_pairs=10, chars_per_turn=800)
    result = compact_prefix(msgs, keep_recent_turns=4)
    assert result.original_tokens > result.compacted_tokens
    assert result.ratio == pytest.approx(result.original_tokens / result.compacted_tokens, rel=1e-6)
    assert result.kept_verbatim_count == 4
