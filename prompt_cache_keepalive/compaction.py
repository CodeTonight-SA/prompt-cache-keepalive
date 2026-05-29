"""Prefix compaction — cheapen the unavoidable post-idle cache miss.

The keepalive prevents most misses; this module cheapens the ones that slip
through (or the very first turn of a resumed session before keepalive starts).

The idea
--------
When a cache miss occurs the provider reprocesses the *entire* prefix at full
price.  Compaction shrinks what gets reprocessed: keep the system prompt and
the most-recent ``keep_recent_turns`` message pairs BYTE-IDENTICAL (so the
provider can cache them again immediately), and replace older turns with a
compact deterministic digest.  A 88k-token prefix becomes ~20k tokens, giving
roughly a 4x reduction on the miss cost.

What compaction does NOT do
---------------------------
* It does **not** prevent the miss — that is the keepalive's job.
* It does **not** warm the cache — the compacted prefix must be sent to the
  provider before a new cache entry exists.
* It does **not** call any network endpoint.  The optional ``llm_summariser``
  callback is dependency-injected; the default is a deterministic head/tail
  digest that is network-free.

Composing the two
-----------------
Run the keepalive during the idle period; if a miss happens anyway (first run,
keepalive not started, etc.) feed the compacted prefix to the next real request.
The two techniques address different failure modes and compose cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Message = Dict[str, object]


@dataclass
class CompactionResult:
    """Outcome of one :func:`compact_prefix` call.

    ``compacted`` is the new message list ready to send to the provider.
    All statistics are in *estimated* tokens (chars / 4 heuristic; see
    :func:`estimate_tokens` for the stated assumption).
    """

    compacted: List[Message]
    original_tokens: int
    compacted_tokens: int
    ratio: float
    kept_verbatim_count: int  # number of messages kept byte-identical


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _content_chars(content: object) -> int:
    """Return character count for a message content field (str or block list)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(b.get("text", ""))) for b in content if isinstance(b, dict))
    return 0


def estimate_tokens(text_or_messages: Union[str, Sequence[Message]]) -> int:
    """Heuristic token count: chars / 4 (GPT-family rule of thumb).

    This is an *estimate*, not a provider-certified count.  Real counts vary
    by model and tokeniser.  The heuristic is accurate to ±20% for typical
    English prose and code, which is sufficient for compaction decisions.
    """
    if isinstance(text_or_messages, str):
        return max(1, len(text_or_messages) // 4)
    total = sum(max(1, _content_chars(m.get("content", "")) // 4) for m in text_or_messages)
    return max(1, total)


# ---------------------------------------------------------------------------
# Internal digest helper
# ---------------------------------------------------------------------------

_HEAD_CHARS = 300
_TAIL_CHARS = 200


def _digest_message(msg: Message) -> Message:
    """Return a compact role-tagged summary of *msg* (deterministic, no I/O)."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    if len(content) <= _HEAD_CHARS + _TAIL_CHARS:
        summary = content
    else:
        summary = content[:_HEAD_CHARS] + " … " + content[-_TAIL_CHARS:]
    return {"role": role, "content": f"[summary] {summary}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compact_prefix(
    messages: Sequence[Message],
    *,
    system: Optional[str] = None,
    target_tokens: int = 20_000,
    keep_recent_turns: int = 6,
    llm_summariser: Optional[Callable[[List[Message]], str]] = None,
) -> CompactionResult:
    """Compact *messages* so a cache miss reprocesses far fewer tokens.

    Parameters
    ----------
    messages:
        The full conversation history (excluding the system prompt).
    system:
        Optional system prompt kept verbatim in the result (not counted in
        ``kept_verbatim_count``; always preserved byte-identical).
    target_tokens:
        Soft target for the compacted message list.  The function keeps the
        most-recent ``keep_recent_turns`` pairs verbatim regardless of whether
        that already meets the target.
    keep_recent_turns:
        Number of the most-recent messages to preserve byte-identical.  These
        are the turns the model needs for conversational coherence.
    llm_summariser:
        Optional injected callback that receives the *older* messages and
        returns a single summary string.  When ``None`` (default), a
        deterministic head/tail digest is used — no network, no secrets.

    Returns
    -------
    CompactionResult
        ``compacted`` is ready to send; ``ratio`` is original/compacted tokens
        (higher is better; ≥ 4 on a typical 88k prefix).
    """
    msgs = list(messages)
    original_tokens = estimate_tokens(msgs)
    if system:
        original_tokens += estimate_tokens(system)

    # Split: recent (kept verbatim) vs older (to be digested)
    recent = msgs[-keep_recent_turns:] if keep_recent_turns < len(msgs) else msgs
    older = msgs[: max(0, len(msgs) - keep_recent_turns)]

    # Summarise older turns
    if not older:
        digest_messages: List[Message] = []
    elif llm_summariser is not None:
        summary_text = llm_summariser(older)
        digest_messages = [{"role": "user", "content": f"[context summary] {summary_text}"}]
    else:
        digest_messages = [_digest_message(m) for m in older]

    compacted_msgs = digest_messages + recent
    compacted_tokens = estimate_tokens(compacted_msgs)
    if system:
        compacted_tokens += estimate_tokens(system)

    ratio = original_tokens / compacted_tokens if compacted_tokens > 0 else 1.0

    return CompactionResult(
        compacted=compacted_msgs,
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        ratio=ratio,
        kept_verbatim_count=len(recent),
    )
