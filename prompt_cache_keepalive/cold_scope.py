"""Scope allowlist — the LOAD-BEARING control for unshaped client data.

Why this, not redaction, is primary
-----------------------------------
A verbatim COLD dump of a client-scoped session is dominated by data with NO
secret shape: proper-noun client names, bare-integer day-rates, matter
references, decision prose. A regex redactor cannot catch that class (it is
recall-bounded). The sound control is EXCLUSION: deny the whole scope rather than
try to scrub its content. (GRIP's own ``no-sensitive-financial-in-grip`` rule
mandates exclusion, not redaction, for exactly this reason.)

Fail-closed by construction
---------------------------
A :class:`ColdScopePolicy` is ``is_cold_safe(scope) -> bool``. The default
:func:`deny_unknown` policy DENIES any scope carrying a client / financial /
named-org marker AND denies anything it does not positively recognise as safe.
Better to forgo the token saving than persist client bytes to a local store.

In the GRIP shim (deliverable B) this binds to ``lib/channel_scope_gate.py`` so
real client-scope detection drives the verdict; the OSS default takes an injected
``is_cold_safe`` predicate.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Pattern

# Markers that, if present in a scope tag, mean "client/sensitive — never COLD".
# Generic risk CATEGORIES only — present in every deployment, never a specific
# client or product name. A deployment's own private names load at runtime from
# PROMPT_CACHE_DENYLIST_FILE instead (see ``_private_deny_markers`` below); they
# never belong in this public source file.
_DENY_MARKERS: List[Pattern[str]] = [
    re.compile(r"\bclient\b", re.IGNORECASE),
    re.compile(r"\bfinanc", re.IGNORECASE),       # financ(e|ial)
    re.compile(r"\binvoice\b", re.IGNORECASE),
    re.compile(r"\bmatter\b", re.IGNORECASE),      # legal matter
    re.compile(r"\bnda\b", re.IGNORECASE),
]


def _private_deny_markers() -> List[Pattern[str]]:
    """Additional deny markers from a private, out-of-repo file (optional).

    ``PROMPT_CACHE_DENYLIST_FILE`` names a text file, one bare word per line
    (blank lines and ``#`` comments ignored). Each word compiles to the same
    ``\\bword\\b`` case-insensitive shape as the built-in markers above, so a
    deployment can deny its own client/product names without ever writing them
    into this shipped source file.

    Absent env var, or a missing/unreadable file, yields no private markers —
    the generic built-in list above still applies either way; this layer is
    additive, never required for the fail-closed default to hold."""
    path = os.environ.get("PROMPT_CACHE_DENYLIST_FILE")
    if not path:
        return []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    markers = []
    for line in lines:
        word = line.strip()
        if not word or word.startswith("#"):
            continue
        markers.append(re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
    return markers


@dataclass(frozen=True)
class ColdScopePolicy:
    """Fail-closed allowlist deciding whether a scope may be COLD-persisted."""

    is_cold_safe: Callable[[str], bool]


def _has_deny_marker(scope: str) -> bool:
    return any(p.search(scope) for p in _DENY_MARKERS + _private_deny_markers())


@dataclass(frozen=True)
class _AllowlistChecker:
    """Deny-on-unknown checker: a scope is safe only if it is on ``safe_scopes``
    AND carries no deny marker. Anything unrecognised is denied."""

    safe_scopes: frozenset

    def __call__(self, scope: str) -> bool:
        if _has_deny_marker(scope):
            return False
        return scope in self.safe_scopes


def deny_unknown(safe_scopes: frozenset = frozenset({"default", "grip", "hal", "session"})) -> ColdScopePolicy:
    """Default policy: allow only explicitly-safe scopes; deny everything else.

    A scope is COLD-safe iff it is in *safe_scopes* and carries no client/
    financial/named-org marker. Unknown scope -> denied (fail-closed)."""
    return ColdScopePolicy(is_cold_safe=_AllowlistChecker(safe_scopes=safe_scopes))


def allow_all_unsafe() -> ColdScopePolicy:
    """ESCAPE HATCH for tests / single-user non-sensitive use ONLY.

    Allows every scope EXCEPT those carrying a deny marker. Never use this where
    client data may flow — the deny markers are a backstop, not an allowlist."""
    return ColdScopePolicy(is_cold_safe=lambda scope: not _has_deny_marker(scope))
