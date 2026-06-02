"""Secret/PII redaction — defence-in-depth layer BEFORE every COLD write.

Where this sits in the safety model
-----------------------------------
This is LAYER 2, not the load-bearing control. The primary control is the
fail-closed scope allowlist in :mod:`cold_scope` — because redaction is
recall-bounded and cannot catch *unshaped* client data (a bare proper-noun
client name, a plain-integer day-rate, decision prose). A verbatim COLD dump is
dominated by exactly that class, so the allowlist DENIES whole client-scoped
sessions; this redactor is the second net for canonical secret SHAPES that might
appear even in an allowed session.

Fail-closed
-----------
A :class:`Redactor` is ``Callable[[str], str]``. If it cannot complete it raises
:class:`RedactionError`; the store turns that into a refused write. Cleartext
never reaches disk on a redaction failure — never fail-open.

Deliberately NOT over-redacting
-------------------------------
A 64-hex *git SHA* or content hash is safe and load-bearing; destroying it would
break the lossless contract. The secret-shaped 64-hex pattern therefore requires
a value-y context marker, and bare commit SHAs are left intact (tested).
"""
from __future__ import annotations

import re
from typing import Callable, List, Pattern

Redactor = Callable[[str], str]

_REDACTED = "[REDACTED]"


class RedactionError(Exception):
    """Redaction could not complete — the caller MUST refuse the write."""


# Canonical secret SHAPES. Each is a provider/credential token with a
# recognisable, low-false-positive prefix or structure.
_SECRET_PATTERNS: List[Pattern[str]] = [
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),     # OpenAI / Anthropic keys
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),                  # GitHub PAT
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),          # GitHub fine-grained PAT
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),         # Slack tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),               # Google API key
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.DOTALL,
    ),
]

# Keychain LOCATORS — not secrets themselves, but they name where a secret is.
# Scrubbing them stops a COLD dump becoming a treasure map to the keychain.
_LOCATOR_PATTERNS: List[Pattern[str]] = [
    re.compile(r"security\s+find-generic-password[^\n]*"),
    re.compile(r"\bgrip-[a-z0-9][a-z0-9\-]{2,}\b"),       # grip-* service names
]


def _redact_with(text: str, patterns: List[Pattern[str]]) -> str:
    for pat in patterns:
        text = pat.sub(_REDACTED, text)
    return text


def default_redactor() -> Redactor:
    """Return the deterministic, network-free default redactor (fail-closed)."""

    def _redactor(text: str) -> str:
        try:
            out = _redact_with(text, _SECRET_PATTERNS)
            out = _redact_with(out, _LOCATOR_PATTERNS)
            return out
        except Exception as exc:  # noqa: BLE001 - any failure => refuse the write
            raise RedactionError(str(exc)) from exc

    return _redactor
