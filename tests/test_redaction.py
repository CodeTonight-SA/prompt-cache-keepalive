"""CBC tests for the secret-shape redactor — layer-2 defence, fail-closed.

Goodhart-resistant: per-class recall is asserted with planted secrets of each
shape (a mutation dropping a pattern fails); over-redaction is forbidden (a bare
git SHA must survive, or the lossless contract breaks); a redactor that raises
must REFUSE the write (the store turns it into a PermissionError), never write
cleartext.
"""
from __future__ import annotations

import pytest

from prompt_cache_keepalive import RedactionError, default_redactor
from prompt_cache_keepalive.coldstore import FileRingStore, RingConfig
from prompt_cache_keepalive.cold_scope import allow_all_unsafe

R = default_redactor()


# Fixtures are deliberately NON-credential placeholders ("EXAMPLE" / "NOTAREAL"
# segments) so they are genuinely not real tokens — yet still match the
# redactor's structural patterns. This is the sanctioned response to a secret
# scanner (make the value honestly fake), NEVER string-splitting to evade it.
@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-EXAMPLEEXAMPLEEXAMPLENOTAREAL",
        "sk-proj-EXAMPLENOTAREALEXAMPLE",
        "ghp_EXAMPLENOTAREALEXAMPLENOTAREAL00",
        "github_pat_11EXAMPLENOTAREAL_EXAMPLENOTAREALxx",
        "xoxb-EXAMPLE-EXAMPLE-NOTAREALTOKENEXAMPLE",
        "AKIAEXAMPLENOTAREAL00",
        "AIzaEXAMPLENOTAREALEXAMPLENOTAREALEXAMPLE0",
    ],
)
def test_canonical_secret_shapes_are_redacted(secret):
    out = R(f"here is the key {secret} use it")
    assert secret not in out
    assert "[REDACTED]" in out


def test_pem_private_key_block_is_redacted():
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ\n"
        "-----END PRIVATE KEY-----"
    )
    out = R(f"cert:\n{pem}\nend")
    assert "MIIEvQ" not in out
    assert "[REDACTED]" in out


def test_keychain_locator_is_scrubbed():
    out = R("token via security find-generic-password -s grip-sudonum-slack -w")
    assert "grip-sudonum-slack" not in out
    assert "find-generic-password" not in out


def test_bare_git_sha_is_NOT_over_redacted():
    # A 40-hex commit SHA and a 64-hex content hash are safe + load-bearing.
    # Destroying them would break losslessness — they must survive intact.
    sha40 = "a" * 40
    sha64 = "deadbeef" * 8
    text = f"commit {sha40} blob {sha64}"
    out = R(text)
    assert sha40 in out
    assert sha64 in out


def test_redaction_failure_refuses_the_write(tmp_path):
    # A redactor that raises must cause the store to refuse — no cleartext on disk.
    def boom(_text: str) -> str:
        raise RedactionError("boom")

    store = FileRingStore(
        cold_dir=tmp_path / "cold",
        redactor=boom,
        scope_policy=allow_all_unsafe(),
        config=RingConfig(shred_on_pageback=False),
    )
    with pytest.raises(PermissionError):
        store.put("k", "secret content sk-ant-xxxxxxxxxxxxxxxx", scope="session")
    assert not list((tmp_path / "cold").glob("*.cold"))  # nothing written


def test_redactor_writes_redacted_not_raw_to_store(tmp_path):
    store = FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=allow_all_unsafe(),
        config=RingConfig(shred_on_pageback=False),
    )
    ref = store.put("k", "key sk-ant-api03-SECRETSECRETSECRET1234 end", scope="session")
    out = store.get(ref, scope="session")
    assert "SECRETSECRET" not in out
    assert "[REDACTED]" in out
