"""CBC tests for the scope allowlist — the LOAD-BEARING fail-closed control.

Goodhart-resistant: a client/financial/named-org scope MUST be denied a COLD
write (the unshaped-data class redaction cannot catch); an unknown scope MUST
default to deny (fail-closed); a planted client-scope marker on a store.put MUST
abort. A mutation that flips deny->allow, or makes unknown default to allow,
fails the suite.
"""
from __future__ import annotations

import pytest

from prompt_cache_keepalive import allow_all_unsafe, deny_unknown
from prompt_cache_keepalive.coldstore import FileRingStore, RingConfig

POLICY = deny_unknown()


@pytest.mark.parametrize(
    "scope",
    [
        "client",
        "financials",
        "client-acme",
        "matter-1234",
        "nda-review",
        "invoice-batch",
    ],
)
def test_client_or_financial_scopes_are_denied(scope):
    assert POLICY.is_cold_safe(scope) is False


def test_private_denylist_file_adds_a_marker_at_runtime(tmp_path, monkeypatch):
    # Without PROMPT_CACHE_DENYLIST_FILE set, an arbitrary non-generic scope
    # carries none of the built-in markers, so the backstop allows it.
    monkeypatch.delenv("PROMPT_CACHE_DENYLIST_FILE", raising=False)
    assert allow_all_unsafe().is_cold_safe("examplecorp-session") is True

    # Pointing the env var at a private file naming "examplecorp" extends the
    # deny set at runtime -- a deployment's own client/product names never
    # need to live in this shipped source file.
    denylist = tmp_path / "private-denylist.txt"
    denylist.write_text("# comment lines and blanks are ignored\n\nexamplecorp\n")
    monkeypatch.setenv("PROMPT_CACHE_DENYLIST_FILE", str(denylist))
    assert allow_all_unsafe().is_cold_safe("examplecorp-session") is False


def test_missing_private_denylist_file_falls_back_to_the_generic_default(monkeypatch):
    # An unset or dangling env var is a silent no-op: the generic built-in
    # categories still deny, and an ordinary scope is still allowed.
    monkeypatch.setenv("PROMPT_CACHE_DENYLIST_FILE", "/nonexistent/private-denylist.txt")
    assert allow_all_unsafe().is_cold_safe("financials") is False
    assert allow_all_unsafe().is_cold_safe("ordinary-session") is True


def test_unknown_scope_defaults_to_deny():
    assert POLICY.is_cold_safe("some-random-unrecognised-scope") is False


@pytest.mark.parametrize("scope", ["default", "grip", "hal", "session"])
def test_explicitly_safe_scopes_are_allowed(scope):
    assert POLICY.is_cold_safe(scope) is True


def test_store_refuses_client_scoped_write(tmp_path):
    # The allowlist is the primary control: a client-scoped session is DENIED a
    # COLD write outright — better to forgo the saving than persist client bytes.
    store = FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=deny_unknown(),
        config=RingConfig(shred_on_pageback=False),
    )
    with pytest.raises(PermissionError):
        store.put("k", "matter re: Acme Corp, day-rate 1200", scope="client-acme")
    assert not list((tmp_path / "cold").glob("*.cold"))


def test_safe_scope_write_succeeds(tmp_path):
    store = FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=deny_unknown(),
        config=RingConfig(shred_on_pageback=False),
    )
    ref = store.put("k", "ordinary grip session context", scope="grip")
    assert store.get(ref, scope="grip") == "ordinary grip session context"


def test_cross_scope_read_is_denied(tmp_path):
    # Channel-scope isolation holds at the cache layer: a ref minted under one
    # scope cannot be read under another.
    store = FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=deny_unknown(),
        config=RingConfig(shred_on_pageback=False),
    )
    ref = store.put("k", "grip ctx", scope="grip")
    with pytest.raises(PermissionError):
        store.get(ref, scope="hal")  # different scope -> denied
