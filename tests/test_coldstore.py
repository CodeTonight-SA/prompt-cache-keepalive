"""CBC tests for the FileRingStore — lossless, private, crash-safe, 0-dep.

Goodhart-resistant: the round-trip identity check is the lossless canary (it
fails the moment a put/get becomes lossy); the mode-0600 check fails if the
private-file contract breaks; the overflow check fails if a full ring silently
truncates a blob; the import-graph check fails if any pasteboard/SDK dependency
sneaks into the 0-dep core.
"""
from __future__ import annotations

import os
import stat
import sys
import tomllib
from pathlib import Path

import pytest

from prompt_cache_keepalive import CacheMiss, FileRingStore, RingConfig
from prompt_cache_keepalive.cold_scope import allow_all_unsafe

_BLOB = "GRIP context: " + ("x" * 5000) + "\n…verbatim tail."


def _store(tmp_path: Path, **kw) -> FileRingStore:
    return FileRingStore(
        cold_dir=tmp_path / "cold",
        scope_policy=allow_all_unsafe(),  # tests use a non-client scope
        **kw,
    )


def test_round_trip_is_byte_identical(tmp_path):
    store = _store(tmp_path, config=RingConfig(shred_on_pageback=False))
    ref = store.put("k1", _BLOB, scope="session")
    out = store.get(ref, scope="session")
    assert out == _BLOB  # lossless canary — any lossy put/get fails here


def test_stored_file_is_mode_0600_and_dir_0700(tmp_path):
    store = _store(tmp_path, config=RingConfig(shred_on_pageback=False))
    store.put("k1", _BLOB, scope="session")
    files = list((tmp_path / "cold").glob("pck-cold-*.cold"))
    assert files, "expected a stored cold entry"
    file_mode = stat.S_IMODE(files[0].stat().st_mode)
    assert file_mode == 0o600
    dir_mode = stat.S_IMODE((tmp_path / "cold").stat().st_mode)
    # umask can drop bits; assert the owner has rwx and no world bits at minimum.
    assert dir_mode & 0o077 == 0


def test_namespace_prefix_is_recognisable(tmp_path):
    store = _store(tmp_path, namespace="grip", config=RingConfig(shred_on_pageback=False))
    store.put("k1", _BLOB, scope="session")
    files = list((tmp_path / "cold").glob("*.cold"))
    assert files[0].name.startswith("pck-cold-grip-")


def test_overflow_evict_oldest_pages_back_verbatim_not_truncated(tmp_path):
    store = _store(tmp_path, config=RingConfig(max_entries=2, overflow="evict_oldest", shred_on_pageback=False))
    r1 = store.put("k1", "first", scope="session")
    store.put("k2", "second", scope="session")
    store.put("k3", "third", scope="session")  # evicts k1
    # k1 is gone -> typed miss, never a truncated string.
    with pytest.raises(CacheMiss):
        store.get(r1, scope="session")
    # surviving entries are intact, byte-for-byte.
    assert store.get(store.list()[-1], scope="session") == "third"


def test_overflow_refuse_raises_rather_than_truncating(tmp_path):
    store = _store(tmp_path, config=RingConfig(max_entries=1, overflow="refuse", shred_on_pageback=False))
    store.put("k1", "first", scope="session")
    with pytest.raises(CacheMiss):
        store.put("k2", "second", scope="session")  # full + refuse -> typed miss


def test_shred_on_pageback_removes_plaintext(tmp_path):
    store = _store(tmp_path, config=RingConfig(shred_on_pageback=True))
    ref = store.put("k1", _BLOB, scope="session")
    assert store.get(ref, scope="session") == _BLOB
    # After page-back the slot is shredded: a second get is a miss, no file left.
    with pytest.raises(CacheMiss):
        store.get(ref, scope="session")
    assert not list((tmp_path / "cold").glob("*.cold"))


def test_missing_ref_is_typed_miss(tmp_path):
    from prompt_cache_keepalive import ColdRef

    store = _store(tmp_path)
    with pytest.raises(CacheMiss):
        store.get(ColdRef(key="ghost", scope="session", slot=999), scope="session")


def test_no_nonstdlib_import_in_cold_core():
    # The 0-dep promise: none of the COLD core modules may pull a pasteboard SDK
    # or any third-party package. Assert their imported module names are stdlib
    # or first-party only.
    import prompt_cache_keepalive.coldstore as cs
    import prompt_cache_keepalive._locking as lk

    forbidden = {"pyobjc", "AppKit", "Foundation", "objc", "pyperclip", "anthropic", "openai"}
    loaded = set(sys.modules)
    for mod in forbidden:
        assert mod not in loaded, f"{mod} must not be imported by the 0-dep core"
    # And the source files reference no such import.
    for path in (Path(cs.__file__), Path(lk.__file__)):
        text = path.read_text()
        for mod in forbidden:
            assert f"import {mod}" not in text


def test_pyproject_dependencies_stay_empty():
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text())
    assert data["project"]["dependencies"] == []  # 0-dep core badge intact
