"""Audit 2026-10-01 P1-4: postgres_vector must sanitize collection identifiers.

The turso twin has enforced `_s()` since the backend abstraction work; the PG
impl interpolated collection names straight into f-string SQL. These tests
pin the ported sanitizer and prove the public methods reject a hostile name
BEFORE any connection/pool access.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from shared.impls.postgres_vector import PostgresVectorStore, _s  # noqa: E402
from shared.impls.turso_vector import _s as _turso_s  # noqa: E402


def test_s_normalizes_separators_like_turso():
    for name in ("fastmcp-code", "folder.to.index-database-code",
                 "odysseus-live-sweep-local-bge-m3-1024", "plain"):
        assert _s(name) == _turso_s(name), name


@pytest.mark.parametrize("bad", [
    "a; DROP TABLE vec_x",
    "a b",
    "x'y",
    "",
    "col; --",
    'col"',
])
def test_s_rejects_non_barewords(bad):
    with pytest.raises(ValueError):
        _s(bad)


def _bare_store() -> PostgresVectorStore:
    """An instance whose pool was never opened — _s must raise first."""
    return object.__new__(PostgresVectorStore)


@pytest.mark.parametrize("method,args,kwargs", [
    ("upsert", ("evil; DROP TABLE x", []), {}),
    ("set_payload", ("evil; DROP TABLE x", {}), {"ids": ["id"]}),
    ("delete", ("evil; DROP TABLE x",), {}),
    ("delete_collection", ("evil; DROP TABLE x",), {}),
    ("query_dense", ("evil; DROP TABLE x", [0.1, 0.2]), {}),
    ("retrieve", ("evil; DROP TABLE x", ["id"]), {}),
    ("scroll", ("evil; DROP TABLE x",), {}),
    ("get_collection", ("evil; DROP TABLE x",), {}),
    ("iter_all", ("evil; DROP TABLE x",), {}),
    ("bulk_upsert", ("evil; DROP TABLE x", []), {}),
])
def test_public_methods_reject_hostile_names_before_pool(method, args, kwargs):
    with pytest.raises(ValueError, match="invalid collection name"):
        getattr(_bare_store(), method)(*args, **kwargs)


def test_ensure_collection_rejects_hostile_name():
    with pytest.raises(ValueError, match="invalid collection name"):
        _bare_store().ensure_collection("evil; DROP TABLE x", dense_dim=1024)
