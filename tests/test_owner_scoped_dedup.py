"""
P1 owner-scoped SQL dedup tests (plans/odysseus-reuse-2026-10-06.md).

Layers:
1. Pure text_hash owner-mixing cases (no DB).
2. Real in-memory libSQL contract tests (same pattern as test_turso_stores.py):
   two owners' identical text+type must stay TWO rows — the pre-P1 defect was
   that ON CONFLICT (text_hash, memory_type) mutated user A's row and remapped
   user B's id. Also: legacy rows coexist, same-owner re-upsert dedups, and
   bulk_upsert preserves exported hashes (owner-mixed hashes are not
   recoverable from text alone).
3. mergeDuplicates cross-owner guard (monkeypatched Null store, dry run).

Run: python -m pytest tests/test_owner_scoped_dedup.py -v
"""
import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# repo root first: memory_core does `from tools.shared.server_factory import ...`
# (namespace package) — same recipe as test_memory_autouse.py.
for p in (str(REPO), str(REPO / "tools"), str(REPO / "tools" / "memorymcp")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.hashing import text_hash

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

_libsql_required = pytest.mark.skipif(
    not HAS_LIBSQL, reason="libsql_experimental not installed"
)


@pytest.fixture
def turso_sql_store():
    from shared.impls.turso_sql import TursoSqlStore
    store = TursoSqlStore(url="file::memory:")
    yield store


# ---------------------------------------------------------------------------
# pure hash cases
# ---------------------------------------------------------------------------

class TestTextHashOwner:
    def test_none_owner_is_exactly_the_legacy_hash(self):
        # golden: sha256 of stripped+lowercased text — the pre-P1 format
        expected = hashlib.sha256("hello world".encode()).hexdigest()[:40]
        assert text_hash("Hello World") == expected
        assert text_hash("  Hello World  ") == expected
        assert text_hash("Hello World", None) == expected

    def test_owner_mixed_hash_differs_from_legacy(self):
        assert text_hash("same text", "alice") != text_hash("same text")
        assert text_hash("same text", "alice") != text_hash("same text", "bob")
        assert text_hash("same text", "alice") == text_hash("same text", "alice")

    def test_owner_hash_still_normalizes_whitespace_and_case(self):
        assert text_hash("  Same TEXT ", "alice") == text_hash("same text", "alice")

    def test_owner_hash_shape(self):
        h = text_hash("x", "user-42")
        assert len(h) == 40
        int(h, 16)  # hex


# ---------------------------------------------------------------------------
# two-owner contract (in-memory libSQL)
# ---------------------------------------------------------------------------

@_libsql_required
class TestTwoOwnerUpsert:
    def test_two_users_same_text_stay_two_rows(self, turso_sql_store):
        id_a = turso_sql_store.upsert_memory(
            "uuid-a", "prefer port 8401 for the pod UI", "decision",
            "agent_action", [], None, None, None, "low", "auto-delete",
            owner="alice",
        )
        id_b = turso_sql_store.upsert_memory(
            "uuid-b", "prefer port 8401 for the pod UI", "decision",
            "agent_action", [], None, None, None, "low", "auto-delete",
            owner="bob",
        )
        assert id_a == "uuid-a"
        assert id_b == "uuid-b"
        # both rows exist, each with its own text
        assert turso_sql_store.get_memory("uuid-a")["text"] == "prefer port 8401 for the pod UI"
        assert turso_sql_store.get_memory("uuid-b")["text"] == "prefer port 8401 for the pod UI"

    def test_same_owner_reupsert_still_dedups(self, turso_sql_store):
        id1 = turso_sql_store.upsert_memory(
            "uuid-1", "same text", "concept",
            "agent", [], None, None, None, "low", "auto-delete", owner="alice",
        )
        id2 = turso_sql_store.upsert_memory(
            "uuid-2", "same text", "concept",
            "agent", [], None, None, None, "low", "auto-delete", owner="alice",
        )
        assert id1 == id2 == "uuid-1"

    def test_legacy_row_coexists_with_owned_row(self, turso_sql_store):
        legacy_id = turso_sql_store.upsert_memory(
            "uuid-legacy", "shared sentence", "concept",
            "agent", [], None, None, None, "low", "auto-delete",
        )
        owned_id = turso_sql_store.upsert_memory(
            "uuid-owned", "shared sentence", "concept",
            "agent", [], None, None, None, "low", "auto-delete", owner="alice",
        )
        assert legacy_id == "uuid-legacy"
        assert owned_id == "uuid-owned"
        assert turso_sql_store.get_memory("uuid-legacy") is not None
        assert turso_sql_store.get_memory("uuid-owned") is not None

    def test_all_ids_and_bulk_upsert_lockstep(self, turso_sql_store):
        """Export → import must not re-dup: bulk_upsert prefers the exported
        hash (owner-mixed hashes are unrecoverable from text)."""
        turso_sql_store.upsert_memory(
            "uuid-a", "alice fact", "concept",
            "agent", [], None, None, None, "low", "auto-delete", owner="alice",
        )
        turso_sql_store.upsert_memory(
            "uuid-b", "alice fact", "concept",
            "agent", [], None, None, None, "low", "auto-delete", owner="bob",
        )
        exported = list(turso_sql_store.iter_all())
        assert len(exported) == 2
        hashes = {r["id"]: r["text_hash"] for r in exported}
        assert hashes["uuid-a"] != hashes["uuid-b"]

        # fresh store = simulated re-import on another backend
        from shared.impls.turso_sql import TursoSqlStore
        fresh = TursoSqlStore(url="file::memory:")
        fresh.bulk_upsert(exported)
        assert sorted(fresh.get_all_memory_ids()) == ["uuid-a", "uuid-b"]

    def test_bulk_upsert_without_hash_still_recomputes(self, turso_sql_store):
        """Rows lacking text_hash (hand-built, pre-P1 exports) behave as before."""
        count = turso_sql_store.bulk_upsert([
            {"id": "uuid-1", "text": "Hello World", "memory_type": "concept"},
        ])
        assert count == 1
        row = turso_sql_store.get_memory("uuid-1")
        assert row["text_hash"] == text_hash("Hello World")


# ---------------------------------------------------------------------------
# mergeDuplicates cross-owner guard
# ---------------------------------------------------------------------------

class _Point:
    def __init__(self, pid, payload, vector=None):
        self.id = pid
        self.payload = payload
        self.vector = vector


class _FakeVectorStore:
    def __init__(self, points):
        self._points = points

    def iter_all(self, collection, *, with_vectors=False):
        yield from self._points


def test_merge_duplicates_never_merges_cross_owner(monkeypatch):
    import memory_tools as mt

    text = "identical memory text from two users"
    points = [
        _Point("id-a", {"text": text, "memory_type": "concept", "owner": "alice", "usage_count": 1}),
        _Point("id-b", {"text": text, "memory_type": "concept", "owner": "bob", "usage_count": 1}),
    ]
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))

    result = asyncio.run(mt.mergeDuplicates(threshold=0.5, dry_run=True))
    # pre-P1 this pair scored 1.0 (identical text, jaccard fallback) and would merge
    assert "Would merge 0 duplicate pairs" in result
    assert "Cross-owner pairs skipped: 1" in result


def test_merge_duplicates_still_merges_same_owner(monkeypatch):
    import memory_tools as mt

    text = "identical memory text from one user"
    points = [
        _Point("id-a", {"text": text, "memory_type": "concept", "owner": "alice", "usage_count": 3}),
        _Point("id-b", {"text": text, "memory_type": "concept", "owner": "alice", "usage_count": 1}),
    ]
    monkeypatch.setattr(mt, "vector_store", _FakeVectorStore(points))

    result = asyncio.run(mt.mergeDuplicates(threshold=0.5, dry_run=True))
    assert "Would merge 1 duplicate pairs" in result


def test_mergeable_owners_helper():
    from text_utils import mergeable_owners
    assert mergeable_owners({"owner": "a"}, {"owner": "a"})
    assert mergeable_owners({}, {})           # both legacy
    assert mergeable_owners(None, None)
    assert not mergeable_owners({"owner": "a"}, {"owner": "b"})
    assert not mergeable_owners({"owner": "a"}, {})
    assert not mergeable_owners(None, {"owner": "b"})
