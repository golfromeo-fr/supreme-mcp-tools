"""
P3 hybrid recall tests (plans/odysseus-reuse-2026-10-06.md).

Layers:
1. Pure scorer helpers — normalize_keyword_rows (PG trgm native 0-1 vs
   Turso bm25 negative / ILIKE zero rank-normalization, OQ5), type_boost,
   blend_hybrid (kw_weight=0 must reproduce the pre-P3 score exactly).
2. queryMemory integration with faked stores: keyword-only candidates
   surface, kw_weight=0 skips the lane entirely, no-SQL degrades to the
   Jaccard signal and counts the fallback.

Run: python -m pytest tests/test_hybrid_recall.py -v
"""
import asyncio
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tools"), str(REPO / "tools" / "memorymcp")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.relevance_scorer import blend_hybrid, normalize_keyword_rows, type_boost


# Review fix 2026-10-18: _memory_caller fails closed (scoped "unknown" owner)
# for unresolvable identities under the user-store verifier. These integration
# tests exercise recall mechanics as an IDENTIFIED admin.
@pytest.fixture(autouse=True)
def _identified_admin(monkeypatch):
    import memory_tools as mt
    monkeypatch.setattr(mt, "_memory_caller", lambda: ("admin", True, True))


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

class TestNormalizeKeywordRows:
    def test_empty(self):
        assert normalize_keyword_rows([]) == {}

    def test_pg_trgm_native_scores_used_directly(self):
        rows = [{"id": "a", "sim_score": 0.42}, {"id": "b", "sim_score": 0.11}]
        assert normalize_keyword_rows(rows) == {"a": 0.42, "b": 0.11}

    def test_turso_bm25_negatives_rank_normalized(self):
        # bm25: more negative = better match; rows arrive best-first
        rows = [{"id": "best", "sim_score": -5.2}, {"id": "mid", "sim_score": -2.1},
                {"id": "worst", "sim_score": -0.3}]
        out = normalize_keyword_rows(rows)
        assert out["best"] == 1.0
        assert out["mid"] == pytest.approx(1 - 1 / 3)
        assert out["worst"] == pytest.approx(1 - 2 / 3)

    def test_ilike_all_zeros_rank_normalized(self):
        rows = [{"id": "x", "sim_score": 0.0}, {"id": "y", "sim_score": 0.0}]
        out = normalize_keyword_rows(rows)
        assert out == {"x": 1.0, "y": 0.5}

    def test_single_row_gets_full_score(self):
        assert normalize_keyword_rows([{"id": "only", "sim_score": -3.0}]) == {"only": 1.0}


class TestTypeBoost:
    def test_overlap_boosts_to_cap(self):
        assert type_boost("architectural_decision", "which decision did we make about ports") == pytest.approx(0.10)

    def test_no_overlap_is_zero(self):
        assert type_boost("architectural_decision", "how to chunk plsql files") == 0.0

    def test_unknown_type_uses_its_own_label(self):
        assert type_boost("lesson", "a lesson learned here") == pytest.approx(0.10)

    def test_cap_env_override(self, monkeypatch):
        monkeypatch.setenv("MEMORY_TYPE_BOOST_CAP", "0.25")
        assert type_boost("plan", "the plan for launch") == pytest.approx(0.25)

    def test_empty_inputs(self):
        assert type_boost(None, "query") == 0.0
        assert type_boost("plan", "") == 0.0


class TestBlendHybrid:
    def test_formula(self):
        assert blend_hybrid(0.8, 0.5, 0.25) == pytest.approx(0.75 * 0.8 + 0.25 * 0.5)

    def test_kw_weight_zero_is_identity(self):
        assert blend_hybrid(0.73, 0.9, 0.0) == pytest.approx(0.73)

    def test_clamping(self):
        assert blend_hybrid(1.5, -0.2, 0.5) == pytest.approx(0.5 * 1.5)
        assert blend_hybrid(0.5, 2.0, 0.5) == pytest.approx(0.25 + 0.5)


# ---------------------------------------------------------------------------
# queryMemory integration (faked stores)
# ---------------------------------------------------------------------------

class _Point:
    def __init__(self, pid, payload, vector=None, score=0.0):
        self.id = pid
        self.payload = payload
        self.vector = vector
        self.score = score


class _FakeVectorStore:
    def __init__(self, dense_points=None, stored=None):
        self.dense_points = dense_points or []
        self.stored = stored or {}   # id -> _Point (for retrieve)
        self.set_payload_calls = []

    def query_dense(self, collection, vec, *, limit=10, filter=None, using=None):
        return self.dense_points[:limit]

    def retrieve(self, collection, ids, *, with_payload=True, with_vectors=False):
        return [self.stored[i] for i in ids if i in self.stored]

    def set_payload(self, collection, payload, *, ids):
        self.set_payload_calls.append((payload, list(ids)))


class _FakeSqlStore:
    def __init__(self, available=True, rows=None):
        self.is_available = available
        self.rows = rows or []
        self.search_calls = []

    def search_text(self, query, limit=10, *, memory_type=None, tags=None, agent_id=None):
        self.search_calls.append(query)
        return self.rows


def _payload(pid, text, **extra):
    p = {"id": pid, "text": text, "memory_type": "concept", "source": "agent_action",
         "tags": [], "usage_count": 1, "created_at": "2026-10-01T00:00:00Z"}
    p.update(extra)
    return p


@pytest.fixture
def mt(monkeypatch):
    import memory_tools as memory_tools_mod
    monkeypatch.setattr(memory_tools_mod, "generate_embedding", lambda q: [1.0, 0.0])
    memory_tools_mod._recall_stats.update({"kw_lane_candidates": 0, "vector_only_fallback_count": 0})
    return memory_tools_mod


def test_kw_only_candidate_surfaces(mt, monkeypatch):
    vec = _FakeVectorStore(
        dense_points=[_Point("vec-1", _payload("vec-1", "unrelated vector hit"), vector=[1.0, 0.0], score=0.9)],
        stored={"kw-1": _Point("kw-1", _payload("kw-1", "the pinned port decision text"),
                               vector=[1.0, 0.0])},
    )
    sql = _FakeSqlStore(rows=[{"id": "kw-1", "sim_score": -4.0}])  # Turso bm25 shape
    monkeypatch.setattr(mt, "vector_store", vec)
    monkeypatch.setattr(mt, "sql_store", sql)

    result = asyncio.run(mt.queryMemory("pinned port decision", k=5, kw_weight=0.6))
    assert "pinned port decision text" in result  # kw-only candidate surfaced
    assert sql.search_calls  # lane was queried


def test_kw_weight_zero_skips_lane(mt, monkeypatch):
    vec = _FakeVectorStore(
        dense_points=[_Point("vec-1", _payload("vec-1", "some memory"), vector=[1.0, 0.0], score=0.8)],
    )
    sql = _FakeSqlStore(rows=[{"id": "kw-1", "sim_score": -1.0}])
    monkeypatch.setattr(mt, "vector_store", vec)
    monkeypatch.setattr(mt, "sql_store", sql)

    result = asyncio.run(mt.queryMemory("some memory", k=5, kw_weight=0.0))
    assert not sql.search_calls  # lane skipped entirely
    assert "some memory" in result


def test_no_sql_degrades_to_jaccard_and_counts_fallback(mt, monkeypatch):
    text = "launcher restart race pinned mgmt port"
    vec = _FakeVectorStore(
        dense_points=[_Point("vec-1", _payload("vec-1", text), vector=[1.0, 0.0], score=0.7)],
    )
    sql = _FakeSqlStore(available=False)
    monkeypatch.setattr(mt, "vector_store", vec)
    monkeypatch.setattr(mt, "sql_store", sql)

    result = asyncio.run(mt.queryMemory("launcher restart", k=5, kw_weight=0.5))
    assert "launcher restart" in result
    assert mt._recall_stats["vector_only_fallback_count"] == 1


def test_kw_lane_hits_counted(mt, monkeypatch):
    vec = _FakeVectorStore(
        dense_points=[_Point("vec-1", _payload("vec-1", "t1"), vector=[1.0, 0.0], score=0.9)],
        stored={"kw-1": _Point("kw-1", _payload("kw-1", "t2"), vector=[0.0, 1.0])},
    )
    sql = _FakeSqlStore(rows=[{"id": "kw-1", "sim_score": 0.55}])  # PG trgm shape
    monkeypatch.setattr(mt, "vector_store", vec)
    monkeypatch.setattr(mt, "sql_store", sql)

    asyncio.run(mt.queryMemory("t2 exact", k=5, kw_weight=0.5))
    assert mt._recall_stats["kw_lane_candidates"] == 1


def test_type_boost_applied_in_blend(mt, monkeypatch):
    # Same query, same kw_weight — the ONLY difference is memory_type.
    # decision overlaps the query tokens → +0.10 boost on the keyword side;
    # with kw_weight=0.5 the final score must rise by exactly 0.5·0.10.
    sql = _FakeSqlStore(rows=[{"id": "d-1", "sim_score": 0.6}])

    def run(mtype):
        vec = _FakeVectorStore(
            dense_points=[_Point("d-1", _payload("d-1", "use x", memory_type=mtype),
                                 vector=[1.0, 0.0], score=0.5)],
        )
        monkeypatch.setattr(mt, "vector_store", vec)
        out = asyncio.run(mt.queryMemory("which decision", k=5, kw_weight=0.5))
        return float(out.split("(score: ")[1].split(")")[0])

    boosted = run("decision")
    plain = run("concept")
    assert boosted == pytest.approx(plain + 0.05)


# ---------------------------------------------------------------------------
# M1 (review 2026-10-24): upsertMemory SQL-dedup must not leave a duplicate
# vector point under the fresh uuid.
# ---------------------------------------------------------------------------

class _RecordingVectorStore(_FakeVectorStore):
    def __init__(self):
        super().__init__()
        self.upserts = []

    def upsert(self, collection, points):
        for p in points:
            self.upserts.append(str(p.id))
            self.stored[str(p.id)] = p


class _DedupSqlStore(_FakeSqlStore):
    """upsert_memory always resolves to one fixed existing row (dedup hit)."""
    def __init__(self, existing_id):
        super().__init__()
        self.existing_id = existing_id
        self.calls = []

    def upsert_memory(self, memory_id, **kw):
        self.calls.append(memory_id)
        return self.existing_id


def test_upsert_dedup_no_duplicate_vector_point(mt, monkeypatch):
    vec = _RecordingVectorStore()
    sql = _DedupSqlStore(existing_id="existing-row-id")
    monkeypatch.setattr(mt, "vector_store", vec)
    monkeypatch.setattr(mt, "sql_store", sql)

    r1 = asyncio.run(mt.upsertMemory("identical memory text", memory_type="concept"))
    r2 = asyncio.run(mt.upsertMemory("identical memory text", memory_type="concept"))

    assert r1 == "existing-row-id" and r2 == "existing-row-id"
    # Every vector upsert lands on the RESOLVED id — the fresh uuids the old
    # order leaked never reach the store.
    assert set(vec.upserts) == {"existing-row-id"}
    assert len(vec.upserts) == 2


def test_upsert_vector_only_when_sql_fails(mt, monkeypatch):
    class _BrokenSql(_FakeSqlStore):
        def upsert_memory(self, memory_id, **kw):
            raise RuntimeError("sql backend down")

    vec = _RecordingVectorStore()
    monkeypatch.setattr(mt, "vector_store", vec)
    monkeypatch.setattr(mt, "sql_store", _BrokenSql())

    result = asyncio.run(mt.upsertMemory("memory text", memory_type="concept"))
    # Memory still stored vector-only (fail-open, no data loss) under its own id
    assert result not in ("Error",) and not str(result).startswith("Error")
    assert len(vec.upserts) == 1
