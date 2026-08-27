"""
Cross-backend contract tests for the VectorStore / SqlStore abstraction.

These tests exist because the per-impl tests (test_turso_stores.py, etc.) pass
in isolation but missed divergences in how the tools consume the stores. Each
test below corresponds to a real bug that shipped and was fixed:

  set_payload_merge       — C1: Turso/PG overwrote instead of merging
  set_payload_shallow     — C1b: Turso json_patch deep-merged nested objects;
                            now json_set → shallow, matching Qdrant/PG
  iter_all_with_vectors   — C2: Turso dropped vectors on export
  scroll_pagination       — H1: Turso truncated collections > limit
  list_collections_str    — C3: ragmcp called .name on a str
  query_sparse_query_text — M5: sparse search matched nothing on Turso/PG
  delete_memory_fts       — M1: Turso orphaned FTS5 rows on delete

The Turso paths run against in-memory libSQL (no network, no containers).
Postgres paths auto-run when a Postgres is configured (via the standard
``POSTGRES_*`` vars or ``POSTGRES_TEST_DSN``) AND reachable; otherwise they
skip cleanly. The reachability probe lives in the session-scoped ``pg_dsn``
fixture in ``conftest.py`` — so the same contract is enforced automatically
in CI/local whenever PG is up, with no per-test env-var dance.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from shared.store_models import (
    PointStruct, SparseVector, Filter, FieldCondition, MatchValue,
)

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

try:
    import psycopg  # noqa: F401
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

# PG reachability is resolved at runtime by the `pg_dsn` session fixture in
# conftest.py (DSN auto-derived from POSTGRES_* vars or POSTGRES_TEST_DSN, then
# probed). The PG tests skip via that fixture — there is no module-level gate.


# ---------------------------------------------------------------------------
# Fixtures: yield a freshly-initialized VectorStore per backend
# ---------------------------------------------------------------------------

@pytest.fixture
def turso_vec():
    from shared.impls.turso_vector import TursoVectorStore
    return TursoVectorStore(url="file::memory:")


@pytest.fixture
def pg_vec(pg_dsn):
    """PostgresVectorStore against a real DB. Auto-runs when PG is reachable.

    ``pg_dsn`` (session fixture in conftest.py) resolves a DSN from the standard
    POSTGRES_* vars (or POSTGRES_TEST_DSN) and probes it; if PG isn't configured
    or is down, the test is skipped here rather than failing the suite.

    Each test gets an isolated collection name so tests don't collide; tables
    are created via ensure_collection and dropped in teardown.
    """
    if pg_dsn is None:
        pytest.skip(
            "Postgres not configured or unreachable "
            "(set POSTGRES_HOST/USER/... or POSTGRES_TEST_DSN in .env)"
        )
    from shared.impls.postgres_vector import PostgresVectorStore
    store = PostgresVectorStore(dsn=pg_dsn)
    created = []
    yield store, created
    for name in created:
        try:
            store.delete_collection(name)
        except Exception:
            pass


VEC_BACKENDS = []
if HAS_LIBSQL:
    VEC_BACKENDS.append("turso")
if HAS_PSYCOPG:
    VEC_BACKENDS.append("pg")


def _make_vec(backend, pg_fixture):
    """Resolve a (store, created_list) pair for parametrized vector tests."""
    if backend == "turso":
        from shared.impls.turso_vector import TursoVectorStore
        return TursoVectorStore(url="file::memory:"), []
    elif backend == "pg":
        return pg_fixture
    pytest.skip(f"unknown backend {backend}")


DIM = 4


# ---------------------------------------------------------------------------
# C1: set_payload must merge, not replace
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_set_payload_merges_turso(turso_vec):
    """set_payload must merge — keys absent from the patch survive."""
    turso_vec.ensure_collection("c1", dense_dim=DIM)
    turso_vec.upsert("c1", [
        PointStruct(id="p1", vector=[0.1] * DIM, payload={"text": "hello", "usage_count": 0}),
    ])
    # Partial update — must NOT wipe "text"
    turso_vec.set_payload("c1", {"usage_count": 5}, ids=["p1"])
    pts = turso_vec.retrieve("c1", ["p1"], with_payload=True)
    assert pts[0].payload == {"text": "hello", "usage_count": 5}


@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_set_payload_shallow_replaces_nested_turso(turso_vec):
    """Nested object values must be REPLACED (shallow), not deep-merged.

    Qdrant set_payload and PG jsonb ``||`` replace a top-level object value
    wholesale. SQLite ``json_patch`` would deep-merge it — so the Turso impl uses
    ``json_set`` for bareword keys to stay shallow. This guards that behaviour:
    'meta' is overwritten entirely ('b' gone), while the sibling 'keep' survives.
    """
    turso_vec.ensure_collection("csh", dense_dim=DIM)
    turso_vec.upsert("csh", [
        PointStruct(id="p1", vector=[0.1] * DIM, payload={"meta": {"a": 1, "b": 2}, "keep": "me"}),
    ])
    turso_vec.set_payload("csh", {"meta": {"c": 3}}, ids=["p1"])
    pts = turso_vec.retrieve("csh", ["p1"], with_payload=True)
    assert pts[0].payload == {"meta": {"c": 3}, "keep": "me"}
    # 'b' must NOT have been deep-merged back in
    assert "b" not in pts[0].payload["meta"]


@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_set_payload_shallow_multi_key_turso(turso_vec):
    """A multi-key shallow patch preserves untouched keys and replaces each listed one."""
    turso_vec.ensure_collection("cmulti", dense_dim=DIM)
    turso_vec.upsert("cmulti", [
        PointStruct(id="p1", vector=[0.1] * DIM,
                    payload={"a": 1, "b": 2, "c": 3, "keep": "yes"}),
    ])
    turso_vec.set_payload("cmulti", {"a": 10, "c": 30}, ids=["p1"])
    pts = turso_vec.retrieve("cmulti", ["p1"], with_payload=True)
    assert pts[0].payload == {"a": 10, "b": 2, "c": 30, "keep": "yes"}


def test_set_payload_merges_pg(pg_vec):
    store, created = pg_vec
    created.append("c1pg")
    store.ensure_collection("c1pg", dense_dim=DIM)
    store.upsert("c1pg", [
        PointStruct(id="p1", vector=[0.1] * DIM, payload={"text": "hello", "usage_count": 0}),
    ])
    store.set_payload("c1pg", {"usage_count": 5}, ids=["p1"])
    pts = store.retrieve("c1pg", ["p1"], with_payload=True)
    assert pts[0].payload["text"] == "hello"
    assert pts[0].payload["usage_count"] == 5


# ---------------------------------------------------------------------------
# C2: iter_all must honor with_vectors (migration exports vectors)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_iter_all_with_vectors_turso(turso_vec):
    """iter_all(with_vectors=True) must round-trip the embedding (was always [])."""
    turso_vec.ensure_collection("c2", dense_dim=DIM)
    vec = [0.2, 0.4, 0.6, 0.8]
    turso_vec.upsert("c2", [
        PointStruct(id="p1", vector=vec, payload={"text": "x"}),
    ])
    rows = list(turso_vec.iter_all("c2", with_vectors=True))
    assert len(rows) == 1
    assert len(rows[0].vector) == DIM
    for a, b in zip(rows[0].vector, vec):
        assert a == pytest.approx(b, abs=1e-6)

    # And the default (with_vectors=False) still yields empty vectors
    rows_nv = list(turso_vec.iter_all("c2"))
    assert rows_nv[0].vector == []


def test_iter_all_with_vectors_pg(pg_vec):
    store, created = pg_vec
    created.append("c2pg")
    store.ensure_collection("c2pg", dense_dim=DIM)
    vec = [0.2, 0.4, 0.6, 0.8]
    store.upsert("c2pg", [PointStruct(id="p1", vector=vec, payload={"text": "x"})])
    rows = list(store.iter_all("c2pg", with_vectors=True))
    assert len(rows) == 1
    assert len(rows[0].vector) == DIM


# ---------------------------------------------------------------------------
# H1: scroll must paginate (was always returning next_offset=None → truncation)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_scroll_paginates_turso(turso_vec):
    """scroll must return next_offset so a loop retrieves every point."""
    turso_vec.ensure_collection("c3", dense_dim=DIM)
    for i in range(25):
        turso_vec.upsert("c3", [
            PointStruct(id=f"p{i}", vector=[float(i)] * DIM, payload={"i": i}),
        ])

    collected = []
    offset = None
    while True:
        batch, offset = turso_vec.scroll("c3", limit=10, offset=offset)
        collected.extend(batch)
        if not offset or not batch:
            break
    # All 25 must be retrieved (was truncated to 10 before the fix)
    assert len(collected) == 25
    ids = {p.id for p in collected}
    assert ids == {f"p{i}" for i in range(25)}


# ---------------------------------------------------------------------------
# C3: list_collections returns list[str] (ragmcp called .name on these)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_list_collections_returns_strings(turso_vec):
    """list_collections returns plain strings; reverse-maps sanitized names."""
    turso_vec.ensure_collection("my.collection-name", dense_dim=DIM)
    colls = turso_vec.list_collections()
    assert all(isinstance(c, str) for c in colls)
    # Sanitized name round-trips to the original (dots/hyphens preserved)
    assert "my.collection-name" in colls


# ---------------------------------------------------------------------------
# M5: query_sparse with query_text must actually match (was always empty)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_query_sparse_matches_via_query_text_turso(turso_vec):
    """query_sparse(query_text=...) runs a real FTS5 MATCH and returns hits."""
    turso_vec.ensure_collection("c5", dense_dim=DIM, sparse=True)
    turso_vec.upsert("c5", [
        PointStruct(id="p1", vector=[0.1] * DIM, payload={"codeChunk": "python flask web framework"}),
        PointStruct(id="p2", vector=[0.1] * DIM, payload={"codeChunk": "rust systems programming language"}),
    ])
    empty_sparse = SparseVector(indices=[], values=[])
    results = turso_vec.query_sparse(
        "c5", empty_sparse, limit=5, query_text="python",
    )
    assert len(results) >= 1
    assert results[0].id == "p1"
    # score is similarity (higher = better) — must be positive for a match
    assert results[0].score > 0


@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_query_sparse_text_key_agnostic_turso(turso_vec):
    """Index population reads 'text' (memorymcp) OR 'codeChunk' (ragmcp)."""
    turso_vec.ensure_collection("c5b", dense_dim=DIM, sparse=True)
    turso_vec.upsert("c5b", [
        PointStruct(id="m1", vector=[0.1] * DIM, payload={"text": "memorymcp uses the text key"}),
    ])
    empty_sparse = SparseVector(indices=[], values=[])
    results = turso_vec.query_sparse("c5b", empty_sparse, limit=5, query_text="memorymcp")
    assert len(results) == 1
    assert results[0].id == "m1"


def test_query_sparse_matches_via_query_text_pg(pg_vec):
    store, created = pg_vec
    created.append("c5pg")
    store.ensure_collection("c5pg", dense_dim=DIM, sparse=True)
    store.upsert("c5pg", [
        PointStruct(id="p1", vector=[0.1] * DIM, payload={"codeChunk": "python flask web framework"}),
    ])
    empty_sparse = SparseVector(indices=[], values=[])
    results = store.query_sparse("c5pg", empty_sparse, limit=5, query_text="python flask")
    assert len(results) >= 1
    assert results[0].id == "p1"


# ---------------------------------------------------------------------------
# M1: delete_memory must clean up FTS5 (no orphan rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def turso_sql():
    from shared.impls.turso_sql import TursoSqlStore
    return TursoSqlStore(url="file::memory:")


@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_delete_memory_removes_from_fts(turso_sql):
    """After delete, the memory must not appear in search_text results."""
    mid = turso_sql.upsert_memory(
        "uuid-del", "unique searchable phrase zebra", "concept",
        "agent", [], None, None, "a", "low", "auto-delete",
    )
    # Confirm it's findable before delete
    hits = turso_sql.search_text("zebra")
    assert any(h["id"] == mid for h in hits)

    assert turso_sql.delete_memory(mid) is True

    # After delete, it must be gone from FTS search
    hits_after = turso_sql.search_text("zebra")
    assert not any(h["id"] == mid for h in hits_after)
    # And gone from the table
    assert turso_sql.get_memory(mid) is None


# ---------------------------------------------------------------------------
# N4: the inline schema fallback stays in sync with the .sql file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")
def test_inline_schema_matches_sql_file():
    """The _INLINE_SCHEMA fallback must define the same columns as the .sql."""
    from shared.impls.turso_sql import _INLINE_SCHEMA
    sql_path = Path(__file__).resolve().parent.parent / "tools" / "shared" / "impls" / "turso_sql_schema.sql"
    assert sql_path.exists(), "turso_sql_schema.sql must ship alongside the module"
    file_sql = sql_path.read_text()
    # Both must define the same core CREATE TABLE and FTS5 virtual table
    for marker in ("CREATE TABLE IF NOT EXISTS memories", "memories_fts USING fts5", "UNIQUE(text_hash, memory_type)"):
        assert marker in _INLINE_SCHEMA, f"_INLINE_SCHEMA missing: {marker}"
        assert marker in file_sql, f"turso_sql_schema.sql missing: {marker}"


# ---------------------------------------------------------------------------
# Dedup helper sanity (guards the extracted escape_fts5_query)
# ---------------------------------------------------------------------------

def test_escape_fts5_query():
    from shared.text_search_utils import escape_fts5_query
    # Each term is quoted with internal quotes doubled (FTS5 escaping rule),
    # terms joined with OR — BM25 recall, not whole-string phrase
    assert escape_fts5_query("hello") == '"hello"'
    assert escape_fts5_query('say "hi"') == '"say" OR """hi"""'
    # Operator words are inert quoted terms, not FTS5 operators (no injection)
    assert escape_fts5_query("a OR b") == '"a" OR "OR" OR "b"'
    assert escape_fts5_query("") == '""'
    assert escape_fts5_query("   ") == '""'
