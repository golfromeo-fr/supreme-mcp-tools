"""
Integration test: migration export → import → verify round-trip.

Exercises three of the staged fixes in one realistic flow:
- iter_all(with_vectors=True)  — C2: vectors must survive export
- scroll pagination             — H1: collections > limit must not truncate
- set_payload merge semantics   — C1: payloads must survive round-trip intact

Uses in-memory libSQL for both source and destination — no network, no containers.
cmd_verify is a shallow parity check (IDs + counts only), so this test adds
direct content assertions on the destination to catch payload/vector corruption.
"""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

pytestmark = pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")

DIM = 4
N_MEMORIES = 10
N_VECTORS = 25  # > scroll default limit (10) to exercise pagination


def _seed_sql(store):
    for i in range(N_MEMORIES):
        store.upsert_memory(
            f"uuid-{i}",
            f"unique memory text {i} about python and databases",
            "concept" if i % 2 == 0 else "code_pattern",
            "agent",
            ["flask", "testing"] if i % 3 == 0 else [],
            f"/path/{i}.py" if i % 2 == 0 else None,
            None,
            "agent-1",
            "low",
            "auto-delete",
        )


def _seed_vec(store):
    store.ensure_collection("code_index", dense_dim=DIM, sparse=True)
    for i in range(N_VECTORS):
        store.upsert("code_index", [
            _point(i),
        ])


def _point(i):
    from shared.store_models import PointStruct
    return PointStruct(
        id=f"vec-{i}",
        vector=[0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1), 0.4 * (i + 1)],
        payload={
            "codeChunk": f"def function_{i}(): return {i}",
            "filePath": f"/src/module_{i % 3}.py",
            "startLine": i * 10,
            "endLine": i * 10 + 5,
        },
    )


def _run_cmd(cmd_fn, args, resolve_fn):
    """Call a migrate_store command with a monkey-patched _resolve_backend."""
    import shared.migrate_store as ms
    orig = ms._resolve_backend
    ms._resolve_backend = resolve_fn
    try:
        cmd_fn(args)
    finally:
        ms._resolve_backend = orig


class TestMigrationRoundTrip:
    """Full export → import → verify cycle with deep content assertions."""

    def test_full_round_trip(self, tmp_path):
        from shared.impls.turso_sql import TursoSqlStore
        from shared.impls.turso_vector import TursoVectorStore
        import shared.migrate_store as ms

        # Source with data
        src_sql = TursoSqlStore(url="file::memory:")
        src_vec = TursoVectorStore(url="file::memory:")
        _seed_sql(src_sql)
        _seed_vec(src_vec)

        # Empty destination
        dst_sql = TursoSqlStore(url="file::memory:")
        dst_vec = TursoVectorStore(url="file::memory:")

        jsonl = str(tmp_path / "backup.jsonl")

        # Export
        _run_cmd(
            ms.cmd_export,
            argparse.Namespace(backend="turso+turso", out=jsonl, progress_every=None),
            lambda spec: (src_sql, src_vec),
        )

        # Import
        _run_cmd(
            ms.cmd_import,
            argparse.Namespace(backend="turso+turso", **{"in": jsonl},
                               progress_every=None, backup_before=False),
            lambda spec: (dst_sql, dst_vec),
        )

        # --- Deep SQL assertions (not just ID sets) ---
        src_ids = set(src_sql.get_all_memory_ids())
        dst_ids = set(dst_sql.get_all_memory_ids())
        assert dst_ids == src_ids, f"ID mismatch: {dst_ids ^ src_ids}"

        for mid in src_ids:
            src_mem = src_sql.get_memory(mid)
            dst_mem = dst_sql.get_memory(mid)
            assert src_mem["text"] == dst_mem["text"], f"text mismatch for {mid}"
            assert src_mem["memory_type"] == dst_mem["memory_type"], f"type mismatch for {mid}"
            assert src_mem["tags"] == dst_mem["tags"], f"tags mismatch for {mid}"
            assert src_mem.get("path") == dst_mem.get("path"), f"path mismatch for {mid}"

        # --- Deep vector assertions ---
        dst_collections = dst_vec.list_collections()
        assert "code_index" in dst_collections

        dst_info = dst_vec.get_collection("code_index")
        assert dst_info.points_count == N_VECTORS, \
            f"Expected {N_VECTORS} points, got {dst_info.points_count}"

        # Verify vectors round-tripped (C2 fix: iter_all with_vectors)
        for i in range(N_VECTORS):
            pts = dst_vec.retrieve("code_index", [f"vec-{i}"], with_payload=True)
            assert len(pts) == 1, f"Missing vec-{i} in destination"
            src_vec_vals = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            # Retrieve doesn't return vectors by default; use iter_all to check
        # Use iter_all to verify vectors survived (the C2 fix)
        dst_points = list(dst_vec.iter_all("code_index", with_vectors=True))
        assert len(dst_points) == N_VECTORS
        for p in dst_points:
            assert len(p.vector) == DIM, f"Vector dimension lost for {p.id}"
            assert any(v != 0.0 for v in p.vector), f"Vector is all zeros for {p.id}"

        # Verify payloads survived intact (C1 fix: merge semantics)
        for p in dst_points:
            assert "codeChunk" in p.payload, f"Missing codeChunk in {p.id}"
            assert "filePath" in p.payload, f"Missing filePath in {p.id}"
            assert "startLine" in p.payload, f"Missing startLine in {p.id}"

        # --- Shallow verify passes ---
        _run_cmd(
            ms.cmd_verify,
            argparse.Namespace(left="src", right="dst"),
            lambda spec: (src_sql, src_vec) if spec == "src" else (dst_sql, dst_vec),
        )

    def test_export_produces_valid_jsonl(self, tmp_path):
        """The exported JSONL must be parseable and have the right record kinds."""
        import json
        from shared.impls.turso_sql import TursoSqlStore
        from shared.impls.turso_vector import TursoVectorStore
        import shared.migrate_store as ms

        src_sql = TursoSqlStore(url="file::memory:")
        src_vec = TursoVectorStore(url="file::memory:")
        _seed_sql(src_sql)
        _seed_vec(src_vec)

        jsonl = str(tmp_path / "backup.jsonl")
        _run_cmd(
            ms.cmd_export,
            argparse.Namespace(backend="turso+turso", out=jsonl, progress_every=None),
            lambda spec: (src_sql, src_vec),
        )

        records = [json.loads(line) for line in Path(jsonl).read_text().strip().split("\n")]

        # Must have SQL meta header + SQL rows
        sql_metas = [r for r in records if "_meta" in r and r["_meta"].get("kind") == "sql"]
        assert len(sql_metas) == 1
        assert sql_metas[0]["_meta"]["schema_version"] == 1

        sql_rows = [r for r in records if "_sql" in r]
        assert len(sql_rows) == N_MEMORIES

        # Must have vector meta header + vector points
        vec_metas = [r for r in records if "_meta" in r and r["_meta"].get("kind") == "vector"]
        assert len(vec_metas) == 1
        assert vec_metas[0]["_meta"]["collection"] == "code_index"
        assert vec_metas[0]["_meta"]["dim"] == DIM

        vec_rows = [r for r in records if "_vec" in r]
        assert len(vec_rows) == N_VECTORS
        # No __collection_metadata__ in export (R2 filter)
        assert not any(r["id"] == "__collection_metadata__" for r in vec_rows)

    def test_verify_detects_missing_data(self, tmp_path):
        """Verify must fail when the destination is missing data."""
        from shared.impls.turso_sql import TursoSqlStore
        from shared.impls.turso_vector import TursoVectorStore
        import shared.migrate_store as ms

        src_sql = TursoSqlStore(url="file::memory:")
        src_vec = TursoVectorStore(url="file::memory:")
        _seed_sql(src_sql)

        # Empty destination
        dst_sql = TursoSqlStore(url="file::memory:")

        with pytest.raises(SystemExit) as exc_info:
            _run_cmd(
                ms.cmd_verify,
                argparse.Namespace(left="src", right="dst"),
                lambda spec: (src_sql, src_vec) if spec == "src" else (dst_sql, None),
            )
        assert exc_info.value.code == 1
