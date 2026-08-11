"""
Tests for the migration tool.

Tests:
- JSONL export/import round-trip (Turso → file → Turso → verify)
- __collection_metadata__ filtering (R2)
- Verify command (parity check)
- Schema version in meta header (R11)

Uses in-memory libSQL — no network, no containers.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False

pytestmark = pytest.mark.skipif(not HAS_LIBSQL, reason="libsql_experimental not installed")


@pytest.fixture
def source_sql_store():
    """Create a TursoSqlStore with test data."""
    from shared.impls.turso_sql import TursoSqlStore
    store = TursoSqlStore(url="file::memory:")
    for i in range(10):
        store.upsert_memory(
            f"uuid-{i}",
            f"memory text {i} about python and databases",
            "concept" if i % 2 == 0 else "code_pattern",
            "agent",
            ["flask"] if i % 3 == 0 else [],
            f"/path/{i}.py" if i % 2 == 0 else None,
            None,
            "agent-1",
            "low",
            "auto-delete",
        )
    yield store


@pytest.fixture
def source_vector_store():
    """Create a TursoVectorStore with test data."""
    from shared.impls.turso_vector import TursoVectorStore
    from shared.store_models import PointStruct
    store = TursoVectorStore(url="file::memory:")
    store.ensure_collection("test_coll", dense_dim=4)
    store.upsert("test_coll", [
        PointStruct(
            id=f"vec-{i}",
            vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
            payload={"text": f"chunk {i}", "filePath": f"/src/file{i}.py"},
        )
        for i in range(1, 6)
    ])
    # Insert a __collection_metadata__ point (should be filtered on export)
    store.upsert("test_coll", [
        PointStruct(
            id="__collection_metadata__",
            vector=[0.0, 0.0, 0.0, 0.0],
            payload={"_is_metadata": True, "embedding_model": "bge-m3"},
        )
    ])
    yield store


class TestJsonlExportImport:
    """Test JSONL export/import round-trip."""

    def test_sql_round_trip(self, source_sql_store, tmp_path):
        """Export SQL data → JSONL → import into fresh Turso → verify IDs match."""
        from shared.impls.turso_sql import TursoSqlStore
        from shared.migrate_store import cmd_export, cmd_import
        import argparse
        import shared.migrate_store as ms

        jsonl_path = str(tmp_path / "export.jsonl")

        # Export from source (monkey-patch _resolve_backend)
        original_resolve = ms._resolve_backend
        ms._resolve_backend = lambda spec: (source_sql_store, None)
        try:
            cmd_export(argparse.Namespace(
                backend="turso+turso", out=jsonl_path, progress_every=None,
            ))
        finally:
            ms._resolve_backend = original_resolve

        # Verify JSONL has expected structure
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert any(l.get("_meta", {}).get("kind") == "sql" for l in lines)
        assert sum(1 for l in lines if "_sql" in l) == 10

        # Import into fresh store
        dest = TursoSqlStore(url="file::memory:")
        ms._resolve_backend = lambda spec: (dest, None)
        try:
            cmd_import(argparse.Namespace(
                backend="turso+turso", **{"in": jsonl_path},
                progress_every=None, backup_before=False,
            ))
        finally:
            ms._resolve_backend = original_resolve

        # Verify parity
        source_ids = set(source_sql_store.get_all_memory_ids())
        dest_ids = set(dest.get_all_memory_ids())
        assert source_ids == dest_ids

    def test_vector_round_trip(self, source_vector_store, tmp_path):
        """Export vector data → JSONL → import into fresh Turso → verify."""
        from shared.impls.turso_vector import TursoVectorStore
        from shared.migrate_store import cmd_export, cmd_import
        import argparse

        jsonl_path = str(tmp_path / "vec_export.jsonl")

        # Export (only vector data — monkey-patch to skip SQL)
        import shared.migrate_store as ms
        original_resolve = ms._resolve_backend
        ms._resolve_backend = lambda spec: (None, source_vector_store)
        try:
            cmd_export(argparse.Namespace(
                backend="turso+turso", out=jsonl_path, progress_every=None,
            ))
        finally:
            ms._resolve_backend = original_resolve

        # Verify JSONL structure
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert any(l.get("_meta", {}).get("kind") == "vector" for l in lines)
        vec_lines = [l for l in lines if "_vec" in l]
        # 5 real points + 0 metadata (R2: filtered)
        assert len(vec_lines) == 5
        # R2: __collection_metadata__ should NOT be in export
        assert all(l["id"] != "__collection_metadata__" for l in vec_lines)

        # Import into fresh store
        dest = TursoVectorStore(url="file::memory:")
        ms._resolve_backend = lambda spec: (None, dest)
        try:
            cmd_import(argparse.Namespace(
                backend="turso+turso", **{"in": jsonl_path},
                progress_every=None, backup_before=False,
            ))
        finally:
            ms._resolve_backend = original_resolve

        # Verify parity
        source_info = source_vector_store.get_collection("test_coll")
        dest_info = dest.get_collection("test_coll")
        # Source has 6 (5 real + 1 metadata), dest has 5 (metadata filtered)
        assert dest_info.points_count == 5


class TestMetadataFiltering:
    """R2: __collection_metadata__ must be filtered on export."""

    def test_metadata_not_in_export(self, source_vector_store, tmp_path):
        """The __collection_metadata__ point must not appear in export JSONL."""
        from shared.migrate_store import cmd_export
        import argparse

        jsonl_path = str(tmp_path / "meta_test.jsonl")

        import shared.migrate_store as ms
        original_resolve = ms._resolve_backend
        ms._resolve_backend = lambda spec: (None, source_vector_store)
        try:
            cmd_export(argparse.Namespace(
                backend="turso+turso", out=jsonl_path, progress_every=None,
            ))
        finally:
            ms._resolve_backend = original_resolve

        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                if "_vec" in record:
                    assert record["id"] != "__collection_metadata__", \
                        "R2 violation: __collection_metadata__ found in export"


class TestSchemaVersion:
    """R11: schema_version must be present in meta headers."""

    def test_sql_meta_has_schema_version(self, source_sql_store, tmp_path):
        from shared.migrate_store import cmd_export
        import argparse
        import shared.migrate_store as ms

        jsonl_path = str(tmp_path / "schema_test.jsonl")
        original_resolve = ms._resolve_backend
        ms._resolve_backend = lambda spec: (source_sql_store, None)
        try:
            cmd_export(argparse.Namespace(
                backend="turso+turso", out=jsonl_path, progress_every=None,
            ))
        finally:
            ms._resolve_backend = original_resolve

        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                if "_meta" in record and record["_meta"]["kind"] == "sql":
                    assert "schema_version" in record["_meta"]
                    assert record["_meta"]["schema_version"] == 1
                    return
        pytest.fail("No SQL meta header found")


class TestVerifyCommand:
    """Test the verify command."""

    def test_verify_parity_pass(self, source_sql_store):
        """Two stores with same IDs should pass verification."""
        from shared.impls.turso_sql import TursoSqlStore
        from shared.migrate_store import cmd_verify
        import argparse

        # Create dest with same data
        dest = TursoSqlStore(url="file::memory:")
        for i in range(10):
            dest.upsert_memory(
                f"uuid-{i}",
                f"memory text {i} about python and databases",
                "concept" if i % 2 == 0 else "code_pattern",
                "agent",
                ["flask"] if i % 3 == 0 else [],
                None, None, "agent-1", "low", "auto-delete",
            )

        import shared.migrate_store as ms
        original_resolve = ms._resolve_backend
        ms._resolve_backend = lambda spec: (
            (source_sql_store, None) if spec == "left+turso" else (dest, None)
        )
        try:
            # Should not raise (parity passes)
            cmd_verify(argparse.Namespace(left="left+turso", right="right+turso"))
        except SystemExit as e:
            assert e.code == 0, f"Verify failed unexpectedly: {e}"
        finally:
            ms._resolve_backend = original_resolve

    def test_verify_parity_fail(self, source_sql_store):
        """Two stores with different IDs should fail verification."""
        from shared.impls.turso_sql import TursoSqlStore
        from shared.migrate_store import cmd_verify
        import argparse

        dest = TursoSqlStore(url="file::memory:")
        # Only add 5 records (source has 10)
        for i in range(5):
            dest.upsert_memory(
                f"uuid-{i}", f"text {i}", "concept",
                "agent", [], None, None, "a", "low", "auto-delete",
            )

        import shared.migrate_store as ms
        original_resolve = ms._resolve_backend
        ms._resolve_backend = lambda spec: (
            (source_sql_store, None) if spec == "left+turso" else (dest, None)
        )
        try:
            cmd_verify(argparse.Namespace(left="left+turso", right="right+turso"))
            pytest.fail("Verify should have failed")
        except SystemExit as e:
            assert e.code == 1, f"Verify should exit 1 on mismatch, got {e.code}"
        finally:
            ms._resolve_backend = original_resolve
