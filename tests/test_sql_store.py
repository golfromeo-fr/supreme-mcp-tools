"""
Unit tests for SqlStore implementations.

Tests both the abstract SqlStore contract and the concrete impls:
- PostgresSqlStore (requires psycopg + a PG connection)
- TursoSqlStore (uses in-memory libSQL)

Phase 5 cleanup: replaces tests/test_pg_store.py which tested the old
pg_store module-level functions against internal globals.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


# ---------------------------------------------------------------------------
# Text hashing (shared, no DB needed)
# ---------------------------------------------------------------------------

class TestTextHash(unittest.TestCase):
    """text_hash is a pure function — no DB needed."""

    def test_deterministic(self):
        from shared.hashing import text_hash
        self.assertEqual(text_hash("Hello World"), text_hash("Hello World"))

    def test_case_insensitive(self):
        from shared.hashing import text_hash
        self.assertEqual(text_hash("Hello World"), text_hash("hello world"))

    def test_whitespace_normalized(self):
        from shared.hashing import text_hash
        self.assertEqual(text_hash("Hello World"), text_hash("  Hello World  "))

    def test_empty_string(self):
        from shared.hashing import text_hash
        # Empty string and whitespace-only produce the same hash
        self.assertEqual(text_hash(""), text_hash("   "))

    def test_length_40(self):
        from shared.hashing import text_hash
        self.assertEqual(len(text_hash("test")), 40)


# ---------------------------------------------------------------------------
# Abstract SqlStore contract (applied to NullSqlStore — no DB needed)
# ---------------------------------------------------------------------------

class TestNullSqlStoreContract(unittest.TestCase):
    """NullSqlStore is the no-op fallback when no backend is configured.

    It satisfies the SqlStore Protocol and returns safe defaults.
    """

    def setUp(self):
        from shared.sql_store import NullSqlStore
        self.store = NullSqlStore()

    def test_not_available(self):
        self.assertFalse(self.store.is_available)

    def test_upsert_returns_id_unchanged(self):
        result = self.store.upsert_memory(
            "uuid-1", "text", "concept", "agent", [],
            None, None, None, "low", "auto-delete",
        )
        self.assertEqual(result, "uuid-1")

    def test_get_returns_none(self):
        self.assertIsNone(self.store.get_memory("uuid-1"))

    def test_delete_returns_false(self):
        self.assertFalse(self.store.delete_memory("uuid-1"))

    def test_search_returns_empty_list(self):
        self.assertEqual(self.store.search_text("query"), [])

    def test_metrics_returns_empty_dict(self):
        self.assertEqual(self.store.get_metrics(), {})

    def test_decay_returns_zero(self):
        self.assertEqual(self.store.decay_memories(30, 0), 0)

    def test_get_all_ids_returns_empty(self):
        self.assertEqual(self.store.get_all_memory_ids(), [])

    def test_iter_all_yields_nothing(self):
        self.assertEqual(list(self.store.iter_all()), [])

    def test_bulk_upsert_returns_zero(self):
        self.assertEqual(self.store.bulk_upsert([{"id": "x"}]), 0)


# ---------------------------------------------------------------------------
# TursoSqlStore (in-memory libSQL — no network, no containers)
# ---------------------------------------------------------------------------

try:
    import libsql_experimental  # noqa: F401
    HAS_LIBSQL = True
except ImportError:
    HAS_LIBSQL = False


@unittest.skipUnless(HAS_LIBSQL, reason="libsql_experimental not installed")
class TestTursoSqlStoreContract(unittest.TestCase):
    """TursoSqlStore satisfies the full SqlStore contract in-memory."""

    def setUp(self):
        from shared.impls.turso_sql import TursoSqlStore
        self.store = TursoSqlStore(url="file::memory:")

    def test_is_available(self):
        self.assertTrue(self.store.is_available)

    def test_upsert_and_get(self):
        mid = self.store.upsert_memory(
            "uuid-1", "hello world", "concept", "agent",
            ["flask"], None, None, "agent-1", "low", "auto-delete",
        )
        mem = self.store.get_memory(mid)
        self.assertIsNotNone(mem)
        self.assertEqual(mem["text"], "hello world")
        self.assertEqual(mem["tags"], ["flask"])

    def test_dedup(self):
        id1 = self.store.upsert_memory(
            "uuid-1", "same text", "concept", "agent",
            [], None, None, "a", "low", "auto-delete",
        )
        id2 = self.store.upsert_memory(
            "uuid-2", "same text", "concept", "agent",
            [], None, None, "a", "low", "auto-delete",
        )
        self.assertEqual(id1, id2)

    def test_delete(self):
        mid = self.store.upsert_memory(
            "uuid-1", "to be deleted", "concept", "agent",
            [], None, None, "a", "low", "auto-delete",
        )
        self.assertTrue(self.store.delete_memory(mid))
        self.assertIsNone(self.store.get_memory(mid))

    def test_fts5_search(self):
        self.store.upsert_memory(
            "uuid-1", "python web framework", "concept", "agent",
            [], None, None, "a", "low", "auto-delete",
        )
        results = self.store.search_text("python")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("python", results[0]["text"].lower())

    def test_metrics(self):
        self.store.upsert_memory(
            "uuid-1", "test", "concept", "agent",
            [], None, None, "a", "low", "auto-delete",
        )
        m = self.store.get_metrics()
        self.assertEqual(m["total"], 1)
        self.assertIn("concept", m["by_type"])

    def test_iter_all_streams(self):
        for i in range(5):
            self.store.upsert_memory(
                f"uuid-{i}", f"text {i}", "concept", "agent",
                [], None, None, "a", "low", "auto-delete",
            )
        rows = list(self.store.iter_all())
        self.assertEqual(len(rows), 5)


# ---------------------------------------------------------------------------
# PostgresSqlStore (requires psycopg + a PG connection — skipped if unavailable)
# ---------------------------------------------------------------------------

try:
    import psycopg  # noqa: F401
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


@unittest.skipUnless(HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresSqlStoreWithMock(unittest.TestCase):
    """PostgresSqlStore can be constructed with a mocked connection.

    We monkey-patch the connection pool after construction so we can test
    without a real PG database.
    """

    def test_constructor_creates_pool(self):
        from unittest.mock import MagicMock, patch

        with patch("shared.impls.postgres_sql._PsycopgPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value = None
            mock_pool_cls.return_value = mock_pool

            from shared.impls.postgres_sql import PostgresSqlStore
            store = PostgresSqlStore(dsn="host=fake")

            self.assertTrue(store.is_available)
            self.assertIs(store._pool, mock_pool)


if __name__ == "__main__":
    unittest.main()
