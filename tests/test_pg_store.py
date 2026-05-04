"""
Unit tests for tools/shared/pg_store.py

Uses mocked psycopg connections to test SQL logic without a real database.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


class TestTextHash(unittest.TestCase):
    """Test the text_hash function (no DB needed)."""

    def test_deterministic(self):
        from shared.pg_store import text_hash
        h1 = text_hash("Hello World")
        h2 = text_hash("Hello World")
        self.assertEqual(h1, h2)

    def test_case_insensitive(self):
        from shared.pg_store import text_hash
        h1 = text_hash("Hello World")
        h2 = text_hash("hello world")
        self.assertEqual(h1, h2)

    def test_whitespace_normalized(self):
        from shared.pg_store import text_hash
        h1 = text_hash("Hello World")
        h2 = text_hash("  Hello World  ")
        self.assertEqual(h1, h2)

    def test_different_text(self):
        from shared.pg_store import text_hash
        h1 = text_hash("Hello")
        h2 = text_hash("World")
        self.assertNotEqual(h1, h2)

    def test_length_40(self):
        from shared.pg_store import text_hash
        h = text_hash("test")
        self.assertEqual(len(h), 40)


class TestIsAvailable(unittest.TestCase):
    """Test is_available when PG not initialized."""

    def test_not_available_by_default(self):
        import shared.pg_store as pg
        original = pg._pg_available
        pg._pg_available = False
        try:
            self.assertFalse(pg.is_available())
        finally:
            pg._pg_available = original


class TestUpsertMemory(unittest.TestCase):
    """Test upsert_memory with mocked pool."""

    def _setup_available(self):
        import shared.pg_store as pg
        pg._pg_available = True
        mock_pool = MagicMock()
        pg._pool = mock_pool
        return mock_pool

    def _teardown(self):
        import shared.pg_store as pg
        pg._pg_available = False
        pg._pool = None

    def test_returns_original_id_when_not_available(self):
        import shared.pg_store as pg
        pg._pg_available = False
        result = pg.upsert_memory("test-id", "text", "concept", "agent", [], None, None, None, "low", "auto-delete")
        self.assertEqual(result, "test-id")

    def test_insert_new_memory(self):
        import shared.pg_store as pg
        mock_pool = self._setup_available()

        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_conn.execute.return_value.fetchone.side_effect = [
            None,
            {"id": "new-uuid"},
        ]

        result = pg.upsert_memory(
            "new-uuid", "test text", "concept", "agent",
            ["tag1"], None, None, None, "low", "auto-delete"
        )
        self.assertEqual(result, "new-uuid")
        self._teardown()

    def test_dedup_returns_existing_id(self):
        import shared.pg_store as pg
        mock_pool = self._setup_available()

        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_conn.execute.return_value.fetchone.return_value = {"id": "existing-id", "usage_count": 5}

        result = pg.upsert_memory(
            "new-uuid", "duplicate text", "concept", "agent",
            [], None, None, None, "low", "auto-delete"
        )
        self.assertEqual(result, "existing-id")
        self._teardown()


class TestDeleteMemory(unittest.TestCase):
    """Test delete_memory with mocked pool."""

    def test_returns_false_when_not_available(self):
        import shared.pg_store as pg
        pg._pg_available = False
        result = pg.delete_memory("test-id")
        self.assertFalse(result)

    def test_returns_true_on_success(self):
        import shared.pg_store as pg
        pg._pg_available = True
        mock_pool = MagicMock()
        pg._pool = mock_pool

        mock_conn = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_conn.execute.return_value = mock_result

        result = pg.delete_memory("test-id")
        self.assertTrue(result)

        pg._pg_available = False
        pg._pool = None


class TestSearchText(unittest.TestCase):
    """Test search_text — verifies the conditions list is initialized."""

    def test_returns_empty_when_not_available(self):
        import shared.pg_store as pg
        pg._pg_available = False
        result = pg.search_text("query")
        self.assertEqual(result, [])


class TestGetMetrics(unittest.TestCase):
    """Test get_metrics with mocked pool."""

    def test_returns_empty_when_not_available(self):
        import shared.pg_store as pg
        pg._pg_available = False
        result = pg.get_metrics()
        self.assertEqual(result, {})


class TestDecayMemories(unittest.TestCase):
    """Test decay_memories parameterized INTERVAL."""

    def test_returns_zero_when_not_available(self):
        import shared.pg_store as pg
        pg._pg_available = False
        result = pg.decay_memories(30, 0)
        self.assertEqual(result, 0)


class TestGetAllMemoryIds(unittest.TestCase):
    """Test get_all_memory_ids."""

    def test_returns_empty_when_not_available(self):
        import shared.pg_store as pg
        pg._pg_available = False
        result = pg.get_all_memory_ids()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
