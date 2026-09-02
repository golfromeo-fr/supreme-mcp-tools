"""
Unit tests for the two tracked PostgreSQL-impl bugs, against the live
implementation in tools/shared/impls/postgres_sql.py (the old pg_store.py
concerns moved here during the backend abstraction):

- HIGH-14: pg_trgm must be feature-scoped — schema init succeeds without the
  extension and search_text degrades instead of failing on similarity().
- LOW-5: DSN passwords must never surface in logs — _masked_dsn()/_safe_error()
  strip them from keyword-form and URL-form conninfo (and from exception
  messages that embed the conninfo, as psycopg OperationalError does).

All tests run WITHOUT a live Postgres: schema-init behaviour is exercised
through a mocked connection (same pattern as TestPostgresSqlStoreWithMock in
tests/test_sql_store.py).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from shared.impls.postgres_sql import _masked_dsn, _safe_error  # noqa: E402

try:
    import psycopg  # noqa: F401
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


# ---------------------------------------------------------------------------
# LOW-5: DSN password masking helpers
# ---------------------------------------------------------------------------

class TestMaskedDsn(unittest.TestCase):
    """_masked_dsn strips password=... and URL credentials, nothing else."""

    def test_keyword_form_masked(self):
        dsn = "host=db.example.com port=5432 user=app password=s3cret dbname=tools"
        out = _masked_dsn(dsn)
        self.assertNotIn("s3cret", out)
        self.assertIn("password=***", out)
        self.assertIn("host=db.example.com", out)
        self.assertIn("dbname=tools", out)

    def test_quoted_keyword_form_masked(self):
        dsn = "host=db password='two words' user=app"
        out = _masked_dsn(dsn)
        self.assertNotIn("two words", out)
        self.assertIn("password=***", out)
        self.assertIn("host=db", out)

    def test_url_form_masked(self):
        dsn = "postgresql://app:sup3rsecret@db.example.com:5432/tools"
        out = _masked_dsn(dsn)
        self.assertNotIn("sup3rsecret", out)
        self.assertIn("postgresql://app:***@db.example.com:5432/tools", out)

    def test_url_password_only_masked(self):
        dsn = "postgresql://:pwonly@db.example.com:5432/tools"
        out = _masked_dsn(dsn)
        self.assertNotIn("pwonly", out)
        self.assertIn("db.example.com", out)

    def test_url_without_password_unchanged(self):
        dsn = "postgresql://db.example.com:5432/tools"
        self.assertEqual(_masked_dsn(dsn), dsn)

    def test_non_dsn_text_unchanged(self):
        text = 'relation "memories" does not exist'
        self.assertEqual(_masked_dsn(text), text)


class TestSafeError(unittest.TestCase):
    """_safe_error masks exception messages that embed the conninfo."""

    def test_keyword_conninfo_in_exception_masked(self):
        # psycopg OperationalError messages can embed the conninfo verbatim
        err = Exception("connection failed: host=db user=app password=hunter2")
        out = _safe_error(err)
        self.assertNotIn("hunter2", out)
        self.assertIn("password=***", out)

    def test_url_conninfo_in_exception_masked(self):
        err = Exception("could not connect to postgresql://app:hunter2@db:5432/x")
        out = _safe_error(err)
        self.assertNotIn("hunter2", out)

    def test_plain_exception_unchanged(self):
        err = Exception("duplicate key value violates unique constraint")
        self.assertEqual(_safe_error(err), str(err))


# ---------------------------------------------------------------------------
# Mock plumbing shared by the HIGH-14 tests
# ---------------------------------------------------------------------------

class _FakeConn:
    """Records executed SQL, optionally failing on pg_trgm statements."""

    def __init__(self, fail_on_trgm=False, search_rows=None):
        self.sqls: list[str] = []
        self.params: list[tuple] = []
        self.fail_on_trgm = fail_on_trgm
        self.search_rows = search_rows or []

    def execute(self, query, params=None):
        q = " ".join(query.split())
        self.sqls.append(q)
        self.params.append(params)
        if self.fail_on_trgm and ("pg_trgm" in q or "gin_trgm_ops" in q):
            raise Exception('permission denied to create extension "pg_trgm"')
        result = MagicMock()
        result.fetchone.return_value = None
        result.fetchall.return_value = list(self.search_rows)
        return result


def _make_store(conn):
    """Build a PostgresSqlStore whose pool serves *conn* (no real PG)."""
    pool = MagicMock()
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)
    with patch("shared.impls.postgres_sql._PsycopgPool", return_value=pool):
        from shared.impls.postgres_sql import PostgresSqlStore
        return PostgresSqlStore(dsn="host=fake password=hunter2")


# ---------------------------------------------------------------------------
# HIGH-14: pg_trgm feature-scoped at schema init
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_PSYCOPG, reason="psycopg not installed")
class TestHIGH14TrgmFeatureScoped(unittest.TestCase):
    """Schema init must not require pg_trgm; search degrades without it."""

    def test_ddl_runs_before_trgm_attempt(self):
        conn = _FakeConn()
        store = _make_store(conn)
        self.assertTrue(store.is_available)
        ddl_idx = next(i for i, s in enumerate(conn.sqls) if "CREATE TABLE" in s)
        self.assertLess(ddl_idx, conn.sqls.index("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        self.assertIn("idx_memories_text_trgm", conn.sqls[ddl_idx + 2])

    def test_trgm_available_on_success(self):
        store = _make_store(_FakeConn())
        self.assertTrue(store._trgm_available)

    def test_schema_init_succeeds_without_pg_trgm(self):
        conn = _FakeConn(fail_on_trgm=True)
        store = _make_store(conn)  # must not raise
        self.assertTrue(store.is_available)
        self.assertFalse(store._trgm_available)
        # core DDL still executed despite the extension failure
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS memories" in s for s in conn.sqls))

    def test_trgm_failure_logs_masked_error(self):
        with self.assertLogs("shared.impls.postgres_sql", level="WARNING") as logs:
            _make_store(_FakeConn(fail_on_trgm=True))
        joined = "\n".join(logs.output)
        self.assertIn("degrade to substring match", joined)
        self.assertNotIn("hunter2", joined)

    def test_search_uses_similarity_with_trgm(self):
        conn = _FakeConn(search_rows=[{"id": "u1", "text": "needle"}])
        store = _make_store(conn)
        self.assertTrue(store._trgm_available)
        rows = store.search_text("needle")
        sql = conn.sqls[-1]
        self.assertIn("similarity(text, %s) > 0.1", sql)
        self.assertIn("similarity(text, %s) AS sim_score", sql)
        self.assertIn("ORDER BY sim_score DESC", sql)
        # query appears twice (SELECT + WHERE) before the limit param
        self.assertEqual(conn.params[-1], ["needle", "needle", 10])
        self.assertEqual(len(rows), 1)

    def test_search_trgm_param_order_with_filter(self):
        """Regression: SELECT similarity placeholder is textual-first.

        The old param order ([*params, *sim_params, limit]) bound the filter
        value to the SELECT placeholder and the query to memory_type whenever
        a filter was present — silently wrong results (values coincide only
        in the no-filter case, which is why the mock test above passed).
        """
        conn = _FakeConn(search_rows=[])
        store = _make_store(conn)
        store.search_text("needle", memory_type="lesson")
        # Textual placeholder order: SELECT sim_q, WHERE memory_type, WHERE sim_q, LIMIT
        self.assertEqual(
            conn.params[-1], ["needle", "lesson", "needle", 10]
        )

    def test_search_degrades_to_ilike_without_trgm(self):
        conn = _FakeConn(search_rows=[{"id": "u1", "text": "needle", "sim_score": 0.0}])
        store = _make_store(conn)
        store._trgm_available = False
        rows = store.search_text("needle")
        sql = conn.sqls[-1]
        self.assertIn("text ILIKE %s", sql)
        self.assertNotIn("similarity(", sql)
        self.assertIn("ORDER BY usage_count DESC, created_at DESC", sql)
        self.assertEqual(conn.params[-1], ["%needle%", 10])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sim_score"], 0.0)

    def test_degrade_warning_logged_once(self):
        store = _make_store(_FakeConn())
        store._trgm_available = False
        with self.assertLogs("shared.impls.postgres_sql", level="WARNING") as logs:
            store.search_text("a")
            store.search_text("b")
        warnings = [o for o in logs.output if "degrading to" in o]
        self.assertEqual(len(warnings), 1)


# ---------------------------------------------------------------------------
# LOW-5: init-failure log path masks the DSN end-to-end
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_PSYCOPG, reason="psycopg not installed")
class TestLOW5InitFailureLogMasked(unittest.TestCase):
    """PostgresSqlStore init failures must not log the raw password."""

    def test_init_failure_logs_masked_dsn(self):
        with patch(
            "shared.impls.postgres_sql._PsycopgPool",
            side_effect=Exception(
                "connection failed: host=db user=app password=hunter2"
            ),
        ):
            from shared.impls.postgres_sql import PostgresSqlStore
            with self.assertLogs("shared.impls.postgres_sql", level="WARNING") as logs:
                store = PostgresSqlStore(dsn="host=db user=app password=hunter2")
        self.assertFalse(store.is_available)
        joined = "\n".join(logs.output)
        self.assertNotIn("hunter2", joined)
        self.assertIn("password=***", joined)


if __name__ == "__main__":
    unittest.main()
