"""Project-wide pytest configuration.

Two jobs:

1. Load the project ``.env`` once before collection so tests see the same
   backend config the running tools use (e.g. ``POSTGRES_HOST``). Mirrors how
   every tool starts up via ``dotenv``.

2. Provide a ``pg_dsn`` session fixture that resolves a Postgres DSN *and probes
   reachability*, so the optional Postgres contract tests run automatically
   whenever a Postgres is configured and up — and skip cleanly (no failures)
   when it isn't. This is the "run if applicable" behaviour: no special
   ``POSTGRES_TEST_DSN`` var is required if the standard ``POSTGRES_*`` vars are
   present (e.g. the app's own config block in ``.env``).

DSN resolution priority:
    POSTGRES_TEST_DSN  >  assembled from POSTGRES_HOST/PORT/USER/PASSWORD/DB
"""
import os
from pathlib import Path

import pytest

# Load .env once, before collection. Best-effort: a missing/odd .env must never
# break test collection.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


def _assemble_pg_dsn() -> str | None:
    """Resolve a PG DSN: explicit override wins, else build from POSTGRES_* vars."""
    explicit = os.getenv("POSTGRES_TEST_DSN")
    if explicit:
        return explicit
    host = os.getenv("POSTGRES_HOST")
    if not host:
        return None
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", user)
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{port}/{db}"


@pytest.fixture(scope="session")
def pg_dsn():
    """Return a PG DSN if Postgres is configured AND reachable, else None.

    Session-scoped so the reachability probe runs exactly once for the whole
    suite (a 3s connect timeout keeps a down server from stalling collection).
    """
    dsn = _assemble_pg_dsn()
    if not dsn:
        return None
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            pass
    except Exception:
        return None
    return dsn


@pytest.fixture()
def db_backend(monkeypatch):
    """db-mode state_docs over the shared fake SqlStore (M5: one definition
    for test_state_docs / test_cluster_state / test_env_manager_h2b)."""
    fake = FakeSqlStore()
    monkeypatch.setenv("MCP_STATE_BACKEND", "db")
    monkeypatch.setattr("tools.shared.sql_store.get_sql_store", lambda: fake)
    from tools.shared import state_docs
    monkeypatch.setattr(state_docs, "_conn_singleton", None)
    monkeypatch.setattr(state_docs, "_init_done", False)
    return fake


# ---------------------------------------------------------------------------
# Canonical fake SQL backend (M5 — was duplicated ×3 across test files).
# Implements exactly the surface state_docs.shared_exec() relies on since the
# SqlStore.execute consolidation: the STORE exposes execute() and delegates
# to the fake connection ('?'-placeholders, CAS/race semantics, statement
# recording). Lives here because a site-packages module named ``tests``
# shadows the tests/ namespace package, breaking ``tests._fakes`` imports.
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, state, params):
        self._state, self._params = state, params

    def fetchone(self):
        sql = self._state["last_sql"]
        if sql.startswith("SELECT data"):
            name = self._state.get("last_name")
            return (self._state["rows"][name]
                    if name in self._state["rows"] else None)
        return None


class FakeConn:
    def __init__(self):
        self.state = {"rows": {}, "last_sql": "", "last_name": None,
                      "statements": []}

    def execute(self, sql, params=()):
        self.state["statements"].append((sql, params))
        self.state["last_sql"] = sql
        rows = self.state["rows"]
        if sql.startswith("INSERT INTO mcp_state_docs"):
            (name, data, ts) = params
            # plain CAS insert has no ON CONFLICT — a duplicate name is the
            # create race and must fail like a real unique violation
            if "ON CONFLICT" not in sql and name in rows:
                raise RuntimeError("UNIQUE constraint failed: mcp_state_docs.name")
            rows[name] = (data, ts)
        elif sql.startswith("UPDATE mcp_state_docs"):
            (data, ts, name, expected) = params
            if name in rows and rows[name][1] == expected:
                rows[name] = (data, ts)
        if sql.startswith("SELECT data"):
            self.state["last_name"] = params[0]
        return FakeCursor(self.state, params)


class FakeSqlStore:
    """Minimal SqlStore double: just what state_docs.shared_exec() needs."""

    is_available = True

    def __init__(self):
        self._conn = FakeConn()

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)
