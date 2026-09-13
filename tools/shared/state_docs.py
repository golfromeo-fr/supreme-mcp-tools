"""M4/H2 — shared network-backed document store (cluster state).

Small extension of the users_store db-backend pattern: named JSON documents
(masks+inventory, later env/auth kv) stored in ONE table of the shared SQL
backend, so every launcher node reads/writes the same cluster state.

Selection: ``MCP_STATE_BACKEND=json`` (default, local files — callers keep
their file logic) or ``db``. In db mode ``load_doc``/``save_doc`` are the
source of truth; when the backend is unavailable they return ``None``/
``False`` so callers fall back to their file paths with a loud log (same
fail-open-to-local contract as users_store).

Reads are always fresh (no cache): edits propagate to all nodes on the
next call. ``save_doc`` is last-writer-wins; ``save_doc_cas`` adds an
optimistic lock (compare-and-swap on ``updated_at``) so concurrent
read-modify-writers retry on the winner's copy instead of clobbering it.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def _backend() -> str:
    return os.environ.get("MCP_STATE_BACKEND", "json").strip().lower()


def is_db_mode() -> bool:
    """True when MCP_STATE_BACKEND selects the shared db state plane.

    Single definition of the db-mode gate: cluster helpers, the launcher,
    tools_config and function_masks all ask here instead of re-reading the
    env var, so the name and normalization live in exactly one place."""
    return _backend() == "db"

_TABLE = "mcp_state_docs"
_SCHEMA = (
    f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
    " name TEXT PRIMARY KEY,"
    " data TEXT NOT NULL,"
    " updated_at TEXT NOT NULL)",
)

_conn_singleton: object | None = None
_init_done = False


class _TursoExec:
    """libsql/sqlite: '?' placeholders, autocommit per statement."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)


class _PgExec:
    """psycopg_pool: '%s' placeholders, borrowed pooled connections with
    explicit commit. The '?' of the shared call convention is converted
    here — callers never need to know the dialect."""

    def __init__(self, pool):
        self._pool = pool

    def execute(self, sql, params=()):
        pg_sql = sql.replace("?", "%s")
        with self._pool.connection() as conn:
            cur = conn.execute(pg_sql, params)
            rows = cur.fetchall() if cur.description else []
            conn.commit()
        # The pool is built with dict_row: rows are dicts keyed by column
        # name — tuple(dict) would yield the KEYS (live-found 2026-09-10).
        # Normalize to positional tuples in column order.
        return _PgResult([
            tuple(r.values()) if isinstance(r, dict) else tuple(r)
            for r in rows
        ])


class _PgResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


def shared_exec():
    """Dialect-agnostic executor over the shared SqlStore, or None to stay
    on local files. Callers always use '?' placeholders. Writes commit
    immediately; reads are fresh (no cache)."""
    try:
        # sql_store imports the bare ``shared`` package, which needs tools/
        # on sys.path (same fix as users_store._db_conn).
        tools_dir = str(Path(__file__).resolve().parents[1])
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from tools.shared.sql_store import get_sql_store

        store = get_sql_store()
        if type(store).__name__ == "NullSqlStore":
            logger.warning(
                "MCP_STATE_BACKEND=db but no SQL backend is configured "
                "(POSTGRES_* / TURSO_DATABASE_URL) - staying on local files"
            )
            return None
        if not getattr(store, "is_available", False) and \
                hasattr(store, "_connect") and not store._connect():
            logger.warning(
                "shared SQL backend unreachable - staying on local files")
            return None
        pool = getattr(store, "_pool", None)
        if pool is not None:
            return _PgExec(pool)
        conn = getattr(store, "_conn", None)
        if conn is not None:
            return _TursoExec(conn)
        raise RuntimeError("SqlStore impl exposes neither _pool nor _conn")
    except Exception as e:
        logger.warning(
            f"shared SQL backend init failed ({type(e).__name__}: {e}) - "
            "staying on local files"
        )
        return None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _conn():
    """State-docs executor, initialized once with its schema."""
    global _conn_singleton, _init_done
    if _init_done:
        return _conn_singleton
    _init_done = True
    try:
        ex = shared_exec()
        if ex is None:
            return None
        for stmt in _SCHEMA:
            ex.execute(stmt)
        _conn_singleton = ex
        logger.warning("[M4] state docs: SQL backend active")
    except Exception as e:
        logger.warning(
            f"MCP_STATE_BACKEND=db init failed ({type(e).__name__}: {e}) - "
            "staying on local files"
        )
        _conn_singleton = None
    return _conn_singleton


def backend_active() -> bool:
    """True once the db backend is selected AND initialized successfully."""
    return is_db_mode() and _conn() is not None


def load_doc(name: str) -> dict | None:
    """Named document from the shared backend; None = no row / backend
    unavailable / json mode (caller decides its local fallback)."""
    if not is_db_mode():
        return None
    conn = _conn()
    if conn is None:
        return None
    try:
        cur = conn.execute(
            f"SELECT data FROM {_TABLE} WHERE name = ?", (name,))
        row = cur.fetchone()
        if not row:
            return None
        doc = json.loads(row[0])
        return doc if isinstance(doc, dict) else None
    except Exception as e:
        logger.warning(
            f"state doc '{name}' unreadable ({type(e).__name__}: {e}) - "
            "caller should fall back to its local file"
        )
        return None


def save_doc(name: str, doc: dict) -> bool:
    """Store a named document; True = stored centrally, False = backend
    unavailable or json mode (caller should fall back to its local file)."""
    if not is_db_mode():
        return False
    conn = _conn()
    if conn is None:
        return False
    try:
        conn.execute(
            f"INSERT INTO {_TABLE} (name, data, updated_at) VALUES (?, ?, ?) "
            f"ON CONFLICT(name) DO UPDATE SET data = excluded.data, "
            f"updated_at = excluded.updated_at",
            (name, json.dumps(doc, ensure_ascii=False), _now_iso()),
        )
        return True
    except Exception as e:
        logger.warning(
            f"state doc '{name}' write failed ({type(e).__name__}: {e}) - "
            "caller should fall back to its local file"
        )
        return False


def load_doc_versioned(name: str) -> tuple[dict | None, str | None]:
    """``(doc, updated_at)`` for a named document; ``(None, None)`` = no row
    / backend unavailable / json mode. The stamp is the optimistic-lock
    version for ``save_doc_cas``."""
    if not is_db_mode():
        return None, None
    conn = _conn()
    if conn is None:
        return None, None
    try:
        cur = conn.execute(
            f"SELECT data, updated_at FROM {_TABLE} WHERE name = ?", (name,))
        row = cur.fetchone()
        if not row:
            return None, None
        doc = json.loads(row[0])
        return (doc if isinstance(doc, dict) else None), row[1]
    except Exception as e:
        logger.warning(
            f"state doc '{name}' unreadable ({type(e).__name__}: {e}) - "
            "caller should fall back to its local file"
        )
        return None, None


def save_doc_cas(name: str, doc: dict, expected_updated_at: str | None) -> bool:
    """Optimistic-lock store: write ``doc`` only if the row still carries
    ``expected_updated_at`` (``None`` = the row must not exist yet).
    True = stored; False = a concurrent writer got in first (retry on the
    fresh copy), or json mode / backend unavailable."""
    if not is_db_mode():
        return False
    conn = _conn()
    if conn is None:
        return False
    payload = json.dumps(doc, ensure_ascii=False)
    try:
        if expected_updated_at is None:
            conn.execute(
                f"INSERT INTO {_TABLE} (name, data, updated_at) VALUES (?, ?, ?)",
                (name, payload, _now_iso()),
            )
        else:
            conn.execute(
                f"UPDATE {_TABLE} SET data = ?, updated_at = ? "
                f"WHERE name = ? AND updated_at = ?",
                (payload, _now_iso(), name, expected_updated_at),
            )
        # Verify-read instead of trusting cursor.rowcount (libsql cursors
        # don't reliably expose it): success = our exact payload now sits in
        # the row. A writer that raced past the guard leaves different
        # bytes there, which reports as a lost race.
        row = conn.execute(
            f"SELECT data FROM {_TABLE} WHERE name = ?", (name,)).fetchone()
        return bool(row) and row[0] == payload
    except Exception as e:
        # Duplicate-key on the create path = the expected create race.
        logger.debug(
            f"state doc '{name}' CAS write rejected ({type(e).__name__}: {e})")
        return False
