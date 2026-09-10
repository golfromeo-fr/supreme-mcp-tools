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
next call. Writes are last-writer-wins per document.
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

_TABLE = "mcp_state_docs"
_SCHEMA = (
    f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
    " name TEXT PRIMARY KEY,"
    " data TEXT NOT NULL,"
    " updated_at TEXT NOT NULL)",
)

_conn_singleton: object | None = None
_init_done = False


def _PH(conn) -> str:
    """Placeholder style: psycopg wants %s, libsql/sqlite want ?."""
    return "%s" if type(conn).__module__.startswith("psycopg") else "?"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _conn():
    """Shared SqlStore connection, or None to stay on local files."""
    global _conn_singleton, _init_done
    if _init_done:
        return _conn_singleton
    _init_done = True
    try:
        # state_docs lives in tools/shared; sql_store imports the bare
        # ``shared`` package, which needs tools/ on sys.path (same fix as
        # users_store._db_conn).
        tools_dir = str(Path(__file__).resolve().parents[1])
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from tools.shared.sql_store import get_sql_store

        store = get_sql_store()
        if type(store).__name__ == "NullSqlStore" or not getattr(
                store, "is_available", False):
            logger.warning(
                "MCP_STATE_BACKEND=db but no SQL backend is configured "
                "(POSTGRES_* / TURSO_DATABASE_URL) - staying on local files"
            )
            return None
        conn = getattr(store, "_conn", None)
        if conn is None:
            raise RuntimeError("SqlStore impl exposes no _conn")
        for stmt in _SCHEMA:
            conn.execute(stmt)
        _conn_singleton = conn
        logger.warning(
            f"[M4] state docs: SQL backend active "
            f"({type(conn).__module__}.{type(conn).__name__})")
    except Exception as e:
        logger.warning(
            f"MCP_STATE_BACKEND=db init failed ({type(e).__name__}: {e}) - "
            "staying on local files"
        )
        _conn_singleton = None
    return _conn_singleton


def backend_active() -> bool:
    """True once the db backend is selected AND initialized successfully."""
    return _backend() == "db" and _conn() is not None


def load_doc(name: str) -> dict | None:
    """Named document from the shared backend; None = no row / backend
    unavailable / json mode (caller decides its local fallback)."""
    if _backend() != "db":
        return None
    conn = _conn()
    if conn is None:
        return None
    ph = _PH(conn)
    try:
        cur = conn.execute(
            f"SELECT data FROM {_TABLE} WHERE name = {ph}", (name,))
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
    if _backend() != "db":
        return False
    conn = _conn()
    if conn is None:
        return False
    ph = _PH(conn)
    try:
        conn.execute(
            f"INSERT INTO {_TABLE} (name, data, updated_at) VALUES ({ph}, {ph}, {ph}) "
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
