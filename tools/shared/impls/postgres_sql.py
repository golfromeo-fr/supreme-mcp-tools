"""
PostgresSqlStore — SqlStore implementation backed by PostgreSQL.

Wraps the same logic as the former pg_store.py (removed after Phase 5) as
instance methods.
This is used by the get_sql_store() factory when backend="postgres".

Phase 1 of the backend abstraction plan.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any
from datetime import datetime, timezone

from shared.hashing import text_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSN secret masking (LOW-5)
#
# DSNs carry the password in keyword form ("password=...") or URL form
# ("postgresql://user:password@host/db"). psycopg OperationalError messages can
# embed the conninfo verbatim, so every log site that interpolates an exception
# funnels it through _safe_error() and the password never reaches the logs.
# ---------------------------------------------------------------------------

# Keyword form: password=secret / password='two words' / password="quoted"
_PASSWORD_KW_RE = re.compile(
    r"(password\s*=\s*)('(?:[^']|'')*'|\"[^\"]*\"|[^\s]+)", re.IGNORECASE
)
# URL form: scheme://[user[:password]@]host — mask only the password segment
_PASSWORD_URL_RE = re.compile(
    r"((?:[A-Za-z][A-Za-z0-9+.\-]*://)[^:/@\s]*:)([^@\s/]+)@"
)


def _masked_dsn(text: str) -> str:
    """Return *text* with any DSN/conninfo password replaced by '***'."""
    text = _PASSWORD_KW_RE.sub(r"\1***", text)
    text = _PASSWORD_URL_RE.sub(r"\1***@", text)
    return text


def _safe_error(e: BaseException) -> str:
    """str(e) with any embedded DSN/conninfo password masked."""
    return _masked_dsn(str(e))


# ---------------------------------------------------------------------------
# Internal helpers (mirroring the former pg_store.py, kept private)
# ---------------------------------------------------------------------------

class _PsycopgPool:
    """Connection pool using psycopg_pool if available, else per-connection fallback."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._real_pool = None
        try:
            from psycopg_pool import ConnectionPool
            self._real_pool = ConnectionPool(
                conninfo=dsn,
                min_size=1,
                max_size=5,
                open=False,
                kwargs={"row_factory": __import__('psycopg.rows', fromlist=['dict_row']).dict_row},
            )
            self._real_pool.open(wait=True)
            logger.info("Using psycopg_pool.ConnectionPool")
        except (ImportError, Exception) as e:
            logger.info(f"psycopg_pool not available, using per-connection fallback: {_safe_error(e)}")

    def connection(self):
        if self._real_pool is not None:
            return self._real_pool.connection()
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(self._dsn, row_factory=dict_row)
        conn.autocommit = True
        return _Ctx(conn)


class _Ctx:
    """Context manager for a single connection."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *args):
        self._conn.close()


_SCHEMA_DDL = """
    CREATE TABLE IF NOT EXISTS memories (
        id          UUID PRIMARY KEY,
        text        TEXT NOT NULL,
        text_hash   TEXT NOT NULL,
        memory_type TEXT NOT NULL DEFAULT 'concept',
        source      TEXT NOT NULL DEFAULT 'agent_action',
        tags        JSONB NOT NULL DEFAULT '[]',
        path        TEXT,
        commit      TEXT,
        agent_id    TEXT,
        sensitivity TEXT NOT NULL DEFAULT 'low',
        retention_policy TEXT NOT NULL DEFAULT 'auto-delete',
        usage_count  INTEGER NOT NULL DEFAULT 0,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_accessed TIMESTAMPTZ,
        provenance   JSONB NOT NULL DEFAULT '{}',
        metadata     JSONB NOT NULL DEFAULT '{}',
        UNIQUE(text_hash, memory_type)
    );

    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
    CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
    CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(text_hash);
"""


# ---------------------------------------------------------------------------
# PostgresSqlStore
# ---------------------------------------------------------------------------

class PostgresSqlStore:
    """
    SqlStore backed by PostgreSQL.

    Provides: CRUD, dedup via text_hash, full-text search via pg_trgm
    (degrading to ILIKE substring match when the extension is unavailable),
    metrics, decay, and migration support (iter_all / bulk_upsert).
    """

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Any = None
        self.is_available: bool = False
        # Feature-scoped pg_trgm availability (HIGH-14): set during schema
        # init, consulted by search_text. Never blocks initialization.
        self._trgm_available: bool = False
        self._trgm_degrade_warned: bool = False
        self._init_lock = threading.Lock()
        self._connect()

    def _connect(self) -> bool:
        """Initialize connection pool and ensure schema exists."""
        if self.is_available:
            return True

        with self._init_lock:
            if self.is_available:
                return True

            try:
                pool = _PsycopgPool(self._dsn)
                with pool.connection() as conn:
                    conn.execute("SELECT 1")
                    self._ensure_schema(conn)

                self._pool = pool
                self.is_available = True
                logger.info("PostgresSqlStore initialized")
                return True

            except ImportError:
                logger.warning("psycopg not installed, using Qdrant-only mode")
                return False
            except Exception as e:
                logger.warning(f"PostgresSqlStore init failed: {_safe_error(e)}")
                self._pool = None
                self.is_available = False
                return False

    def _ensure_schema(self, conn) -> None:
        """Create tables and indexes if they don't exist.

        pg_trgm is feature-scoped (HIGH-14): the core schema DDL never depends
        on it. We attempt the extension + trigram GIN index as an optional
        feature and record availability; a failure (e.g. non-superuser role)
        only disables similarity ranking, never schema init.
        """
        conn.execute(_SCHEMA_DDL)

        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_text_trgm "
                "ON memories USING gin (text gin_trgm_ops)"
            )
            self._trgm_available = True
            logger.info("pg_trgm available: trigram full-text search enabled")
        except Exception as e:
            self._trgm_available = False
            logger.warning(
                "pg_trgm extension not available (similarity search will "
                f"degrade to substring match): {_safe_error(e)}"
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert_memory(
        self,
        memory_id: str,
        text: str,
        memory_type: str,
        source: str,
        tags: list[str],
        path: str | None,
        commit: str | None,
        agent_id: str | None,
        sensitivity: str,
        retention_policy: str,
    ) -> str:
        """Upsert a memory. Uses text_hash + memory_type as dedup key."""
        if not self.is_available:
            return memory_id

        thash = text_hash(text)
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags)

        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT id, usage_count FROM memories WHERE text_hash = %s AND memory_type = %s",
                    (thash, memory_type),
                ).fetchone()

                if row:
                    existing_id = str(row["id"])
                    conn.execute("""
                        UPDATE memories SET
                            text = %s,
                            tags = %s,
                            source = %s,
                            path = COALESCE(%s, path),
                            commit = COALESCE(%s, commit),
                            agent_id = COALESCE(%s, agent_id),
                            sensitivity = %s,
                            retention_policy = %s,
                            usage_count = usage_count + 1,
                            last_accessed = %s
                        WHERE id = %s
                    """, (text, tags_json, source, path, commit, agent_id,
                          sensitivity, retention_policy, now, existing_id))
                    logger.info(f"PG: dedup update for memory {existing_id}")
                    return existing_id
                else:
                    row = conn.execute("""
                        INSERT INTO memories (id, text, text_hash, memory_type, source, tags,
                            path, commit, agent_id, sensitivity, retention_policy, created_at, last_accessed)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (text_hash, memory_type) DO UPDATE SET
                            text = EXCLUDED.text,
                            tags = EXCLUDED.tags,
                            usage_count = memories.usage_count + 1,
                            last_accessed = EXCLUDED.last_accessed
                        RETURNING id
                    """, (memory_id, text, thash, memory_type, source, tags_json,
                          path, commit, agent_id, sensitivity, retention_policy, now, now)).fetchone()
                    actual_id = str(row["id"]) if row else memory_id
                    logger.info(f"PG: inserted memory {actual_id}")
                    return actual_id

        except Exception as e:
            logger.error(f"PG upsert failed: {_safe_error(e)}")
            return memory_id

    def get_memory(self, memory_id: str) -> dict | None:
        """Get a memory by ID, also increments usage_count."""
        if not self.is_available:
            return None

        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM memories WHERE id = %s", (memory_id,)
                ).fetchone()
                if row is None:
                    return None
                conn.execute("""
                    UPDATE memories SET usage_count = usage_count + 1, last_accessed = NOW()
                    WHERE id = %s
                """, (memory_id,))
                return dict(row)
        except Exception as e:
            logger.error(f"PG get failed: {_safe_error(e)}")
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self.is_available:
            return False

        try:
            with self._pool.connection() as conn:
                result = conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"PG delete failed: {_safe_error(e)}")
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        limit: int = 10,
        *,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
    ) -> list[dict]:
        """Full-text search.

        Uses pg_trgm similarity when the extension is available (HIGH-14);
        otherwise degrades to an ILIKE substring match ordered by usage and
        recency instead of failing on the missing similarity() function.
        Results keep the same shape (sim_score is 0.0 in the degraded path).
        """
        if not self.is_available:
            return []

        try:
            with self._pool.connection() as conn:
                conditions: list[str] = []
                params: list[Any] = []

                if memory_type:
                    conditions.append("memory_type = %s")
                    params.append(memory_type)
                if agent_id:
                    conditions.append("agent_id = %s")
                    params.append(agent_id)
                if tags:
                    for tag in tags:
                        conditions.append("tags @> %s::jsonb")
                        params.append(json.dumps([tag]))

                if self._trgm_available:
                    conditions.append("similarity(text, %s) > 0.1")
                    params.append(query)
                    sim_select = "similarity(text, %s) AS sim_score"
                    sim_params = [query]
                    order_by = "sim_score DESC"
                else:
                    if not self._trgm_degrade_warned:
                        logger.warning(
                            "pg_trgm unavailable: search_text degrading to "
                            "ILIKE substring match (no similarity ranking)"
                        )
                        self._trgm_degrade_warned = True
                    conditions.append("text ILIKE %s")
                    params.append(f"%{query}%")
                    sim_select = "0.0 AS sim_score"
                    sim_params = []
                    order_by = "usage_count DESC, created_at DESC"

                where = " AND ".join(conditions)

                # Placeholder order is textual: sim_select's %s comes first,
                # then the WHERE params (which already end with the similarity/
                # ILIKE argument), then LIMIT.
                rows = conn.execute(f"""
                    SELECT id, text, memory_type, tags, source, created_at, last_accessed, usage_count,
                           {sim_select}
                    FROM memories
                    WHERE {where}
                    ORDER BY {order_by}
                    LIMIT %s
                """, [*sim_params, *params, limit]).fetchall()

                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"PG search failed: {_safe_error(e)}")
            return []

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Get memory system metrics."""
        if not self.is_available:
            return {}

        try:
            with self._pool.connection() as conn:
                total = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
                by_type = conn.execute("""
                    SELECT memory_type, COUNT(*) as cnt FROM memories GROUP BY memory_type ORDER BY cnt DESC
                """).fetchall()
                by_agent = conn.execute("""
                    SELECT agent_id, COUNT(*) as cnt FROM memories GROUP BY agent_id ORDER BY cnt DESC LIMIT 10
                """).fetchall()
                total_usage = conn.execute("SELECT SUM(usage_count) as cnt FROM memories").fetchone()

                return {
                    "total": total["cnt"] if total else 0,
                    "by_type": {r["memory_type"]: r["cnt"] for r in by_type},
                    "by_agent": {r["agent_id"]: r["cnt"] for r in by_agent},
                    "total_usage": total_usage["cnt"] if total_usage and total_usage["cnt"] else 0,
                }
        except Exception as e:
            logger.error(f"PG metrics failed: {_safe_error(e)}")
            return {}

    def decay_memories(
        self,
        ttl_days: int,
        min_usage_count: int,
        retention_policy: str | None = None,
    ) -> int:
        """Delete expired memories based on TTL and usage. Returns count deleted."""
        if not self.is_available:
            return 0

        try:
            with self._pool.connection() as conn:
                conditions = ["retention_policy != 'permanent'"]
                params: list[Any] = []

                interval_str = f"{int(ttl_days)} days"
                conditions.append("(last_accessed < NOW() - INTERVAL %s OR created_at < NOW() - INTERVAL %s)")
                params.append(interval_str)
                params.append(interval_str)
                conditions.append("usage_count < %s")
                params.append(min_usage_count)

                if retention_policy:
                    conditions.append("retention_policy = %s")
                    params.append(retention_policy)

                where = " AND ".join(conditions)
                result = conn.execute(f"DELETE FROM memories WHERE {where}", params)
                return result.rowcount
        except Exception as e:
            logger.error(f"PG decay failed: {_safe_error(e)}")
            return 0

    def get_all_memory_ids(self) -> list[str]:
        """Get all memory IDs (for reindex coordination)."""
        if not self.is_available:
            return []

        try:
            with self._pool.connection() as conn:
                rows = conn.execute("SELECT id FROM memories").fetchall()
                return [str(r["id"]) for r in rows]
        except Exception as e:
            logger.error(f"PG get_all_ids failed: {_safe_error(e)}")
            return []

    # ------------------------------------------------------------------
    # Migration support (iter_all / bulk_upsert)
    # ------------------------------------------------------------------

    def iter_all(self):
        """Stream all memory records. Yields one dict at a time."""
        if not self.is_available:
            return

        try:
            with self._pool.connection() as conn:
                cursor = conn.execute("SELECT * FROM memories")
                cols = [d[0] for d in cursor.description] if cursor.description else []
                while True:
                    batch = cursor.fetchmany(100)
                    if not batch:
                        break
                    for row in batch:
                        yield dict(row) if not isinstance(row, dict) else row
        except Exception as e:
            logger.error(f"PG iter_all failed: {_safe_error(e)}")

    def bulk_upsert(self, rows) -> int:
        """Bulk upsert memory records. Returns count inserted."""
        count = 0
        for row in rows:
            self.upsert_memory(
                memory_id=row.get("id", ""),
                text=row.get("text", ""),
                memory_type=row.get("memory_type", "concept"),
                source=row.get("source", "agent_action"),
                tags=row.get("tags", []),
                path=row.get("path"),
                commit=row.get("commit"),
                agent_id=row.get("agent_id"),
                sensitivity=row.get("sensitivity", "low"),
                retention_policy=row.get("retention_policy", "auto-delete"),
            )
            count += 1
        return count
