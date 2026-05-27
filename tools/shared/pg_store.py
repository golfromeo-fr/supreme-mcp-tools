"""
PostgreSQL Store - Metadata and full-text search for memories.

Works alongside Qdrant (vectors) to provide:
- Persistent metadata storage with true upsert
- Full-text search via pg_trgm similarity
- Deduplication via text hashing
- Easy backup via pg_dump

Optional: if PostgreSQL is not configured, gracefully falls back to Qdrant-only mode.
"""

import os
import json
import logging
import hashlib
import threading
from typing import Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_pool: Any = None
_pg_available: bool = False
_init_lock = threading.Lock()


def _get_pg_dsn() -> str | None:
    """Build PostgreSQL DSN from env vars."""
    host = os.getenv("POSTGRES_HOST")
    if not host:
        return None
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "gr")
    password = os.getenv("POSTGRES_PASSWORD", "")
    dbname = os.getenv("POSTGRES_DB", "memorymcp")
    return f"host={host} port={port} user={user} password={password} dbname={dbname}"


def is_available() -> bool:
    """Check if PostgreSQL store is available."""
    return _pg_available


def init_pg() -> bool:
    """Initialize PostgreSQL connection pool and ensure schema exists."""
    global _pool, _pg_available

    if _pg_available:
        return True

    with _init_lock:
        if _pg_available:
            return True

        dsn = _get_pg_dsn()
        if not dsn:
            logger.info("PostgreSQL not configured (POSTGRES_HOST not set), using Qdrant-only mode")
            return False

        try:
            import psycopg
            from psycopg.rows import dict_row

            pool = psycopg_pool(dsn)

            with pool.connection() as conn:
                conn.execute("SELECT 1")
                _ensure_schema(conn)

            _pool = pool
            _pg_available = True
            logger.info("PostgreSQL store initialized")
            return True

        except ImportError:
            logger.warning("psycopg not installed, using Qdrant-only mode")
            return False
        except Exception as e:
            logger.warning(f"PostgreSQL init failed, using Qdrant-only mode: {e}")
            _pool = None
            _pg_available = False
            return False


class psycopg_pool:
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
            logger.info(f"psycopg_pool not available, using per-connection fallback: {e}")

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


def _ensure_schema(conn) -> None:
    """Create tables and indexes if they don't exist."""
    conn.execute("""
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
    """)

    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_text_trgm "
            "ON memories USING gin (text gin_trgm_ops)"
        )
    except Exception as e:
        logger.warning(f"pg_trgm extension not available (full-text search disabled): {e}")


def text_hash(text: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:40]


def upsert_memory(
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
    """
    Upsert a memory into PostgreSQL.

    Uses text_hash + memory_type as dedup key.
    Returns the actual memory_id used (may differ if dedup matched).
    """
    if not _pg_available:
        return memory_id

    thash = text_hash(text)
    now = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(tags)

    try:
        with _pool.connection() as conn:
            # Check for existing memory with same hash + type
            row = conn.execute(
                "SELECT id, usage_count FROM memories WHERE text_hash = %s AND memory_type = %s",
                (thash, memory_type)
            ).fetchone()

            if row:
                # Update existing - increment usage, update text/tags if changed
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
                # Insert new (ON CONFLICT handles race condition)
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
        logger.error(f"PG upsert failed: {e}")
        return memory_id


def get_memory(memory_id: str) -> dict | None:
    """Get a memory by ID, also increments usage_count."""
    if not _pg_available:
        return None

    try:
        with _pool.connection() as conn:
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
        logger.error(f"PG get failed: {e}")
        return None


def delete_memory(memory_id: str) -> bool:
    """Delete a memory by ID."""
    if not _pg_available:
        return False

    try:
        with _pool.connection() as conn:
            result = conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            return result.rowcount > 0
    except Exception as e:
        logger.error(f"PG delete failed: {e}")
        return False


def search_text(query: str, limit: int = 10, memory_type: str | None = None,
                tags: list[str] | None = None, agent_id: str | None = None) -> list[dict]:
    """
    Full-text search using pg_trgm similarity.

    Returns list of {id, text, memory_type, tags, similarity} sorted by relevance.
    """
    if not _pg_available:
        return []

    try:
        with _pool.connection() as conn:
            conditions = []
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

            conditions.append("similarity(text, %s) > 0.1")
            params.append(query)

            where = " AND ".join(conditions)

            # query appears twice: once for WHERE similarity(), once for SELECT similarity()
            params.append(query)
            params.append(limit)

            rows = conn.execute(f"""
                SELECT id, text, memory_type, tags, source, created_at, last_accessed, usage_count,
                       similarity(text, %s) AS sim_score
                FROM memories
                WHERE {where}
                ORDER BY sim_score DESC
                LIMIT %s
            """, params).fetchall()

            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"PG search failed: {e}")
        return []


def get_metrics() -> dict:
    """Get memory system metrics from PostgreSQL."""
    if not _pg_available:
        return {}

    try:
        with _pool.connection() as conn:
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
        logger.error(f"PG metrics failed: {e}")
        return {}


def decay_memories(ttl_days: int, min_usage_count: int, retention_policy: str | None = None) -> int:
    """
    Delete expired memories based on TTL and usage.

    Returns count of deleted memories.
    """
    if not _pg_available:
        return 0

    try:
        with _pool.connection() as conn:
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
        logger.error(f"PG decay failed: {e}")
        return 0


def get_all_memory_ids() -> list[str]:
    """Get all memory IDs (for reindex coordination)."""
    if not _pg_available:
        return []

    try:
        with _pool.connection() as conn:
            rows = conn.execute("SELECT id FROM memories").fetchall()
            return [str(r["id"]) for r in rows]
    except Exception as e:
        logger.error(f"PG get_all_ids failed: {e}")
        return []
