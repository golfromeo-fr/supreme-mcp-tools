"""
TursoSqlStore — SqlStore implementation backed by Turso / libSQL.

Provides the same SqlStore interface as PostgresSqlStore but uses
libSQL's FTS5 for full-text search (token-based, not fuzzy).

Phase 3 of the backend abstraction plan.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Iterator, Iterable
from datetime import datetime, timezone

from shared.hashing import text_hash
from shared.text_search_utils import escape_fts5_query as _escape_fts5_query

logger = logging.getLogger(__name__)


def _exec_statements(conn, sql: str) -> None:
    """
    Execute a multi-statement SQL script by splitting on semicolons.
    libsql_experimental's executescript is unreliable, so we iterate manually.
    Strips line comments and blank lines before execution.
    """
    # Strip line comments (-- ...) to avoid the "commit" reserved-word trap
    lines = []
    for line in sql.split("\n"):
        # Remove trailing -- comments
        comment_pos = line.find("--")
        if comment_pos >= 0:
            # Check if -- is inside a string (rough heuristic)
            # Just strip comments that aren't inside strings
            before_comment = line[:comment_pos]
            # Count quotes in before_comment
            if before_comment.count("'") % 2 == 0 and before_comment.count('"') % 2 == 0:
                line = before_comment
        lines.append(line)
    cleaned = "\n".join(lines)

    for stmt in cleaned.split(";"):
        stripped = stmt.strip()
        if not stripped:
            continue
        try:
            conn.execute(stripped)
        except Exception as e:
            err_str = str(e).lower()
            if "already exists" not in err_str and "duplicate" not in err_str:
                logger.debug(f"Schema stmt failed (may be OK): {e}")


class TursoSqlStore:
    """SqlStore backed by Turso / libSQL."""

    def __init__(self, url: str, auth_token: str | None = None):
        import libsql_experimental as libsql

        # Concurrency: a single connection is shared across all requests. This is
        # safe because libsql_experimental's C binding serializes statements via
        # an internal mutex (verified: 4 threads x 50 interleaved INSERT/SELECT
        # ops on one connection produced 200/200 correct rows, no "database is
        # locked"). autocommit=True keeps each statement its own transaction so
        # one thread's implicit transaction can't block another. Unlike oraclemcp
        # we therefore do NOT add a threading.Lock here — it would only reduce
        # throughput without improving correctness. If a future build drops the
        # internal mutex, switch to a small Python connection pool (libsql
        # supports multiple connect()s to the same URL).
        if auth_token:
            self._conn = libsql.connect(url, auth_token=auth_token)
        else:
            self._conn = libsql.connect(url)
        self._conn.autocommit = True
        self.is_available = True
        self._ensure_schema()
        logger.info(f"TursoSqlStore initialized (url={url[:30]}...)")

    def _ensure_schema(self) -> None:
        """
        Run the DDL. libsql_experimental doesn't reliably create triggers via
        executescript — so we split statements and create tables only.
        FTS5 sync is handled manually in upsert_memory/delete_memory (no triggers).

        The canonical schema lives in ``turso_sql_schema.sql`` next to this module;
        ``_INLINE_SCHEMA`` below is a last-resort fallback for environments where
        the .sql file is not packaged (e.g. frozen/imported standalone). Keep the
        two in sync — a test in tests/test_turso_stores.py asserts parity.
        """
        schema_path = os.path.join(os.path.dirname(__file__), "turso_sql_schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                schema_sql = f.read()
        else:
            schema_sql = _INLINE_SCHEMA

        # Strip line comments (-- ...) to avoid "commit" reserved-word trap
        lines = []
        for line in schema_sql.split("\n"):
            comment_pos = line.find("--")
            if comment_pos >= 0:
                before_comment = line[:comment_pos]
                if before_comment.count("'") % 2 == 0 and before_comment.count('"') % 2 == 0:
                    line = before_comment
            lines.append(line)
        cleaned = "\n".join(lines)

        # Execute only CREATE TABLE and CREATE INDEX (skip triggers, BEGIN/END fragments)
        for stmt in cleaned.split(";"):
            stripped = stmt.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            # Skip triggers and trigger body statements
            if "TRIGGER" in upper or upper in ("BEGIN", "END"):
                continue
            # Skip statements that reference new./old. (trigger body fragments)
            if "NEW." in upper or "OLD." in upper:
                continue
            # Skip comment-only lines
            if stripped.startswith("--"):
                continue
            try:
                self._conn.execute(stripped)
            except Exception as e:
                err_str = str(e).lower()
                if "already exists" not in err_str and "duplicate" not in err_str:
                    logger.debug(f"Schema stmt failed (may be OK): {e}")

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
        """Upsert a memory. Uses text_hash + memory_type as dedup key.

        Note: FTS5 sync is done manually (libsql doesn't support triggers reliably).
        """
        if not self.is_available:
            return memory_id

        thash = text_hash(text)
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags)

        try:
            # Check dedup
            row = self._conn.execute(
                "SELECT id, rowid, usage_count FROM memories WHERE text_hash = ? AND memory_type = ?",
                (thash, memory_type),
            ).fetchone()

            if row:
                existing_id = str(row[0])
                existing_rowid = row[1]
                self._conn.execute("""
                    UPDATE memories SET
                        text = ?, tags = ?, source = ?,
                        path = COALESCE(?, path), "commit" = COALESCE(?, "commit"),
                        agent_id = COALESCE(?, agent_id),
                        sensitivity = ?, retention_policy = ?,
                        usage_count = usage_count + 1, last_accessed = ?
                    WHERE id = ?
                """, (text, tags_json, source, path, commit, agent_id,
                      sensitivity, retention_policy, now, existing_id))
                # Update FTS5 entry (delete + insert, since content table sync)
                self._conn.execute(
                    "INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', ?, ?)",
                    (existing_rowid, text),
                )
                self._conn.execute(
                    "INSERT INTO memories_fts(rowid, text) VALUES (?, ?)",
                    (existing_rowid, text),
                )
                logger.info(f"Turso: dedup update for memory {existing_id}")
                return existing_id

            # Insert new (ON CONFLICT handles race)
            self._conn.execute("""
                INSERT INTO memories (id, text, text_hash, memory_type, source, tags,
                    path, "commit", agent_id, sensitivity, retention_policy, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(text_hash, memory_type) DO UPDATE SET
                    text = excluded.text, tags = excluded.tags,
                    usage_count = usage_count + 1, last_accessed = excluded.last_accessed
            """, (memory_id, text, thash, memory_type, source, tags_json,
                  path, commit, agent_id, sensitivity, retention_policy, now, now))
            # Add to FTS5 (get rowid from the just-inserted row)
            fts_row = self._conn.execute(
                "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if fts_row:
                self._conn.execute(
                    "INSERT INTO memories_fts(rowid, text) VALUES (?, ?)",
                    (fts_row[0], text),
                )
            return memory_id

        except Exception as e:
            logger.error(f"Turso upsert failed: {e}")
            return memory_id

    def get_memory(self, memory_id: str) -> dict | None:
        """Get a memory by ID, also increments usage_count."""
        if not self.is_available:
            return None

        try:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                return None
            cols = [d[0] for d in self._conn.execute("SELECT * FROM memories LIMIT 0").description]
            result = dict(zip(cols, row))
            self._conn.execute(
                "UPDATE memories SET usage_count = usage_count + 1, last_accessed = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), memory_id),
            )
            # Parse JSON fields
            try:
                result["tags"] = json.loads(result.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                result["tags"] = []
            try:
                result["provenance"] = json.loads(result.get("provenance", "{}"))
            except (json.JSONDecodeError, TypeError):
                result["provenance"] = {}
            return result
        except Exception as e:
            logger.error(f"Turso get failed: {e}")
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID, keeping the FTS5 index in sync.

        libsql doesn't reliably support the sync triggers in the schema, so the
        FTS5 'delete' command is issued explicitly before the base row goes away.
        Without this the FTS index accumulates orphan rows for deleted memories.
        """
        if not self.is_available:
            return False

        try:
            row = self._conn.execute(
                "SELECT rowid, text FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return False
            # Remove from FTS5 first (needs the row's text for the content table)
            try:
                self._conn.execute(
                    "INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', ?, ?)",
                    (row[0], row[1]),
                )
            except Exception as e:
                logger.debug(f"FTS5 delete-side sync skipped: {e}")
            result = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Turso delete failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Search (FTS5 token-based, NOT fuzzy — S3 escape for injection)
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
        """FTS5 full-text search (S3: user query is escaped)."""
        if not self.is_available:
            return []

        try:
            conditions: list[str] = []
            params_list: list = []

            fts_query = _escape_fts5_query(query)
            conditions.append("memories_fts MATCH ?")
            params_list.append(fts_query)

            if memory_type:
                conditions.append("m.memory_type = ?")
                params_list.append(memory_type)
            if agent_id:
                conditions.append("m.agent_id = ?")
                params_list.append(agent_id)
            if tags:
                for tag in tags:
                    # JSON1 array contains check (S3-compatible equivalent of MatchContains)
                    conditions.append("EXISTS (SELECT 1 FROM json_each(m.tags) WHERE value = ?)")
                    params_list.append(tag)

            params_list.append(limit)

            where = " AND ".join(conditions)
            rows = self._conn.execute(f"""
                SELECT m.id, m.text, m.memory_type, m.tags, m.source,
                       m.created_at, m.last_accessed, m.usage_count,
                       bm25(memories_fts) AS sim_score
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE {where}
                ORDER BY sim_score
                LIMIT ?
            """, tuple(params_list)).fetchall()

            results = []
            for row in rows:
                result = {
                    "id": row[0],
                    "text": row[1],
                    "memory_type": row[2],
                    "tags": row[3],
                    "source": row[4],
                    "created_at": row[5],
                    "last_accessed": row[6],
                    "usage_count": row[7],
                    "similarity": -row[8],  # S4: bm25 returns negative; negate to similarity
                }
                # Parse JSON tags
                try:
                    result["tags"] = json.loads(result["tags"])
                except (json.JSONDecodeError, TypeError):
                    pass
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Turso search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Get memory system metrics."""
        if not self.is_available:
            return {}

        try:
            total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type_rows = self._conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type ORDER BY COUNT(*) DESC"
            ).fetchall()
            by_agent_rows = self._conn.execute(
                "SELECT agent_id, COUNT(*) FROM memories GROUP BY agent_id ORDER BY COUNT(*) DESC LIMIT 10"
            ).fetchall()
            total_usage = self._conn.execute("SELECT COALESCE(SUM(usage_count), 0) FROM memories").fetchone()[0]

            return {
                "total": total,
                "by_type": {r[0]: r[1] for r in by_type_rows},
                "by_agent": {r[0]: r[1] for r in by_agent_rows},
                "total_usage": total_usage,
            }
        except Exception as e:
            logger.error(f"Turso metrics failed: {e}")
            return {}

    def decay_memories(
        self,
        ttl_days: int,
        min_usage_count: int,
        retention_policy: str | None = None,
    ) -> int:
        """Delete expired memories based on TTL and usage (julianday arithmetic)."""
        if not self.is_available:
            return 0

        try:
            conditions = ["retention_policy != 'permanent'"]
            params_list: list = []

            conditions.append("(julianday('now') - julianday(last_accessed) > ? OR julianday('now') - julianday(created_at) > ?)")
            params_list.extend([ttl_days, ttl_days])
            conditions.append("usage_count < ?")
            params_list.append(min_usage_count)

            if retention_policy:
                conditions.append("retention_policy = ?")
                params_list.append(retention_policy)

            where = " AND ".join(conditions)
            result = self._conn.execute(f"DELETE FROM memories WHERE {where}", tuple(params_list))
            return result.rowcount
        except Exception as e:
            logger.error(f"Turso decay failed: {e}")
            return 0

    def get_all_memory_ids(self) -> list[str]:
        """Get all memory IDs (for reindex coordination)."""
        if not self.is_available:
            return []

        try:
            rows = self._conn.execute("SELECT id FROM memories").fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"Turso get_all_ids failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Migration support — stream without materializing (R1, R15)
    # ------------------------------------------------------------------

    def iter_all(self) -> Iterator[dict]:
        """Stream all memory records."""
        if not self.is_available:
            return

        try:
            cursor = self._conn.execute("SELECT * FROM memories")
            cols = [d[0] for d in cursor.description] if cursor.description else []
            while True:
                batch = cursor.fetchmany(100)
                if not batch:
                    break
                for row in batch:
                    d = dict(zip(cols, row))
                    try:
                        d["tags"] = json.loads(d.get("tags", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        d["tags"] = []
                    try:
                        d["provenance"] = json.loads(d.get("provenance", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        d["provenance"] = {}
                    yield d
        except Exception as e:
            logger.error(f"Turso iter_all failed: {e}")

    def bulk_upsert(self, rows: Iterable[dict]) -> int:
        """Bulk upsert memory records."""
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


# Inline schema for in-memory mode (when the .sql file isn't available).
_INLINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    text_hash       TEXT NOT NULL,
    memory_type     TEXT NOT NULL DEFAULT 'concept',
    source          TEXT NOT NULL DEFAULT 'agent_action',
    tags            TEXT NOT NULL DEFAULT '[]',
    path            TEXT,
    "commit"        TEXT,
    agent_id        TEXT,
    sensitivity     TEXT NOT NULL DEFAULT 'low',
    retention_policy TEXT NOT NULL DEFAULT 'auto-delete',
    usage_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_accessed   TEXT,
    provenance      TEXT NOT NULL DEFAULT '{}',
    metadata        TEXT NOT NULL DEFAULT '{}',
    UNIQUE(text_hash, memory_type)
);

CREATE INDEX IF NOT EXISTS idx_memories_type   ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_agent  ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_hash   ON memories(text_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""
