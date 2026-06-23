-- Turso/libSQL schema for SqlStore memory metadata.
-- This is the SQL DDL for the TursoSqlStore. Run it once at startup.
-- In-memory or local-file libSQL: just executes this script.
-- Cloud Turso: apply with `turso db shell <db> < turso_sql_schema.sql`.

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

-- FTS5 full-text index (replaces pg_trgm similarity search).
-- Token-based, NOT fuzzy. For fuzzy matching, PG is the better backend.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Keep FTS5 in sync with the memories table.
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
