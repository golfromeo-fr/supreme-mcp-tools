# MemoryMCP PostgreSQL Integration

PostgreSQL works **alongside Qdrant** as an optional metadata and deduplication layer.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  MemoryMCP                       │
├────────────────────┬────────────────────────────┤
│  Qdrant (required) │  PostgreSQL (optional)      │
│  ────────────────  │  ──────────────────────     │
│  • Vector storage   │  • Metadata (source, tags,  │
│  • Semantic search  │    path, agent_id, ...)     │
│  • Embeddings       │  • Full-text search (trgm)  │
│                     │  • Deduplication (text hash) │
│                     │  • Usage counters            │
│                     │  • Retention policy tracking  │
│                     │  • Easy backup (pg_dump)     │
└────────────────────┴────────────────────────────┘
```

- **Qdrant** handles vector similarity search (required)
- **PostgreSQL** handles metadata, dedup, and text search (optional — graceful fallback to Qdrant-only)

## Setup

### 1. Run PostgreSQL (Podman/Docker)

```bash
podman run -d \
  --name postgresql \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=memorymcp \
  -e POSTGRES_USER=gr \
  -e POSTGRES_DB=memorymcp \
  -v /nvme2/gr/postgresql_data:/var/lib/postgresql/data \
  docker.io/library/postgres:16
```

### 2. Configure `.env`

```bash
POSTGRES_HOST=192.168.0.1
POSTGRES_PORT=5432
POSTGRES_USER=gr
POSTGRES_PASSWORD=memorymcp
POSTGRES_DB=memorymcp
```

If `POSTGRES_HOST` is not set, memorymcp runs in **Qdrant-only mode** automatically.

### 3. Install dependency

```bash
pip install "psycopg[binary]>=3.1.0"
```

### 4. Start memorymcp

On startup, logs will show:

```
PostgreSQL store available for metadata and dedup
```

or if not configured:

```
PostgreSQL not available, using Qdrant-only mode
```

No further action needed — schema (tables, indexes, `pg_trgm` extension) is created automatically.

## What PostgreSQL Adds

| Feature | Qdrant-only | With PostgreSQL |
|---|---|---|
| Vector search | ✅ | ✅ |
| Metadata storage | In payload (bloat) | ✅ Dedicated columns |
| **Deduplication** | ❌ Always creates new | ✅ Hash-based upsert |
| Full-text search | ❌ | ✅ `pg_trgm` fuzzy match |
| Usage counters | Approximate (payload) | ✅ Accurate (ACID) |
| Backup | Qdrant snapshot | `pg_dump` + Qdrant snapshot |
| Retention policy | Scroll + delete | SQL `DELETE WHERE` |

## How Deduplication Works

On `upsertMemory`, PostgreSQL computes a SHA-256 hash of `(normalized_text, memory_type)`:

- **New content** → inserted into both PG and Qdrant
- **Duplicate content** → PG returns the existing ID, usage counter incremented, Qdrant updated in-place

This means calling `upsertMemory` with the same insight twice won't create duplicates.

## Backup

```bash
# Full backup
pg_dump -h 192.168.0.1 -U gr memorymcp > memorymcp_backup.sql

# Restore
psql -h 192.168.0.1 -U gr memorymcp < memorymcp_backup.sql
```

For a complete backup, also snapshot Qdrant:

```bash
curl -X POST http://192.168.0.1:6333/collections/memory-store/snapshots
```

## Schema

Auto-created on first connection:

```sql
CREATE TABLE memories (
    id              UUID PRIMARY KEY,
    text            TEXT NOT NULL,
    text_hash       TEXT NOT NULL,          -- SHA-256 of normalized text
    memory_type     TEXT NOT NULL DEFAULT 'concept',
    source          TEXT NOT NULL DEFAULT 'agent_action',
    tags            JSONB NOT NULL DEFAULT '[]',
    path            TEXT,
    commit          TEXT,
    agent_id        TEXT,
    sensitivity     TEXT NOT NULL DEFAULT 'low',
    retention_policy TEXT NOT NULL DEFAULT 'auto-delete',
    usage_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed   TIMESTAMPTZ,
    provenance      JSONB NOT NULL DEFAULT '{}',
    metadata        JSONB NOT NULL DEFAULT '{}',
    UNIQUE(text_hash, memory_type)
);

-- Indexes
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_hash ON memories(text_hash);

-- Full-text search (pg_trgm)
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_memories_text_trgm ON memories USING gin (text gin_trgm_ops);
```

## Useful Queries

```sql
-- Memory count by type
SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type ORDER BY count DESC;

-- Most accessed memories
SELECT id, text, usage_count FROM memories ORDER BY usage_count DESC LIMIT 20;

-- Find duplicates (same hash)
SELECT text_hash, COUNT(*) FROM memories GROUP BY text_hash HAVING COUNT(*) > 1;

-- Memories not accessed in 30 days
SELECT id, text, last_accessed FROM memories
WHERE last_accessed < NOW() - INTERVAL '30 days'
AND retention_policy != 'permanent';

-- Full-text search
SELECT id, text, similarity(text, 'authentication pattern') AS score
FROM memories
WHERE similarity(text, 'authentication pattern') > 0.1
ORDER BY score DESC LIMIT 10;
```
