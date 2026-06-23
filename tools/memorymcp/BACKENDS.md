# Backend Configuration & Migration

MemoryMCP and RAGMCP support pluggable storage backends. You can mix-and-match
any SQL backend with any Vector backend, or use a single backend for both.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Tool Layer                             │
│  (memory_tools.py, ragmcp_fastmcp.py, indexer)            │
├────────────────────┬─────────────────────────────────────┤
│   SqlStore (ABC)   │      VectorStore (ABC)              │
│   ──────────────   │      ──────────────                 │
│   • CRUD           │      • ANN query (dense/sparse)     │
│   • Dedup          │      • Hybrid search                │
│   • Full-text      │      • Payload filter               │
│   • Metrics        │      • Scroll / iterate             │
│   • Decay          │                                     │
├─────────┬──────────┼──────────┬──────────┬───────────────┤
│ PG      │ Turso    │ Qdrant   │ Turso    │ PG (pgvector) │
│ SQL     │ SQL      │ Vector   │ Vector   │ Vector        │
└─────────┴──────────┴──────────┴──────────┴───────────────┘
```

## Supported Combos

| Combo | SQL | Vector | Containers needed | Use case |
|-------|-----|--------|-------------------|----------|
| 1 (default) | PostgreSQL | Qdrant | PG + Qdrant | Full-featured |
| 2 | Turso | Turso | **None** | Single embedded process |
| 3 | PostgreSQL | Turso | PG | Drop Qdrant |
| 4 | Turso | Qdrant | Qdrant | Drop PG |
| 5 | (none) | Qdrant | Qdrant | Qdrant-only (existing) |
| 6 | PostgreSQL | PG (pgvector) | PG | Single PG container |
| 7 | Turso | PG (pgvector) | PG | Turso SQL + PG vectors |

## Configuration

### Per-tool `config.json` (recommended)

```json
{
  "auth": { "api_key": "..." },
  "storage": {
    "sql":    { "backend": "postgres" },
    "vector": { "backend": "qdrant" }
  }
}
```

Turso single-backend example:

```json
{
  "storage": {
    "sql":    { "backend": "turso", "url_env": "TURSO_DATABASE_URL", "token_env": "TURSO_AUTH_TOKEN" },
    "vector": { "backend": "turso", "url_env": "TURSO_DATABASE_URL", "token_env": "TURSO_AUTH_TOKEN" }
  }
}
```

### Environment variables (auto-detected if config absent)

| Variable | Detects | Notes |
|----------|---------|-------|
| `POSTGRES_HOST` | PostgreSQL SQL backend | Also needs `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` |
| `QDRANT_HOST` | Qdrant Vector backend | Also `QDRANT_PORT` (default 6333) |
| `TURSO_DATABASE_URL` | Turso SQL + Vector | `file:local.db` for local, `libsql://...` for cloud |
| `TURSO_AUTH_TOKEN` | Turso auth | Required for cloud, omitted for local file |

Priority: **explicit config.json > env vars > default (qdrant + optional pg)**.

---

## Setup Guides

### PG + Qdrant (default combo)

```bash
# PostgreSQL
podman run -d --name postgresql -p 5432:5432 \
  -e POSTGRES_PASSWORD=memorymcp -e POSTGRES_USER=gr -e POSTGRES_DB=memorymcp \
  -v /nvme2/gr/postgresql_data:/var/lib/postgresql/data \
  docker.io/library/postgres:16

# Qdrant
podman run -d --name qdrant -p 6333:6333 \
  -v /nvme2/gr/qdrant_data:/qdrant/storage \
  docker.io/qdrant/qdrant

# Start tools (auto-detects both)
export POSTGRES_HOST=127.0.0.1
export QDRANT_HOST=127.0.0.1
python -m launcher --tools memorymcp,ragmcp
```

### Turso (single backend, no containers)

```bash
# Option A: local file (zero-network, great for dev)
export TURSO_DATABASE_URL="file:/path/to/memory.db"

# Option B: Turso cloud
export TURSO_DATABASE_URL="libsql://your-db.turso.io"
export TURSO_AUTH_TOKEN="your-token"

# Start tools
python -m launcher --tools memorymcp,ragmcp
```

Install the libSQL client:

```bash
pip install libsql-experimental
```

### PG + PG via pgvector (single container)

Requires the `pgvector` extension on your PostgreSQL server.

```bash
# In config.json for memorymcp:
{
  "storage": {
    "sql":    { "backend": "postgres" },
    "vector": { "backend": "postgres" }
  }
}

# Enable pgvector extension (one-time)
psql -h localhost -U gr -d memorymcp -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## Migration Tool

Move data between any backend combo.

### CLI

```bash
# Export current config to a JSONL backup file
python -m tools.shared.migrate_store export --out backup.jsonl

# Export from a specific backend
python -m tools.shared.migrate_store export \
    --backend postgres+qdrant \
    --out backup.jsonl \
    --progress-every 1000

# Import into Turso+Turso (combo 2)
python -m tools.shared.migrate_store import \
    --backend turso+turso \
    --in backup.jsonl \
    --backup-before

# Direct pipe (no intermediate file)
python -m tools.shared.migrate_store pipe \
    --from postgres+qdrant \
    --to turso+turso

# Verify two backends hold the same data
python -m tools.shared.migrate_store verify \
    --left postgres+qdrant \
    --right turso+turso
```

The `--backend` spec format is `sql+vector` (e.g. `postgres+qdrant`, `turso+turso`, `none+qdrant`).

### JSONL format

```jsonl
{"_meta": {"kind": "sql", "schema_version": 1, "exported_at": "...", "source": "postgres"}}
{"_sql": {"id": "uuid-1", "text": "...", "memory_type": "concept", "tags": [...], ...}}
{"_sql": {"id": "uuid-2", "text": "...", ...}}
{"_meta": {"kind": "vector", "collection": "memory-store", "dim": 1024, "has_sparse": false}}
{"_vec": "memory-store", "id": "uuid-1", "vector": [0.1, ...], "payload": {...}}
```

- Self-describing headers (`_meta`) mark each section
- SQL and Vector sections can be in the same file or separate
- Embeddings stored inline so the destination rebuilds ANN indexes without re-running the embedder
- `__collection_metadata__` special points are filtered on export (recorded in header)
- Import is idempotent (uses dedup keys — safe to re-run)
- `--backup-before` exports the destination before overwriting

### MCP tools

Available when the migration module is registered on the FastMCP instance:

```
migrateMemoryBackend(
    export_backend="postgres+qdrant",
    import_backend="turso+turso"
)

verifyBackendParity(
    left_backend="postgres+qdrant",
    right_backend="turso+turso"
)
```

### Common migration scenarios

| From | To | Why | Command |
|------|----|-----|---------|
| PG + Qdrant | Turso + Turso | Drop both containers | `pipe --from postgres+qdrant --to turso+turso` |
| Turso + Turso | PG + Qdrant | Escape hatch | `pipe --from turso+turso --to postgres+qdrant` |
| PG + Qdrant | PG + Turso | Drop Qdrant only | `pipe --from postgres+qdrant --to postgres+turso` |
| PG + Qdrant | Turso + Qdrant | Drop PG only | `pipe --from postgres+qdrant --to turso+qdrant` |

Always run `verify` after migration to confirm data parity.

---

## Backend-Specific Notes

### PostgreSQL

- Full-text search via `pg_trgm` (fuzzy, character-level matching)
- Dedup via `text_hash + memory_type` unique constraint
- JSONB for tags with `@>` containment
- Connection pool via `psycopg_pool` (min_size=1, max_size=5)

### Turso / libSQL

- Full-text search via FTS5 (token-based, NOT fuzzy — typos won't match)
- Same dedup logic as PG (`text_hash + memory_type`)
- JSON1 for tags with `json_each()` containment
- Vector search via `vector_distance_cos()` (cosine distance)
- HNSW index for ANN if libSQL version supports it (falls back to brute-force)
- Local file mode (`file:path.db`) — zero-network, zero-containers
- `file::memory:` for tests — in-memory, no disk

### Qdrant

- Native sparse vectors for hybrid search
- HNSW for dense ANN
- Payload filters for metadata filtering
- Named vectors for multi-model collections

### pgvector

- HNSW index via `USING hnsw (embedding vector_cosine_ops)`
- `tsvector` generated column for BM25-like sparse search
- Requires PostgreSQL 12+ (for `GENERATED ALWAYS AS ... STORED`)

## Dependencies

```toml
# Required for PG SQL backend
psycopg >= 3.1
psycopg_pool >= 3.1

# Required for Qdrant Vector backend
qdrant-client >= 1.7

# Required for Turso SQL + Vector backend
libsql-experimental >= 0.0.30

# Required for pgvector Vector backend
# (no Python package — the extension runs server-side)
# PostgreSQL 12+ with pgvector extension installed
```
