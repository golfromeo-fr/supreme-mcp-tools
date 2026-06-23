# Backend Abstraction Plan: SQL + Vector stores with PG / Qdrant / Turso

## Goals

1. **Rename** `pg_store` to a backend-neutral name (current name lies the moment Turso is added).
2. **Finer-grained abstraction** — split into two orthogonal interfaces:
   - `SqlStore` — relational/metadata/full-text (CRUD, dedup, FTS, metrics, decay)
   - `VectorStore` — vector index (ANN query, hybrid, payload filter, scroll)
3. **Three implementations** of each, users can pair any SQL with any Vector:
   - PostgreSQL (SQL only), Qdrant (Vector only), Turso/libSQL (both)
4. **Migration tool** — move data between any (SQL, Vector) pair → any other pair.
5. Keep the current PG+Qdrant default working unchanged behind the new facade.

## Why split SQL and Vector (not one "backend" enum)

The trio has asymmetric capabilities:

| Component | SQL (relational, FTS, JSON, dedup) | Vector (ANN, hybrid) |
|---|---|---|
| PostgreSQL | ✅ (+ pgvector if wanted) | ⚠️ via pgvector only |
| Qdrant | ❌ (payload filters, not SQL) | ✅ (+ native sparse) |
| Turso (libSQL) | ✅ (FTS5, JSON1) | ✅ (`vector_distance_cos`) |

So the user can mix-and-match — 7 viable combos:

| Combo | SQL | Vector | Use case |
|---|---|---|---|
| 1 (current default) | PG | Qdrant | Full-featured, two containers |
| 2 | Turso | Turso | **Single embedded process, no containers** |
| 3 | PG | Turso | Keep PG, drop Qdrant container |
| 4 | Turso | Qdrant | Keep Qdrant, drop PG container |
| 5 | (none) | Qdrant | Qdrant-only mode (already supported) |
| 6 | PG | PG (pgvector) | Single PG container for everything |
| 7 | Turso | PG (pgvector) | Turso for metadata, PG for vectors |

Combo 2 is the headline: a single local file (or Turso cloud) replaces two containers — perfect for the "lighter, embedded" goal. Combo 6 (pgvector) gives a single-container PG setup for users who already run Postgres.

> **No SQLite impl** — Turso IS libSQL (a SQLite fork) and supports local-file mode (`file:test.db`) with zero network. This already covers the "zero-container laptop dev" and "CI testing" use cases a standalone SQLite impl was meant for.

## Naming / rename

### Files

| Old | New | Why |
|---|---|---|
| `tools/shared/pg_store.py` | `tools/shared/sql_store.py` | Module now defines the `SqlStore` ABC + impls; no PG-ism in the name |
| `tests/test_pg_store.py` | `tests/test_sql_store.py` | Follows source rename |

### Symbols inside

| Old | New |
|---|---|
| `class psycopg_pool` (internal) | stays in `postgres_impl.py` (PG-specific, private) |
| `init_pg()` | `PostgresSqlStore.connect()` (instance method) |
| `is_available()` | `SqlStore.is_available` (property) |
| `upsert_memory(...)`, `get_memory(...)`, etc. | methods on `SqlStore` ABC |
| `text_hash(...)` | moves to `tools/shared/hashing.py` (pure function, no DB) |

### Backward-compat shim

`tools/shared/pg_store.py` stays as a **thin re-export module** for one release cycle so `from shared import pg_store` keeps working in un-migrated callers (and `tests/test_pg_store.py` until renamed):

```python
# tools/shared/pg_store.py — DEPRECATED shim
"""Deprecated. Use sql_store; kept for backward-compat re-exports."""
from .sql_store import get_sql_store as _get
from .hashing import text_hash

# Module-level functions that delegate to the default SqlStore instance.
def upsert_memory(*a, **kw): return _get().upsert_memory(*a, **kw)
def get_memory(*a, **kw):    return _get().get_memory(*a, **kw)
# ... etc
def is_available(): return _get().is_available
def init_pg():      return _get().is_available
```

Shim gets removed in the cleanup phase.

## Target module layout

```
tools/shared/
  store_models.py         # NEW — backend-neutral types: PointStruct, Filter,
                          #   ScoredPoint, CollectionInfo, SparseVector, Range
  hashing.py              # NEW — text_hash (extracted, no DB deps)
  sql_store.py            # NEW — SqlStore ABC + factory get_sql_store()
  vector_store.py         # NEW — VectorStore ABC + factory get_vector_store()
  store_factory.py        # NEW — reads tool config, returns (SqlStore, VectorStore)
  impls/
    __init__.py
    postgres_sql.py       # NEW — PostgresSqlStore(SqlStore)
    turso_sql.py          # NEW — TursoSqlStore(SqlStore)
    qdrant_vector.py      # NEW — QdrantVectorStore(VectorStore) [thin adapter over qdrant_client]
    turso_vector.py       # NEW — TursoVectorStore(VectorStore)
    postgres_vector.py    # NEW — PostgresVectorStore(VectorStore) [via pgvector extension]

  pg_store.py             # DEPRECATED shim (re-exports from sql_store), removed in Phase 5
  memory_models.py        # unchanged (MemoryItem, MemoryHit, enums)
  relevance_scorer.py     # unchanged
  pii_redactor.py         # unchanged
```

`impls/` keeps concrete backends private — callers only see the ABCs and the factories.

## Interface contracts

### `SqlStore` (abstracts `pg_store.py`)

Methods = current module-level functions, no signature drift except instance-style:

```python
class SqlStore(Protocol):
    is_available: bool

    def upsert_memory(self, memory_id, text, memory_type, source, tags,
                      path, commit, agent_id, sensitivity, retention_policy) -> str: ...
    def get_memory(self, memory_id: str) -> dict | None: ...      # increments usage
    def delete_memory(self, memory_id: str) -> bool: ...
    def search_text(self, query: str, limit: int = 10, *,
                    memory_type: str | None = None,
                    tags: list[str] | None = None,
                    agent_id: str | None = None) -> list[dict]: ...
    def get_metrics(self) -> dict: ...
    def decay_memories(self, ttl_days: int, min_usage_count: int,
                       retention_policy: str | None = None) -> int: ...
    def get_all_memory_ids(self) -> list[str]: ...

    # migration support
    def iter_all(self) -> Iterator[dict]: ...                     # streaming dump
    def bulk_upsert(self, rows: Iterable[dict]) -> int: ...       # streaming load
```

### Neutral types (live in `tools/shared/store_models.py`)

These are **our own** types. Each impl translates to/from its native representation (qdrant_client models, libSQL rows). This is what stops `memory_tools.py` from `from qdrant_client.models import Filter`.

> All code blocks below assume: `from dataclasses import dataclass, field` and `from typing import Any`.

#### `CollectionInfo` (read schema back)

Replaces ad-hoc reaches into `info.config.params.vectors` / `info.config.params.sparse_vectors` (currently 12+ sites in `ragmcp_fastmcp.py`).

```python
@dataclass
class CollectionInfo:
    name: str
    points_count: int
    named_vectors: dict[str, int]     # name -> dim; empty if single/unnamed
    has_sparse: bool                  # any named vector is sparse
    distance: str = "Cosine"          # "Cosine" | "Euclid" | "Dot"
    @property
    def dim(self) -> int | None:
        return next(iter(self.named_vectors.values()), None)
```

#### `SparseVector` (sparse vector shape)

Used in `PointStruct.sparse_vector` and in `query_sparse` / `query_hybrid`. Matches Qdrant's sparse representation; Turso translates to FTS5 token frequencies.

```python
@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]
```

#### `PointStruct` (write shape)

Hybrid points need both dense (named or unnamed) and optional sparse. Today `incremental_indexer.py:617-626` builds Qdrant hybrid points as `vector={"dense": [...], "sparse": SparseVector(...)}`. The neutral type preserves both layouts:

```python
@dataclass
class PointStruct:
    id: str | int
    vector: list[float] | dict[str, list[float]]  # dense, or named-dense
    sparse_vector: SparseVector | None = None
    payload: dict | None = None
```

#### `ScoredPoint` (query result shape)

Returned by `query_dense`, `query_sparse`, `query_hybrid`. Carries the raw impl score (Qdrant cosine / Turso `vector_distance_cos` / FTS5 `bm25`) — score fusion for hybrid happens inside the impl, not at the call site.

```python
@dataclass
class ScoredPoint:
    id: str | int
    score: float
    payload: dict | None = None
    vector: list[float] | dict[str, list[float]] | None = None  # only if with_vectors=True
```

#### `Filter` and match operators

Must support all three match kinds observed in the codebase: `MatchValue` (most calls), `MatchText` (`incremental_indexer.py:687` for stale-file deletion), `MatchAny` (forward-looking). Plus a `MatchContains` for the tag-array case where Qdrant / PG JSONB / SQLite JSON1 all need different SQL/marshalling.

```python
@dataclass
class Range:
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

@dataclass
class Filter:
    must: list[FieldCondition] = field(default_factory=list)
    should: list[FieldCondition] = field(default_factory=list)
    must_not: list[FieldCondition] = field(default_factory=list)

@dataclass
class FieldCondition:
    key: str
    match: MatchValue | MatchText | MatchAny | MatchContains | None = None
    range_: Range | None = None

@dataclass
class MatchValue:
    value: Any

@dataclass
class MatchText:
    text: str

@dataclass
class MatchAny:
    values: list[Any]

@dataclass
class MatchContains:
    value: Any   # field is array/scalar; semantics: "array contains this"
```

#### `Filter` semantics parity

These four match kinds are **not** 1:1 across backends:

| Match           | Qdrant (payload)              | PG (JSONB)            | Turso (JSON1)                                       |
|-----------------|-------------------------------|-----------------------|------------------------------------------------------|
| `MatchValue`    | exact equality on scalar      | exact equality        | exact equality                                       |
| `MatchText`     | `MatchText` (string full-text)| `ILIKE %x%`           | FTS5 `MATCH`                                         |
| `MatchAny`      | any-of                        | `IN (...)`            | `IN (...)`                                           |
| `MatchContains` | array-contains-scalar         | `field @> '[x]'::jsonb` | `EXISTS (SELECT 1 FROM json_each(field) WHERE value=?)` |

Tools that build filters must use the operator that matches their intent. Today's `memory_tools.py:208-210` uses `MatchValue` for a tag list — that's a latent bug if the stored payload is a JSON array; should be `MatchContains` after the refactor. Capture this in the contract test.

### `VectorStore` (abstracts everything `qdrant_client.X` is used for)

Derived from the ~50 actual call sites in `memory_tools.py`, `memory_graph.py`, `ragmcp_fastmcp.py`, `indexer/incremental_indexer.py`. Three named query methods (not one with `using=`) so Qdrant impl can use `using=` and Turso impl can use FTS5 + `vector_distance_cos` + Python RRF without the caller caring.

```python
class VectorStore(Protocol):
    # lifecycle
    def ensure_collection(
        self, name: str, *,
        dense_dim: int | None = None,   # None = sparse-only
        sparse: bool = False,
        distance: str = "Cosine",
    ) -> None: ...

    # writes
    def upsert(self, collection: str, points: list[PointStruct]) -> None: ...
    def set_payload(self, collection: str, payload: dict, *, ids: list[str]) -> None: ...
    def delete(self, collection: str, *, ids: list[str] | None = None,
               filter: Filter | None = None) -> None: ...
    def delete_collection(self, name: str) -> None: ...

    # reads — three named methods, not one with `using`
    def query_dense(self, collection: str, vec: list[float], *,
                    limit: int = 10, filter: Filter | None = None,
                    using: str | None = None) -> list[ScoredPoint]: ...
    def query_sparse(self, collection: str, sparse: SparseVector, *,
                     limit: int = 10, filter: Filter | None = None) -> list[ScoredPoint]: ...
    def query_hybrid(self, collection: str, dense: list[float],
                     sparse: SparseVector, *, limit: int = 10,
                     filter: Filter | None = None) -> list[ScoredPoint]: ...

    def retrieve(self, collection: str, ids: list[str], *,
                 with_payload: bool = True, with_vectors: bool = False) -> list[PointStruct]: ...
    def scroll(self, collection: str, *, limit: int = 1000, offset=None,
               with_payload: bool = True,
               filter: Filter | None = None,
    ) -> tuple[list[PointStruct], Any]: ...
    def get_collection(self, name: str) -> CollectionInfo: ...
    def list_collections(self) -> list[str]: ...

    # migration
    def iter_all(self, collection: str, *, with_vectors: bool = True
    ) -> Iterator[PointStruct]: ...
    def bulk_upsert(self, collection: str, points: Iterable[PointStruct]) -> int: ...
```

**Why split `query` into three:** ragmcp has three search paths (`_do_dense_search`, `_do_sparse_search`, `_do_hybrid_search`). Qdrant implements sparse via `using="sparse"` which is impl-specific. A neutral API with three named methods hides the impl choice.

**Why only `query_dense` has `using`:** Qdrant collections can have multiple named dense vectors (e.g., `"dense"` + `"image"`). `using` selects which to search. `query_sparse` and `query_hybrid` don't need it because sparse vectors live under a single conventional name (`"sparse"`). If a future use case requires multiple sparse indexes or named-vectors in hybrid, add `using` to those methods then — don't speculative-generalize now.

## Config schema

Per-tool `config.json` gains a `storage` block. Backward-compat: absent = current behavior (Qdrant + optional PG via env).

```json
{
  "auth": { "api_key": "..." },
  "storage": {
    "sql":     { "backend": "postgres" },          // "postgres" | "turso" | "sqlite" | "none"
    "vector":  { "backend": "qdrant" }             // "qdrant"   | "turso" | "none"
  }
}
```

Turso example (single-backend, combo 2):

```json
{
  "storage": {
    "sql":    { "backend": "turso", "url_env": "TURSO_DATABASE_URL", "token_env": "TURSO_AUTH_TOKEN" },
    "vector": { "backend": "turso", "url_env": "TURSO_DATABASE_URL", "token_env": "TURSO_AUTH_TOKEN" }
  }
}
```

Env-based auto-config still works: if `POSTGRES_HOST` is set and `storage.sql.backend` is unset → postgres; if `TURSO_DATABASE_URL` set → turso; etc. `store_factory.py` resolves in priority: **explicit config > env vars > default (qdrant + optional pg)**.

## Migration tool

`tools/shared/migrate_store.py` — CLI + importable function.

### Format

JSON Lines, one record per line. Self-describing header line:

```jsonl
{"_meta": {"kind": "sql", "schema_version": 1, "exported_at": "...", "source": "postgres://..."}}
{"id": "...", "text": "...", "memory_type": "concept", "tags": [...], "embedding": null, ...}
{"_meta": {"kind": "vector", "collection": "memory-store", "dim": 1024}}
{"id": "...", "vector": [...], "payload": {...}}
```

- Two sections (sql + vector) can be in the same file or split.
- Embeddings stored inline in the vector section so the destination can rebuild ANN indexes without re-running the embedder.
- Streaming: tool reads/writes line-by-line, no full dataset in memory.

### CLI

```bash
# Export from current config to a file
python -m tools.shared.migrate_store export --out backup.jsonl

# Export from a specific backend
python -m tools.shared.migrate_store export \
    --sql-backend postgres --vector-backend qdrant \
    --out backup.jsonl

# Import into Turso+Turso (combo 2), then user can switch config
python -m tools.shared.migrate_store import \
    --sql-backend turso --vector-backend turso \
    --in backup.jsonl

# Direct pipe, no intermediate file
python -m tools.shared.migrate_store pipe \
    --from "postgres+qdrant" --to "turso+turso"

# Verify two backends hold the same data (row counts, id sets, sample checks)
python -m tools.shared.migrate_store verify \
    --left "postgres+qdrant" --right "turso+turso"
```

### Idempotency

`import` uses the SqlStore's dedup (`text_hash + memory_type` for SQL, point id for vectors) so re-running is safe. `--replace` flag truncates destination collections first.

### Path coverage

| From | To | Notes |
|---|---|---|
| PG+Qdrant | Turso+Turso | headline case — drops both containers |
| Turso+Turso | PG+Qdrant | escape hatch if Turso limits hit |
| PG+Qdrant | PG+Turso | drop Qdrant only |
| PG+Qdrant | Turso+Qdrant | drop PG only |
| any | any | general — both stores' `iter_all`/`bulk_upsert` cover it |

## Phased rollout

### Phase 0 — prep (no behavior change)
- Add `tools/shared/store_models.py` (neutral types)
- Add `tools/shared/hashing.py` (extract `text_hash`)
- Add empty `tools/shared/impls/` package

### Phase 1 — SQL refactor (rename + abstract)
- Write `SqlStore` ABC in `sql_store.py`
- Move PG code into `impls/postgres_sql.py` as `PostgresSqlStore`
- `pg_store.py` becomes the deprecation shim (re-exports)
- `memory_core.py` switches from `from shared import pg_store` to `get_sql_store()`
- All other call sites stay on `pg_store.*` (shim works)
- Rename `tests/test_pg_store.py` → `tests/test_sql_store.py`; add `tests/test_sql_store_contract.py` parametrized over impls
- Run full test suite — must pass unchanged

### Phase 2 — Vector refactor (abstract qdrant_client out)
- Write `VectorStore` ABC in `vector_store.py` (with the `query_dense` / `query_sparse` / `query_hybrid` split)
- Add `impls/qdrant_vector.py` as a thin adapter (delegates to `qdrant_client`)
- Edit in this order, keeping the suite green between each:
  1. `tools/memorymcp/memory_core.py` — replace `qdrant_client` import + init
  2. `tools/memorymcp/memory_graph.py` — 6 call sites
  3. `tools/memorymcp/memory_tools.py` — ~20 call sites, includes the tag-array latent bug fix (`MatchValue` → `MatchContains` for `tags`)
  4. `tools/ragmcp/ragmcp_fastmcp.py` — ~30 call sites, biggest churn, `info.config.params.*` accesses become `CollectionInfo` reads
  5. `tools/ragmcp/indexer/incremental_indexer.py` — ~15 call sites, plus the `MatchText` usage for stale-file deletion
  6. `tools/ragmcp/copilot_context_injector.py` — verify no direct `qdrant_client` use; if present, refactor
- Replace every `from qdrant_client.models import Filter` / `FieldCondition` / `MatchValue` → `from shared.store_models import Filter` / `FieldCondition` / `MatchValue` (neutral types)
- Tests pass

> Note: `tools/memorymcp/memory_text.py` is text-pure — grep confirmed zero `qdrant_client` references; no Phase 2 work needed.

### Phase 3 — Turso + pgvector implementations
- `impls/turso_sql.py` (FTS5 for `search_text`, JSON1 for tags, `vector_distance_cos` not used here)
- `impls/turso_vector.py` (`vector` column type, `vector_distance_cos` for `query_dense`, FTS5 sidecar for `query_hybrid`)
- `tools/shared/impls/turso_schema.sql` — DDL for both layers
- `impls/postgres_vector.py` (pgvector extension: `vector` column type, `<=>` cosine operator, HNSW index for ANN)
- `tools/shared/impls/postgres_vector_schema.sql` — DDL (`CREATE EXTENSION pgvector`, table + index creation)
- Add `tests/test_turso_sql_store.py`, `tests/test_turso_vector_store.py` (use local libSQL `file:` mode, no network)
- Add `tests/test_postgres_vector_store.py` (requires pgvector extension in PG container)
- `store_factory.py` resolves Turso + pgvector config
- **No standalone SQLite impl** — Turso local-file mode (`file:test.db`) covers CI/laptop dev

### Phase 4 — migration tool
- `tools/shared/migrate_store.py` with export/import/pipe/verify CLI subcommands
- MCP tool wrappers registered on the shared FastMCP instance:
  - `migrateMemoryBackend(export_backend, import_backend, options)` — export from one backend pair, import to another
  - `verifyBackendParity(left_backend, right_backend)` — verify two backends hold the same data (id sets, row counts, sample diffs)
- Add `tests/test_migrate_store.py` (round-trip PG+Qdrant → Turso-local-file → verify)
- Document in `tools/memorymcp/POSTGRESQL.md` rewrite → `tools/memorymcp/BACKENDS.md`

### Phase 5 — cleanup
- Remove `pg_store.py` shim
- Remove direct `qdrant_client` references in tool source (keep only inside `impls/qdrant_vector.py`)
- Update `AGENTS.md` conventions block

## Test plan

| Layer | Test |
|---|---|
| Pure types | `tests/test_store_models.py` — round-trip `PointStruct`/`Filter`/`CollectionInfo` serialization; assert all 4 match kinds (Value/Text/Any/Contains) survive transport |
| Hashing | `tests/test_hashing.py` (moved from `test_pg_store.py::test_text_hash`) |
| SQL contract | `tests/test_sql_store_contract.py` — parametrized over `[postgres, turso]`; skips impls whose backend isn't installed; one test per `SqlStore` method |
| Vector contract | `tests/test_vector_store_contract.py` — parametrized over `[qdrant, turso, postgres]`; **asserts `CollectionInfo` shape parity**: for a hybrid collection, `has_sparse == True` and `named_vectors` has both `"dense"` and `"sparse"` keys, on ALL backends |
| Filter parity | `tests/test_filter_semantics.py` — every `Match*` kind returns the same id set on Qdrant and Turso for a fixed dataset; catches the tag-array latent bug |
| Factory | `tests/test_store_factory.py` — config precedence (explicit > env > default), invalid combos raise, single-instance-per-process (call `get_vector_store()` twice → same id) |
| Migration | `tests/test_migrate_store.py` — export → import → verify, every From×To combo reachable via Turso-local-file in CI; special handling for `__collection_metadata__` point (recorded in header, not re-inserted verbatim) |
| Existing suite | `tests/test_memorymcp.py`, `tests/test_pg_store.py` (until renamed) — must stay green at every phase |

## Turso-specific compromises to acknowledge

1. **Hybrid search** — Qdrant's native sparse vectors vs Turso's FTS5 sidecar table. Same interface, different fidelity. Documented in `BACKENDS.md`.
2. **Trigram fuzzy match** — PG `pg_trgm similarity()` has no Turso equivalent. `search_text` falls back to FTS5 `MATCH` (token-based, not fuzzy). Memory search will be slightly less forgiving of typos in Turso mode. Acceptable per earlier discussion; revisit if it bites.
3. **Vector distance metric** — Turso supports cosine via `vector_distance_cos`; L2/dot available via different functions. ABC exposes `query_dense(...)` with cosine default; per-call metric override is a future addition, not in scope.
4. **Sparse vector dimension discovery** — Qdrant stores sparse as `{indices, values}`. Turso has no sparse type → sparse lives in an FTS5 virtual table, dense in a `vector` column. `ensure_collection(dense_dim=..., sparse=True)` creates both in Turso impl.
5. **Concurrency** — PG uses `psycopg_pool`. Turso libSQL client has HTTP-based concurrency, no pooling needed. ABC stays silent on pooling — each impl manages its own.
6. **Process-scoped instance reuse** — the factory returns a single per-process store instance, not a new client per call. Today `incremental_indexer.py:187` re-instantiates `QdrantClient(...)` inside `get_qdrant_files` on every call — a code smell the abstraction fixes. `vector_store.iter_all(collection)` replaces the lazy client pattern.
7. **Special point: `__collection_metadata__`** — `incremental_indexer.py:824-825` writes a special point with `id="__collection_metadata__"` to track embedding model/version. The migration tool should NOT try to re-insert this verbatim — it should record the metadata in the JSONL header and re-insert as part of `import` (or skip if the destination has a different embedding model).

## Review notes (counter-review additions)

### Smaller polish items

1. **`copilot_context_injector.py`** — it lives in `tools/ragmcp/` and may touch `qdrant_client`. Added to Phase 2 file list above.
2. **`memory_text.py` is text-pure** — grep confirmed zero `qdrant_client` references; no Phase 2 work needed. Documented inline in Phase 2 above.

### Code references (evidence backing the new sections)

| Finding | File:line evidence |
|---|---|
| `CollectionInfo` needed (12+ sites) | `ragmcp_fastmcp.py:337, 357, 552, 596, 827-828, 919, 1495, 1585, 1781` |
| `MatchText` in stale-file delete | `indexer/incremental_indexer.py:687` |
| Hybrid point shape | `indexer/incremental_indexer.py:617-626, 824-828` |
| Tag-array latent bug | `memory_tools.py:208-210` |
| `using="sparse"` for sparse search | `ragmcp_fastmcp.py:938, 984, 1139` |
| Re-instantiated QdrantClient | `indexer/incremental_indexer.py:187, 762` |
| Sparse-only collection creation | `indexer/incremental_indexer.py:775-783` |
| Hybrid collection creation | `indexer/incremental_indexer.py:787-799` |
| Dense-only collection creation | `indexer/incremental_indexer.py:804-807` |
| `__collection_metadata__` special point | `indexer/incremental_indexer.py:824-825` |
| `pg_store` import path | `memory_core.py:63` |
| `pg_store` test imports | `tests/test_pg_store.py:19-184` (19 lines) |
| `pg_store._init_lock` (used in test) | `tests/test_bug_fixes_low.py:76` |
| Qdrant call count by file | `memory_core.py:2`, `memory_tools.py:~20`, `memory_graph.py:~10`, `ragmcp_fastmcp.py:~30`, `indexer/incremental_indexer.py:~15` |

## Decisions (locked)

1. **Naming** — **LOCKED: `sql_store.py` + `vector_store.py` + `impls/`**. The `pg_store` shim handles the deprecation window.

2. **Shim removal window** — **LOCKED: one release cycle, then remove**. `pg_store.py` is deleted in Phase 5.

3. **SQLite impl** — **LOCKED: skip**. Turso IS libSQL and supports local-file mode (`file:test.db`) with zero network — covers CI and laptop dev without a separate impl.

4. **pgvector** — **LOCKED: include `impls/postgres_vector.py` in Phase 3**. PG can serve as a Vector backend (combo 6: PG+PG single container, combo 7: Turso SQL + PG vectors).

5. **Config location** — **LOCKED: per-tool `config.json`** with a `storage` block, plus env-var fallback (`POSTGRES_HOST`, `TURSO_DATABASE_URL`, `QDRANT_HOST`). No global `config/storage.json`.

6. **Migration tool surface** — **LOCKED: CLI + MCP tools from the start**. Both `python -m tools.shared.migrate_store` CLI and `migrateMemoryBackend` / `verifyBackendParity` MCP tools ship in Phase 4.

---

## Phase 2 sub-plan: QdrantVectorStore adapter + migration checklist

This section details the riskiest phase. The `QdrantVectorStore` adapter is the bridge between neutral types and `qdrant_client`. If its design is wrong, all 80+ call sites need rework.

### `QdrantVectorStore` adapter skeleton

```python
# tools/shared/impls/qdrant_vector.py
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from shared.store_models import (
    PointStruct, ScoredPoint, Filter, FieldCondition,
    MatchValue, MatchText, MatchAny, MatchContains, Range,
    CollectionInfo, SparseVector,
)


def _to_qdrant_filter(f: Filter | None) -> qm.Filter | None:
    """Translate neutral Filter → qdrant_client.models.Filter."""
    if f is None:
        return None
    return qm.Filter(
        must=[_to_qdrant_condition(c) for c in f.must],
        should=[_to_qdrant_condition(c) for c in f.should],
        must_not=[_to_qdrant_condition(c) for c in f.must_not],
    )


def _to_qdrant_condition(c: FieldCondition) -> qm.FieldCondition:
    """Translate one FieldCondition. MatchContains → MatchValue (Qdrant payload semantics)."""
    if isinstance(c.match, MatchValue):
        match = qm.MatchValue(value=c.match.value)
    elif isinstance(c.match, MatchText):
        match = qm.MatchText(text=c.match.text)
    elif isinstance(c.match, MatchAny):
        match = qm.MatchAny(any=c.match.values)
    elif isinstance(c.match, MatchContains):
        # Qdrant payload arrays: MatchValue already does array-contains on list fields
        match = qm.MatchValue(value=c.match.value)
    else:
        match = None

    range_ = None
    if c.range_:
        range_ = qm.Range(
            gt=c.range_.gt, gte=c.range_.gte,
            lt=c.range_.lt, lte=c.range_.lte,
        )

    return qm.FieldCondition(key=c.key, match=match, range=range_)


def _to_qdrant_point(p: PointStruct) -> qm.PointStruct:
    """Translate neutral PointStruct → qdrant_client.models.PointStruct."""
    vector = p.vector
    if p.sparse_vector is not None:
        # Hybrid: merge dense + sparse into a named-vector dict
        if isinstance(vector, list):
            vector = {"dense": vector}
        vector["sparse"] = qm.SparseVector(
            indices=p.sparse_vector.indices,
            values=p.sparse_vector.values,
        )
    return qm.PointStruct(id=p.id, vector=vector, payload=p.payload or {})


def _from_qdrant_scored(sp) -> ScoredPoint:
    """Translate qdrant ScoredPoint → neutral ScoredPoint."""
    return ScoredPoint(
        id=str(sp.id),
        score=float(sp.score),
        payload=sp.payload,
        vector=getattr(sp, 'vector', None),
    )


def _from_qdrant_collection_info(name, info) -> CollectionInfo:
    """Translate qdrant collection info → neutral CollectionInfo."""
    vc = info.config.params.vectors
    if isinstance(vc, dict):
        named_vectors = {k: v.size for k, v in vc.items() if hasattr(v, 'size')}
    elif vc and hasattr(vc, 'size'):
        named_vectors = {"": vc.size}
    else:
        named_vectors = {}

    has_sparse = bool(info.config.params.sparse_vectors)
    distance = "Cosine"
    if vc and hasattr(vc, 'distance'):
        distance = str(vc.distance).capitalize()
    elif isinstance(vc, dict):
        for v in vc.values():
            if hasattr(v, 'distance'):
                distance = str(v.distance).capitalize()
                break

    return CollectionInfo(
        name=name,
        points_count=info.points_count or 0,
        named_vectors=named_vectors,
        has_sparse=has_sparse,
        distance=distance,
    )


class QdrantVectorStore:
    """Thin adapter: delegates to qdrant_client, translates types at the boundary."""

    def __init__(self, host: str, port: int, timeout: int = 30):
        self._client = QdrantClient(host=host, port=port, timeout=timeout)

    # lifecycle
    def ensure_collection(self, name, *, dense_dim=None, sparse=False, distance="Cosine"):
        try:
            self._client.get_collection(name)
            return
        except Exception:
            pass
        dist = qm.Distance.COSINE if distance == "Cosine" else qm.Distance.EUCLID
        vectors_config = qm.VectorParams(size=dense_dim, distance=dist) if dense_dim else {}
        sparse_config = {"sparse": qm.SparseVectorParams(index=qm.SparseIndexParams())} if sparse else {}
        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config or None,
        )

    # writes
    def upsert(self, collection, points):
        self._client.upsert(collection_name=collection,
                            points=[_to_qdrant_point(p) for p in points])

    def set_payload(self, collection, payload, *, ids):
        self._client.set_payload(collection_name=collection, payload=payload, points=ids)

    def delete(self, collection, *, ids=None, filter=None):
        if ids:
            self._client.delete(collection_name=collection, points_selector=ids)
        elif filter:
            self._client.delete(collection_name=collection,
                                points_selector=_to_qdrant_filter(filter))

    def delete_collection(self, name):
        self._client.delete_collection(name)

    # reads
    def query_dense(self, collection, vec, *, limit=10, filter=None, using=None):
        res = self._client.query_points(
            collection_name=collection, query=vec, using=using,
            query_filter=_to_qdrant_filter(filter), limit=limit, with_payload=True,
        )
        return [_from_qdrant_scored(p) for p in res.points]

    def query_sparse(self, collection, sparse, *, limit=10, filter=None):
        qv = qm.SparseVector(indices=sparse.indices, values=sparse.values)
        res = self._client.query_points(
            collection_name=collection, query=qv, using="sparse",
            query_filter=_to_qdrant_filter(filter), limit=limit, with_payload=True,
        )
        return [_from_qdrant_scored(p) for p in res.points]

    def query_hybrid(self, collection, dense, sparse, *, limit=10, filter=None):
        # Qdrant native hybrid: query with named dense + named sparse
        qv = qm.NamedVector(name="dense", vector=dense)
        qsv = qm.NamedSparseVector(name="sparse", sparse=qm.SparseVector(
            indices=sparse.indices, values=sparse.values))
        # Qdrant's query_points supports prefetch fusion for hybrid
        from qdrant_client.models import FusionQuery, Prefetch
        res = self._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=qv, using="dense", limit=limit * 2),
                Prefetch(query=qsv, using="sparse", limit=limit * 2),
            ],
            query=FusionQuery(fusion=qm.Fusion.RRF),
            query_filter=_to_qdrant_filter(filter),
            limit=limit, with_payload=True,
        )
        return [_from_qdrant_scored(p) for p in res.points]

    def retrieve(self, collection, ids, *, with_payload=True, with_vectors=False):
        results = self._client.retrieve(
            collection_name=collection, ids=ids,
            with_payload=with_payload, with_vectors=with_vectors,
        )
        return [PointStruct(id=str(r.id), vector=r.vector if with_vectors else None,
                            payload=r.payload) for r in results]

    def scroll(self, collection, *, limit=1000, offset=None, with_payload=True, filter=None):
        results, next_offset = self._client.scroll(
            collection_name=collection, limit=limit, offset=offset,
            with_payload=with_payload, with_vectors=False,
            scroll_filter=_to_qdrant_filter(filter),
        )
        points = [PointStruct(id=str(r.id), vector=None, payload=r.payload) for r in results]
        return points, next_offset

    def get_collection(self, name):
        info = self._client.get_collection(name)
        return _from_qdrant_collection_info(name, info)

    def list_collections(self):
        return [c.name for c in self._client.get_collections().collections]

    # migration
    def iter_all(self, collection, *, with_vectors=True):
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=collection, limit=1000, offset=offset,
                with_payload=True, with_vectors=with_vectors,
            )
            for r in results:
                yield PointStruct(id=str(r.id), vector=r.vector, payload=r.payload)
            if not offset or not results:
                break

    def bulk_upsert(self, collection, points):
        batch = []
        count = 0
        for p in points:
            batch.append(_to_qdrant_point(p))
            if len(batch) >= 100:
                self._client.upsert(collection_name=collection, points=batch)
                count += len(batch)
                batch = []
        if batch:
            self._client.upsert(collection_name=collection, points=batch)
            count += len(batch)
        return count
```

### Translation patterns (before → after)

Each pattern shows the exact code change at call sites.

**Pattern 1: Query (dense)**
```python
# BEFORE (memory_tools.py:216)
results = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=k,
    query_filter=search_filter,
    with_payload=True,
)

# AFTER
results = vector_store.query_dense(
    COLLECTION_NAME,
    query_embedding,
    limit=k,
    filter=search_filter,  # neutral Filter, not qdrant Filter
)
```

**Pattern 2: Filter construction**
```python
# BEFORE (memory_tools.py:205-212)
from qdrant_client.models import Filter, FieldCondition, MatchValue
conditions = []
if memory_type:
    conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
if tags:
    for tag in tags:
        conditions.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
search_filter = Filter(must=conditions) if conditions else None

# AFTER
from shared.store_models import Filter, FieldCondition, MatchValue, MatchContains
conditions = []
if memory_type:
    conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
if tags:
    for tag in tags:
        conditions.append(FieldCondition(key="tags", match=MatchContains(value=tag)))  # FIX: was MatchValue
search_filter = Filter(must=conditions) if conditions else None
```

**Pattern 3: Upsert**
```python
# BEFORE (memory_tools.py:131-138)
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[{"id": memory_id, "vector": embedding, "payload": payload}]
)

# AFTER
from shared.store_models import PointStruct
vector_store.upsert(
    COLLECTION_NAME,
    [PointStruct(id=memory_id, vector=embedding, payload=payload)]
)
```

**Pattern 4: Retrieve + set_payload (usage tracking)**
```python
# BEFORE (memory_tools.py:291-313)
results = qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
payload = results[0].payload
payload["id"] = str(results[0].id)
qdrant_client.set_payload(collection_name=COLLECTION_NAME, payload={...}, points=[memory_id])

# AFTER
results = vector_store.retrieve(COLLECTION_NAME, [memory_id], with_payload=True)
payload = results[0].payload
payload["id"] = results[0].id
vector_store.set_payload(COLLECTION_NAME, {...}, ids=[memory_id])
```

**Pattern 5: Collection info (ragmcp auto-routing)**
```python
# BEFORE (ragmcp_fastmcp.py:826-834)
info = qdrant_client.get_collection(collection_name)
vectors_config = info.config.params.vectors
has_sparse = bool(info.config.params.sparse_vectors)
if isinstance(vectors_config, dict):
    has_dense = "dense" in vectors_config
else:
    has_dense = vectors_config is not None

# AFTER
info = vector_store.get_collection(collection_name)
has_sparse = info.has_sparse
has_dense = info.dim is not None or len(info.named_vectors) > 0
```

**Pattern 6: Sparse/hybrid query (ragmcp)**
```python
# BEFORE (ragmcp_fastmcp.py:981)
query_response = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_sparse_vec,  # qdrant SparseVector
    using="sparse",
    query_filter=search_filter,
    limit=limit, with_payload=True,
)

# AFTER
results = vector_store.query_sparse(
    collection_name,
    query_sparse_vec,  # neutral SparseVector(indices, values)
    limit=limit,
    filter=search_filter,
)
```

**Pattern 7: Delete by filter (indexer stale files)**
```python
# BEFORE (incremental_indexer.py:681-691)
from qdrant_client.models import Filter, FieldCondition, MatchText
qdrant_client.delete(
    collection_name=collection_name,
    points_selector=Filter(must=[
        FieldCondition(key='filePath', match=MatchText(text=rel_path))
    ])
)

# AFTER
from shared.store_models import Filter, FieldCondition, MatchText
vector_store.delete(
    collection_name,
    filter=Filter(must=[
        FieldCondition(key='filePath', match=MatchText(text=rel_path))
    ])
)
```

**Pattern 8: Scroll (memory_core.scroll_all)**
```python
# BEFORE (memory_core.py:202-218)
def scroll_all(collection_name, **kwargs):
    all_points = []
    offset = None
    while True:
        results, next_offset = qdrant_client.scroll(
            collection_name=collection_name, limit=1000, offset=offset,
            with_payload=True, **kwargs,
        )
        all_points.extend(results)
        if not next_offset or not results:
            break
        offset = next_offset
    return all_points

# AFTER — scroll_all becomes a thin wrapper around vector_store.iter_all
def scroll_all(collection_name, **kwargs):
    return list(vector_store.iter_all(collection_name))
```

### Per-file migration checklist

| File | Call sites | Key changes | Risk |
|---|---|---|---|
| `memory_core.py` | 2 (init + scroll_all) | Replace QdrantClient init with `vector_store = get_vector_store()`. `scroll_all` delegates to `vector_store.iter_all`. Export `vector_store` instead of `qdrant_client`. | Low — init only |
| `memory_graph.py` | 6 (retrieve, set_payload) | Replace `qdrant_client.retrieve/set_payload` → `vector_store.retrieve/set_payload`. Change `points=[id]` → `ids=[id]`. | Low — mechanical |
| `memory_tools.py` | ~20 (upsert, query_points, retrieve, set_payload, delete, get_collection) | All 6 patterns apply. Fix tag-array bug: `MatchValue` → `MatchContains` for tags filter. Replace `info.points_count` → `info.points_count` (same name, neutral type). | Medium — latent bug fix |
| `ragmcp_fastmcp.py` | ~30 (query_points, get_collection, get_collections, delete_collection) | Pattern 5 (collection info) is the biggest win. Three search functions (`_do_dense/_sparse/_hybrid`) switch to `query_dense/sparse/hybrid`. `validate_search_request` reads `CollectionInfo` instead of raw config. | High — most churn |
| `incremental_indexer.py` | ~15 (QdrantClient re-instantiation, upsert, scroll, delete, create_collection) | Remove lazy `QdrantClient()` construction (lines 187, 762). Use injected `vector_store`. `ensure_collection` replaces manual `create_collection` branching (lines 775-810). Pattern 7 for stale-file delete. | Medium — lazy-init fix |
| `copilot_context_injector.py` | verify | Grep for `qdrant_client` — if present, apply patterns. | Unknown — verify first |

---

## Phase 3 sub-plan: Turso + pgvector implementations

Three new impls land in Phase 3: `TursoSqlStore`, `TursoVectorStore`, `PostgresVectorStore`. This section specifies the DDL, SQL dialect differences, and implementation skeletons for each.

### SQL dialect cheat-sheet (PG → Turso/libSQL)

| Operation | PostgreSQL | Turso (libSQL) |
|---|---|---|
| Connection | `psycopg.connect(dsn)` | `libsql_experimental.connect(url, auth_token)` |
| Param placeholder | `%s` | `?` |
| Upsert | `INSERT ... ON CONFLICT (...) DO UPDATE ... RETURNING id` | Same syntax (libSQL supports it) |
| Fuzzy text search | `similarity(text, ?) > 0.1` (pg_trgm) | **No equivalent** — use FTS5 `MATCH` |
| Full-text search | `tsvector`, `ts_rank`, `@@` | `FTS5` virtual table, `bm25()` |
| JSON containment | `tags @> '[x]'::jsonb` | `EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)` |
| Date arithmetic | `NOW() - INTERVAL '30 days'` | `julianday('now') - julianday(col) > 30` |
| UUID type | `UUID` | `TEXT` (store as string) |
| JSONB type | `JSONB` | `TEXT` (store as JSON string, parse with json1) |
| Timestamp | `TIMESTAMPTZ DEFAULT NOW()` | `TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))` |
| Vector distance | `embedding <=> ?` (pgvector) | `vector_distance_cos(embedding, ?)` (libSQL) |
| ANN index | `USING hnsw (embedding vector_cosine_ops)` | HNSW (if libSQL version supports it, else brute-force) |
| Sparse vectors | `tsvector` column + GIN index | FTS5 virtual table sidecar |

### TursoSqlStore implementation

#### DDL: `tools/shared/impls/turso_sql_schema.sql`

```sql
-- Metadata table (mirrors PG schema, adapted for SQLite/libSQL)
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    text_hash       TEXT NOT NULL,
    memory_type     TEXT NOT NULL DEFAULT 'concept',
    source          TEXT NOT NULL DEFAULT 'agent_action',
    tags            TEXT NOT NULL DEFAULT '[]',    -- JSON array
    path            TEXT,
    commit          TEXT,
    agent_id        TEXT,
    sensitivity     TEXT NOT NULL DEFAULT 'low',
    retention_policy TEXT NOT NULL DEFAULT 'auto-delete',
    usage_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_accessed   TEXT,
    provenance      TEXT NOT NULL DEFAULT '{}',    -- JSON
    metadata        TEXT NOT NULL DEFAULT '{}',    -- JSON
    UNIQUE(text_hash, memory_type)
);

CREATE INDEX IF NOT EXISTS idx_memories_type   ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_agent  ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_hash   ON memories(text_hash);

-- FTS5 full-text index (replaces pg_trgm similarity search)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with the memories table
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
```

#### Skeleton: `tools/shared/impls/turso_sql.py`

```python
import json
import logging
from datetime import datetime, timezone
from shared.hashing import text_hash

logger = logging.getLogger(__name__)


class TursoSqlStore:
    """SqlStore implementation backed by Turso / libSQL."""

    def __init__(self, url: str, auth_token: str | None = None):
        import libsql_experimental as libsql
        self._conn = libsql.connect(url, auth_token=auth_token)
        self._conn.autocommit = True
        self.is_available = True
        self._ensure_schema()

    def _ensure_schema(self):
        """Run turso_sql_schema.sql."""
        import os
        schema_path = os.path.join(os.path.dirname(__file__), "turso_sql_schema.sql")
        with open(schema_path) as f:
            self._conn.executescript(f.read())

    # ---- CRUD ----

    def upsert_memory(self, memory_id, text, memory_type, source, tags,
                      path, commit, agent_id, sensitivity, retention_policy) -> str:
        thash = text_hash(text)
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags)
        # Check dedup
        row = self._conn.execute(
            "SELECT id, usage_count FROM memories WHERE text_hash = ? AND memory_type = ?",
            (thash, memory_type)
        ).fetchone()
        if row:
            existing_id = row[0]
            self._conn.execute("""
                UPDATE memories SET
                    text = ?, tags = ?, source = ?,
                    path = COALESCE(?, path), commit = COALESCE(?, commit),
                    agent_id = COALESCE(?, agent_id),
                    sensitivity = ?, retention_policy = ?,
                    usage_count = usage_count + 1, last_accessed = ?
                WHERE id = ?
            """, (text, tags_json, source, path, commit, agent_id,
                  sensitivity, retention_policy, now, existing_id))
            return existing_id
        # Insert new
        self._conn.execute("""
            INSERT INTO memories (id, text, text_hash, memory_type, source, tags,
                path, commit, agent_id, sensitivity, retention_policy, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(text_hash, memory_type) DO UPDATE SET
                text = excluded.text, tags = excluded.tags,
                usage_count = usage_count + 1, last_accessed = excluded.last_accessed
        """, (memory_id, text, thash, memory_type, source, tags_json,
              path, commit, agent_id, sensitivity, retention_policy, now, now))
        return memory_id

    def get_memory(self, memory_id: str) -> dict | None:
        rows = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchall()
        if not rows:
            return None
        cols = [d[0] for d in self._conn.execute("SELECT * FROM memories LIMIT 0").description]
        row = dict(zip(cols, rows[0]))
        self._conn.execute(
            "UPDATE memories SET usage_count = usage_count + 1, last_accessed = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), memory_id))
        row["tags"] = json.loads(row.get("tags", "[]"))
        row["provenance"] = json.loads(row.get("provenance", "{}"))
        return row

    def delete_memory(self, memory_id: str) -> bool:
        result = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return result.rowcount > 0

    # ---- Full-text search (FTS5 replaces pg_trgm) ----

    def search_text(self, query: str, limit: int = 10, *,
                    memory_type: str | None = None,
                    tags: list[str] | None = None,
                    agent_id: str | None = None) -> list[dict]:
        """
        FTS5 token-based search. NOT fuzzy — exact word stems only.
        For fuzzy matching, PG is the better backend.
        """
        conditions = []
        params: list = []
        # FTS5 MATCH on the virtual table
        fts_query = query  # FTS5 query syntax: "search terms" (implicit AND)
        conditions.append("memories_fts MATCH ?")
        params.append(fts_query)
        if memory_type:
            conditions.append("m.memory_type = ?")
            params.append(memory_type)
        if agent_id:
            conditions.append("m.agent_id = ?")
            params.append(agent_id)
        # Tag containment via json_each
        if tags:
            for tag in tags:
                conditions.append("EXISTS (SELECT 1 FROM json_each(m.tags) WHERE value = ?)")
                params.append(tag)
        params.append(limit)
        rows = self._conn.execute(f"""
            SELECT m.id, m.text, m.memory_type, m.tags, m.source,
                   m.created_at, m.last_accessed, m.usage_count,
                   bm25(memories_fts) AS sim_score
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE {' AND '.join(conditions)}
            ORDER BY sim_score
            LIMIT ?
        """, params).fetchall()
        cols = ["id", "text", "memory_type", "tags", "source",
                "created_at", "last_accessed", "usage_count", "similarity"]
        return [dict(zip(cols, r)) for r in rows]

    # ---- Metrics ----

    def get_metrics(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = self._conn.execute(
            "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        by_agent = self._conn.execute(
            "SELECT agent_id, COUNT(*) FROM memories GROUP BY agent_id ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        total_usage = self._conn.execute("SELECT COALESCE(SUM(usage_count), 0) FROM memories").fetchone()[0]
        return {
            "total": total,
            "by_type": {r[0]: r[1] for r in by_type},
            "by_agent": {r[0]: r[1] for r in by_agent},
            "total_usage": total_usage,
        }

    # ---- Decay ----

    def decay_memories(self, ttl_days: int, min_usage_count: int,
                       retention_policy: str | None = None) -> int:
        conditions = ["retention_policy != 'permanent'"]
        params: list = []
        conditions.append("(julianday('now') - julianday(last_accessed) > ? OR julianday('now') - julianday(created_at) > ?)")
        params.extend([ttl_days, ttl_days])
        conditions.append("usage_count < ?")
        params.append(min_usage_count)
        if retention_policy:
            conditions.append("retention_policy = ?")
            params.append(retention_policy)
        where = " AND ".join(conditions)
        result = self._conn.execute(f"DELETE FROM memories WHERE {where}", params)
        return result.rowcount

    # ---- Misc ----

    def get_all_memory_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM memories").fetchall()
        return [r[0] for r in rows]

    def iter_all(self):
        cursor = self._conn.execute("SELECT * FROM memories")
        cols = [d[0] for d in cursor.description]
        while True:
            batch = cursor.fetchmany(100)
            if not batch:
                break
            for row in batch:
                d = dict(zip(cols, row))
                d["tags"] = json.loads(d.get("tags", "[]"))
                d["provenance"] = json.loads(d.get("provenance", "{}"))
                yield d

    def bulk_upsert(self, rows) -> int:
        count = 0
        for row in rows:
            self.upsert_memory(
                memory_id=row["id"], text=row["text"],
                memory_type=row.get("memory_type", "concept"),
                source=row.get("source", "agent_action"),
                tags=row.get("tags", []),
                path=row.get("path"), commit=row.get("commit"),
                agent_id=row.get("agent_id"),
                sensitivity=row.get("sensitivity", "low"),
                retention_policy=row.get("retention_policy", "auto-delete"),
            )
            count += 1
        return count
```

### TursoVectorStore implementation

#### Vector DDL (created dynamically per collection)

```sql
-- Dense vector table (one per "collection")
CREATE TABLE IF NOT EXISTS vec_{name} (
    id          TEXT PRIMARY KEY,
    embedding   VECTOR({dim}),
    payload     TEXT,     -- JSON
    created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at  TEXT
);

-- Sparse (FTS5 sidecar for lexical search)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_{name}_fts USING fts5(
    text_content,
    content='vec_{name}',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 synced
CREATE TRIGGER IF NOT EXISTS vec_{name}_fts_ai AFTER INSERT ON vec_{name} BEGIN
    INSERT INTO vec_{name}_fts(rowid, text_content) VALUES (new.rowid, json_extract(new.payload, '$.text'));
END;
CREATE TRIGGER IF NOT EXISTS vec_{name}_fts_ad AFTER DELETE ON vec_{name} BEGIN
    INSERT INTO vec_{name}_fts(vec_{name}_fts, rowid, text_content) VALUES ('delete', old.rowid, json_extract(old.payload, '$.text'));
END;
CREATE TRIGGER IF NOT EXISTS vec_{name}_fts_au AFTER UPDATE ON vec_{name} BEGIN
    INSERT INTO vec_{name}_fts(vec_{name}_fts, rowid, text_content) VALUES ('delete', old.rowid, json_extract(old.payload, '$.text'));
    INSERT INTO vec_{name}_fts(rowid, text_content) VALUES (new.rowid, json_extract(new.payload, '$.text'));
END;
```

#### Key method patterns

```python
class TursoVectorStore:
    def __init__(self, url: str, auth_token: str | None = None):
        import libsql_experimental as libsql
        self._conn = libsql.connect(url, auth_token=auth_token)
        self._conn.autocommit = True

    def ensure_collection(self, name, *, dense_dim=None, sparse=False, distance="Cosine"):
        if dense_dim:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS vec_{name} (..., embedding VECTOR({dense_dim}), ...)")
        else:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS vec_{name} (id TEXT PRIMARY KEY, payload TEXT, ...)")
        if sparse:
            self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{name}_fts USING fts5(...)")
            # triggers...

    def query_dense(self, collection, vec, *, limit=10, filter=None, using=None):
        # Serialize vector for libSQL: "[0.1, 0.2, ...]"
        vec_str = json.dumps(vec)
        results = self._conn.execute(f"""
            SELECT id, payload, vector_distance_cos(embedding, ?) AS distance
            FROM vec_{collection}
            ORDER BY distance
            LIMIT ?
        """, (vec_str, limit)).fetchall()
        return [ScoredPoint(id=r[0], score=1 - r[2], payload=json.loads(r[1])) for r in results]
        # Note: vector_distance_cos returns DISTANCE (0=same), so similarity = 1 - distance

    def query_sparse(self, collection, sparse_vec, *, limit=10, filter=None):
        # Sparse query = FTS5 search on the sidecar table
        # Build FTS5 query string from sparse vector indices/values (or just use raw text)
        # In practice, sparse_vector_gen produces token frequencies → convert to FTS5 query
        fts_query = self._sparse_to_fts_query(sparse_vec)
        results = self._conn.execute(f"""
            SELECT v.id, v.payload, bm25(vec_{collection}_fts) AS score
            FROM vec_{collection}_fts
            JOIN vec_{collection} v ON v.rowid = vec_{collection}_fts.rowid
            WHERE vec_{collection}_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (fts_query, limit)).fetchall()
        return [ScoredPoint(id=r[0], score=-r[2], payload=json.loads(r[1])) for r in results]
        # Note: bm25 returns NEGATIVE scores (lower = better), negate for "higher = better"

    def query_hybrid(self, collection, dense, sparse, *, limit=10, filter=None):
        # No native hybrid — run both, fuse with RRF in Python
        dense_results = self.query_dense(collection, dense, limit=limit * 2, filter=filter)
        sparse_results = self.query_sparse(collection, sparse, limit=limit * 2, filter=filter)
        return self._rrf_fuse(dense_results, sparse_results, limit=limit)

    @staticmethod
    def _rrf_fuse(dense_hits, sparse_hits, limit=10, k=60):
        """Reciprocal Rank Fusion — combines two ranked lists into one."""
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for rank, hit in enumerate(dense_hits):
            scores[hit.id] = scores.get(hit.id, 0) + 1.0 / (k + rank + 1)
            payloads[hit.id] = hit.payload
        for rank, hit in enumerate(sparse_hits):
            scores[hit.id] = scores.get(hit.id, 0) + 1.0 / (k + rank + 1)
            payloads[hit.id] = hit.payload
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        return [ScoredPoint(id=id, score=score, payload=payloads[id]) for id, score in ranked]

    def upsert(self, collection, points):
        for p in points:
            vec_str = json.dumps(p.vector) if isinstance(p.vector, list) else json.dumps(p.vector.get("dense", []))
            payload_str = json.dumps(p.payload or {})
            self._conn.execute(f"""
                INSERT INTO vec_{collection} (id, embedding, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    embedding = excluded.embedding,
                    payload = excluded.payload,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """, (str(p.id), vec_str, payload_str))

    # retrieve, scroll, delete, set_payload — straightforward SQL
    # get_collection — query table info, return CollectionInfo
    # iter_all, bulk_upsert — same pattern as QdrantVectorStore
```

### PostgresVectorStore implementation (pgvector)

#### DDL: `tools/shared/impls/postgres_vector_schema.sql`

```sql
-- Run once per collection (parameterized by collection name at runtime)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {schema}.vec_{name} (
    id          TEXT PRIMARY KEY,
    embedding   vector({dim}),
    payload     JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ,
    -- tsvector column for sparse/BM25 search (generated from payload text)
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(payload->>'text', ''))
    ) STORED
);

-- HNSW ANN index for dense vector cosine search
CREATE INDEX IF NOT EXISTS idx_{name}_embedding
ON {schema}.vec_{name} USING hnsw (embedding vector_cosine_ops);

-- GIN index for sparse (tsvector) search
CREATE INDEX IF NOT EXISTS idx_{name}_search
ON {schema}.vec_{name} USING gin (search_vector);
```

#### Key method patterns

```python
class PostgresVectorStore:
    """VectorStore backed by PostgreSQL + pgvector extension."""

    def __init__(self, dsn: str):
        from psycopg_pool import ConnectionPool
        from psycopg.rows import dict_row
        self._pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=5,
                                     kwargs={"row_factory": dict_row})
        self._pool.open(wait=True)
        # Ensure pgvector extension exists
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def ensure_collection(self, name, *, dense_dim=None, sparse=False, distance="Cosine"):
        with self._pool.connection() as conn:
            if dense_dim:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS vec_{name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({dense_dim}),
                        payload JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ,
                        search_vector tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(payload->>'text', ''))
                        ) STORED
                    )
                """)
                # HNSW index for ANN
                ops = "vector_cosine_ops" if distance == "Cosine" else "vector_l2_ops"
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{name}_embedding
                    ON vec_{name} USING hnsw (embedding {ops})
                """)
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{name}_search
                    ON vec_{name} USING gin (search_vector)
                """)

    def query_dense(self, collection, vec, *, limit=10, filter=None, using=None):
        vec_str = json.dumps(vec)  # pgvector accepts "[0.1, 0.2, ...]" format
        with self._pool.connection() as conn:
            filter_clause = self._build_pg_filter(filter)
            rows = conn.execute(f"""
                SELECT id, payload, 1 - (embedding <=> %s::vector) AS similarity
                FROM vec_{collection}
                {filter_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, vec_str, limit)).fetchall()
            return [ScoredPoint(id=str(r["id"]), score=float(r["similarity"]),
                                payload=r["payload"]) for r in rows]

    def query_sparse(self, collection, sparse_vec, *, limit=10, filter=None):
        # sparse_vec → FTS query (or use ts_query directly)
        fts_query = self._sparse_to_tsquery(sparse_vec)
        with self._pool.connection() as conn:
            filter_clause = self._build_pg_filter(filter)
            rows = conn.execute(f"""
                SELECT id, payload, ts_rank(search_vector, plainto_tsquery(%s)) AS score
                FROM vec_{collection}
                WHERE search_vector @@ plainto_tsquery(%s)
                {filter_clause.replace('WHERE', 'AND') if filter_clause else ''}
                ORDER BY score DESC
                LIMIT %s
            """, (fts_query, fts_query, limit)).fetchall()
            return [ScoredPoint(id=str(r["id"]), score=float(r["score"]),
                                payload=r["payload"]) for r in rows]

    def query_hybrid(self, collection, dense, sparse, *, limit=10, filter=None):
        # PG can do combined query in SQL (weighted fusion)
        vec_str = json.dumps(dense)
        fts_query = self._sparse_to_tsquery(sparse)
        with self._pool.connection() as conn:
            rows = conn.execute(f"""
                SELECT id, payload,
                    (1 - (embedding <=> %s::vector)) * 0.5 +
                    ts_rank(search_vector, plainto_tsquery(%s)) * 0.5 AS combined_score
                FROM vec_{collection}
                WHERE search_vector @@ plainto_tsquery(%s)
                ORDER BY combined_score DESC
                LIMIT %s
            """, (vec_str, fts_query, fts_query, limit)).fetchall()
            return [ScoredPoint(id=str(r["id"]), score=float(r["combined_score"]),
                                payload=r["payload"]) for r in rows]
```

### Shared helper: sparse vector → text query

Both Turso and PG need to convert a `SparseVector(indices, values)` back into a text query for FTS5 / tsvector search. The sparse vector generator produces token-frequency maps, so the reverse is:

```python
# tools/shared/impls/sparse_to_text.py
from shared.store_models import SparseVector

# Reverse lookup table: sparse index → token string
# This must match the vocabulary used by sparse_vector_gen.py
# For FTS-based backends, we reconstruct the query text from the sparse vector
def sparse_to_fts_query(sparse: SparseVector, vocab: dict[int, str] | None = None) -> str:
    """
    Convert a SparseVector to an FTS5 / tsquery-compatible string.
    Uses token indices to look up words; falls back to OR-ing all known tokens.
    """
    if vocab is None:
        # Without a vocabulary, we can't reconstruct the query text.
        # Callers should pass the original query text instead of the sparse vector
        # for FTS-based backends. This is a known limitation.
        raise ValueError("FTS-based backends need the original query text, not a SparseVector")
    tokens = []
    for idx, val in zip(sparse.indices, sparse.values):
        if val > 0 and idx in vocab:
            tokens.append(vocab[idx])
    return " ".join(tokens) if tokens else ""
```

> **Design note:** FTS-based backends (Turso, PG tsvector) can't reconstruct query text from a SparseVector without a reverse vocabulary. The `query_sparse` method on these backends should accept the **original query string** and build the FTS query from it, bypassing the SparseVector entirely. The `VectorStore` ABC stays unchanged — the Turso/PG impls simply ignore the SparseVector and use a stored query-text parameter internally. This is documented as a known compromise in the "Turso-specific compromises" section.

### Phase 3 test setup

| Test | Backend | Setup |
|---|---|---|
| `test_turso_sql_store.py` | Turso local file | `libsql.connect("file::memory:")` — in-memory, no network |
| `test_turso_vector_store.py` | Turso local file | Same — in-memory libSQL with vector extension |
| `test_postgres_vector_store.py` | PG + pgvector | Requires running PG container with `CREATE EXTENSION vector`; skip if not installed |
| Contract tests | All impls | Parametrized via `[qdrant, turso, postgres]` — same assertions, different backends |

### Phase 3 dependency check

```toml
# Added to tools/memorymcp/requirements.txt and tools/ragmcp/requirements.txt
libsql-experimental>=0.0.30   # Turso/libSQL Python client (optional — only if backend=turso)
# psycopg and psycopg_pool already present for PG
# pgvector requires the PG extension (server-side), no Python package needed
```

---

## Phase 0 sub-plan: concrete module code

Phase 0 creates three files with zero behavior change. All code is self-contained — no existing file is modified yet.

### `tools/shared/store_models.py` (complete)

```python
"""
Backend-neutral types for the storage abstraction layer.

These types are the ONLY shapes that tool code (memory_tools.py, ragmcp_fastmcp.py, etc.)
should import. Concrete backends (qdrant_client, libsql, psycopg) translate to/from
these types at the impl boundary.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Iterable


# ---------------------------------------------------------------------------
# Sparse vector
# ---------------------------------------------------------------------------

@dataclass
class SparseVector:
    """Sparse vector as {indices, values}. Matches Qdrant's representation."""
    indices: list[int]
    values: list[float]


# ---------------------------------------------------------------------------
# Collection metadata (read schema back)
# ---------------------------------------------------------------------------

@dataclass
class CollectionInfo:
    """Schema/capability info for a vector collection."""
    name: str
    points_count: int
    named_vectors: dict[str, int] = field(default_factory=dict)  # name -> dim
    has_sparse: bool = False
    distance: str = "Cosine"  # "Cosine" | "Euclid" | "Dot"

    @property
    def dim(self) -> int | None:
        """Convenience: dimension of the single (or first) dense vector."""
        return next(iter(self.named_vectors.values()), None)

    @property
    def has_dense(self) -> bool:
        """True if any dense vector is configured."""
        return len(self.named_vectors) > 0


# ---------------------------------------------------------------------------
# Write shapes
# ---------------------------------------------------------------------------

@dataclass
class PointStruct:
    """
    A point to upsert into a vector collection.
    vector: list[float] for unnamed dense, dict[str, list[float]] for named dense.
    sparse_vector: optional SparseVector for hybrid collections.
    """
    id: str | int
    vector: list[float] | dict[str, list[float]]
    sparse_vector: SparseVector | None = None
    payload: dict | None = None


# ---------------------------------------------------------------------------
# Query result shape
# ---------------------------------------------------------------------------

@dataclass
class ScoredPoint:
    """A ranked result from a vector query."""
    id: str | int
    score: float
    payload: dict | None = None
    vector: list[float] | dict[str, list[float]] | None = None


# ---------------------------------------------------------------------------
# Filter shapes
# ---------------------------------------------------------------------------

@dataclass
class Range:
    """Numeric range filter on a payload field."""
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None


@dataclass
class MatchValue:
    """Exact equality on a scalar field."""
    value: Any


@dataclass
class MatchText:
    """Full-text / substring match on a string field."""
    text: str


@dataclass
class MatchAny:
    """Any-of match (field value is in the given list)."""
    values: list[Any]


@dataclass
class MatchContains:
    """Array-contains-scalar match (field is a JSON array containing this value)."""
    value: Any


@dataclass
class FieldCondition:
    """A single filter condition on a payload field."""
    key: str
    match: MatchValue | MatchText | MatchAny | MatchContains | None = None
    range_: Range | None = None


@dataclass
class Filter:
    """A boolean filter combining multiple conditions."""
    must: list[FieldCondition] = field(default_factory=list)
    should: list[FieldCondition] = field(default_factory=list)
    must_not: list[FieldCondition] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connection info
# ---------------------------------------------------------------------------

@dataclass
class BackendInfo:
    """Info about the active backend, for logging and diagnostics."""
    sql_backend: str    # "postgres" | "turso" | "none"
    vector_backend: str  # "qdrant" | "turso" | "postgres" | "none"
    sql_available: bool = False
    vector_available: bool = False
```

### `tools/shared/hashing.py` (complete)

```python
"""
Deterministic text hashing for memory deduplication.

Extracted from pg_store.py — pure function, no DB dependencies.
Used by all SqlStore impls (PG, Turso) for dedup key generation.
"""
import hashlib


def text_hash(text: str) -> str:
    """
    Deterministic hash for deduplication.
    SHA256 of stripped+lowercased text, truncated to 40 chars.
    """
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:40]
```

### `tools/shared/impls/__init__.py` (complete)

```python
"""
Concrete backend implementations.

This package is NOT scanned by the tool discovery system.
Callers should never import from here directly — use the factories in
sql_store.py / vector_store.py / store_factory.py instead.
"""
```

### Phase 0 verification

```bash
# After creating the three files:
python -c "from shared.store_models import Filter, MatchContains, PointStruct, ScoredPoint, CollectionInfo, SparseVector, Range; print('OK')"
python -c "from shared.hashing import text_hash; assert text_hash('Hello') == text_hash(' hello '); print('OK')"
python -m pytest tests/ -x -q  # full suite must pass unchanged
```

---

## Phase 1 sub-plan: SqlStore ABC + PostgresSqlStore + shim + factory

### `SqlStore` ABC definition

```python
# tools/shared/sql_store.py
from __future__ import annotations
from typing import Protocol, Iterator, Iterable


class SqlStore(Protocol):
    """
    Relational/metadata/full-text store.

    Implementations: PostgresSqlStore, TursoSqlStore.
    Optional — if no SQL backend is configured, tools fall back to vector-only mode.
    """
    is_available: bool

    # CRUD
    def upsert_memory(self, memory_id: str, text: str, memory_type: str,
                      source: str, tags: list[str], path: str | None,
                      commit: str | None, agent_id: str | None,
                      sensitivity: str, retention_policy: str) -> str: ...
    def get_memory(self, memory_id: str) -> dict | None: ...
    def delete_memory(self, memory_id: str) -> bool: ...

    # Search
    def search_text(self, query: str, limit: int = 10, *,
                    memory_type: str | None = None,
                    tags: list[str] | None = None,
                    agent_id: str | None = None) -> list[dict]: ...

    # Analytics
    def get_metrics(self) -> dict: ...
    def decay_memories(self, ttl_days: int, min_usage_count: int,
                       retention_policy: str | None = None) -> int: ...
    def get_all_memory_ids(self) -> list[str]: ...

    # Migration
    def iter_all(self) -> Iterator[dict]: ...
    def bulk_upsert(self, rows: Iterable[dict]) -> int: ...


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_singleton: SqlStore | None = None


def get_sql_store() -> SqlStore | None:
    """
    Return the process-scoped SqlStore singleton.
    Resolves backend from: config.json storage.sql > env vars > None.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    from shared.store_factory import resolve_sql_backend
    backend = resolve_sql_backend()

    if backend is None:
        _singleton = _NullSqlStore()
        return _singleton

    if backend["name"] == "postgres":
        from shared.impls.postgres_sql import PostgresSqlStore
        _singleton = PostgresSqlStore(dsn=backend["dsn"])
    elif backend["name"] == "turso":
        from shared.impls.turso_sql import TursoSqlStore
        _singleton = TursoSqlStore(url=backend["url"], auth_token=backend.get("auth_token"))
    else:
        _singleton = _NullSqlStore()

    return _singleton


class _NullSqlStore:
    """No-op store for when no SQL backend is configured."""
    is_available = False
    def upsert_memory(self, *a, **kw): return a[0]  # return the memory_id unchanged
    def get_memory(self, *a, **kw): return None
    def delete_memory(self, *a, **kw): return False
    def search_text(self, *a, **kw): return []
    def get_metrics(self, *a, **kw): return {}
    def decay_memories(self, *a, **kw): return 0
    def get_all_memory_ids(self, *a, **kw): return []
    def iter_all(self): return; yield  # empty generator
    def bulk_upsert(self, rows): return 0
```

### `PostgresSqlStore` skeleton (wraps existing pg_store code)

```python
# tools/shared/impls/postgres_sql.py
"""PostgresSqlStore — SqlStore backed by PostgreSQL."""
import json, logging
from datetime import datetime, timezone
from shared.hashing import text_hash

logger = logging.getLogger(__name__)


class PostgresSqlStore:
    """Wraps the existing pg_store psycopg_pool pattern as a SqlStore instance."""

    def __init__(self, dsn: str):
        from psycopg.rows import dict_row
        try:
            from psycopg_pool import ConnectionPool
            self._pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=5,
                                        open=False, kwargs={"row_factory": dict_row})
            self._pool.open(wait=True)
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
                self._ensure_schema(conn)
            self.is_available = True
            logger.info("PostgresSqlStore initialized")
        except Exception as e:
            logger.warning(f"PostgresSqlStore init failed: {e}")
            self.is_available = False
            self._pool = None

    def _ensure_schema(self, conn):
        """Same DDL as pg_store._ensure_schema — CREATE TABLE + pg_trgm + indexes."""
        conn.execute(""" ... """)  # identical to current pg_store.py lines 130-164

    # Every method is a thin rewrite of the current pg_store module-level function,
    # changing `with _pool.connection()` → `with self._pool.connection()`
    # and `return memory_id` stays the same.

    def upsert_memory(self, memory_id, text, memory_type, source, tags,
                      path, commit, agent_id, sensitivity, retention_policy) -> str:
        if not self.is_available:
            return memory_id
        thash = text_hash(text)
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags)
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, usage_count FROM memories WHERE text_hash = %s AND memory_type = %s",
                (thash, memory_type)
            ).fetchone()
            if row:
                # ... same UPDATE as pg_store.py:208-222
                return str(row["id"])
            else:
                # ... same INSERT ... ON CONFLICT as pg_store.py:227-238
                return memory_id

    # get_memory, delete_memory, search_text, get_metrics, decay_memories,
    # get_all_memory_ids, iter_all, bulk_upsert — same pattern, instance methods.
```

### Backward-compat shim (complete)

```python
# tools/shared/pg_store.py — DEPRECATED, removed in Phase 5
"""
DEPRECATED: use sql_store.get_sql_store() instead.
This shim re-exports module-level functions that delegate to the singleton
PostgresSqlStore, keeping `from shared import pg_store` working during migration.
"""
import logging
logger = logging.getLogger(__name__)

_lazy_store = None

def _store():
    global _lazy_store
    if _lazy_store is None:
        from shared.sql_store import get_sql_store
        _lazy_store = get_sql_store()
    return _lazy_store

# Re-export text_hash for existing callers
from shared.hashing import text_hash

# Module-level function shims
def is_available():
    s = _store()
    return s.is_available if s else False

def init_pg():
    """Deprecated — the store auto-initializes on first access."""
    return is_available()

def upsert_memory(*a, **kw):  return _store().upsert_memory(*a, **kw)
def get_memory(*a, **kw):     return _store().get_memory(*a, **kw)
def delete_memory(*a, **kw):  return _store().delete_memory(*a, **kw)
def search_text(*a, **kw):    return _store().search_text(*a, **kw)
def get_metrics(*a, **kw):    return _store().get_metrics(*a, **kw)
def decay_memories(*a, **kw): return _store().decay_memories(*a, **kw)
def get_all_memory_ids(*a, **kw): return _store().get_all_memory_ids(*a, **kw)
```

### `store_factory.py` skeleton

```python
# tools/shared/store_factory.py
"""
Backend factory — resolves SQL + Vector backends from config.

Priority: explicit config.json > env vars > defaults.
"""
import os, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_tool_config(tool_name: str) -> dict:
    """Load config.json for a given tool."""
    config_path = Path(__file__).parent.parent / tool_name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def resolve_sql_backend(tool_name: str = "memorymcp") -> dict | None:
    """
    Resolve SQL backend config.
    Priority: config.json storage.sql.backend > env POSTGRES_HOST > env TURSO_DATABASE_URL > None
    """
    cfg = _load_tool_config(tool_name).get("storage", {}).get("sql", {})
    backend = cfg.get("backend")

    # Explicit config
    if backend == "postgres":
        return {"name": "postgres", "dsn": _pg_dsn_from_env()}
    if backend == "turso":
        return {"name": "turso", "url": cfg.get("url") or os.getenv("TURSO_DATABASE_URL"),
                "auth_token": os.getenv(cfg.get("token_env", "TURSO_AUTH_TOKEN"))}
    if backend == "none":
        return None

    # Env-var auto-detect
    if os.getenv("POSTGRES_HOST"):
        return {"name": "postgres", "dsn": _pg_dsn_from_env()}
    if os.getenv("TURSO_DATABASE_URL"):
        return {"name": "turso", "url": os.getenv("TURSO_DATABASE_URL"),
                "auth_token": os.getenv("TURSO_AUTH_TOKEN")}

    return None


def resolve_vector_backend(tool_name: str = "memorymcp") -> dict | None:
    """
    Resolve Vector backend config.
    Priority: config.json storage.vector.backend > env QDRANT_HOST > env TURSO_DATABASE_URL > None
    """
    cfg = _load_tool_config(tool_name).get("storage", {}).get("vector", {})
    backend = cfg.get("backend")

    if backend == "qdrant":
        return {"name": "qdrant",
                "host": os.getenv("QDRANT_HOST", "qdrant"),
                "port": int(os.getenv("QDRANT_PORT", "6333"))}
    if backend == "turso":
        return {"name": "turso", "url": cfg.get("url") or os.getenv("TURSO_DATABASE_URL"),
                "auth_token": os.getenv(cfg.get("token_env", "TURSO_AUTH_TOKEN"))}
    if backend == "postgres":
        return {"name": "postgres", "dsn": _pg_dsn_from_env()}
    if backend == "none":
        return None

    # Auto-detect
    if os.getenv("QDRANT_HOST"):
        return {"name": "qdrant",
                "host": os.getenv("QDRANT_HOST", "qdrant"),
                "port": int(os.getenv("QDRANT_PORT", "6333"))}
    if os.getenv("TURSO_DATABASE_URL"):
        return {"name": "turso", "url": os.getenv("TURSO_DATABASE_URL"),
                "auth_token": os.getenv("TURSO_AUTH_TOKEN")}

    return None


def _pg_dsn_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "gr")
    password = os.getenv("POSTGRES_PASSWORD", "")
    dbname = os.getenv("POSTGRES_DB", "memorymcp")
    return f"host={host} port={port} user={user} password={password} dbname={dbname}"
```

### Phase 1 migration order

1. Create `store_models.py`, `hashing.py`, `impls/__init__.py` (Phase 0)
2. Create `impls/postgres_sql.py` — copy-paste from `pg_store.py`, wrap as instance methods
3. Create `sql_store.py` — ABC + `get_sql_store()` factory
4. Create `store_factory.py` — config resolver
5. Replace `pg_store.py` with the shim (keep same filename)
6. Edit `memory_core.py` line 63: `from shared import pg_store` → `from shared import sql_store as pg_store` (works because both expose same function names)
7. Run tests — everything passes because the shim delegates to the same PG code
8. Rename `tests/test_pg_store.py` → `tests/test_sql_store.py`

---

## Phase 4 sub-plan: migration tool (CLI + MCP)

### CLI skeleton

```python
# tools/shared/migrate_store.py
#!/usr/bin/env python3
"""
Migration tool for moving data between backend combos.

Usage:
  python -m tools.shared.migrate_store export --out backup.jsonl
  python -m tools.shared.migrate_store import --in backup.jsonl
  python -m tools.shared.migrate_store pipe --from postgres+qdrant --to turso+turso
  python -m tools.shared.migrate_store verify --left postgres+qdrant --right turso+turso
"""
import argparse, json, sys, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def cmd_export(args):
    """Export from current (or specified) backends to a JSONL file."""
    sql_store = _make_sql_store(args.sql_backend)
    vector_store = _make_vector_store(args.vector_backend)

    with open(args.out, 'w') as f:
        # SQL section
        if sql_store and sql_store.is_available:
            f.write(json.dumps({"_meta": {
                "kind": "sql", "schema_version": 1,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "source": args.sql_backend,
            }}) + "\n")
            for row in sql_store.iter_all():
                f.write(json.dumps({"_sql": row}) + "\n")

        # Vector section (per collection)
        if vector_store:
            for coll_name in vector_store.list_collections():
                info = vector_store.get_collection(coll_name)
                f.write(json.dumps({"_meta": {
                    "kind": "vector", "collection": coll_name,
                    "dim": info.dim, "has_sparse": info.has_sparse,
                    "distance": info.distance,
                }}) + "\n")
                for point in vector_store.iter_all(coll_name):
                    f.write(json.dumps({
                        "_vec": coll_name,
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": point.payload,
                    }) + "\n")

    logger.info(f"Export complete: {args.out}")


def cmd_import(args):
    """Import from a JSONL file into current (or specified) backends."""
    sql_store = _make_sql_store(args.sql_backend)
    vector_store = _make_vector_store(args.vector_backend)
    collection_metas: dict[str, dict] = {}
    sql_rows = []
    vec_points: dict[str, list] = {}

    with open(args.in) as f:
        for line in f:
            record = json.loads(line)
            if "_meta" in record:
                meta = record["_meta"]
                if meta["kind"] == "vector":
                    collection_metas[meta["collection"]] = meta
                continue
            if "_sql" in record:
                sql_rows.append(record["_sql"])
            elif "_vec" in record:
                coll = record["_vec"]
                vec_points.setdefault(coll, []).append(record)

    # Bulk insert SQL
    if sql_store and sql_rows:
        count = sql_store.bulk_upsert(sql_rows)
        logger.info(f"Imported {count} SQL rows")

    # Bulk insert vectors
    if vector_store:
        for coll_name, points in vec_points.items():
            meta = collection_metas.get(coll_name, {})
            vector_store.ensure_collection(
                coll_name,
                dense_dim=meta.get("dim"),
                sparse=meta.get("has_sparse", False),
                distance=meta.get("distance", "Cosine"),
            )
            from shared.store_models import PointStruct
            structs = [PointStruct(
                id=p["id"], vector=p["vector"], payload=p["payload"]
            ) for p in points]
            count = vector_store.bulk_upsert(coll_name, structs)
            logger.info(f"Imported {count} vectors into '{coll_name}'")


def cmd_verify(args):
    """Verify two backends hold the same data."""
    left_sql = _make_sql_store(args.left.split("+")[0])
    left_vec = _make_vector_store(args.left.split("+")[1])
    right_sql = _make_sql_store(args.right.split("+")[0])
    right_vec = _make_vector_store(args.right.split("+")[1])

    issues = []

    # SQL parity
    if left_sql and right_sql:
        left_ids = set(left_sql.get_all_memory_ids())
        right_ids = set(right_sql.get_all_memory_ids())
        if left_ids != right_ids:
            missing = left_ids - right_ids
            extra = right_ids - left_ids
            if missing:
                issues.append(f"SQL: {len(missing)} IDs in left but not right")
            if extra:
                issues.append(f"SQL: {len(extra)} IDs in right but not left")

    # Vector parity
    if left_vec and right_vec:
        left_colls = set(left_vec.list_collections())
        right_colls = set(right_vec.list_collections())
        if left_colls != right_colls:
            issues.append(f"Vector: collections differ: {left_colls ^ right_colls}")
        for coll in left_colls & right_colls:
            li = left_vec.get_collection(coll)
            ri = right_vec.get_collection(coll)
            if li.points_count != ri.points_count:
                issues.append(f"Vector '{coll}': point count {li.points_count} vs {ri.points_count}")

    if issues:
        print("VERIFICATION FAILED:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    else:
        print("VERIFICATION PASSED: backends are in parity")


def _make_sql_store(backend: str | None):
    if not backend or backend == "none":
        return None
    # ... resolve from backend name
    pass

def _make_vector_store(backend: str | None):
    if not backend or backend == "none":
        return None
    # ... resolve from backend name
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend migration tool")
    sub = parser.add_subparsers(dest="command")

    p_export = sub.add_parser("export")
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--sql-backend", default=None)
    p_export.add_argument("--vector-backend", default=None)

    p_import = sub.add_parser("import")
    p_import.add_argument("--in", dest="in", required=True)
    p_import.add_argument("--sql-backend", default=None)
    p_import.add_argument("--vector-backend", default=None)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--left", required=True)   # e.g. "postgres+qdrant"
    p_verify.add_argument("--right", required=True)   # e.g. "turso+turso"

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "verify":
        cmd_verify(args)
```

### MCP tool wrappers

```python
# tools/shared/migrate_mcp.py
"""
MCP tool wrappers for the migration tool.
Registered on the shared FastMCP instance during tool startup.
"""
from shared.migrate_store import cmd_export, cmd_import, cmd_verify
import argparse, sys


def register_migrate_tools(mcp):
    """Register migration MCP tools on the given FastMCP instance."""

    @mcp.tool()
    async def migrateMemoryBackend(
        export_backend: str = "auto",
        import_backend: str = "auto",
        file_path: str | None = None,
    ) -> str:
        """
        Migrate memory data from one backend to another.

        Args:
            export_backend: Source combo, e.g. "postgres+qdrant" (default: current config)
            import_backend: Target combo, e.g. "turso+turso" (default: current config)
            file_path: Optional intermediate JSONL file. If omitted, pipes directly.

        Returns:
            Migration summary (rows + vectors moved)
        """
        # ... build args namespace, call cmd_export → cmd_import
        return f"Migration complete: {export_backend} → {import_backend}"

    @mcp.tool()
    async def verifyBackendParity(
        left_backend: str,
        right_backend: str,
    ) -> str:
        """
        Verify two backend combos hold the same data.

        Args:
            left_backend: e.g. "postgres+qdrant"
            right_backend: e.g. "turso+turso"

        Returns:
            Verification result (pass/fail with details)
        """
        # ... call cmd_verify, capture output
        return "PASSED" or f"FAILED: {issues}"
```

---

## Phase 5 sub-plan: cleanup checklist

```text
[ ] Remove tools/shared/pg_store.py (the shim)
[ ] Remove tools/shared/pg_store.pyc
[ ] Update memory_core.py: remove `from shared import pg_store` (if still present)
[ ] Grep entire codebase for any remaining `pg_store` references → migrate
[ ] Grep entire codebase for any remaining `qdrant_client` references outside impls/
[ ] Update AGENTS.md:
    - Add "Backend abstraction" section under Architecture
    - Document config.json storage block
    - Document env-var fallback chain
    - Update "Common pitfalls" with backend-switching notes
[ ] Rename tests/test_pg_store.py → tests/test_sql_store.py (if not done in Phase 1)
[ ] Update tools/memorymcp/POSTGRESQL.md → tools/memorymcp/BACKENDS.md
[ ] Update tools/ragmcp/README.md with backend config section
[ ] Verify: python -m pytest tests/ -x -q passes
[ ] Verify: python -m launcher --list-tools shows all tools
[ ] Verify: no import of qdrant_client outside tools/shared/impls/
[ ] Verify: no import of psycopg outside tools/shared/impls/
[ ] Verify: no import of libsql_experimental outside tools/shared/impls/
```

---

## Expanded testing strategy

### Priority: PG+Qdrant (current default) and Turso-only (headline combo)

These two combos get the deepest test coverage. Other combos (PG+Turso, Turso+Qdrant, PG+PG) get contract-level coverage only.

### Test pyramid

```
                    ┌─────────────────┐
                    │  E2E (manual)   │  ← docker compose up, curl the MCP API
                    ├─────────────────┤
                    │  Integration    │  ← real PG + Qdrant / real Turso-local-file
                    ├─────────────────┤
                    │  Contract       │  ← same assertions, parametrized over impls
                    ├─────────────────┤
                    │  Unit           │  ← pure types, hashing, filter translation
                    └─────────────────┘
```

### 1. Unit tests (fast, no dependencies)

| Test file | What it covers | Depends on |
|---|---|---|
| `tests/test_store_models.py` | All 7 types: construction, serialization (to/from dict), `CollectionInfo.dim`/`has_dense` properties | Nothing |
| `tests/test_hashing.py` | `text_hash` deterministic, case-insensitive, whitespace-stripped | Nothing |
| `tests/test_filter_translation.py` | `_to_qdrant_filter()` maps every `Match*` kind correctly; `MatchContains` → `MatchValue` for Qdrant | `qdrant_client` (import only) |
| `tests/test_store_factory.py` | Config precedence (explicit > env > default), env-var auto-detect, invalid combos raise | Config files (temp) |

### 2. Contract tests (parametrized, skip if backend not installed)

```python
# tests/test_sql_store_contract.py
import pytest

SQL_BACKENDS = []
try:
    from shared.impls.postgres_sql import PostgresSqlStore
    SQL_BACKENDS.append("postgres")
except ImportError:
    pass
try:
    import libsql_experimental
    SQL_BACKENDS.append("turso")
except ImportError:
    pass


@pytest.fixture(params=SQL_BACKENDS)
def sql_store(request):
    if request.param == "postgres":
        # ... create PostgresSqlStore with test DSN
        store = PostgresSqlStore(dsn="host=localhost ...")
        yield store
        # ... cleanup: DROP TABLE memories
    elif request.param == "turso":
        store = TursoSqlStore(url="file::memory:")
        yield store
        # in-memory DB auto-cleanup


class TestSqlStoreContract:
    """Every SqlStore impl must pass these."""

    def test_upsert_and_get(self, sql_store):
        mid = sql_store.upsert_memory("uuid-1", "hello world", "concept", ...)
        assert sql_store.get_memory(mid)["text"] == "hello world"

    def test_dedup(self, sql_store):
        id1 = sql_store.upsert_memory("uuid-1", "same text", "concept", ...)
        id2 = sql_store.upsert_memory("uuid-2", "same text", "concept", ...)
        assert id1 == id2  # same text_hash → same ID

    def test_search_text(self, sql_store):
        sql_store.upsert_memory("uuid-1", "python web framework", "concept", ...)
        sql_store.upsert_memory("uuid-2", "rust systems language", "concept", ...)
        results = sql_store.search_text("python", limit=5)
        assert len(results) >= 1
        assert "python" in results[0]["text"].lower()

    def test_search_by_tags(self, sql_store):
        sql_store.upsert_memory("uuid-1", "tagged memory", "concept", ..., tags=["flask", "web"])
        results = sql_store.search_text("tagged", tags=["flask"])
        assert len(results) == 1

    def test_decay(self, sql_store):
        sql_store.upsert_memory("uuid-1", "old memory", "concept", ...)
        deleted = sql_store.decay_memories(ttl_days=0, min_usage_count=0)
        assert deleted >= 1

    def test_metrics(self, sql_store):
        sql_store.upsert_memory("uuid-1", "test", "concept", ...)
        m = sql_store.get_metrics()
        assert m["total"] >= 1
        assert "concept" in m["by_type"]
```

```python
# tests/test_vector_store_contract.py
VECTOR_BACKENDS = []
try:
    from shared.impls.qdrant_vector import QdrantVectorStore
    VECTOR_BACKENDS.append("qdrant")
except (ImportError, Exception):
    pass
try:
    import libsql_experimental
    VECTOR_BACKENDS.append("turso")
except ImportError:
    pass
try:
    from shared.impls.postgres_vector import PostgresVectorStore
    VECTOR_BACKENDS.append("postgres")
except (ImportError, Exception):
    pass


@pytest.fixture(params=VECTOR_BACKENDS)
def vector_store(request):
    if request.param == "qdrant":
        store = QdrantVectorStore(host="127.0.0.1", port=6333)
    elif request.param == "turso":
        store = TursoVectorStore(url="file::memory:")
    elif request.param == "postgres":
        store = PostgresVectorStore(dsn="host=localhost ...")
    yield store
    # cleanup: delete all test collections


class TestVectorStoreContract:
    """Every VectorStore impl must pass these."""

    DIM = 4  # small for tests

    def test_ensure_collection_and_upsert(self, vector_store):
        vector_store.ensure_collection("test_coll", dense_dim=self.DIM)
        from shared.store_models import PointStruct
        vector_store.upsert("test_coll", [
            PointStruct(id="p1", vector=[0.1]*self.DIM, payload={"text": "hello"}),
            PointStruct(id="p2", vector=[0.9]*self.DIM, payload={"text": "world"}),
        ])
        info = vector_store.get_collection("test_coll")
        assert info.points_count == 2
        assert info.dim == self.DIM

    def test_query_dense(self, vector_store):
        # ... upsert points, query with similar vector, assert top result
        pass

    def test_query_dense_with_filter(self, vector_store):
        # ... upsert points with different memory_type, filter, assert only matching returned
        pass

    def test_collection_info_hybrid(self, vector_store):
        """For hybrid collections, has_sparse must be True."""
        vector_store.ensure_collection("hybrid_coll", dense_dim=self.DIM, sparse=True)
        info = vector_store.get_collection("hybrid_coll")
        assert info.has_sparse == True
        assert info.has_dense == True

    def test_retrieve_and_set_payload(self, vector_store):
        # ... upsert, retrieve, set_payload, retrieve again, assert payload updated
        pass

    def test_scroll_pagination(self, vector_store):
        # ... upsert 2500 points, scroll through all, assert count
        pass

    def test_delete_by_ids(self, vector_store):
        # ... upsert, delete by id, assert gone
        pass

    def test_delete_by_filter(self, vector_store):
        # ... upsert points with filePath payload, delete by MatchText filter
        pass
```

### 3. Integration tests (real backends, real workflows)

```python
# tests/test_integration_pg_qdrant.py
"""
Integration test: PG + Qdrant combo (the current default).
Requires: running PG container + running Qdrant container.
Skips if either is unavailable.
"""

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_HOST") or not os.getenv("QDRANT_HOST"),
    reason="PG+Qdrant not available"
)


class TestMemoryMcpEndToEnd:
    """Full memorymcp workflow against PG+Qdrant."""

    def test_upsert_query_get_delete(self, clean_stores):
        from shared.sql_store import get_sql_store
        from shared.vector_store import get_vector_store
        sql = get_sql_store()
        vec = get_vector_store()

        # Upsert
        mid = sql.upsert_memory("test-uuid", "Qdrant uses HNSW for ANN", "code_pattern", ...)
        vec.ensure_collection("memory-store", dense_dim=1024)
        vec.upsert("memory-store", [PointStruct(id=mid, vector=[0.1]*1024, payload={...})])

        # Query
        results = vec.query_dense("memory-store", [0.1]*1024, limit=5)
        assert any(r.id == mid for r in results)

        # SQL search
        sql_results = sql.search_text("HNSW", limit=5)
        assert any("HNSW" in r["text"] for r in sql_results)

        # Get + usage tracking
        mem = sql.get_memory(mid)
        assert mem["usage_count"] == 1  # incremented by get_memory

        # Delete
        assert sql.delete_memory(mid)
        vec.delete("memory-store", ids=[mid])
        assert sql.get_memory(mid) is None
```

```python
# tests/test_integration_turso_only.py
"""
Integration test: Turso-only combo (single backend for SQL + Vector).
Uses local-file mode — no network, no containers.
"""

@pytest.fixture(scope="module")
def turso_stores():
    """Create in-memory Turso stores for both SQL and Vector."""
    from shared.impls.turso_sql import TursoSqlStore
    from shared.impls.turso_vector import TursoVectorStore
    sql = TursoSqlStore(url="file::memory:")
    vec = TursoVectorStore(url="file::memory:")  # separate in-memory DB
    yield sql, vec


class TestTursoOnlyEndToEnd:
    """Full memorymcp workflow against Turso-only."""

    def test_upsert_query_get_delete(self, turso_stores):
        sql, vec = turso_stores
        # Same workflow as TestMemoryMcpEndToEnd but with Turso
        # ...
```

### 4. Migration round-trip tests

```python
# tests/test_migrate_store.py

class TestMigrationRoundTrip:
    """Export from one backend, import into another, verify parity."""

    @pytest.mark.skipif("postgres" not in SQL_BACKENDS, reason="PG not available")
    @pytest.mark.skipif("turso" not in SQL_BACKENDS, reason="Turso not available")
    def test_sql_pg_to_turso(self, tmp_path):
        """Export PG SQL data → JSONL → import into Turso → verify."""
        # 1. Seed PG with test data
        pg = PostgresSqlStore(dsn=...)
        for i in range(50):
            pg.upsert_memory(f"uuid-{i}", f"memory text {i}", "concept", ...)

        # 2. Export
        jsonl_file = tmp_path / "export.jsonl"
        cmd_export Namespace(sql_backend="postgres", vector_backend=None, out=str(jsonl_file))

        # 3. Import into Turso
        turso = TursoSqlStore(url="file::memory:")
        cmd_import Namespace(sql_backend="turso", vector_backend=None, in_=str(jsonl_file))

        # 4. Verify
        pg_ids = set(pg.get_all_memory_ids())
        turso_ids = set(turso.get_all_memory_ids())
        assert pg_ids == turso_ids

    @pytest.mark.skipif("qdrant" not in VECTOR_BACKENDS, reason="Qdrant not available")
    @pytest.mark.skipif("turso" not in VECTOR_BACKENDS, reason="Turso not available")
    def test_vector_qdrant_to_turso(self, tmp_path):
        """Export Qdrant vectors → JSONL → import into Turso → verify."""
        # Same pattern: seed Qdrant, export, import into Turso, verify point counts + IDs
        pass
```

### 5. Filter parity tests (cross-backend semantics)

```python
# tests/test_filter_semantics.py

class TestFilterParity:
    """
    Every Match* kind must return the same ID set across all available backends.
    This catches the tag-array latent bug and any future semantic drift.
    """

    @pytest.fixture
    def seeded_data(self, available_backends):
        """Seed identical data into all available vector backends."""
        # Upsert same 10 points with known payloads into each backend
        # ...
        yield available_backends

    @pytest.mark.parametrize("filter_spec,expected_ids", [
        # MatchValue
        (Filter(must=[FieldCondition(key="memory_type", match=MatchValue(value="concept"))]),
         {"p1", "p2", "p3"}),
        # MatchText
        (Filter(must=[FieldCondition(key="filePath", match=MatchText(text="src/main"))]),
         {"p4", "p5"}),
        # MatchContains (tag array)
        (Filter(must=[FieldCondition(key="tags", match=MatchContains(value="flask"))]),
         {"p1", "p6"}),
        # MatchAny
        (Filter(must=[FieldCondition(key="memory_type", match=MatchAny(values=["concept", "trick"]))]),
         {"p1", "p2", "p3", "p7"}),
    ])
    def test_same_results_across_backends(self, seeded_data, filter_spec, expected_ids):
        """Run the same filter on every backend, assert same ID set."""
        results_by_backend = {}
        for store in seeded_data:
            hits = store.query_dense("test_coll", [0.5]*4, limit=10, filter=filter_spec)
            results_by_backend[store.name] = {h.id for h in hits}

        # All backends must agree
        first = next(iter(results_by_backend.values()))
        for backend_name, ids in results_by_backend.items():
            assert ids == expected_ids, f"{backend_name} returned {ids}, expected {expected_ids}"
```

### 6. Performance comparison benchmarks

```python
# tests/bench_backends.py
"""
Performance benchmarks: PG+Qdrant vs Turso-only.
Not run in normal CI — use `pytest tests/bench_backends.py --benchmark-only`.
"""

class TestBackendBenchmarks:
    """Compare query latency across backends."""

    @pytest.fixture(scope="class")
    def seeded_backends(self):
        """Seed 1000 memories + vectors into each backend."""
        # ...

    @pytest.mark.parametrize("query_text", [
        "python web framework",
        "qdrant vector search",
        "memory deduplication",
    ])
    def test_search_latency(self, benchmark, seeded_backends, query_text):
        """Benchmark search_text across backends."""
        for name, store in seeded_backends.items():
            result = benchmark(store.search_text, query_text, limit=10)
            assert len(result) > 0
```

### Test CI matrix

```yaml
# Suggested CI matrix (when CI is set up):
matrix:
  include:
    - name: "PG + Qdrant"
      env: { POSTGRES_HOST: localhost, QDRANT_HOST: localhost }
    - name: "Turso local"
      env: { TURSO_DATABASE_URL: "file::memory:" }
    - name: "No SQL + Qdrant"
      env: { QDRANT_HOST: localhost }
```

### Test file summary (all new tests)

| Test file | Phase | Type | Backends needed | Priority |
|---|---|---|---|---|
| `test_store_models.py` | 0 | Unit | None | High |
| `test_hashing.py` | 0 | Unit | None | High |
| `test_filter_translation.py` | 2 | Unit | None (import only) | High |
| `test_store_factory.py` | 1 | Unit | None (temp config) | High |
| `test_sql_store_contract.py` | 1/3 | Contract | PG or Turso (skip if absent) | High |
| `test_vector_store_contract.py` | 2/3 | Contract | Qdrant or Turso or PG (skip) | High |
| `test_filter_semantics.py` | 2 | Cross-backend | ≥2 backends | Medium |
| `test_integration_pg_qdrant.py` | 2 | Integration | PG + Qdrant containers | High |
| `test_integration_turso_only.py` | 3 | Integration | Turso local file | High |
| `test_migrate_store.py` | 4 | Migration | ≥2 backends | Medium |
| `bench_backends.py` | 5 | Benchmark | All | Low |

---

## Counter-review remarks (added by second LLM after full review of all 6 phases)

Reviewed: every phase sub-plan (Phase 0–5), all interface contracts, the expanded testing strategy. The plan is execution-ready. Below are the gaps and risks found, organized by severity.

### Critical (fix before starting Phase 1)

**R1. `iter_all(with_vectors=True)` default risks OOM on large datasets** — the `QdrantVectorStore.iter_all` skeleton at line 244 sets `with_vectors=True` by default. For 100k memories × 1024-dim float32, that's ~400 MB of vector data loaded into Python at once. Memorymcp's typical export use case is text metadata (for migration), not vectors (which are needed only for ANN search).

*Fix:* change `VectorStore` ABC default to `with_vectors=False`:
```python
# Interface contracts section, VectorStore ABC
def iter_all(self, collection: str, *, with_vectors: bool = False) -> Iterator[PointStruct]: ...
def bulk_upsert(self, collection: str, points: Iterable[PointStruct]) -> int: ...
```
Migration tool callers that DO need vectors can pass `with_vectors=True` explicitly. Update `QdrantVectorStore.iter_all` accordingly.

**R2. `__collection_metadata__` not filtered in `cmd_import`** — the export dumps this special point (line 1442 acknowledges it), but the CLI skeleton at line 2082 doesn't filter it on import. If a user exports a ragmcp collection and imports into Turso, the destination will have a literal point with `id="__collection_metadata__"` containing the SOURCE embedding model name — which is wrong after migration.

*Fix:* add to `cmd_import` before `vector_store.bulk_upsert`:
```python
# Strip __collection_metadata__ from vector points — it's recorded in the header
for coll_name in vec_points:
    vec_points[coll_name] = [
        p for p in vec_points[coll_name]
        if str(p.get("id", "")) != "__collection_metadata__"
    ]
```

**R3. Migration tool duplicates factory logic** — `migrate_store.py`'s `_make_sql_store(backend_name)` and `_make_vector_store(backend_name)` (line 2158, 2162) duplicate the resolution logic that's already in `store_factory.resolve_sql_backend()`. Two sources of truth = drift over time.

*Fix:* Phase 1 should expose a public factory function:
```python
# tools/shared/sql_store.py
def make_sql_store(backend_name: str, config: dict | None = None) -> SqlStore:
    """Create a SqlStore by name. Used by tests and the migration tool."""
    if backend_name == "postgres":
        from shared.impls.postgres_sql import PostgresSqlStore
        dsn = config.get("dsn") if config else _pg_dsn_from_env()
        return PostgresSqlStore(dsn=dsn)
    if backend_name == "turso":
        from shared.impls.turso_sql import TursoSqlStore
        url = (config or {}).get("url") or os.getenv("TURSO_DATABASE_URL")
        token = (config or {}).get("auth_token") or os.getenv("TURSO_AUTH_TOKEN")
        return TursoSqlStore(url=url, auth_token=token)
    raise ValueError(f"Unknown SQL backend: {backend_name}")
```
Same for `make_vector_store(backend_name)`. Migration tool imports these instead of duplicating resolution.

### Important (fix before Phase 3)

**R4. libSQL HNSW availability not verified** — Phase 3 sub-plan mentions HNSW indexes for Turso vectors (line 1188, 1198) but doesn't verify which libSQL versions support them. HNSW was added to libSQL relatively recently (2024). If the installed `libsql_experimental` is older, vector queries become O(N) brute-force — fine for 1k points, painful for 100k.

*Fix:* add to Phase 3 verification:
```python
# Check at TursoVectorStore startup
def _supports_vectors(self) -> bool:
    """Detect if the connected libSQL supports the VECTOR column type."""
    try:
        self._conn.execute("CREATE TABLE _probe (v VECTOR(1))")
        self._conn.execute("DROP TABLE _probe")
        return True
    except Exception:
        return False

def _supports_hnsw(self) -> bool:
    """Detect HNSW index support (requires VECTOR support first)."""
    if not self._supports_vectors():
        return False
    try:
        self._conn.execute("CREATE TABLE _probe_hnsw (v VECTOR(1))")
        self._conn.execute("CREATE INDEX _test_hnsw ON _probe_hnsw USING hnsw(v)")
        self._conn.execute("DROP TABLE _probe_hnsw")
        return True
    except Exception:
        return False

# Then in query_dense, log a warning if brute-force:
if not self._supports_hnsw():
    logger.warning(f"libSQL HNSW not available — vector queries on '{collection}' will be O(N). "
                   "Consider upgrading libsql_experimental or reducing collection size.")
```

**R5. Turso connection pooling limitations** — `libsql_experimental.connect()` returns a single connection. If the MCP server handles concurrent requests, only one can use the store at a time. For local-file mode this is fine (SQLite serializes anyway). For Turso cloud, HTTP-based — also fine. But if a future user runs Turso via in-process embedded server, this could be a bottleneck. Note: `libsql_experimental` 0.0.34+ supports native connection pooling — check the installed version before designing around this constraint.

*Fix:* add to Phase 3 testing:
```python
# tests/test_turso_concurrent.py
@pytest.mark.skipif("turso" not in SQL_BACKENDS, reason="Turso not available")
def test_concurrent_reads(sql_store):
    """Turso impl should not deadlock on concurrent reads."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(sql_store.search_text, "test", 5) for _ in range(50)]
        results = [f.result() for f in futures]
    assert all(isinstance(r, list) for r in results)
```
If this fails, document the single-connection limit in `BACKENDS.md`.

**R6. pgvector minimum PG version not called out** — `tsvector GENERATED ALWAYS AS ... STORED` requires PG 12+ (line 1325). If a user's PG is older, the DDL will fail at `ensure_collection` time.

*Fix:* add to `postgres_vector_schema.sql`:
```sql
-- Requires PostgreSQL 12+ (for tsvector GENERATED ... STORED)
-- Check version:
DO $$
BEGIN
    IF current_setting('server_version_num')::int < 120000 THEN
        RAISE EXCEPTION 'PostgresVectorStore requires PostgreSQL 12+ (have %)',
            current_setting('server_version');
    END IF;
END $$;
```

**R7. `pg_store._init_lock` race condition in shim** — Phase 1 shim has `_store()` lazy init but no lock. Two threads racing could both create the singleton. Currently `psycopg_pool.ConnectionPool` is thread-safe, so the race is benign in practice — but `tests/test_bug_fixes_low.py:76` imports `_init_lock` directly, which means the test will fail if the shim doesn't expose it.

*Fix:* add to shim:
```python
import threading
_init_lock = threading.Lock()

def _store():
    global _lazy_store
    with _init_lock:
        if _lazy_store is None:
            from shared.sql_store import get_sql_store
            _lazy_store = get_sql_store()
    return _lazy_store
```
And keep `_init_lock` as a module-level re-export so `tests/test_bug_fixes_low.py:76` keeps working.

### Nice to have (fix anytime)

**R8. `BackendProtocol` Literal type missing** — `BackendInfo.sql_backend: str` accepts any string. Adding `Literal` would catch typos:
```python
# tools/shared/store_models.py
from typing import Literal
SQLBackend = Literal["postgres", "turso", "none"]
VectorBackend = Literal["qdrant", "turso", "postgres", "none"]

@dataclass
class BackendInfo:
    sql_backend: SQLBackend
    vector_backend: VectorBackend
    ...
```

**R9. No progress reporting in migration tool** — for 100k+ memories, `export`/`import` take minutes with no feedback. Add `--progress-every N` flag that prints every N records:
```python
# In cmd_export
if args.progress_every and count % args.progress_every == 0:
    print(f"Exported {count} records...", file=sys.stderr)
```

**R10. Phase 5 rollback note missing** — if cleanup breaks something, how does the user recover? The shim is gone. Add to Phase 5 checklist:
```
[ ] Tag the commit BEFORE removing pg_store.py shim (e.g., v1.x-shim-removal-safe-point)
      so users can rollback by checking out that tag and reinstalling.
[ ] Add deprecation NOTICE to release notes / CHANGELOG.md
```

**R11. Schema migration in `migrate_store` is undocumented** — JSONL records assume a fixed schema. If memorymcp's column layout changes between versions, the migration tool will silently lose data. Add a `schema_version` field to the meta header (already there at line 1982) and a `--allow-version-mismatch` flag (default: reject).

**R12. No backup before destructive ops** — the migration tool's `import` doesn't create a backup of the destination before writing. Add `--backup-before` flag that exports the destination to a timestamped file before importing.

### Cross-phase consistency findings

**R13. `memory_core.py` edited in both Phase 1 and Phase 2** — Phase 1 swaps `from shared import pg_store` to `get_sql_store()`. Phase 2 swaps `qdrant_client` to `get_vector_store()`. The order is correct (1 before 2), but both phases should run the test suite separately so a regression in one is caught before the other starts.

*Fix:* add explicit verification between phases:
```
[ ] Phase 1 done → pytest tests/test_memorymcp.py tests/test_pg_store.py -v  (must pass)
[ ] Phase 2 step 1 (memory_core.py edits) → pytest tests/test_memorymcp.py -v
[ ] Phase 2 step 2 (memory_graph.py edits) → pytest tests/test_memorymcp.py -v
... etc per file
```

**R14. Phase 0 verification doesn't check for pre-existing imports** — if `tests/test_pg_store.py:19-184` (line 2473 reference) or any other test imports from `shared.store_models` before Phase 0 creates it, those tests will fail with `ModuleNotFoundError`.

*Fix:* add to Phase 0 verification:
```bash
rg "from shared.store_models" tools/ tests/ launcher/ 2>/dev/null
# Expected output: empty (no pre-existing imports)
```

**R15. `iter_all` consistency between SqlStore and VectorStore** — `SqlStore.iter_all` yields rows, `VectorStore.iter_all` yields points. Both have similar migration-tool semantics. Make sure both are documented as "streaming — don't materialize" in their docstrings.

*Fix:* add to both ABC methods:
```python
def iter_all(self, ...) -> Iterator[dict]:
    """
    Stream all records. Backends MUST yield one at a time without
    materializing the full result set in memory. The migration tool
    relies on this for 100k+ record exports.
    """
```

### Additional implementation-level findings (S1–S5)

These are code-level issues discovered by reading the actual adapter skeletons and SQL patterns in the plan. They're not structural design problems — they're correctness traps that will bite during implementation if not called out.

**S1. `query_hybrid` depends on qdrant-client 1.7+ API** — the `QdrantVectorStore.query_hybrid` skeleton (Phase 2 sub-plan, ~line 640) uses `Prefetch` and `FusionQuery` from `qdrant_client.models`. These classes were added in qdrant-client 1.7.0. If the installed version is older, the import fails at module load time with `ImportError`, breaking ALL hybrid search.

*Fix:* add a version-guarded import with Python-side RRF fallback:
```python
# tools/shared/impls/qdrant_vector.py — top of file
try:
    from qdrant_client.models import FusionQuery, Prefetch, Fusion
    _NATIVE_HYBRID = True
except ImportError:
    _NATIVE_HYBRID = False
    logger.warning("qdrant-client <1.7: hybrid search falls back to Python RRF")

# In query_hybrid:
if _NATIVE_HYBRID:
    # Use Qdrant's native Prefetch + RRF fusion
    ...
else:
    # Fallback: run query_dense + query_sparse separately, fuse in Python
    dense_hits = self.query_dense(collection, dense, limit=limit*2, filter=filter)
    sparse_hits = self.query_sparse(collection, sparse, limit=limit*2, filter=filter)
    return _rrf_fuse(dense_hits, sparse_hits, limit=limit)
```

**S2. Turso vector serialization format** — the `TursoVectorStore` impl uses `json.dumps(vec)` to serialize vectors for libSQL. `json.dumps` produces `[0.1, 0.2, 0.3]` (with spaces after commas). libSQL's `VECTOR` type parser expects `[0.1,0.2,0.3]` (no spaces). The spaces MAY cause a parse error depending on the libSQL version.

*Fix:* use a tight format string instead of `json.dumps`:
```python
def _serialize_vector(vec: list[float]) -> str:
    """Serialize a vector for libSQL's VECTOR type (no spaces)."""
    return "[" + ",".join(repr(x) for x in vec) + "]"
```
Apply this in `TursoVectorStore.upsert` and `TursoVectorStore.query_dense`.

**S3. Turso FTS5 query injection** — `TursoSqlStore.search_text` passes the raw user query to FTS5 `MATCH`. But FTS5 interprets special syntax: `*` (prefix), `"..."` (phrase), `NEAR(...)`, `column:term`, `AND`/`OR`/`NOT` operators. A user searching for `C++ templates` or `path:src/main` would get unexpected results or errors.

*Fix:* wrap the query in double quotes for safe phrase matching, escaping internal quotes:
```python
def _escape_fts5_query(query: str) -> str:
    """Escape a user query for FTS5 MATCH — wraps in double quotes."""
    escaped = query.replace('"', '""')
    return f'"{escaped}"'
```
Use in `search_text`:
```python
fts_query = _escape_fts5_query(query)
conditions.append("memories_fts MATCH ?")
params.append(fts_query)
```

**S4. `bm25()` returns NEGATIVE scores — ranking trap** — the `TursoVectorStore.query_sparse` implementation (Phase 3, ~line 1170) correctly negates `bm25()` scores (`score=-r[2]`). But this is a **subtle correctness trap**: `bm25()` returns negative values where lower (more negative) = better match. If a future maintainer forgets the negation, sparse search results will be ranked **backwards** (worst results first). This kind of bug is invisible in tests that don't check ranking order.

*Fix:* add a defensive assertion and a comment:
```python
# FTS5 bm25() returns NEGATIVE scores (lower = better match).
# We negate so ScoredPoint.score follows the universal contract: higher = better.
raw_bm25 = r[2]
assert raw_bm25 <= 0, f"bm25 should be <= 0, got {raw_bm25} — check FTS5 version"
score = -raw_bm25  # now: higher = better
```

**S5. `ScoredPoint.score` semantics must be universal** — the three VectorStore impls use different internal score representations:
- **Qdrant**: cosine similarity directly (0–1, higher = better)
- **Turso**: `1 - vector_distance_cos(...)` (0–1, higher = better)
- **pgvector**: `1 - (embedding <=> ?)` (0–1, higher = better)

All three produce "higher = better" at the `ScoredPoint.score` level, but this contract is **implicit**. If someone adds a new impl that returns raw distance (lower = better), every caller that sorts by `score DESC` will break silently.

*Fix:* add an explicit contract to the `ScoredPoint` definition in `store_models.py`:
```python
@dataclass
class ScoredPoint:
    """
    A ranked result from a vector query.

    CONTRACT: `score` is ALWAYS similarity (higher = better), NEVER distance.
    All impls MUST normalize: cosine similarity, negated bm25, negated distance, etc.
    Callers can safely sort by `score` descending.
    """
    id: str | int
    score: float
    payload: dict | None = None
    vector: list[float] | dict[str, list[float]] | None = None
```
Add a contract test in `test_vector_store_contract.py`:
```python
def test_score_is_similarity_not_distance(self, vector_store):
    """All impls must return scores where higher = more similar."""
    vector_store.ensure_collection("score_test", dense_dim=4)
    from shared.store_models import PointStruct
    vector_store.upsert("score_test", [
        PointStruct(id="identical", vector=[1.0, 0.0, 0.0, 0.0], payload={}),
        PointStruct(id="opposite", vector=[0.0, 0.0, 0.0, 1.0], payload={}),
    ])
    results = vector_store.query_dense("score_test", [1.0, 0.0, 0.0, 0.0], limit=2)
    assert results[0].id == "identical"
    assert results[0].score > results[1].score  # higher = more similar
```

### Risk summary table

| # | Risk | Phase | Severity | Mitigation |
|---|------|-------|----------|------------|
| R1 | `iter_all` OOM on large datasets | 2/4 | High | Default to `with_vectors=False` |
| R2 | `__collection_metadata__` not filtered on import | 4 | Medium | Filter in `cmd_import` |
| R3 | Factory logic duplicated | 1/4 | Medium | Expose `make_*_store()` public API |
| R4 | libSQL HNSW availability | 3 | Medium | Detect at startup, warn |
| R5 | Turso single-connection limit | 3 | Low | Test + document; check libsql 0.0.34+ |
| R6 | pgvector needs PG 12+ | 3 | Low | Version check in DDL |
| R7 | Shim `_init_lock` race | 1 | Low | Add `threading.Lock` |
| R8 | No `BackendProtocol` Literal | 0 | Cosmetic | Add typing |
| R9 | No migration progress | 4 | UX | `--progress-every` flag |
| R10 | No Phase 5 rollback | 5 | Low | Tag pre-shim-removal commit |
| R11 | No schema version check | 4 | Low | `schema_version` in meta |
| R12 | No backup before destructive ops | 4 | Low | `--backup-before` flag (import is additive by default) |
| R13 | Phases 1+2 both edit memory_core.py | 1/2 | Medium | Per-step pytest |
| R14 | Phase 0 doesn't check pre-imports | 0 | Low | Add `rg` check |
| R15 | `iter_all` streaming not documented | 1/2 | Low | Add docstrings |
| S1 | `query_hybrid` needs qdrant-client 1.7+ | 2 | High | Version-guarded import + Python RRF fallback |
| S2 | Turso vector serialization spaces | 3 | Medium | Use tight format string, not `json.dumps` |
| S3 | FTS5 query injection | 3 | Medium | Escape user query in double quotes |
| S4 | `bm25()` negation trap | 3 | Medium | Assert + comment in query_sparse |
| S5 | `ScoredPoint.score` semantics implicit | 0/2 | Medium | Explicit docstring + contract test |

### Verdict

**The plan is execution-ready** for the user's two primary combos (PG+Qdrant, Turso-only). 

**Critical fixes (R1, R2, S1) must land before Phase 1 starts:**
- R1: change `iter_all` default to `with_vectors=False` (prevents 400MB OOM)
- R2: filter `__collection_metadata__` in `cmd_import` (prevents metadata corruption)
- S1: version-guard the `Prefetch`/`FusionQuery` import with Python RRF fallback (prevents `ImportError` on older qdrant-client)

**Important fixes (R4, S2, S3, S4, S5) must land before Phase 3:**
- R4: detect HNSW/vector support at TursoVectorStore startup
- S2: use tight vector serialization for libSQL (no `json.dumps` spaces)
- S3: escape user queries for FTS5 `MATCH` (prevents query injection)
- S4: assert + document `bm25()` negation (prevents backwards ranking)
- S5: make `ScoredPoint.score` contract explicit (prevents future impl bugs)

**Nice-to-haves (R8–R12, R14–R15) can be folded in during implementation.**

No structural redesigns needed. No new sections to add. The plan's organization (phases → sub-plans → tests → counter-review) is sound. 20 risks identified (R1–R15 + S1–S5), 3 critical, 8 important, 9 low/cosmetic — all with concrete fixes.
