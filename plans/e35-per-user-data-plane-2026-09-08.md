# E3.5 — Per-user data plane (2026-09-08)

> STATUS: plan for coding (user-directed: "database: preset per user or
> master-admin-only + which db config accessible per user; ragmcp: which rag
> accessible per user; memorymcp per user; plus everything mentioned").
> Branch: `feature/e3-multiuser` (extends E3; same suite gates).
> Foundation already verified: F7 — tool functions can resolve the caller
> (`get_http_request()`), E3 gave every tool the identity gate and the user
> store. What's missing is the DATA-level layer inside each server.

## Goal

Within a granted server, a user sees and touches only the data they are
granted. Tool-level access (E3) says WHICH SERVER; data-plane grants say
WHICH DATA inside it.

Principle (differs from masks by design): **data grants are default-CLOSE**
— a newly indexed RAG collection or newly added preset is NOT automatically
visible to non-admin users; an admin must grant it. (Masks are deny-lists;
data grants are allow-lists. This asymmetry is deliberate: data leakage is
worse than tool absence.)

## Non-goals (v1)

- Per-user DB *credentials* — presets/connect configs remain system-defined
  (master admin, in `.env`); users get or don't get each one.
- Shared memories between users (v1 = strict owner scoping; admin sees all).
- Per-user webmcp search_history/fetch-cache partitioning (single-user data,
  shared cache stays; documented privacy note instead).
- Admin hierarchy (master vs sub-admins) — all admins equal in v1; the
  system key is attributed to the first enabled admin ("master").

## User store schema addition

Each user record gains two grant fields (both optional, default = none):

```json
{
  "db_presets": ["02", "work"],          // databasemcp: presets/connections usable
  "rag_collections": ["team-code"]       // ragmcp: collections usable
}
```

- `db_presets`: preset numbers or NAME aliases (the same identifiers
  `connect_preset`/`query connection=` accept). The tool's SYSTEM default
  connection (legacy env-default, mono-style) is implicitly granted to
  role=admin only.
- `rag_collections`: collection names. Absent = none. New collections
  (created by indexing) are admin-visible only until granted.
- memories: no grant list — **owner scoping** (below). `owner` = username.

Helpers in `users_store.py`:

```python
def get_data_grants(username: str) -> dict      # {"db_presets": [...], "rag_collections": [...]}
def set_db_presets(username, presets: list) -> None
def set_rag_collections(username, collections: list) -> None
```

## Per-server enforcement

### databasemcp — presets & DB configs

- **Presets stay master-admin-defined** in `.env` (`DB_PRESET_<NN>` — the
  easy-to-config part the user wants). Per-user = the GRANT.
- Gate point: the preset bypass inside `REGISTRY.get()`/`tokens_map_for_tool`
  consumers — when resolving a preset for a NON-ADMIN caller, check
  `preset.connection_name ∈ user.db_presets`; otherwise the resolution
  raises `LookupError` (same UX as unknown connection — no existence leak).
- System default connection (legacy env pair): admin-only. Non-admin users
  calling without `connection` and without their own active connection →
  clear error "no accessible connection for your account".
- Named runtime connections (`connect_database`): remain SHARED
  process-wide (v1) — consequence documented: any granted user may use a
  shared named connection and E4 txs are per-entry. v2 option: per-user
  connection namespaces.
- `connect_database` (new arbitrary DBs) stays admin-only via the E3.5
  default mask profile (already masked for role=user).

### ragmcp — collections

- Gate point: inside the rag tools — resolve caller (F7) → allow-list =
  `rag_collections`; admin bypasses.
- Tools touched: `list_collections` (filter), `search`/`search_code`/
  `get_copilot_context` (restrict to granted collections — the collection
  param must be granted; no-param search runs across granted ones),
  `index_code`/`start_indexing` (target collection must be granted; new
  collection names → admin-only to CREATE, i.e. role check), `clear_index`/
  `stop_indexing` (already default-masked for users; owner check: only the
  grantor/admin).
- Identity source: `get_http_request()` (F7) — ragmcp tools become
  identity-aware; mono mode = no identity → no filtering (unchanged).

### memorymcp — owner scoping (the big slice)

- `upsertMemory`: payload gains `owner = <client_id>` (admin may pass an
  explicit `owner` param to create shared memories). Existing records: one
  backfill (owner="admin").
- `queryMemory` / `listMemories` / `getMemory`: filter results to the
  caller's own records; admin sees all. Filtering happens AFTER retrieval
  where the store API lacks a filter (brute-force scan is bounded), or via
  payload filter where supported.
- `deleteMemory` / `attachProvenance` / `auditTrail`: owner or admin.
- `decayOrExpire` / `mergeDuplicates` / `reindexMemory` /
  `migrateMemoryBackend` / `getMemoryMetrics`: admin-only (default-masked
  for users — matches the E3.5 default profile already shipped).
- Legacy records (no `owner`): treated as `owner="admin"` (the
  pre-multi-user corpus), invisible to non-admin users until granted.
- mono mode: no identity → no scoping (exactly today).

### Central + UI

- Users tab: two new editors per user — preset checkboxes (from
  `list_presets`) and collection checkboxes (from ragmcp
  `list_collections` via the MCP surface).
- `by_user` call stats already shipped (request_stats).

## Per-server consequence table (final, supersedes the dig)

| Server | Tool-level (E3, done) | Data-level (E3.5) | Destructive defaults (done) |
|---|---|---|---|
| databasemcp | servers grant + masks | presets per user; shared named conns documented | execute_sql/connect/disconnect masked for users |
| ragmcp | servers grant + masks | collections per user (allow-list, default-closed) | index mutations masked for users |
| memorymcp | servers grant + masks | owner scoping (strict; admin sees all) | deletes/expiry/merge/migration masked for users |
| webmcp | servers grant + masks | none (shared cache/history noted) | — |
| simplemcp | servers grant | n/a | — |

## Tests (each phase)

- users_store: grant CRUD + tolerant load (extend `test_users_store.py`).
- databasemcp: granted user reaches preset 02; ungranted user gets
  unknown-connection error; admin bypass.
- ragmcp: granted user searches only granted collections; ungranted
  collection search → Unknown-tool/no-results semantics; admin all.
- memorymcp: alice creates → alice sees it, bob doesn't; admin sees all;
  legacy records admin-only; delete by non-owner rejected.
- UI: Users tab renders the two new editors.

## Phases (suite-gated; branch continues on `feature/e3-multiuser`)

- **D1 (~half day)** store schema + helpers + Users tab editors (+ tests).
- **D2 (~half day)** databasemcp preset/connection gating (+ tests).
- **D3 (~1 day)** ragmcp collection gating (+ tests).
- **D4 (~1 day)** memorymcp owner scoping + legacy backfill (+ tests).
- **D5 (~half day)** docs (README/AGENTS/CHANGELOG), live sweep with two
  users against the real launcher, browser pass on the Users tab.

Total ~3–3.5 days. Sequencing note: D1/D2 depend on nothing; D3/D4 are
independent of each other.

## Open decisions (defaulted, veto-able)

- Data grants default-CLOSED for new collections/presets (admin grants).
- Memories: strict owner scoping; admin sees all; legacy corpus = admin's.
- Preset definitions stay in `.env` (master admin); only grants are
  per-user. (Per-user *credentials* remains out of scope.)
- `connect_database` (arbitrary new DBs) stays admin-only via default masks.
- webmcp history/cache: shared, documented (no partitioning in v1).
