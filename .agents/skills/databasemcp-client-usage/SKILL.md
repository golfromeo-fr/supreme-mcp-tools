---
name: databasemcp-client-usage
description: Client-side guide for using the databasemcp MCP server well — preset
  nicknames, the lazy-bypass connection model, transactions, and read/write
  discipline. Use whenever a task touches databases through the
  mcp__databasemcp* tools (connecting, running SQL, exploring schemas,
  tuning queries), when picking a connection/preset, or when a preset or
  connection call errors.
---

# databasemcp — client-side usage

The server exposes 18 tools: `connect_database`, `connect_preset`,
`list_presets`, `list_connections`, `use_database`, `disconnect_database`,
`query`, `execute_sql`, `begin_transaction`, `commit_transaction`,
`rollback_transaction`, `explain_plan`, `get_schemas`, `list_tables`,
`get_valid_languages`, `get_proc_rules`, `get_sql_optimization_rules`,
`optimize_sql_with_ai`.

## Golden path (most tasks)

1. `list_presets` — see what connections exist. Entries are NAME-first:
   `- testpg (preset 02) [postgres] — …`.
2. **Skip connecting entirely.** Every tool's `connection` parameter accepts
   an unconnected preset NAME (case-insensitive) or number and connects it
   lazily — the bypass. Prefer the NAME (`connection="testpg"`); the LLM
   should never pass a bare number when a NAME exists.
3. `list_tables` → `get_schemas(table_name=…)` → then `query` / `execute_sql`.

## Connection model (three ways in)

| Route | When | Notes |
|---|---|---|
| lazy bypass via `connection=` param | preferred — zero setup | accepts preset NAME or number, connects on demand |
| `connect_preset("name")` | explicit connect | registers the connection; first one becomes ACTIVE |
| `connect_database(name, db_type, params)` | raw params | **admin-only** (E3.5); presets are the normal route |

- Omitting `connection` uses the ACTIVE connection; switch with
  `use_database(name)`.
- `disconnect_database` is refused while a transaction is open on the
  connection.
- Exactly one open transaction per connection; idle transactions
  auto-rollback after `DB_TX_IDLE_TIMEOUT` (default 300 s).

## Read vs write

- `query` — SELECT/WITH **only** (read-only). `max_rows` default 100.
- `execute_sql` — everything else (INSERT/UPDATE/DELETE/DDL). Either one
  `sql`, or an **atomic batch**: `statements=[...]` (max 50) runs in ONE
  all-or-nothing transaction and reports the failing index on error.
- Multi-step consistency → `begin_transaction` → work → `commit`/`rollback`.

## Presets and nicknames (owner-managed in .env)

- Keys: `DB_PRESET_<NN>=url`, `_DESC` (label), `_NAME` (nickname — the
  preferred handle, case-insensitive), `DB_PRESET_AUTOCONNECT=NN,NN`
  (eager-connect at server startup).
- `.env` is read at process start: adding/renaming presets needs a launcher
  restart — the client cannot do it; tell the user instead of retrying.
- Listings and error messages show the NAME first
  (`testpg/02 (postgres)`). `Unknown preset` errors list every valid handle.

## Safety

- Credentials are masked in every listing — never echo, store, or guess
  them; preset URLs live only in `.env`.
- `execute_sql` writes the REAL database. Confirm scope with the user for
  anything beyond scratch/throwaway tables, and prefer an explicit
  transaction when a mistake must be reversible. Some presets point at live
  stores (e.g. the memory DB) — treat those as production.
- Non-admin users face per-user preset grants (E3.5): `Unknown preset` /
  refusals may be a grant boundary, not a typo. Admins bypass.

## Quick recipes

- **Explore an unfamiliar DB:** `list_presets` → `list_tables`
  (connection=NAME) → `get_schemas` on the interesting tables → `query`.
- **Batch load/fix data:** build `statements=[...]` (≤50, atomic) rather
  than 50 separate `execute_sql` calls.
- **Tune a slow query:** `explain_plan` (dialect-specific output) →
  `get_sql_optimization_rules` → `optimize_sql_with_ai`.
- **Oracle work:** `get_valid_languages` for LANGUES codes; `get_proc_rules`
  for the Pro*C coding rules pack.
