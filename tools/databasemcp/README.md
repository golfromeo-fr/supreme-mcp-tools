# databasemcp — database workbench MCP

Query, explore, and manage **Oracle**, **Postgres**, and **libSQL** (local
file / Turso) databases from any MCP client. Formerly `oraclemcp`; the
multi-database connection registry landed 2026-09-06
(see `plans/databasemcp-overhaul-2026-09-06.md`).

## Connection model

Connections live in a **registry**. Each has a name, a dialect, and its own
lock — queries on different connections run in parallel. One connection is
*active*; the generic tools target it, or accept an explicit `connection`
parameter.

- `connect_database(name, db_type, params)` — connect now; params never echoed
- `disconnect_database(name)` / `list_connections()` / `use_database(name)`
- Generic tools: `query` (read-only, real row limiting), `execute_sql`
  (commit + rowcount, invalidates schema cache), `get_schemas` (cached),
  `list_tables`, `explain_plan`
- Oracle work tools: `get_valid_languages` (LANGUES), `optimize_sql_with_ai`,
  `get_sql_optimization_rules`, `get_proc_rules`

## Dialects

| db_type | Driver | params |
|---|---|---|
| `oracle` | python-oracledb thin (session pool) | user, password, host, port, service_name |
| `postgres` | psycopg 3 + psycopg_pool | host, dbname, user, password (opt: port, connect_timeout) |
| `libsql` | libsql_experimental (no lock; C binding serializes) | url (`file:/path.db`, `file::memory:`, `libsql://…`), optional auth_token |

Testing tip: `libsql` with a `file:` URL is a real-SQL local database —
zero containers. Note: a `file:` path that doesn't exist is created empty.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `USERID` / `DB_HOST` / `DB_PORT` / `DB_SERVICE_NAME` | — | Legacy lazy `default` Oracle connection (created on first query if set) |
| `DB_AUTOCONNECT` | `1` | `0` disables the lazy default entirely |
| `ORACLE_MIN_CONNECTIONS` / `ORACLE_MAX_CONNECTIONS` | 1 / 10 | Oracle session pool sizing |
| `ORACLE_QUERY_TIMEOUT` | 30 s | Oracle call timeout per acquire |
| `DB_QUERY_TIMEOUT_MS` | 30000 | Postgres statement timeout |
| `DB_LOCK_WAIT_S` | 10 | Wait for a busy connection before reporting it busy |
| `AI_API_KEY` / `AI_BASE_URL` / `AI_MODEL` | — | AI SQL optimization endpoint (gpt-4.1 default model) |

## Rules files (user-local)

`optimization.json` and `proc_rules.md` are read from
`~/.config/supreme-mcp-tools/databasemcp/` — copy the templates from
`examples/` there and edit. They never live in the repo.

## Run

Discovered by the launcher from `tools/databasemcp/databasemcp_fastmcp.py`
(ports 8000/8100 from `config/ports.json`), or standalone:
`python tools/databasemcp/databasemcp_fastmcp.py`.
