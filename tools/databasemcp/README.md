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
| `DB_TX_IDLE_TIMEOUT` | 300 s | Idle seconds before the reaper rolls an open transaction back; `0` disables the reaper |

## Transactions (E4)

Every tool call is a committed unit (autocommit; per-call pool release).
Interactive transactions pin a **dedicated** connection per transaction —
one transaction per connection entry — so ordinary traffic is never blocked:

- `begin_transaction(connection?)` → returns a `tx_id`
- `query(..., tx_id)` / `execute_sql(..., tx_id)` — statements inside the
  transaction; **nothing commits implicitly** (visible only through the tx
  until you finish it)
- `commit_transaction(tx_id)` / `rollback_transaction(tx_id)` — finish it and
  release the dedicated connection
- `execute_sql(statements=[...])` — atomic batch: up to 50 statements in one
  all-or-nothing transaction (failing index + undone rowcounts reported)

Guards: `disconnect_database` refuses while a transaction is open;
`reset_connections` (mgmt action) force-rolls-back; the reaper reaps idle
transactions (log line per reap). Oracle caveat: DDL commits implicitly —
a DDL statement inside a transaction ends it server-side (the tool warns).
libSQL caveat: `file:`/Turso transactions need a second connection — handled
by the dialect; Postgres transactions run outside the pool. Long-idle
transactions on PG hold `idle in transaction` sessions — the reaper is the
first line of defense, `idle_in_transaction_session_timeout` the backstop.

## Rules files (user-local)

`optimization.json` and `proc_rules.md` are read from
`~/.config/supreme-mcp-tools/databasemcp/` — copy the templates from
`examples/` there and edit. They never live in the repo.

## Run

Discovered by the launcher from `tools/databasemcp/databasemcp_fastmcp.py`
(ports 8000/8110 from `config/ports.json`), or standalone:
`python tools/databasemcp/databasemcp_fastmcp.py`.
