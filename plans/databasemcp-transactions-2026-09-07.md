# databasemcp transaction management plan — 2026-09-07

> STATUS: plan only — **do NOT implement until `feature/databasemcp` is merged to main**.
> Sequencing: branch `feature/db-transactions` off merged main. Written spec-grade for a
> flash-model implementer: verified facts + edge-case catalogue + exact signatures + failure
> playbook; NO verbatim full code. Tracker: `TODO.md` (evolutions section, item E4).

## Goal

Two tiers of transaction support for databasemcp, in one feature:

- **Tier 1 — atomic batches** (tool-surface only): run several statements in ONE call,
  all-or-nothing.
- **Tier 2 — interactive transactions**: explicit `begin_transaction` → statements with a
  `tx_id` → `commit_transaction` / `rollback_transaction`, with idle reaping.

Primary use case: the Oracle/PG workbench (Pro*C bench, test PG). The live memory store
(preset 03) stays a READ-ONLY target per standing caution — transactions don't change that.

## Non-goals (v1)

- Savepoints / nested transactions; cross-connection or distributed transactions.
- More than ONE active transaction per connection entry (reject the second `begin` with a
  clear message — workbench usage is serial; revisit on real demand).
- No change to the no-lock libsql shared-connection concurrency model for NON-tx traffic.
- No mcp_ui panel (FEF metrics only).

## VERIFIED FACTS (probes, all 2026-09-07)

1. **PG does not survive transactions across calls** — `execute_sql("BEGIN")` on the preset
   connection, then `pg_stat_activity` filtered `state='idle in transaction'` returned ZERO
   rows for our user. Root cause in code: `autocommit=True` at connect
   (`tools/databasemcp/dialects/postgres.py:39`) + per-call pool acquire/release
   (`with handle.connection()` in every method, e.g. :81-94).
2. **Oracle commits per call** — `execute()` calls `conn.commit()` before releasing the
   session to the pool (`oracle.py:64`). One call = one committed unit.
3. **libsql DOES carry a transaction across calls on the shared connection** — live probe on
   a throwaway file DB through three separate MCP calls: `BEGIN` → `INSERT` (landed INSIDE
   the open tx; own uncommitted read visible) → `ROLLBACK` (row gone). ⇒ raw capability
   exists but is UNSAFE as-is: the connection is shared with no lock, so another caller's
   statement would silently join the open transaction (hazard verified, not just inferred).
4. **libsql multi-statement strings are rejected pre-execution** by the quote-aware guard
   (`libsql.py:49-63`; verified live in the 2026-09-07 sweep).
5. **Stateless transport is irrelevant to transactions** — DB handles live in the process
   registry; statelessness is the MCP session layer only.

[documented, not probed]: psycopg3 allows toggling `conn.autocommit` outside a transaction;
psycopg_pool resets (rollbacks) a connection left in-transaction on return; SQLite permits
multiple connections to the same file; oracledb connections default to `autocommit=False`.

## Implementer probes — P0, run BEFORE coding (plan-time could not verify)

- **P0-a**: `libsql_experimental` — does `conn.autocommit = False` work, and do
  `conn.commit()` / `conn.rollback()` methods exist? (If only SQL `COMMIT`/`ROLLBACK`
  works under `autocommit=False`, use SQL text; record which.)
- **P0-b**: open a SECOND libsql connection to the same file while the registry connection
  holds it — verify concurrent read + serialized write, and that an uncommitted write on
  conn B is invisible to conn A until commit.
- **P0-c**: psycopg — confirm `autocommit=False` on a pool-returned connection + explicit
  `commit()`/`rollback()`, then restore `autocommit=True` in `finally` before release
  (defensive even though the pool resets).
- **P0-d**: oracledb — hold `pool.acquire()` open for minutes; confirm POOL_GETMODE_WAIT +
  pool `timeout` interplay is acceptable for max=10 (document exhaustion behavior).

## Design

### Shared core (built once, used by both tiers)

- `TxState` on `ConnectionEntry`: `tx_id: str | None`, `tx_handle`, `tx_opened_at`,
  `tx_last_used`, `tx_lock: threading.Lock` (serializes commit/rollback/reap vs in-flight
  statement — NOT the entry lock; the entry lock must NEVER be held across calls).
- Dialect ABC additions (exact signatures):

| Method | Signature | Semantics |
|---|---|---|
| `open_tx` | `open_tx(params: dict) -> Any` | Open a DEDICATED connection, `autocommit=False` (PG: `psycopg.connect` outside the pool, dict_row + statement_timeout options; Oracle: `handle.acquire()` held + call_timeout; libsql: fresh `libsql.connect(url[, auth_token])`) |
| `execute_tx` | `execute_tx(handle, sql: str) -> int` | Execute WITHOUT commit (Oracle's `execute()` commits — tx statements must use this; PG/libsql tx handles behave like normal execute) |
| `close_tx` | `close_tx(handle) -> None` | Rollback-if-open then close (PG/libsql) or release to pool (Oracle) |

- `run_select` is reused unchanged for tx reads (no commit inside it on any dialect).

### Tier 1 — atomic batch

- `execute_sql` gains optional **`statements: list[str]`** (mutually exclusive with `sql`;
  max 50 items; libsql multi-statement guard still applies to EACH item).
- Implementation: `open_tx` → loop `execute_tx` collecting rowcounts → `commit()` →
  `close_tx`; on ANY error: `rollback()`, `close_tx`, report which statement index failed +
  note that all effects were rolled back. Rowcounts returned per statement.
- No cross-call state, no tx_id, no reaper exposure.

### Tier 2 — interactive transactions

- New tools (3):

| Tool | Params | Returns |
|---|---|---|
| `begin_transaction` | `connection` (entry name or preset — bypass applies) | `tx_id` (uuid4 hex) + connection name |
| `commit_transaction` | `tx_id` | committed confirmation |
| `rollback_transaction` | `tx_id` | rolled-back confirmation |

- `query` and `execute_sql` gain optional **`tx_id`**: routes to the pinned `tx_handle` of
  the matching entry. Passing a `tx_id` together with a DIFFERENT `connection` → error
  ("tx belongs to connection X"). Read-guard applies inside transactions unchanged.
- **Idle reaper** (mandatory): asyncio task started in the entry file next to autoconnect;
  sweeps every 60s; rolls back + closes + logs any tx idle > `DB_TX_IDLE_TIMEOUT`
  (default 300s; `0` disables with a loud startup warning). Reaper takes `tx_lock`
  non-blockingly — if busy, skips and reaps next sweep.
- Oracle DDL caveat: DDL implicitly commits in Oracle — when a DDL keyword is detected in a
  tx statement on the oracle dialect, prepend a warning to the tool output. (Best-effort
  keyword check; documented limitation.)
- `disconnect_database` on an entry with an active tx → refuse (message names the tx age);
  the `reset_connections` mgmt extension force-rolls-back + releases everything (admin path).

## Tool surface

15 → **18** tools (3 new) + 2 extended (`execute_sql` statements/tx_id, `query` tx_id).
Update `config.json` tools list, wiring test tool-count assertion, README table.

## Edge-case catalogue (prescribed; each needs a test)

1. `begin_transaction` on unknown connection / unconnected preset → preset bypass applies
   (connects then opens tx); unknown name → standard error.
2. Second `begin_transaction` while one is active on the same entry → rejected with active
   tx age; different entries can each have one.
3. `commit`/`rollback` with unknown or already-finished `tx_id` → clear error.
4. `commit` after `rollback` (replayed id) → same clear error (tx_id cleared on finish).
5. `query`/`execute_sql` with `tx_id` + mismatched `connection` → error naming the owner.
6. Reaper fires while a statement is in flight → tx_lock busy → skip, reap next sweep.
7. Reaper fires normally → rollback + close + log line; subsequent commit of that tx_id →
   error mentions "reaped".
8. Statement fails mid-`statements` batch → rollback, failed index + rowcounts-so-far
   reported, connection state clean afterwards (next batch works).
9. Empty `statements: []` → parameter error (min 1).
10. libsql `open_tx` on `file:` DB while registry conn holds it → works (P0-b); on failure
    (file lock) → error surfaced with dialect message.
11. Oracle DDL inside tx → warning prefix in output; the implicit commit makes the tx
    "finished server-side" — next commit returns the clear "no transaction" mapping.
12. PG TEMP table created inside a tx → lives on the dedicated connection, dies at
    close_tx (document; do NOT treat as a bug).
13. Server restart with an open tx → handles die with the process; DB server-side session
    left `idle in transaction` until its own timeout — README documents
    `idle_in_transaction_session_timeout` as the operator backstop.
14. `execute_sql` with BOTH `sql` and `statements` → parameter error.

## Failure playbook

- Pool exhaustion (Oracle max held by long tx): documented in tool output at `begin` time
  ("holds 1 of N pool sessions"); POOL_GETMODE_WAIT + timeout make the acquire wait.
- Reaper crash safety: whole sweep body in try/except with `logger.exception`; a dead reaper
  task must be restarted on next sweep tick (resilient loop, not one-shot task).
- libsql file locked / disk error at open_tx → dialect error text (already masked-safe for
  params; never echo auth_token).
- tx_handle dead (DB restarted mid-tx): commit/rollback errors map through
  `format_error` + auto-close_tx so the entry is not stuck.

## Env / config changes

- `DB_TX_IDLE_TIMEOUT` (secs, default 300; 0 = disable reaper with warning).
- `DB_TX_SWEEP_INTERVAL` fixed 60s (constant, not env — avoid knob sprawl).
- `config.json`: document both; bump tools list to 18.

## Tests

- Unit: TxState lifecycle; reaper with injected clock (no real sleeps); parameter
  validation (14 catalogue cases where unit-expressible).
- Dialect: batch all-or-nothing on PG real tables (existing `dbmcp_test_*` pattern) + libsql
  tmp file (P0-b second-conn pattern) + Oracle mocked; execute_tx no-commit assertions.
- Concurrency: tx open + concurrent non-tx traffic on the same entry proceeds (no entry-lock
  hold); two concurrent begins on one entry → exactly one wins.
- Wiring: 18 tools, ports 8000/8110 unchanged, mask test still green.
- Live (`tools/databasemcp/test_tools.py`): gated additions — begin/insert/query-tx/rollback
  round-trip on a throwaway libsql DB; batch commit + batch rollback paths.

## Phases (suite-gated, one commit each)

- **T1** — shared core: TxState + open_tx/execute_tx/close_tx trio (P0 probes first), unit +
  dialect tests. Gate: full suite green.
- **T2** — Tier 1 surface: `statements` on execute_sql (implemented over the core), tests.
- **T3** — Tier 2 surface: 3 new tools + tx_id params + reaper + FEF counters, tests.
- **T4** — docs (README, SKILL.md procedure + cautions, CHANGELOG) + live sweep (native
  calls: batch commit, batch rollback, begin/query/commit, reaper observed via short
  timeout) + extend `tools/databasemcp/test_tools.py`.

## Docs checklist

README (transactions section + env table + Oracle DDL caveat + idle-in-transaction
backstop note) · `.agents/skills/mcp-live-tool-test/SKILL.md` (safe-args: throwaway DB only,
never open txs on live presets) · `tools/databasemcp/config.json` · CHANGELOG · this plan's
Deviations section appended during implementation.

## Open decisions (defaulted veto-able)

- One active tx per entry (v1) — revisit on demand.
- Reaper sweep fixed at 60s, no env knob.
- `reset_connections` force-rolls-back without confirmation (admin-only path).
- Oracle DDL warning is best-effort keyword matching (CREATE/ALTER/DROP/TRUNCATE/GRANT/
  REVOKE/ANALYZE), not a parser.

## Deviations (recorded during implementation, 2026-09-07)

1. **Dialect ABC grew to 6 tx methods** (open_tx/select_tx/execute_tx/commit_tx/
   rollback_tx/close_tx) instead of the plan's 3. The plan's signatures
   (`open_tx(params)`, `close_tx(handle)`) could not reach Oracle's pool —
   `open_tx` takes `(handle, params)` and `close_tx` takes `(handle, tx_handle)`;
   `select_tx` was mandatory because PG's `run_select` uses the pool shim and
   Oracle's `_acquire` would read from a DIFFERENT session than the tx's (breaking
   uncommitted-read visibility). commit/rollback became ABC methods for uniformity.
2. **Reaper starts lazily on the first `begin_transaction`**, not in the entry-file
   startup: `asyncio.create_task` at import time has no running loop, and no
   abandoned tx can predate the process's first begin (txs die with the process).
3. **P0-c correction:** psycopg3 has NO `in_transaction` (psycopg2-ism) — `close_tx`
   rolls back unconditionally (no-op when clean), verified harmless.
4. **P0-d stayed documented-only** (no live Oracle instance); Oracle tx behavior is
   covered by the scripted-fake registry tests, not a real driver.
5. **P0-a/b verified as planned**: libsql `autocommit=False` works, commit/rollback
   methods exist, uncommitted writes invisible cross-connection until commit.
6. libSQL batch/tx `CREATE TABLE` before a mid-batch failure may persist (SQLite
   DDL is non-transactional on file DBs) — the all-or-nothing guarantee covers DML;
   the live test asserts ROWS are rolled back, not the earlier DDL.
