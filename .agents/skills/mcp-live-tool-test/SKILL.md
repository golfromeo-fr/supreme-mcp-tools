---
name: mcp-live-tool-test
description: Live-test the MCP tool servers by having the agent itself call every exposed function through its native mcp__<server>__<function> tools — the same path a real harness client uses. Use whenever the user asks to "test the MCP tools", "do tool calls", verify tools after a launcher restart or upgrade, check that servers/functions work, or says "are the tools up". The agent must make the calls itself, as many functions as possible — not one lazy call per server, and not a Python script doing the calls.
---

# Live MCP tool testing (agent-driven)

The agent makes real MCP tool calls through its native `mcp__<server>__<function>`
tools. That exercises the full client path (harness MCP client → auth → session →
server → tool), which is the thing the user actually wants validated. Scripts and
raw HTTP are only for checks a native call cannot express (see "Complementary
HTTP checks") — never a substitute for native calls.

## Procedure

1. **Enumerate what is exposed.** Scan the session's available tools for
   `mcp__<server>__*`. Cross-check the expected function set against the repo:
   `tools/<name>/config.json` → `"tools"` section lists each tool's name and
   input schema. A server may be running but not wired into this session —
   say so explicitly if a configured server has no native tools.

2. **Call every function** — the user's expectation is a full sweep, not one
   call per server. Safe argument choices:
   - deterministic math/string tools → assert the exact expected value
     (`double(21)` must return 42)
   - fetch/search tools → cheap, harmless targets (`https://example.com`,
     query "smoke test")
   - exercise optional parameters on at least one call (e.g. a `greeting`
     override, a pagination `start_index`) — optional paths break too
   - **skip** mutation-sounding functions (`delete*`, `upsert*`, `create*`,
     `execute*`, `write*`, …) unless the user explicitly asks for them

3. **Verify each result**: a response came back, no error flag, plausible
   content. For deterministic tools, compare the exact value.

4. **Report a table**: function | arguments used | result | PASS/FAIL/SKIP,
   then a one-line summary (X passed / Y failed / Z skipped). Flag anything
   that is an upstream error rather than a server defect.

## Known failure mode: `Session not found` (code -32600)

The harness client holds a **stale `Mcp-Session-Id`** — typically after a
launcher restart or someone calling `POST /admin/flush-sessions` (it terminates
ALL live sessions, including the harness's own). The server answers 404 +
`{"error":{"code":-32600,"message":"Session not found"}}` and ZCode does not
auto-reinitialize mid-turn (see repo root `zcode-mcp-session-churn-bug.md`:
ZCode reconnects at task boundaries, ~4 parallel initializers per server).

Recovery — the CLIENT must re-initialize, nothing server-side fixes it:
1. Usually the **next user turn** (task boundary) re-establishes sessions.
2. Otherwise the user restarts the ZCode session / reconnects MCP servers.
3. Do not retry-loop the same call. Report the cause, ask for a new turn or
   reconnect, then verify recovery with one cheap deterministic call first
   (e.g. `square`) before resuming the sweep.

When testing flush-sessions recovery deliberately: do it LAST, after all
native sweeps are done, and warn that it kills the harness's own sessions.

## Expected arguments (current repo tools)

| Function | Safe args | Deterministic? |
|---|---|---|
| `mcp__simplemcp__double` | `{"value": 21}` | yes → 42 |
| `mcp__simplemcp__square` | `{"value": 7}` | yes → 49 |
| `mcp__simplemcp__greet` | `{"name": "smoke", "greeting": "Howdy"}` | yes → "Howdy, smoke!" |
| `mcp__simplemcp__get_secret` | `{}` | returns the SIMPLEMCP_SECRET fixture |
| webmcp `brave_search_api` | `{"query": "...", "count": 3}` | 429 = upstream rate limit, not a defect |
| webmcp `google_search_api` | `{"query": "..."}` | no |
| webmcp `post_url` | `https://httpbin.org/post` + small JSON body | echoed payload |
| webmcp `fetch_url` | `{"url": "https://example.com"}` | assert: status 200 AND body contains "Example Domain" AND no U+FFFD replacement chars |

For functions not listed: read the input schema from the session's tool
definition or the tool's `config.json`, pick minimal safe args, and skip with
a note if none are safe.

### databasemcp notes (added 2026-09-06 — multi-DB workbench, ex-oraclemcp)

Native-call procedure, in this order (the registry is stateful per server):

1. `mcp__databasemcp-stateless__list_connections` `{}` — baseline (usually "none").
2. `mcp__databasemcp-stateless__list_presets` `{}` — shows `.env` presets
   (masked). If it errors "Unknown tool", the launcher predates P7 — say so
   and skip the preset steps.
3. `mcp__databasemcp-stateless__connect_database`
   `{"name": "sweep", "db_type": "libsql", "params": {"url": "file:/tmp/databasemcp_sweep.db"}}`
   — throwaway local DB; `file:` to a nonexistent path creates it (SQLite semantics).
4. `mcp__databasemcp-stateless__execute_sql`
   `{"sql": "CREATE TABLE IF NOT EXISTS sweep_t (id INTEGER PRIMARY KEY, label TEXT)"}`
   then one INSERT — default connection = the one just connected.
5. `mcp__databasemcp-stateless__query` `{"sql": "SELECT * FROM sweep_t", "max_rows": 10}`
6. `mcp__databasemcp-stateless__get_schemas` `{"table_name": "sweep_t"}` — call
   twice: second response contains "(cached)".
7. `mcp__databasemcp-stateless__explain_plan` `{"sql": "SELECT * FROM sweep_t"}`
8. `mcp__databasemcp-stateless__disconnect_database` `{"name": "sweep"}`

Safe-args extras:

| Function | Safe args | Notes |
|---|---|---|
| `query` + `connection` | `{"sql": "SELECT 1", "connection": "<preset NN or NAME>"}` | **preset bypass**: an unconnected preset number/alias connects lazily |
| `query` (guard) | `{"sql": "DELETE FROM x"}` | must answer "query() is read-only" — a success here is a defect |
| `use_database` | `{"name": "<connected name>"}` | switches the active connection |
| `list_presets` | `{}` | passwords always masked (`***`); a raw password in output is a defect |

#### Transaction testing (added 2026-09-07 — self-gating on E4)

**Current server (E4 not yet built) — transaction semantics probes, THROWAWAY
DB only:**

1. `execute_sql` `{"sql": "BEGIN"}` → `{"sql": "INSERT INTO txp VALUES (2, 'in-tx')"}` →
   `query` `{"sql": "SELECT * FROM txp"}` — on libsql the uncommitted row IS visible:
   the shared connection carries an explicit BEGIN/ROLLBACK across separate MCP calls
   (verified 2026-09-07).
2. `execute_sql` `{"sql": "ROLLBACK"}` → `query` again — the in-tx row must be GONE.
   A row that survives ROLLBACK is a defect.
3. On a Postgres connection, do NOT chain BEGIN/COMMIT across calls — `BEGIN` is
   discarded when the call ends (autocommit + pool release). Optional read-only probe:
   `query` `{"sql": "SELECT count(*) AS open_txs FROM pg_stat_activity WHERE state = 'idle in transaction' AND usename = current_user"}`
   right after an `execute_sql("BEGIN")` → must be **0** (verified 2026-09-07).

**After E4 (IMPLEMENTED 2026-09-07 — branch feature/db-transactions):** if
`begin_transaction` appears in `tools/list`, run the full procedure on the
throwaway DB (skip silently on an older launcher, like the preset steps):

1. `begin_transaction` `{"connection": "sweep"}` → returns
   `"Transaction <tx_id> opened on '<name>' (<dialect>)"` — take the `tx_id`.
2. `execute_sql` `{"sql": "INSERT …", "tx_id": "…"}` → "OK (in transaction,
   not committed)"; `query` `{"sql": "SELECT …", "tx_id": "…"}` — the
   uncommitted row is visible INSIDE the tx, invisible on the plain connection.
3. `rollback_transaction` `{"tx_id": "…"}` → `query` WITHOUT tx_id shows the
   row gone.
4. `begin_transaction` again → INSERT → `commit_transaction` → row visible
   WITHOUT tx_id.
5. Atomic batch: `execute_sql` `{"statements": ["INSERT …", "INSERT …"]}` →
   "OK. Batch committed: N statement(s), rowcounts […]"; with one bad
   statement among good ones → "batch ROLLED BACK at statement <i>" and the
   good statements' effects are UNDONE (verify with query).
6. Guards: a second `begin_transaction` on the same connection answers
   "already has an active transaction"; `commit`/`rollback` with an unknown
   or already-finished `tx_id` answers "Unknown or already-finished";
   `disconnect_database` on the connection answers "active transaction";
   `query` with `tx_id` + a mismatched `connection` answers "belongs to
   connection".
7. Always finish every opened tx in the same sweep; the reaper itself is NOT
   exercised in a standard sweep (needs a scratch launcher with a shortened
   `DB_TX_IDLE_TIMEOUT`) — observe it only in a dedicated E4 test session.

Cautions:
- **Never `execute_sql` against a preset pointing at the live Turso memory
  store** unless the user asked for it — preset 03-style entries target real
  data; the sweep flow uses the throwaway `sweep` connection only.
- **Never open transactions on live presets** — a transaction pins a
  dedicated connection and holds locks until commit/rollback; tx probes run
  on the throwaway `sweep` connection only.
- `query` accepts SELECT/WITH only (lexical guard); DML/DDL goes through
  `execute_sql` (which commits).
- Ports: MCP 8000, mgmt 8110 (pinned via `databasemcp_mgmt` in ports.json — ABOVE the auto-allocation corridor; a pin at 8100 collided with simplemcp's floor grab, 2026-09-07).

### webmcp notes (updated 2026-08-27)

- **`brave_search_web` is NOT TESTED — user directive (2026-08-27).** The
  function is not meant to be used but cannot be hidden server-side, so it is
  removed from all testing: never call it in sweeps, do not re-add it.
- The Content-Encoding decode bug (manual `Accept-Encoding: ..., br` sent
  without the brotli package installed → raw compressed bytes as text) was
  FIXED 2026-08-27 by letting httpx set `Accept-Encoding` itself. The
  `fetch_url` assertions above are its regression probe: a mojibake failure
  there is now an unexpected regression, not a known defect.

## Companion scripts (run these instead of retyping probes)

Both run against a launcher you started, using the fastmcp >=4 client in the
active interpreter; ports/keys resolve from repo config automatically.

- **`scripts/sweep_all.py`** — full safe-function sweep of every running
  server (`--only simplemcp,webmcp` to narrow). Encodes the expectations
  table above: brave_search_web never called (user directive), fetch_url
  asserted for clean decode. Exit code = unexpected failures, so it is
  scriptable.
- **`scripts/probe_eras.py [server] [--wire]`** — simulates a NEWEST-era MCP
  client (fastmcp default mode) plus a legacy control on one server;
  `--wire` additionally shows raw HTTP behavior (Mcp-Session-Id issuance,
  stateless vs stateful transport).

```bash
python .agents/skills/mcp-live-tool-test/scripts/sweep_all.py
python .agents/skills/mcp-live-tool-test/scripts/probe_eras.py simplemcp --wire
```

## Complementary HTTP checks (only when explicitly wanted)

Native calls cannot express these; use a short inline `python` + `httpx`
snippet (repo has httpx):

- **Auth enforced**: POST `/mcp` `initialize` without auth header → expect
  HTTP 401. A 200 here is an auth hole — report immediately.
- **Flush recovery**: initialize (get session id) → `POST /admin/flush-sessions`
  with `X-API-Key` → old session id must now get HTTP 404 → fresh initialize
  yields a new id. Remember: this terminates all live sessions of that tool.

Ports come from `config/ports.json` (`assignments.mcp`), keys from
`tools/<name>/config.json` (`auth.api_key`).
