# databasemcp overhaul — implementation spec (2026-09-06)

**Status: APPROVED-FOR-HANDOFF — written for a smaller implementing model (GLM-5.3-flash). Follow EXACTLY. Every design decision is already made; deviate only via the Failure Playbook + Deviations log at the bottom.**

Renames `tools/oraclemcp` → `tools/databasemcp`: a connection registry holding named heterogeneous connections (Oracle / Postgres / libSQL), optional launch-time Oracle connection (lazy, current semantics preserved), runtime connect/disconnect tools, dialect-dispatched query tools, and every verified bug of the current tool fixed.

## Decisions (final)

1. **Name: `databasemcp`** (discovery derives the tool name from the `databasemcp_fastmcp.py` file stem).
2. **Launch: lazy env-default, deactivatable.** If `USERID` + `DB_HOST` env vars are set AND `DB_AUTOCONNECT` is not `"0"` (default: enabled), a `default` Oracle connection is created lazily on first use — exactly today's semantics (server always starts; unreachable DB never blocks startup; error surfaces on first tool call). **`DB_AUTOCONNECT=0` (env or `.env`) is the explicit kill-switch**: the lazy default is never created; every DB tool answers with the "use connect_database(...)" instruction until a connection is made. All other connections come from the new `connect_database` tool at runtime. No new secrets on disk.
3. **The tool joins `startlauncher`** as the 5th live tool.
4. **Rules files are user-local**: `~/.config/supreme-mcp-tools/databasemcp/{optimization.json,proc_rules.md}`; templates ship in `tools/databasemcp/examples/`.

## Binding implementer rules

- **Branch: work on `feature/databasemcp`** (repo convention: `feature/<topic>`). Step 0: `git checkout main && git pull && git checkout -b feature/databasemcp`. All phase commits land there. **Do NOT merge to main yourself** — after P6 the live sweep passes, the user (or a reviewed PR) merges. If the branch already exists, reuse it (`git checkout feature/databasemcp`).
- Names/files/signatures EXACTLY as specced. No renaming, no improvements, no extra abstractions.
- One phase = one commit. Gate every commit: `/home/gr/env_python/bin/python -m pytest tests/ -q` → all green (live-server tests in `tools/simplemcp/test_tools.py` skip themselves when the launcher is down — C5 pattern; never "fix" them by deletion).
- No new dependencies. Verified installed: `oracledb` 3.4.2, `psycopg` 3.3.3 + `psycopg-pool` 3.3.1 + `psycopg-binary`, `libsql-experimental` 0.0.55.
- Never log or return passwords/params. All param output goes through `_mask_params` (spec below).
- Do NOT touch: `tools/shared/function_masks.py`, `tools/shared/server_factory.py`, `tools/shared/atomic_io.py`, any tool other than databasemcp, launcher internals beyond the listed rename edits.
- Step 0 of implementation: create TODO entries; at the end update `TODO.md` + AGENTS.md only as listed in P6.

## VERIFIED FACTS (probe outputs, 2026-09-06 — bake in, do not re-derive)

**A — libsql_experimental 0.0.55** (probed on `file::memory:`, autocommit=True):
- A1 `cursor.rowcount` after INSERT = 1. A3 `cursor.fetchmany(n)` EXISTS and works; SELECT `cursor.description` = 7-tuples `(name, None×6)` — dict rows via `[dict(zip([c[0] for c in cur.description], row)) for row in rows]`.
- A4 `PRAGMA table_info("t")` rows = `(cid, name, type, notnull, dflt_value, pk)`; missing table → `[]` (A7). A8 PRAGMA cursors DO have `description` (`cid,name,type,notnull,dflt_value,pk`).
- A5 `PRAGMA foreign_key_list("t")` rows = `(id, seq, table, from, to, on_update, on_delete, match)` → map `name=f"fk_{id}"`, `column=from` (index 3), `ref_table=table` (index 2), `ref_column=to` (index 4).
- A6 `EXPLAIN QUERY PLAN …` rows = 4-tuples, text is the LAST element → join `[r[-1] for r in rows]`.
- A9 **multi-statement strings are ACCEPTED by one `execute()`** ("CREATE TABLE a(x); CREATE TABLE b(y)" ran both) — no splitting needed in `execute_sql`.

**B — oracledb 3.4.2 pools:** `oracledb.SessionPool.__init__` signature is `(self, dsn, params, cache_name, kwargs)` — do NOT construct SessionPool directly. **USE `oracledb.create_pool(user=…, password=…, dsn=…, min=…, max=…, getmode=oracledb.POOL_GETMODE_WAIT, timeout=…)`** — verified to accept exactly these kwargs; returns a pool with `.acquire()` context manager. `oracledb.POOL_GETMODE_WAIT` exists.

**C — Postgres catalog SQL** (verified live against `POSTGRES_TEST_DSN`, psycopg 3.3.3, `dict_row`):
- C1 list_tables SQL (spec below) runs; `comment` comes back `str`. **TEMP tables are NOT listed** (they live in `pg_temp_N`) — correct for a workbench; document, don't chase.
- C4 the information_schema FK join (spec below) WORKS, including on temp tables: returned `{constraint_name, column_name, table_name, ref_column}`.
- **C-correction: `information_schema.columns` and `pg_constraint … current_schema()` return EMPTY for TEMP tables.** Therefore the P5 Postgres contract tests must use REAL tables named `dbmcp_test_*` in the default schema with `DROP TABLE … CASCADE IF EXISTS` teardown — never temp tables.
- C5 `EXPLAIN (FORMAT TEXT) SELECT 1` returns one text row per line.

**D — psycopg error codes:** `psycopg.errors.QueryCanceled.sqlstate == "57014"` (statement_timeout), `psycopg.errors.SyntaxError.sqlstate == "42601"`. Map via `getattr(e, "sqlstate", None)`.

## File tree (final)

```
tools/databasemcp/
  databasemcp_fastmcp.py  # entry: path bootstrap → import core, db_tools; export setup_extensions;
                          # apply_function_masks(mcp, TOOL_NAME); app = get_transport_app(mcp)
  core.py                 # TOOL_NAME="databasemcp"; ports guard; mcp; logger; metrics
  connections.py          # ConnectionEntry, ConnectionRegistry, REGISTRY, _mask_params, legacy default
  dialects/__init__.py    # DbDialect ABC, DIALECTS dict, get_dialect()
  dialects/oracle.py  dialects/postgres.py  dialects/libsql.py
  db_tools.py             # ALL @mcp.tool() registrations + setup_extensions
  rules.py                # user-local rules loader
  config.json  requirements.txt  README.md  examples/optimization.json  examples/proc_rules.md
```

## core.py (exact)

- Port the ports.json guard VERBATIM from current `oraclemcp_fastmcp.py:29-40`: env `MCP_PORT`/`MCP_MGMT_PORT` override, else `config/ports.json` `assignments.mcp["databasemcp"]` (+ mgmt), `sys.exit(1)` with a clear message if missing. Default MCP_PORT=8000, MGMT_PORT=8100.
- `mcp = create_fastmcp_server(TOOL_NAME)` (from `tools.shared.server_factory`), same as today (`oraclemcp_fastmcp.py:349`).
- Logger + `FileHandler(SCRIPT_DIR / "databasemcp.log")` ported from current 51-58.
- `metrics: dict` ported verbatim from 82-91 (keys: query_count, query_errors, total/min/max query_time_ms, connection_count, connection_errors, schema_lookups).

## connections.py (exact)

```python
@dataclass
class ConnectionEntry:
    name: str; dialect: str; params: dict; handle: Any
    state: str = "CONNECTED"                 # CONNECTED | ERROR | CLOSED
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_error: str | None = None
    schema_cache: dict[str, dict] = field(default_factory=dict)   # table -> describe_table() result
    lock: threading.Lock = field(default_factory=threading.Lock)  # per-entry: guards handle use

class ConnectionRegistry:
    def __init__(self): self._entries = {}; self._active = None; self._map_lock = threading.Lock()
    def connect(self, name, db_type, params) -> ConnectionEntry:
        # under _map_lock: duplicate name -> ValueError("Connection '<name>' already exists (dialect <x>). Use disconnect_database first or another name.")
        # dialect = get_dialect(db_type)  (ValueError lists supported)
        # handle = dialect.connect(params)   # connect errors propagate wrapped: RuntimeError(f"Connect failed: {e}") — never include params
        # entry stored CONNECTED; if self._active is None: self._active = name
    def disconnect(self, name) -> str:
        # entry = self._entries[name] or LookupError listing names
        # if not entry.lock.acquire(timeout=1): return busy message (edge case 2)
        # try: dialect.close(entry.handle) finally: release lock
        # del self._entries[name]; if self._active == name: self._active = next(iter(self._entries), None)
    def get(self, name: str | None = None) -> ConnectionEntry:
        # name given: self._entries[name] or LookupError("Unknown connection '<n>'. Available: [...]")
        # name None, DOUBLE-CHECKED under _map_lock (edge case 1):
        #   if (not self._entries) and os.environ.get("DB_AUTOCONNECT", "1") != "0"
        #      and env USERID and env DB_HOST:
        #       params = {user, password} from USERID.split("/", 1) (OSError if no "/"),
        #       host=DB_HOST, port=int(env DB_PORT, 1521), service_name=env DB_SERVICE_NAME
        #       handle = DIALECTS["oracle"].connect(params)  -> entry "default" (this replaces get_db_connection 200-232)
        #   return self._entries.get(self._active) or raise LookupError(
        #       "No database connection. Use connect_database(name, db_type, params) — db_type: oracle | postgres | libsql")
    def set_active(self, name)          # LookupError if unknown; sets _active
    def list(self) -> list[dict]        # [{name, dialect, state, active: bool, cached_tables, created_at, last_used, last_error}] — NEVER params
    def close_all(self) -> tuple[int, int]   # (closed, skipped_busy) — same acquire(timeout=1) policy per entry

def _mask_params(params: dict) -> dict  # copy; any key in {password, userid, auth_token, token, secret, key} (case-insensitive) -> "***"
REGISTRY = ConnectionRegistry()         # module singleton
```

## dialects/__init__.py (exact protocol)

```python
class DbDialect(ABC):
    name: str; REQUIRED_PARAMS: tuple[str, ...]
    def connect(self, params: dict) -> Any                    # pool or conn handle; raise on failure
    def close(self, handle) -> None
    def ping(self, handle) -> None                            # raise on failure
    def run_select(self, handle, sql, max_rows) -> tuple[list[dict], bool]
        # fetchmany(max_rows + 1); rows beyond max_rows -> truncated=True, drop extras
    def execute(self, handle, sql) -> int                     # commit; rowcount (-1 acceptable)
    def list_tables(self, handle) -> list[dict]               # [{"name": str, "comment": str}]
    def describe_table(self, handle, table) -> dict           # below
    def explain(self, handle, sql) -> str
    def format_error(self, e) -> dict                         # {"error": str, "code": str|None, "message": str, "offset": int|None}

DIALECTS = {"oracle": OracleDialect(), "postgres": PostgresDialect(), "libsql": LibsqlDialect()}
def get_dialect(db_type: str) -> DbDialect                    # ValueError(f"Unsupported db_type '{x}'. Supported: oracle, postgres, libsql")
```
`describe_table` return contract (all dialects): `{"columns": [{"name","type","nullable","comment"}], "constraints": [{"name","type"}], "foreign_keys": [{"name","column","ref_table","ref_column"}]}` — types normalized to `PRIMARY|UNIQUE|FOREIGN|CHECK`.

## dialects/oracle.py (exact)

- REQUIRED_PARAMS = `("user","password","host","port","service_name")`.
- `connect`: **`oracledb.create_pool(user=params["user"], password=params["password"], dsn=oracledb.makedsn(params["host"], int(params["port"]), service_name=params["service_name"]), min=int(os.environ.get("ORACLE_MIN_CONNECTIONS","1")), max=int(os.environ.get("ORACLE_MAX_CONNECTIONS","10")), getmode=oracledb.POOL_GETMODE_WAIT, timeout=int(os.environ.get("ORACLE_QUERY_TIMEOUT","30"))*3)`** (VERIFIED fact B — create_pool, not SessionPool).
- All ops: `with handle.acquire() as conn:` then set `conn.call_timeout = int(os.environ.get("ORACLE_QUERY_TIMEOUT","30")) * 1000`; `cur = conn.cursor()`.
- ping: `SELECT 1 FROM DUAL`.
- run_select: `cur.execute(sql)`; rows = `cur.fetchmany(max_rows+1)`; dicts via `cur.description`; truncated = len(rows) > max_rows.
- execute: `cur.execute(sql)`; `conn.commit()`; return `cur.rowcount`.
- list_tables: `SELECT table_name, NVL(comments, '') FROM user_tab_comments ORDER BY table_name` (name lowercased key).
- describe_table: port the THREE catalog queries VERBATIM from current `fetch_schema_from_cache` (`oraclemcp_fastmcp.py:246-271`) with `:table_name` named binds (columns+comments; constraints with P→PRIMARY, U→UNIQUE, R→FOREIGN, C→CHECK; 4-way FK self-join). DELETE the gatekeeper block (239-242) — caching is the caller's job.
- explain: port the PLAN_TABLE block VERBATIM from 655-668: `DELETE FROM PLAN_TABLE` → `EXPLAIN PLAN FOR {sql}` → `SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY())` with fallback plain `SELECT * FROM PLAN_TABLE`.
- format_error: port `format_oracle_error` VERBATIM from 291-314 (ORA-code parse, offset).

## dialects/postgres.py (exact)

- REQUIRED_PARAMS = `("host","dbname","user","password")`; optional `port` (default 5432), `connect_timeout` (default 10).
- `connect`: kwargs-form (NEVER a DSN string — passwords may contain URL-special chars, edge 14):
  ```python
  from psycopg_pool import ConnectionPool
  from psycopg.rows import dict_row
  ConnectionPool(kwargs={"host":…, "port":…, "dbname":…, "user":…, "password":…,
      "row_factory": dict_row,
      "options": f"-c statement_timeout={os.environ.get('DB_QUERY_TIMEOUT_MS','30000')}"},
      min_size=1, max_size=5, open=False); pool.open(wait=True)
  ```
  On ANY pool failure: log warning, fall back to per-call `psycopg.connect(**same_kwargs_without_timeout_options, connect_timeout=…)` + autocommit, wrapped in a tiny context-manager shim exposing `.connection()`. (Pattern proven in `tools/shared/impls/postgres_sql.py:59-86`.)
- Ops: `with handle.connection() as conn: with conn.cursor() as cur:` (dict_row already returns dicts).
- ping `SELECT 1`; run_select fetchmany as protocol (rowcount guard identical); execute: autocommit pool → implicit commit; return `cur.rowcount`.
- list_tables (VERIFIED C1): `SELECT c.relname AS name, COALESCE(obj_description(c.oid), '') AS comment FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = current_schema() AND c.relkind = 'r' ORDER BY 1`.
- describe_table columns (VERIFIED C2 shape): `SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = %s ORDER BY ordinal_position` (nullable already "YES"/"NO" — map to bool; comment "" for now).
- constraints (VERIFIED C3 shape): `SELECT conname, contype FROM pg_constraint con JOIN pg_class rel ON rel.oid = con.conrelid JOIN pg_namespace ns ON ns.oid = rel.relnamespace WHERE ns.nspname = current_schema() AND rel.relname = %s`; map contype p→PRIMARY u→UNIQUE f→FOREIGN c→CHECK.
- foreign_keys (VERIFIED C4 WORKS): `SELECT tc.constraint_name, kcu.column_name, ccu.table_name, ccu.column_name AS ref_column FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = tc.constraint_name JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s`.
- explain (VERIFIED C5): `EXPLAIN (FORMAT TEXT) {sql}` → join row texts with "\n".
- format_error: `code = getattr(e, "sqlstate", None)` (VERIFIED D: QueryCanceled→57014, syntax→42601); message = first line of `str(e)`.

## dialects/libsql.py (exact)

- REQUIRED_PARAMS = `("url",)` (`file:` / `file::memory:` / `libsql://`); optional `auth_token`.
- `connect`: `libsql_experimental.connect(url, auth_token=…)` (auth_token only if provided); `conn.autocommit = True`; **NO lock** — the C binding serializes statements internally (copy the explanatory comment from `tools/shared/impls/turso_sql.py:62-71`).
- ping `SELECT 1`. run_select/execute: `cur = handle.execute(sql)` (VERIFIED A9: multi-statement accepted as-is); fetchmany per protocol; dicts via `cur.description` 7-tuples (VERIFIED A2/A8). execute returns `cur.rowcount` (VERIFIED A1; -1 acceptable).
- list_tables: `SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name`, comment "".
- describe_table: FIRST reject table names containing `"` or `\0` (PRAGMA interpolation guard). columns via `PRAGMA table_info("<t>")` — rows `(cid,name,type,notnull,dflt,pk)` (VERIFIED A4): nullable = not notnull; comment "". constraints: pk flag → one `PRIMARY` entry. foreign_keys via `PRAGMA foreign_key_list("<t>")` — rows `(id,seq,table,from,to,…)` (VERIFIED A5): `name=f"fk_{id}", column=from(idx3), ref_table=table(idx2), ref_column=to(idx4)`.
- explain: `EXPLAIN QUERY PLAN {sql}` → `"\n".join(r[-1] for r in cur.fetchall())` (VERIFIED A6).
- format_error: code=None, message=str(e) (single line).

## db_tools.py (exact signatures; every tool `async def`, returns `str`, NEVER raises; DB errors → dialect.format_error rendered as today's tool strings, e.g. `"Oracle Error ORA-00942: …"`; the shared helper logs `logger.info(f"[SQL] {sql[:100]}")` BEFORE executing and updates the `metrics` timing block ported from `execute_query` 317-340)

```python
async def connect_database(name: str, db_type: str, params: dict) -> str
    # validate name against ^[A-Za-z0-9_-]{1,32}$ → else "Invalid connection name …"
    # REQUIRED_PARAMS check → "Missing required params for <db_type>: […]"
    # REGISTRY.connect in asyncio.to_thread → "Connected '<name>' (<db_type>). Active: '<active>'"
    # failure → "Connect failed: <e.first-line>" (e NEVER includes params — connections.connect guarantees)
    # tool description MUST note: libsql file: URLs to a nonexistent path create an empty DB (SQLite semantics)
async def disconnect_database(name: str) -> str   # busy → its message; ok → "Disconnected '<n>'. Active now: '<x>'|none"
async def list_connections() -> str               # REGISTRY.list rendered: "- name (dialect, STATE, cached=N)[ *ACTIVE*]"; empty → "none — use connect_database(...)"
async def use_database(name: str) -> str          # "Active connection: '<n>' (<dialect>)"
async def query(sql: str, max_rows: int = 100, connection: str | None = None) -> str
    # empty-sql guard (edge 3); max_rows clamp <1→1, >5000→5000 (edge 4)
    # read-guard (edge 6): s=sql.strip(); while s.startswith("("): s=s[1:].lstrip()
    #   first word upper() in {SELECT, WITH} else "query() is read-only (got '<word>'). Use execute_sql."
    # run_select → render rows like today's query tool; truncated → append "(truncated at <max_rows> rows)"
async def execute_sql(sql: str, connection: str | None = None) -> str
    # execute → "OK. Rows affected: N"; THEN entry.schema_cache.clear() (edge 10)
async def get_schemas(table_name: str, connection: str | None = None) -> str
    # entry.schema_cache hit → render + " (cached)"; miss → describe_table → cache → render
    # empty columns (edge 7, dialect-agnostic) → "Table '<t>' not found on connection '<n>'."
    # render columns + constraints + foreign_keys (INCLUDES FKs — fixes current 391-394 dropping them)
async def list_tables(connection: str | None = None) -> str        # renamed from list_user_tables_with_descriptions
async def explain_plan(sql: str, connection: str | None = None) -> str
async def get_valid_languages() -> str
    # entry = REGISTRY.get(None); entry.dialect != "oracle" → "Oracle-only tool (queries the LANGUES table); active connection is <dialect>."
    # else port current SQL verbatim (397-436: LANGUES, ROWNUM <= 10)
async def optimize_sql_with_ai(sql: str) -> str
    # port 694-792; base_url = os.environ["AI_BASE_URL"] (unset → clear error naming the var; replaces the hard-coded placeholder);
    # schema context: for tables parsed by the EXISTING regex (709-711) → active entry describe_table (live, works now);
    # the blocking OpenAI call inside asyncio.to_thread
async def get_sql_optimization_rules() -> str   # rules.load("optimization.json") → render; missing →
async def get_proc_rules() -> str               # "File not found: <path>. Copy the template from tools/databasemcp/examples/ and edit."
```
Shared helper:
```python
def _with_entry(connection: str | None, fn) -> Any   # runs fn(entry); sync fn called via asyncio.to_thread
    entry = REGISTRY.get(connection)
    if not entry.lock.acquire(timeout=int(os.environ.get("DB_LOCK_WAIT_S", "10"))):
        raise TimeoutError(f"connection '{entry.name}' is busy")
    entry.last_used = time.time()
    try: return fn(entry)
    except Exception as e: entry.last_error = str(e)[:200]; raise
    finally: entry.lock.release()
```
`setup_extensions(registry=None)`: keep the current structure/idempotence (`fef_setup_done`) and the `Extension(...)` registration shape (port from 838-953); register FIVE: `query_stats` (metrics as today), `connections` (REGISTRY.list() + pool kind per dialect), `schema_cache` (per-entry cached-table counts), `reset_connections` (REGISTRY.close_all → `{"success": True, "message": "closed N, skipped M busy"}`), `clear_cache` (clear every entry's schema_cache).

`rules.py`: `RULES_DIR = Path.home() / ".config" / "supreme-mcp-tools" / "databasemcp"`; `load(name) -> str` reading `RULES_DIR / name` (optimization.json rendered as-is; proc_rules.md as-is); FileNotFoundError → the template message above.

## EDGE-CASE CATALOGUE (each must have the prescribed behavior — flash implements ALL)

1. **First-use race**: legacy `default` creation is double-checked under `_map_lock` → two concurrent first calls create exactly ONE entry.
2. **Disconnect while busy**: `entry.lock.acquire(timeout=1)` fails → return "Connection '<n>' is busy (a query is running); retry after it completes." Never blocks indefinitely.
3. Empty/whitespace SQL → "Empty SQL statement."
4. max_rows clamp: <1 → 1; >5000 → 5000.
5. Unknown `connection` name → LookupError listing available names.
6. Read-guard is LEXICAL best-effort (first keyword). Known accepted hole: PG CTE-DML (`WITH x AS (DELETE…)`) — documented in the tool docstring; guard prevents accidents, not attacks (client already holds execute_sql).
7. Missing table: Oracle raises ORA-00942; PG/libsql return empty columns → NORMALIZE: empty columns ⇒ "Table '<t>' not found on connection '<n>'." (dialect-agnostic contract; libsql PRAGMA-empty per VERIFIED A7).
8. No connection at all → the uniform LookupError message with the connect_database example (see connections.get).
9. Timeout fires: Oracle call_timeout → oracledb error → format_error path (entry stays CONNECTED; pool revalidates); PG statement_timeout → sqlstate 57014 (VERIFIED D).
10. `execute_sql` clears the active entry's schema_cache after success (staleness guard on DDL).
11. libsql `file:` to nonexistent path creates an empty DB — SQLite semantics; note in connect_database description.
12. Duplicate connect name → the ValueError message from connections.connect.
13. Extra params keys: allowed, stored, masked like required ones.
14. Passwords with URL-special chars: postgres + oracle use kwargs forms only — NEVER DSN/URL strings.
15. After disconnecting the last connection WITH legacy env present (and `DB_AUTOCONNECT` enabled), the next `REGISTRY.get(None)` lazily recreates `default` (matches today's reconnect semantics).
16. **`DB_AUTOCONNECT=0` kill-switch**: `REGISTRY.get(None)` never creates the env default — even with `USERID`/`DB_HOST` set, a query with zero connections returns the connect_database instruction. Test: set env creds + `DB_AUTOCONNECT=0`, call `query` → instruction message; flip to `1` → default connects.

## PORT-VERBATIM TABLE (current `oraclemcp_fastmcp.py` lines → destination; "verbatim" = copy the SQL/logic, adapt names only)

29-40 ports guard → core.py · 51-58 logging → core.py (`databasemcp.log`) · 82-91 metrics → core.py · 156-163 get_pool_config → DELETED (env reads move into OracleDialect) · 170-193 lock comment → superseded by registry docs · 200-232 get_db_connection → legacy-default assembly in connections.get + create_pool in OracleDialect · 235-288 fetch_schema_from_cache → OracleDialect.describe_table (DELETE gatekeeper 239-242; the `global connection` bug at 284 dies by design) · 291-314 format_oracle_error → OracleDialect.format_error · 317-340 execute_query → `_with_entry` + dialect.run_select (KEEP `[SQL]` log + metrics timing) · 352-394 get_schemas → rewritten (cache + describe + render INCLUDING FKs) · 397-436 get_valid_languages → kept + oracle guard · 439-500 query → generic (read-guard + fetchmany) · 503-559 execute_sql → generic (+ cache clear) · 562-601 list_user_tables → list_tables per dialect · 604-634 + 795-825 rules tools → rules.py (user-local dir) · 637-691 explain_plan → dialect dispatch (Oracle body verbatim) · 694-792 optimize_sql_with_ai → kept + AI_BASE_URL + to_thread + live describe context · 838-953 setup_extensions → same shape, 5 reworked extensions · 961-983 lifespan → DELETED (dead code — never wired) · `table_columns_cache` → DELETED (never populated) · dead imports (openai stays — used by optimize tool; httpx/python-dotenv dropped from tool requirements.txt).

## FAILURE PLAYBOOK

- Spec SQL fails on a live engine → fix the SQL minimally, ADD the failing case as a test, record both in the commit body.
- oracledb create_pool rejects a kwarg at runtime → drop `timeout` first, then `getmode` (defaults are acceptable); never switch back to SessionPool-direct.
- A rename-batch test references deleted code → rewrite to the new equivalent; never delete coverage silently.
- Suite fails on unrelated live-server tests → check the launcher is up (`pgrep -f "[l]aunchmcp.py"`); C5 skipif handles down; do not modify those tests.
- Anything else surprising → STOP the current item, append one line to Deviations below, continue other items; surface it in the phase commit message.

## Rename checklist (mechanical, exact)

1. `git mv tools/oraclemcp tools/databasemcp && git mv tools/databasemcp/oraclemcp_fastmcp.py tools/databasemcp/databasemcp_fastmcp.py`; `rm -f tools/databasemcp/oraclemcp*.log; rm -rf tools/databasemcp/__pycache__` (untracked).
2. `config/ports.json`: `assignments.mcp` key `oraclemcp`→`databasemcp` (keep 8000); `assignments.mgmt` same (keep 8100). VERIFY: `python launchmcp.py --dry-run databasemcp` shows MCP 8000 + mgmt 8100; if mgmt shows a different port, add key `databasemcp_mgmt` (allocation looks up `f"{name}_mgmt"` — launchmcp.py:327).
3. `config/launcher_config.json` toolDirectories path; `launcher/launcher_config.py` DEFAULT_CONFIG.toolDirectories + legacy `ports`/`managementPorts`/`manualPorts` maps (~lines 103,115,122,133).
4. `config/monitoring_config.json:~129` tools key rename.
5. `tools/databasemcp/config.json` (machine-written — PRESERVE structure + auth block): name, script, transports.script, `tools` list = the 13-tool surface above, `environment_variables` = USERID/DB_HOST/DB_PORT/DB_SERVICE_NAME (legacy default) + **DB_AUTOCONNECT (new, "0" disables the lazy env default; default "1")** + ORACLE_MIN_CONNECTIONS/ORACLE_MAX_CONNECTIONS/ORACLE_QUERY_TIMEOUT (now real) + DB_QUERY_TIMEOUT_MS (new, default 30000) + DB_LOCK_WAIT_S (new, default 10) + AI_API_KEY + AI_BASE_URL (new).
6. `startlauncher`: append ` databasemcp` to the tool list.
7. `~/.config/supreme-mcp-tools/tools_config.json` (python -c json edit): `tools.oraclemcp` → `tools.databasemcp` (new list); `disabled_tools.oraclemcp` → `disabled_tools.databasemcp`.
8. Tests: `tests/test_regression.py` set `oraclemcp`→`databasemcp` (still 6 tools); `tests/test_review_fixes.py` TOOLS list; `tests/test_fastmcp_critical_fixes.py:126-141` → point at `tools/databasemcp/db_tools.py` (KEEP the `[SQL]` + no-legacy-names assertions); DELETE `tests/test_oracle_thread_safety.py` (superseded by test_db_registry/concurrency); `tests/fef_v3/test_runner.py` oraclemcp refs → databasemcp, drop the stale `pool_config` extension expectation.
9. `.agents/skills/mcp-live-tool-test/scripts/sweep_all.py`: oraclemcp entry → databasemcp flow: `list_connections` → `connect_database(name="sweep", db_type="libsql", params={"url": "file:/tmp/databasemcp_sweep.db"})` → `execute_sql(CREATE TABLE sweep_t…)` → INSERT → `query` → `list_tables` → `get_schemas("sweep_t")` → `explain_plan` → `disconnect_database("sweep")` + rm the db file; remove the ORA-skip markers.
10. Rewrite `tools/databasemcp/README.md` (env table incl. legacy default + new vars, connect tools, three dialects with examples, link to examples/); root `README.md` tool-table row rename; CHANGELOG entry. Historical mentions in plans/, docs/, CHANGELOG history: LEAVE AS-IS.
11. `tools/databasemcp/requirements.txt`: `oracledb>=1.0.0` (keep) + `psycopg[binary]>=3.1.0` + `psycopg-pool>=3.1.0` + `libsql-experimental>=0.0.30`; drop unused httpx/python-dotenv. Regenerate `requirements.merged.txt` via `python requirements_manager.py` (check `--help` first).

## Tests (new files, exact cases)

- `tests/test_db_registry.py` — FakeDialect (dict-backed, implements the protocol): connect/duplicate-reject/disconnect-active-fallback/get(None)→active/no-env-no-entries→LookupError message/legacy-env default created once under concurrency (two threads, one entry)/list masking asserts `***`/close_all busy-skip.
- `tests/test_db_dialects_libsql.py` — `pytest.mark.skipif(not HAS_LIBSQL)`; tmp_path file URL: ping/INSERT rowcount/5-rows-truncated-at-3 (fetchmany proof)/list_tables/describe with FK (assert fk mapping per VERIFIED A5)/explain text (A6)/missing-table empty (A7)/format_error shape/multi-statement execute (A9).
- `tests/test_db_dialects_postgres.py` — conftest `pg_dsn` fixture (skip if None); **REAL tables `dbmcp_test_*` + DROP CASCADE teardown** (VERIFIED C-correction — never temp tables); same contract cases.
- `tests/test_db_dialects_oracle.py` — patch `tools.databasemcp.dialects.oracle.oracledb` (MagicMock): create_pool called with env min/max/getmode (VERIFIED B)/ping SQL string/ping fail raises/describe uses `:table_name` binds (assert execute args)/format_error ORA-01234 parse/call_timeout set on acquire.
- `tests/test_db_tools_generic.py` — await tool functions directly against two libsql tmp files: connect both → use_database switch → execute_sql CREATE+INSERT → query truncation note → get_schemas twice (2nd contains "(cached)") → list_tables → explain_plan → read-guard rejects "DELETE FROM …" → get_valid_languages on libsql → "Oracle-only" message → disconnect active → auto-fallback to other → list_connections format.
- `tests/test_db_concurrency.py` — FakeDialect.run_select sleeps 0.4s; two entries; two threads release together; BOTH complete < 0.7s (per-entry locks don't serialize across connections).
- `tests/test_databasemcp_wiring.py` — ports.json has databasemcp 8000 + mgmt 8100; zero `oraclemcp` in tracked configs (ports/launcher_config/monitoring — source-scan); `apply_function_masks` honored via `MCP_TOOLS_CONFIG_PATH` tmp config disabling `query` on databasemcp → `mcp.list_tools()` hides it.

## Phases (one commit each, suite-gated; P2 starts with probes re-run + results pasted into commit body)

- **P1 rename + split, behavior-preserving** (Oracle-only through the legacy-default path). Gate: suite green; dry-run shows 8000/8100; `grep -rn oraclemcp tools/ config/ launcher/ tests/ | grep -v plans/ | grep -v docs/` empty.
- **P2 registry + dialects + connect/disconnect/list/use tools.** Gate: registry + all dialect unit tests green.
- **P3 generic tools** (query/execute_sql/get_schemas/list_tables/explain_plan + read-guard + caching). Gate: test_db_tools_generic green.
- **P4 polish**: timeouts, rules.py + examples/, optimize_sql_with_ai rework, FEF extensions rework, README, dead code gone, requirements. Gate: full suite; source-assert tests repointed.
- **P5 contract suites + concurrency + wiring.** Gate: full suite green (pg suite may skip when Postgres down).
- **P6 integration**: startlauncher, tools_config cleanup, requirements.merged regen, CHANGELOG, AGENTS.md conventions entry (mirroring the function-masks entry style). Then USER restarts the launcher; live native-call sweep = the sweep_all databasemcp flow end-to-end; TODO/memory close-out.

**Out of scope:** per-user access (E3), additional dialects, Oracle container CI, cross-connection cache sharing, fixing `get_valid_languages`' ROWNUM-pre-sort quirk (kept verbatim — work DB behavior).

## Deviations (implementer appends here; empty at handoff)

- (none)
