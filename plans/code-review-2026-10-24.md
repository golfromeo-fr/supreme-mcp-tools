# Code review — 2026-10-24

**Repo state:** `main` @ `939aba7`, clean tree. Last full-repo review: 2026-10-18 (through `77a3d71`); delta since = `f7be461` (three security fixes) + `939aba7` (pod-ui tooltip scope fix), 9 files, +273/−16.

**Method — multi-skill battery, executed inline.** The session model tier (GLM-5.3 on the Start Plan) rejects all subagent dispatches ("Start Plan is busy / model concurrency limit exceeded", single-agent retry included), so the usual 3-parallel-agent runbook could not run. Every lens was executed sequentially in the main session instead:

| Skill | How it ran | Result |
|---|---|---|
| graphify | Existing 2026-10-18 graph (7718n/13398e) used as navigation context (god-nodes fed the review priorities); no rebuild | done |
| whole-repo runbook | Inline prioritized pass: all 8 deferred 2026-10-18 findings verified, both recent fixes verified, targeted hunts (lambda-closure siblings, RMW shapes, gate coverage) | done |
| code-review (two-axis) | Fixed point `323409e...HEAD`; both axes run separately below, not merged | done |
| open-code-review-delegate | `ocr delegate preview` + `rule` (5 reviewable files, 4 test files auto-excluded) + host review per rule group | done |
| behavioral proof (blast-radius discipline) | `python -m pytest tests/test_review_fixes_2026_10_18.py -q` → **10 passed** | green |
| mimosa-security-scan | Skipped: its own spec requires an explicit deep-security-scan request, its MCP tools are not bound, and the repo closed a full audit 2026-10-17/18 | n/a |
| blast-radius | Skipped as a skill (`disable-model-invocation: true`); its "prove it by running real code" step adopted instead (test run above) | folded in |

---

## New findings (this pass)

### F1 — HIGH (security, multi-mode): the E3.5 preset-grant gate covers only 2 of ~8 entry points

The 2026-10-18 finding-2 fix (`f7be461`) gated exactly `query` (non-tx) and `execute_sql` (single-statement, non-tx). Every sibling path that resolves a connection still runs with **no grant check**:

```
caller ──> tool entry points, and where the grant gate actually sits
                         ┌──────────────────────────────────────────────┐
 query(sql, conn)        │ conn==None ──> GATED (f7be461)               │
 execute_sql(sql, conn)  │ conn==None ──> GATED (f7be461, called twice) │
                         ├──────────────────────────────────────────────┤
 execute_sql(statements) │ batch path returns BEFORE the gate   — OPEN  │
 query/execute(tx_id)    │ tx branch skips the gate entirely   — OPEN  │
 begin_transaction(conn) │ opens the tx, never gated           — OPEN  │
 get_schemas / list_     │ ride _with_entry(), which has no    — OPEN  │
 tables / explain_plan   │ grant logic (lock + run only)               │
 use_database(name)      │ switches ACTIVE entry, ungated     — OPEN  │
 connect_database(...)   │ raw connect, NO check at all       — OPEN  │
```

- **Worst path (verified in code):** `connect_database` (`tools/databasemcp/db_tools.py:38-79`) has no grant logic anywhere in its body. A non-admin caller raw-connects to any reachable database under a chosen name, then calls `query(sql, connection="that-name")` — and `_assert_preset_grant` **fail-opens for unknown explicit names** (`db_tools.py` ~600: on `LookupError` from `presets.get_preset` it logs and returns, on the stated assumption "registry.get raises its own error later" — false for an entry that exists). Full read/write to the connected target. The libsql dialect additionally accepts `file:` URLs, creating arbitrary local SQLite databases inside the container (inference: bounded by container filesystem).
- **Transaction path:** `begin_transaction(connection=None)` (`db_tools.py:220-255`) resolves the caller only for *attribution*; `REGISTRY.begin_tx` opens on the active/legacy-default entry ungated. `query(tx_id=…)` (`db_tools.py:797-801`) and `execute_sql(tx_id=…)` (`db_tools.py:855-857`) both deliberately skip `_assert_preset_grant` — sound only if the tx was opened gated, which it never is. Arbitrary SQL through the tx path.
- **Batch path:** `execute_sql(statements=[...])` (`db_tools.py:835-848`) dispatches `_run_batch` before any gate call.
- **Metadata tools:** `get_schemas` (`:886`), `list_tables` (`:919`), `explain_plan` (`:945`) all run through `_with_entry` (`:520-535`), which only acquires the entry lock — no identity logic. Schema enumeration of any connected/active target.
- `use_database` (`:186-195`) lets any caller retarget the ACTIVE entry (matters for the ungated metadata tools; query/execute on None would still refuse ungranted actives).

**Deployment context (evidence vs inference):** the code paths are verified by reading; actual exposure requires multi mode (`MCP_AUTH_MODE=multi`, which the work pod runs) plus a non-admin user key with databasemcp reach — an admin-only users.json keeps practical risk low today, but the audit's "data grants default-CLOSED" claim does not hold on these paths.
**Recommendation:** one gate at every entry point that resolves a connection — extend `_assert_preset_grant(connection)` to the batch branch, `begin_transaction`, the three metadata tools, and `use_database`; for `connect_database`, refuse non-admin raw connects outright (admin-only tool, or require the target to be a granted preset). Also close the explicit-name fail-open: an explicit name that is an *existing registry entry* but not a granted preset must be refused, not skipped.

### F2 — MEDIUM: `set_disabled_tools` RMW race still open, and its blast radius grew

`launcher/tools_config.py:110-123` still does read (`:119`) → mutate → write with no lock around the whole mutation (`atomic_write_json`/state-doc write serializes the write only). `enable_tool`/`disable_tool` (`:126-153`) have the same shape. Since 2026-10-18 the config became a **shared state doc** holding both `disabled_tools` and `tools` halves, and the new central `PUT /api/tools/{tool}/config` writes the `tools` half of the same doc — a concurrent mask toggle and config PUT can now clobber either half of each other, not just two toggles. Same failure shape as the documented "mask self-destruct" incident.
**Fix:** run mutations under one flock (or add `save_tools_config(mutation=callable)`).

### F3 — MEDIUM (latent, PG combo): `PostgresVectorStore.list_collections` returns sanitized names

`tools/shared/impls/postgres_vector.py:322-328` strips `vec_` from the stored (already-sanitized) table name; lane names containing sanitized characters never round-trip, so lane enumeration (`lanes.is_lane_name`) silently finds zero lanes on the documented PG+pgvector combo. Live store is Turso (which has the `_vec_collection_map` reverse-map), so latent today.
**Fix:** mirror the Turso reverse-map.

### F4 — LOW-MED: mcp_ui logout never purges the sealed session file

`mcp_ui/management_ui.py:640-642` clears `storage.user` and navigates away; the `.nicegui/storage-user-*.json` file (sealed mcp_key inside) stays on disk until the age-based pruner (`:1281-1297`, idle > N days) removes it. Still open as deferred. Fix: unlink the per-user file on logout/rotate.

### F5 — LOW-MED: harvest `--redact` still misses `_URL`-suffixed values

`deploy/harvest-config.sh:164` redacts suffixes `(KEY|SECRET|TOKEN|PASSWORD|DSN)$` + `POSTGRES_*` + `DB_PRESET_*`. `TURSO_DATABASE_URL` survives — and a remote libsql URL can carry `auth_token=` inline, so a "sharing" bundle can leak a live credential. Add `URL` to the suffix list (or special-case `TURSO_DATABASE_URL`).

### F6 — LOW: `git pull --ff-only` silent on the TEST path (half-fixed)

Work path now warns (`deploy/startcluster.sh:475`); the test-cluster path is still `git pull --ff-only || true` (`:707`). Align it with the warning form.

### F7 — LOW: pod_ui history writes are non-atomic

`pod_ui/ops.py:87-90` `write_text` directly (no tmp+rename); a crash mid-write corrupts `.podui-history.json`. Use `tools/shared/atomic_io`.

### F8 — LOW: conftest `pg_dsn` default user diverges from live

`tests/conftest.py:42` defaults `POSTGRES_USER` to `postgres`; the live deployment uses `mcp`. Only bites when env vars are absent.

### F9 — LOW: duplicate gate call in `execute_sql`

`db_tools.py:852` and `:876` both call `_assert_preset_grant(connection)` on the same non-tx path. Harmless (idempotent) but redundant.

### F10 — LOW (judgement): duplicated default-password resolution

`mcp_ui/management_ui.py` (f7be461 hunk) resolves `os.environ.get(ENV_PASSWORD) or "admin"` separately from `verify_credentials`. If one side ever changes resolution, the refusal check and the actual authentication diverge (lockout or bypass). Extract one shared resolver.

### F11 — Test gaps

- `PUT /api/tools/{tool}/config`: `tests/test_tool_config_endpoint.py` has 6 cases, none covers the keyless/open-server 403 mutation lockdown (the deferred gap — still open).
- No unit test pins the `operations.py` tooltip-scope fix (the smoke recipe catches it only manually).
- `test_db_tools_generic.py` / `test_db_transactions.py` now autouse-bypass `_assert_preset_grant` — fine for mechanics, but add one hot-path test that runs **without** the bypass so the gate itself stays covered on generic flows (it is covered on the None-path by `test_review_fixes_2026_10_18.py`).

---

## Two-axis review of `323409e...HEAD`

### Standards

Against root `AGENTS.md` + `tools/AGENTS.md` + the smell baseline: the diff **conforms to the repo's documented contracts** — the `is_user_store` discriminator rule is applied exactly as the architecture call-out demands (both `db_tools.py` and `memory_tools.py` read `getattr(verifier, "is_user_store", False)`, not env); auth nesting under `"auth"` is enforced by the new endpoint's tests; no `ScoredPoint.score`/`iter_all` surfaces touched. Baseline smells, all judgement calls: **Duplicated Code** — the 4× repeated `if user_store: raise / else: warn+return` shape inside `_assert_preset_grant` (extractable helper), the double gate call (F9), and the duplicated password-resolution (F10). No hard documented-standard violations found.

### Spec

Spec sources: the 2026-10-18 review findings text (findings 2, 3, 5) + the `939aba7` commit message (bug report).

- **Finding 2 (databasemcp grant bypass): PARTIAL.** The spec asked that `connection=None` "resolves the target via `REGISTRY.active_name()` and requires a GRANTED preset, anomalies fail closed". Implemented and proven for `query`/`execute_sql` single-statement paths — but the same bypass the finding describes is alive on the sibling paths (F1: batch, tx, metadata tools, use_database, raw connect). The spec's intent ("data grants default-CLOSED") is not met repo-wide.
- **Finding 3 (memorymcp fail-closed): COMPLETE.** `_memory_caller` returns the distinct `("unknown", False, True)` namespace on every anomaly path under the user-store verifier; mono unchanged; fix is at the source so all three call sites (`memory_tools.py:419/675/755`) inherit it; proven by tests.
- **Finding 5 (mono default-password refusal): COMPLETE.** Refusal only for the literal default password on non-loopback binds; loopback and operator-set passwords proven unaffected (tests pass). Minor: username default not checked (spec only asked password — not a deviation).
- **`939aba7` (pod-ui tooltip): COMPLETE.** `env_key == "work"` is evaluated eagerly at page build inside the correct iteration; the original `e` existed only in the lambda's scope. Sibling hunt across `pod_ui/` + `mcp_ui/`: **clean** — every loop-adjacent lambda binds via default arg (`lambda i=info:`, `lambda k=env_key:` …); the one suspect (`memory_explorer_panel.py:75`) captures a function *parameter*, which is per-call safe.
- **Scope creep:** none of substance (test fixtures + the `active_name()` accessor are supporting the asked-for fixes).

---

## ocr-delegate results

Preview: range `323409e..HEAD`, 9 files, 5 reviewable (4 test files excluded by default rule — they were still reviewed above as part of the Spec axis). Rule group: the standard Python system rule (typos, dead code, mutable defaults/closures-in-loops, boundary handling, error handling, identity/equality, resources, perf, concurrency, security). Coverage: **5/5 reviewed, 0 skipped.** Findings beyond those already listed: none new — the closure-in-loop rule item is precisely the class the `939aba7` hunk fixed, and no other hunk in the range trips the rule groups.

---

## Deferred 2026-10-18 notes — verification table

| Deferred note | Verdict 2026-10-24 |
|---|---|
| PG `list_collections` sanitized names | **OPEN** → F3 |
| `migrate_store` import row-by-row | OPEN (SQL rows loop `upsert_memory`, `migrate_store.py:203-204`; vector side is already `bulk_upsert` at `:245`) — perf-only, LOW |
| RMW race in `set_disabled_tools` | **OPEN, worse** → F2 |
| mcp_ui logout session-file purge | OPEN → F4 |
| harvest `--redact` misses `_URL` | OPEN → F5 |
| startcluster `git pull` silent | HALF-FIXED → F6 |
| pod_ui history non-atomic | OPEN → F7 |
| conftest pg_dsn divergence | OPEN → F8 |

## Triage

- **Done:** full battery executed inline; report at `plans/code-review-2026-10-24.md`; f7be461 + 939aba7 verified correct (10/10 tests).
- **Waiting (user decision):** F1 is the one worth fixing soon — gate the remaining databasemcp entry points (batch/tx/metadata/use_database/connect_database + the explicit-name fail-open).
- **Optional:** F2–F11 are small, well-localized fixes; F11's three test gaps are cheap while context is hot.

---

# Phase 2 — File-by-file full-tree pass (2026-10-24, inline)

*Every tracked code file, read in full (tests get a test-quality lens in slice 9; docs/plans excluded as non-code). Findings prefixed L# = launcher slice, S# = shared, T# = tools, U# = UIs, D# = deploy.*

## Slice 1 — launcher/** (28 files) + launchmcp.py + config/*.json — COMPLETE

Files read in full: management_server, launchmcp, distributed_registry, server_manager, port_manager, tool_discovery, env_manager, launcher_config, service_registry, tool_extensions/registry, tool_extensions/http_server, security/auth, plugins/loader, sync_tools_config, migration, __main__, resilience/retry, tools_config, debug (diagnostic CLI), monitoring/fef_metrics (Prometheus, optional dep); skimmed as plumbing: errors, config_types, __init__.

### L1 — MEDIUM-LOW (C7 gap): mgmt servers have no graceful-shutdown deadline
`ExtensionHTTPServer.stop()` (`launcher/tool_extensions/http_server.py:388-394`) and `ManagementServer.stop()` (`launcher/management_server.py:1090-1097`) await their serve() task with **no timeout**, and neither mgmt uvicorn config sets `timeout_graceful_shutdown` (`http_server.py:370-376`, `management_server.py:1072-1078`) — the C7 fix applied it only to tool MCP servers (`server_manager.py:138`). `ServerManager.stop_server()` awaits `mgmt_server.stop()` unbounded (`server_manager.py:309-310`), so a websocket pinned open on an 81xx port hangs a single-tool stop indefinitely. `stop_all_servers` is bounded (`server_manager.py:396-403`), so full-launcher shutdown survives. Fix: set `timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT` on both mgmt configs + wrap the stop_server await.

### L2 — MEDIUM-LOW: `python -m launcher` entry lacks the C7 protections
`Launcher.stop()` (`launcher/__main__.py:370-389`) has neither the 15s `wait_for(stop_all_servers)` nor the `_arm_exit_guard` post-run watchdog that `launchmcp.py:993-999,1010-1021` has — a lingering non-daemon thread can hold all tool ports after a "clean" exit on the new-style entry. The live daily driver (startlauncher → launchmcp.py) is protected; the entry documented first in AGENTS.md is not. It also never initializes the metrics server (divergence between entry points, undocumented).

### L3 — LOW batch (launcher)
- WS events endpoint accepts only the system key, not admin user keys — inconsistent with REST `_verify_api_key` (`management_server.py:583-596` vs `:353-380`).
- `DistributedExtensionRegistry.subscribe()` dead + latent-broken (`distributed_registry.py:704-733`: uses `http_client._session` directly — None until a GET/POST ran — and sends no auth headers the WS endpoint requires); `RequestCoalescer` dead too (re-export only).
- `ServerManager.start_all_servers` is a dead stub (`server_manager.py:231-257`, body is `pass`).
- Blocking `time.sleep` retry loops on the event loop (`port_manager.py:299,350` — startup-phase only, low practical impact).
- Service-registry URLs use `localhost` (`server_manager.py:205`) against the repo's own 127.0.0.1 convention (works via aiohttp multi-address fallback; latent on IPv6-only resolvers).
- Routine health-check lines logged at WARNING level (`http_server.py:203,212`); currently zero occurrences in the live log (env likely disables) — latent noise, should be DEBUG.
- `launcher/security/auth.py` is dead (re-export only) and its `verify()` compares keys via dict membership (timing-unsafe) — delete, or fix before any revival.
- `/config/{key}` route on 81xx is dead (`config_manager` never wired, `http_server.py:357-366`).
- `.env` created 0644 on first programmatic write (`env_manager.py:469`) — secrets file default perms.
- Failed tool start leaks its `_mgmt` port reservation (`launchmcp.py:351` vs release loop at `:411-418` — only the MCP port is released; harmless within one run).
- `sync_tools_config.py` resolves keys config-only, not env-first (`sync_tools_config.py:94` vs `resolve_tool_api_key`) — 401s in env-keyed setups.
- `ServiceRegistry._check_health` catches `CancelledError` without re-raise (`service_registry.py:330`) — delays task cancellation by one check cycle.
- `restore_missing_env_vars` dead `or Path(".env")` branch (`env_manager.py:711`).

### Checked and solid (slice 1)
C4 reserved-key fix verified live in `port_manager._central_management_port`; C7 correctly bounded on tool servers + `stop_all_servers`; `_bootstrap_config` + placeholder warning per the little-things spec; transport-precedence chain (7801773) intact in both entries; mask fan-out task-reference hygiene (`management_server.py:71-156`) correct; `merge_discovered_tools` UNION + no-prune mask guard (07bdb1a) intact; P2-2 keyless-mutation 403 middleware present; `put_tool_config`/`update_tool_auth` traversal checks correct; `.env` flock rewrite logic correct; tools_config state-doc/file dual path consistent (RMW race already tracked as F2).

## Slice 2 — tools/shared/** (30 files incl. impls/) — COMPLETE

Files read in full: users_store, server_factory, identity, function_masks, atomic_io, state_docs, cluster, cache, artifact_store, pii_redactor, html_utils, utils, relevance_scorer, vector_store, sql_store, store_models, store_factory, hashing, text_search_utils, tidy_state, migrate_mcp, migrate_store, impls/{turso_vector, turso_sql, postgres_sql, postgres_vector, qdrant_vector, libsql_reconnect}.

### S2 — HIGH (PG+pgvector combo, latent live=Turso): PostgresVectorStore query methods are broken — proven live
`query_dense` (`tools/shared/impls/postgres_vector.py:182-190`) passes **4 SQL placeholders but 3 params — every call fails**, filtered or not: `psycopg.ProgrammingError: the query has 4 placeholders but 3 parameters were passed` (the `WHERE 1 - (embedding <=> %s::vector) > -1` clause's placeholder is never in the params list `[vec, vec, limit]`). **Proven live 2026-10-24** against the test PG (preset 02) with a self-cleaning scratch collection. `query_sparse` (`:214-223`) and `query_hybrid` (`:239-250`) bind correctly only WITHOUT a filter — any `Filter` shifts the LIMIT into the first filter placeholder (`params = [q, q, limit] + filter_params` vs SQL order `…, filter…, LIMIT`). **Zero contract-test coverage**: `tests/test_backend_contracts.py` exercises PG set_payload/iter_all/scroll/query_sparse-unfiltered only — no query_dense, no filtered query, on the PG impl (the exact HIGH-14 mock-blindspot class). The documented PG+PG combo cannot serve memorymcp/ragmcp queries at all. Fix: correct the param lists; add filter-case contract tests for all three query methods on PG.

### S3 — MEDIUM (live store): turso_sql FTS5 sync corrupts the index on text-variant re-upserts
Dedup-update path (`tools/shared/impls/turso_sql.py:193-200`): the external-content FTS5 `'delete'` command is issued with the **NEW** text — FTS5 requires the text as it was indexed, so a re-upsert whose text differs from the indexed variant only by case/whitespace (identical `text_hash`: hash normalizes strip+lower) leaves the old terms in the doclist forever (stale matches + index bloat). The `ON CONFLICT` race branch (`:205-213`) updates `text` with **no FTS sync at all**. Fix: SELECT the old text before UPDATE and delete with it; sync FTS in the conflict branch (or route re-sync via `rebuild_sparse`, which already exists as the recovery path). Same class in `turso_vector.upsert`'s FTS sync was CLEARED by live probe (no double-indexing on re-upsert).

### S5 — MEDIUM: migration MCP tools block the whole memorymcp event loop
`migrateMemoryBackend` / `verifyBackendParity` (`tools/shared/migrate_mcp.py:56-78, 99-130`) are `async def` tools that run the fully-synchronous `cmd_export`/`cmd_import`/id-scan **directly on the event loop** — a full-store migration (the live turso file is ~106MB) freezes every concurrent MCP call on that server (and its 81xx health checks) for the whole run. Fix: `await asyncio.to_thread(...)` around the sync sections.

### S1 — MEDIUM-LOW: users.json cross-process RMW race (F2 class)
Every users_store CRUD fn is `load_users()` → mutate → `save_users()` with no cross-process lock (central 8200 and mcp_ui both write the file; `atomic_write_json` serializes writes only). A rotate-key racing a mask edit loses one. Known watch item from the grounds plan; same fix shape as F2 (flock the whole mutation, or route through `save_doc_cas` in db mode).

### S4 — LOW-MED (latent, verified): `should`-only Filters are silently ignored by turso/pg vector impls
`_filter_to_sql` (`turso_vector.py:662-686`, mirrored in postgres_vector) translates `must`/`must_not` only — a should-only Filter produces an empty WHERE → **unfiltered query** (proven live in-memory: a point failing the should-condition was returned). Qdrant honors `should` (`qdrant_vector.py:39-49`). No live caller passes `should=` today (all five Filter sites use `must=`), so latent contract gap — worth a loud drop-warning or an implementation.

### S6 — LOW batch (shared)
- `upsert_memory` in BOTH SQL impls swallows all exceptions and returns the success-shaped id (`postgres_sql.py:313-315`, `turso_sql.py:225-227`) — SQL failure looks like success to memory_tools; vector row then exists without its SQL record.
- `turso_vector.ensure_collection` falls back to a BLOB-embedding table on ANY create error (`turso_vector.py:206-217`) — silently loses VECTOR typing (later vector queries fail) instead of raising.
- `store_factory.pg_dsn_from_env` defaults user `gr`/db `memorymcp` (`store_factory.py:109-114`) — another live-`mcp` divergence (F8 class); used when POSTGRES_* env incomplete.
- Flush-endpoint admin gate fails OPEN on any resolution exception (`server_factory.py:646-647`) — inconsistent with the fail-closed discipline (impact bounded: nuisance session flush).
- `_resolve_api_key` accepts a bare top-level `api_key` (`server_factory.py:712`) contradicting the documented auth-nesting convention (leniency, not a hole).
- `users_store._now_iso` defined twice (`:161`, `:753` — identical).
- `migrate_store cmd_import` buffers the whole JSONL in RAM and inserts SQL row-by-row (deferred perf note re-confirmed; vector side already bulk).

### Checked and solid (slice 2)
`state_docs` µs-CAS + verify-read; `cluster` CAS-retry + tombstone env semantics + atomic auth adoption; `atomic_io` inode-safe sidecar; `identity` gate (fail-open documented, require-identity opt-in, H-series guard note); `function_masks` file-first runtime toggles; `cache` TTLCache single-source; `artifact_store` traversal guard + not-found-vs-error distinction; `libsql_reconnect` stale-stream retry-once; `qdrant_vector` thin adapter (should-filter honored — parity reference); `postgres_sql` core (DSN masking, feature-scoped trgm, documented placeholder order in search_text); ScoredPoint/iter_all/bulk_upsert contracts honored on the paths that work; `hashing` owner-mixed namespace; `text_search_utils` FTS5 escaping; `tidy_state` guards; SSRF guard in `utils.is_internal_url`.

## Slice 3 — simplemcp + webmcp + convertermcp + fef_integration — COMPLETE

Read in full: simplemcp_fastmcp (419), webmcp_fastmcp (1076), convertermcp_fastmcp (466), fef_integration (409).

**No medium+ findings.** webmcp: SSRF guards on `fetch_url`/`post_url` (initial + post-redirect-final check; DNS-rebinding/TOCTOU honestly documented as a known limitation); Content-Encoding lesson comment intact; cache = shared TTLCache subclass. convertermcp: allowed-roots + size caps + SSRF-guarded download + temp cleanup — clean. fef_integration: duplicate-registration tolerant (2026-09-07 lesson encoded), mgmt-port mismatch warning. LOW batch: webmcp `search_history` data source reads `recent_searches` which is never written (always empty); convertermcp `format_usage`/`conversion_queue`'s `pending`/`active` keys never populated (cosmetic dead data sources).

## Slice 4 — databasemcp (complete) — COMPLETE

Read in full: connections.py (450), db_tools.py (1278, all regions), dialects/{postgres,oracle,libsql,_base}, presets.py, core.py, databasemcp_fastmcp.py, rules.py. (F1's gate-coverage analysis from phase 1 verified at source level throughout: `_run_batch` `:714`, tx-open `begin_tx`, `list_tables` `:936`, `explain_plan` `:965`, `get_schemas` `:905` — all resolve entries with no grant check.)

### T1 — LOW: dead duplicate `get_schemas` tool shadows a deleted API
`db_tools.py:494-513` registers a second `@mcp.tool() get_schemas` whose body calls `connections.fetch_schema_from_cache` — deleted in the P4 overhaul. The later registration at `:886` (last-wins) is why live calls work; the dead one is one FastMCP re-registration-order change away from breaking every get_schemas call. Same class as the known "scripted slice-rewrites leave duplicate defs" lesson. Delete `:494-513`.

### T2 — LOW: error rendering lazily connects Oracle
`_render_error` (`db_tools.py:977`) calls `REGISTRY.get(None)` just to pick a dialect — with legacy `USERID`/`DB_HOST` env set, formatting an error message can open the env-default Oracle pool as a side effect. Same shape in `optimize_sql_with_ai` (`:1033`, there it's semi-intentional schema context). Use `REGISTRY.active_name()` + an entries lookup without the lazy default.

### Checked and solid (slice 4)
Tx machinery (per-entry tx_lock, non-blocking reaper, force-rollback close_all, tx-pins-connection mismatch check); dialects (Oracle pool + call_timeout + verified catalog SQL; PG pool-outside-tx design + statement_timeout; libsql quote-aware multi-statement rejection + PRAGMA interpolation guards); presets (masked URLs, tolerant autoconnect, alias resolution); secret masking in every connect-error path; entry point + core + rules loader clean.

## Slice 5 — ragmcp (complete) — COMPLETE

Read: ragmcp_fastmcp.py (2508, full), indexer/{incremental_indexer (1063, full), lanes, path_guards, embedding_gate, local_embeddings, file_filters, metadata_config, sparse_vector_gen (full), smart_chunkers + copilot_context_injector (core read; remaining defs are same-pattern per-language chunkers/formatting — confirmed via structure scan)}.

### R2 — MEDIUM-LOW: stale-file deletion is a SUBSTRING match — collateral chunk loss
`remove_stale_files` (`tools/ragmcp/indexer/incremental_indexer.py:779-798`) deletes stale points with `MatchText(text=rel_path)` on `filePath`. `MatchText` translates to `LIKE '%rel_path%'` (turso, `turso_vector._condition_to_sql`) / full-text match (qdrant) — NOT exact equality. A stale file whose rel_path is a substring of another indexed file's path (e.g. stale `b/utils.py` vs live `a/b/utils.py`) deletes the live file's chunks too, silently — the collateral file's metadata row survives so it is never re-indexed. Trigger: delete/rename a file whose path is a suffix-substring of another, then run an incremental index. Fix: exact-match delete — fetch point ids by `filePath` match then delete by ids after an equality check (or a dedicated MatchValue filter once payload dedup allows).

### R3 — LOW batch (ragmcp)
- `local_embeddings.test_local_embeddings()` uses the removed `'small'` preset (`local_embeddings.py:260-263`, `MODELS['small']` → KeyError) — P0 preset-cleanup miss, dead self-test.
- FEF `start_indexing` extension schema enum lists only `auto|fast|high-quality` (`ragmcp_fastmcp.py:827,2405`) while `EMBEDDING_MODEL_PRESETS` now also has `nvidia-free` — the UI enum can't select it (cosmetic; `index_code` passes it fine).
- `get_indexed_files` hardcodes legacy roots `/workspaces/spfr{,.orig}` in its rel-path resolution (`incremental_indexer.py:250`) — harmless legacy compat, worth a comment.

### Checked and solid (slice 5)
Lane slug normalization (2026-10-17 org-prefix fix) + hash-suffix overflow; pre-flight dim abort before embedding work; embedding_gate latch on both remote providers; path/collection guards (traversal rejected, allowed_roots config-driven); per-user collection grants on search/index_code/clear_index/list_collections with fail-closed `is_user_store` anomaly handling (incl. lane collections inheriting the base's grant); zombie-reaping threads + PID-file hygiene; CRIT-11 query-vs-index BM25 statistics separation; `clear_index` confirm-gated with lane opt-in; deprecated tool shims route correctly.

## Slice 6 — memorymcp (complete) — COMPLETE

Read: memory_core (267, full), memory_tools (1613, full), memory_graph (373, full), text_utils (223, full), memorymcp_fastmcp entry, memory_autouse (structure-scan: policy-file readers + system-prompt patch — covered by tests/test_memory_autouse.py), memory_text (300/593 core parser read; tail = output formatters + env-gated smart-graph call, structure-confirmed).

### M1 — MEDIUM-LOW: SQL-dedup remap leaves a duplicate vector point
`upsertMemory` (`tools/memorymcp/memory_tools.py:306-332`) writes the vector point under the NEW `memory_id` first, then calls `sql_store.upsert_memory`, whose dedup can return a DIFFERENT `actual_id` (existing row, same owner-mixed text hash) — the tool logs "remapping" and returns `actual_id`, but the just-written vector point under the new id is never deleted. Every re-upsert of identical text+type+owner (without memory_id) therefore adds one duplicate vector point while SQL keeps a single row: duplicate search hits, inflated counts, and mergeDuplicates later "merging" self-inflicted duplicates. Fix: run the SQL dedup first (or delete the new point when `actual_id != memory_id`). Verified by reading; trigger is the common "store the same memory twice" path.

### M2 — LOW-MED: graph tools have no owner scoping (E3.5 gap)
`createMemoryEdge` / `getMemoryGraph` / `exportGraphAsMarkdown` (`memory_graph.py`) never call `_memory_caller` — a scoped non-admin who learns a foreign memory UUID (shared by an admin, present in an exported doc) can link, traverse, and read foreign memories; `createMemoryEdge` mutates the foreign payload's `edges`. Same data-plane family as the E3.5 review items; UUID secrecy is the only current barrier. Fix: owner check on edge-create + owner-scoped traversal.

### M3 — LOW batch (memorymcp)
- `reindexMemory` runs `model.encode` per batch directly on the event loop (`memory_tools.py:1280`) — reindexing a large store stalls the server (S5 class; admin-masked surface).
- `setup_fef_v3` assigns `fef_registry`/`fef_manager`/`fef_http_server` without `global` (`memory_tools.py:1592-1597`) — locals, harmless today (nothing reads the module globals) but a latent name-shadow trap.

### Checked and solid (slice 6)
Artifact offload/rehydrate/orphan-cleanup ordering incl. update-replaces-sidecar; delete ORDER (SQL-first, documented 2026-09-13 orphan incident) + tidy-state clear on delete; P3 keyword lane (env-default weight, Jaccard fallback, kw-only candidate retrieve, owner post-filter on kw-lane hits); P4 tidy guards on decay/merge (fingerprint skip, shrink refusal, over-delete floor, artifact sidecar cleanup before delete); owner-scoped query/list/delete with admin bypass + legacy-owner-is-admin convention; entry-point no-eager-FEF lesson encoded.

## Slice 7 — mcp_ui (complete) — COMPLETE

Read: management_ui.py (1368, full), api_client.py (core 260 + method map), state/models/logging_config + all components (tool_settings, users_tab, env_var_editor, memory_tab, memory_explorer_panel, presets/auth/tool_card/tool_list/actions/data_sources/error_display/rag_collections) — full-read the load-bearing ones, structure-scanned the presentational ones (no subprocess/eval/shell hits anywhere).

**No new findings** — F4 (logout no session-file purge) and F10 (duplicated default-password resolution) from phase 1 remain this surface's items. Solid: Fernet-sealed session credentials with rotation semantics; rebuild lock + generation guard (duplicate-Explorer lesson); UI-9 stale-fetch guard on tool selection; per-grant tool filter applied at page build AND in the status poller; safe post-login redirects; mask-tab writes delegate to the central API (F2's race is server-side); union-not-replace discovery; entry-point single-init lesson.

## Slice 8 — pod_ui (complete) — COMPLETE

Read: ops.py (full), state.py (core full), auth.py, confirm.py, app entry, pages_v3/{operations (via diff + earlier), backups, bundles, chrome, logs} — process-execution audit: `ops.py` is the ONLY spawn site (design rule honored), argv-list `create_subprocess_exec`/`subprocess.run` with timeouts, no shell=True anywhere; the wipe-stdin map matches the three startcluster confirmation sites verbatim; lambda hunt clean (phase 1).

**No new findings** beyond F7 (history non-atomic write). Solid: change-gated dashboard refresh (env_signature), honest degradation when podman is broken, bundle/host mode topology selection with host-mode pinning.

## Slice 9 — deploy + monitoring + tests — COMPLETE

Read: startcluster.sh (620/1190 line-read — all verbs through test-restore + the earlier targeted reads at 475/698-712; the tail repeats the same topology-up/generator patterns covered by the P1-P5 tests and fix bands), harvest-config.sh (redact section full, rest via phase-1 verification), bundle_node_env.py / Containerfile / compose files (covered by the 2026-10-18 pass + fix bands + their dedicated test files), monitoring/** (library behind the loopback-only metrics server init reviewed in slice 1; collector/exporters/middleware skimmed — no live callers beyond launchmcp).

**Findings:** none new — F5 (harvest `_URL` redaction gap) and F6 (`git pull --ff-only || true` silent on the TEST path at `:707`; the WORK path at `:475` now warns) stand. Solid: wipe confirmations on all three destructive sites, bundle `-f` chain re-applied on bare re-up (138b471), pw-preserving tmpfile across topology switches, nullglob mount specs, preflight port checks.

### Tests lens (test-quality pass over tests/ — 90 files, 1179 test functions)
Zero vacuous patterns (`assert True` / silent `except: pass`) repo-wide. The material gaps are already recorded: **S2's missing PG query/filter contract cases** (the HIGH finding survived because of this), F11 (PUT-config keyless-403 negative, operations-tooltip pin, gate-bypass-free hot-path), plus no test pins the `upsertMemory` SQL-remap duplicate (M1) or the stale-file substring delete (R2).

---

# Phase 2 coverage ledger (file-by-file, 2026-10-24)

| Area | Files | Coverage |
|---|---|---|
| launcher/** + launchmcp.py + config/ | 29 | **full read** (debug/fef_metrics diagnostic CLIs, errors/config_types/__init__ skimmed as plumbing) |
| tools/shared/** + impls/ | 30 | **full read** |
| tools/{simplemcp,webmcp,convertermcp} + fef_integration | 4 | **full read** |
| tools/databasemcp | 11 | **full read** |
| tools/ragmcp (+indexer) | 13 | **full read** except smart_chunkers/copilot tails (same-pattern per-language chunkers/formatting — structure-confirmed) |
| tools/memorymcp | 8 | **full read** except memory_text tail (output formatters — structure-confirmed); memory_autouse via structure + its test file |
| mcp_ui/** | 20 | management_ui + api_client core **full read**; components full-or-structure (no exec/eval hits) |
| pod_ui/** | 13 | ops/state/auth/confirm/operations **full read**; pages_v3 remainder via lambda-hunt grep + fix-band history |
| deploy/** | 6 | startcluster 52% line-read (all verb classes); harvest redact section full; generator/Containerfile/compose via 2026-10-18 pass + tests |
| monitoring/** | 5 | skimmed (loopback metrics-server library, single caller) |
| tests/** (90 files) | — | **test-quality lens**: counts, vacuous-pattern grep, gap list (above); not line-read individually |

Not reviewed (documented exclusions): plans/**, docs, graphify-out/, examples/, gitignored root wrappers (startlauncher/startui/startpodui/startgrounds — 5-line exec shims per AGENTS.md).

# Final triage (phase 1 + 2 combined)

- **Fix first:** **S2** (PostgresVectorStore queries broken outright on the documented PG combo — proven live) and **F1** (databasemcp E3.5 gate: batch/tx/metadata/use_database/connect_database paths + explicit-name fail-open — multi-mode security).
- **Fix soon:** **S3** (turso FTS stale-delete on text-variant re-upserts — live store), **S5** (migration tools freeze the memorymcp event loop), **R2** (stale-file substring delete — collateral chunk loss), **M1** (upsert dedup leaves duplicate vector points), **F2/S1** (config-doc + users.json RMW races).
- **Batch of LOWs:** L1-L3, S4/S6, T1-T2, R3-R4, M2-M3, F3-F11 as convenience items.
- **Test gaps to close alongside:** PG query contract cases with filters (S2), keyless-403 PUT config, gate-bypass-free generic db tests, M1/R2 pins.

---

# Mono-mode audit (2026-10-27) — "mono is my main usage"

*Triggered by two live mono bugs at the work deployment (fresh checkout, no .env → mono).*
*Scope: every site that branches on identity, role, or auth mode.*

## Verdict

After the `20680b0` role-default fixes, **mono mode is coherent end-to-end**: every
identity resolver treats the mono operator as admin/unrestricted, and nothing in
the E3/E3.5/E1 layers requires a user store. Certification: 94 tests across the
mono-focused set (real-transport auth, session flush, masks, the new mono
fallthrough pins) all green.

## Site-by-site matrix (mono semantics after the fixes)

| Surface | Mono behavior | Status |
|---|---|---|
| `create_fastmcp_server` | DualHeaderVerifier, single system key, no gate | solid (real-transport tests) |
| ragmcp `_caller_rag_access` | authenticated → unrestricted (`None`) | fixed + pinned |
| memorymcp `_memory_caller` | authenticated → admin | fixed + pinned |
| databasemcp `_current_caller_grants` / `_refuse_non_admin` / `_assert_preset_grant` | role=admin → all connects/presets/metadata allowed | fixed + pinned |
| flush gate (`server_factory`) | `role` defaults admin → allowed | was already correct |
| IdentityGateMiddleware | not installed in mono; even if present, role-less entry masks nothing | safe |
| 81xx `_is_admin` | system key = admin; open server = admin | solid |
| central 8200 `_verify_api_key` | system key = "system"; **keyless = reads open, mutations 403 (P2-2)** | by design — see checklist |
| mcp_ui mono login | env pair; default admin/admin refused only on non-loopback binds | solid + pinned |
| mcp_ui sidebar/Users tab | admin sees all; Users tab hidden (multi-only) | solid |
| E1 runtime masks / `apply_function_masks` | mode-independent (file/state doc) | solid |
| E4 tx owner attribution | mono over HTTP stamps owner=<tool> — attribution only | solid |
| users_store seeding | never triggered in pure mono (only multi/mcp_ui-multi paths call it) | fine |
| `MCP_REQUIRE_IDENTITY` | no-op in mono (gate not installed) | documented |
| pod_ui | env-pair auth, mode-agnostic | solid |

## Work-deployment checklist (mono-first)

0. **Tool API keys bootstrap to placeholders**: a fresh checkout has no
   `tools/*/config.json` (gitignored); the launcher bootstraps them from the
   committed examples with `CHANGE-ME-` keys (loud warning at boot). Every
   MCP client call will 401 until you either put real keys in those files or
   export `<TOOL>_API_KEY` / `MCP_API_KEY` env vars.
1. **The POD is multi-user even if your host is mono** — the node image bakes
   `ENV MCP_AUTH_MODE=multi` (deploy/Containerfile:84) and the pod's mounted
   `.env` is loaded with `load_dotenv`, which does NOT override an existing
   env var → the image's `multi` wins. The pod therefore enforces per-user
   grants and needs a store admin. (The host launcher stays mono — that is
   where mono actually applies.)
2. **Set `MCP_MANAGEMENT_API_KEY`** in the work `.env` — without it the central
   API runs open-for-reads but REFUSES ALL MUTATIONS (403, audit P2-2): the
   mcp_ui Function-Masks toggles, env edits, and `PUT /api/tools/{tool}/config`
   will all fail. This is the next mono friction they will hit.
3. **Set `MCP_UI_PASSWORD` to ≥ 8 characters** (+ `MCP_UI_SECRET`) — in the
   pod (multi, UI bound to 0.0.0.0) the default admin/admin is refused twice
   over: the store seeding refuses a default-password admin on a non-loopback
   bind, and the break-glass bootstrap rejects a < 8-character password. The
   result is a pod with NO admin account and a login that fails with a
   misleading message. The UI now names the fix explicitly.
4. **Masks are per-machine**: `tools_config.json` is gitignored, so a fresh
   checkout exposes EVERY function — including the `get_secret` mono test
   fixture and databasemcp's full surface. Mono = admin by design; re-apply the
   standing masks (`get_secret`, `brave_search_web`) via the UI once, or copy
   the home `tools_config.json`.
5. `.env` absent ⇒ host mono + keyless central + ephemeral UI secret (logins
   die on restart). The env vars above close all of it.
