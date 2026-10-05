# Changelog

All notable changes to the MCP Launcher will be documented in this file.

## [Unreleased]

### 2026-10-05 — feat(grounds): probe suite deepened (ragmcp indexer, databasemcp, mask push, automated switch) + code-drift detection
- **Three new pod probes** (15 total per ground): (12) the ragmcp INDEXER SUBPROCESS actually starts and produces its collection — it previously died silently at import (`tiktoken` missing from the pod image; the bare import also defeated the indexer's own documented fallback — now guarded AND declared in requirements.txt); (13) databasemcp executes real SQL against the ground's own state db (connect_database + query + disconnect; a raw DSN in `query`'s connection= is a NAME not a DSN — live-verified); (14) central mask-push propagation: unauthenticated writes rejected, a pushed mask reaches the RUNNING node's tools/list, revert restores it (the E1 control plane, live-proven on a pod).
- **Automated switch round trip** (tests/test_pod_switch.py): pg → turso → pg on work-layout grounds, password + users preserved — replaces the retired manual-rehearsal idea; runs inside `./startgrounds probe` for work-layout grounds (2m39s live).
- **`./startgrounds probe --all`**: sequential ensure+probe across every spec; all three grounds green (39s / 68s / 3m15s).
- **Code-drift detection**: create stamps the repo HEAD in the registry; ensure treats a moved HEAD as drift and recreates — a ground testing a stale image would lie about the current tree (found live: two grounds were running pre-fix images).
- **Fixed**: TursoVectorStore missing-collection message aligned to the ecosystem string contract ("doesn't exist") — ragmcp's auto-create path string-matches it, and my previous wording ("does not exist") turned index_code into a hard error.
- **Tests**: 1076 passed / 0 failed; probe --all green across g1-pg, g2-pg, gwork-pg.

### 2026-10-05 — fix(grounds): two-node custom clusters could not coexist
- **Fixed**: `gen_custom_compose.py` published node2's tool ports identical to node1's — with `--nodes 2` the two nodes could never run simultaneously (never hit: nodes=2 had zero usage). node2 now offsets its tool publishes by +1000, mirroring central2 = central + 1000; the registry records the node2 map as `ports.tools2`; the create port-reservation window covers base+1002..base+1007 for two-node clusters. Proposed g2 ground: probe 11 pins the multihost invariant (the same admin key authenticates on BOTH nodes via the shared state plane).
- **Tests**: 1076 passed. Probe 11 added (gated on two-node grounds).

### 2026-10-05 — feat(grounds): three-tier testing + recreatable real-pod grounds + BASIC topology switch
- **Three-tier test shape**: tier 1 = the hermetic suite (unchanged); tier 2 = NEW real-pod smoke probes run against disposable testing grounds (`./startgrounds probe <name>`, opt-in via `POD_GROUNDS=<name>`, 11 probes, ~30s); tier 3 = manual live sweeps. The tier rule: a test belongs on a real pod iff its claim mentions a container-boundary fact. Motivated by a real catch: TEST-node composes had no HF-cache mount, so local BGE-M3 embeddings could never load there (fixed in Phase 0).
- **Grounds**: `./startgrounds` (ensure/recreate/switch/status/probe/stop/clean) manages disposable pods created via the existing `startcluster create` registry from pinned bundles. `ensure` recreates on drift; `probe` never does (evidence preservation). v1 grounds are port-shifted (coexist with everything); `--layout work` grounds use the canonical ports + host networking with the three-way mutual-exclusion preflight. Ground data planes are ephemeral (container-local turso file) — zero host-data mounts.
- **BASIC topology switch**: `startcluster work pg|turso` changes the identity-plane engine (pgvector ↔ sqld on 127.0.0.1:9080) with VOLUMES KEPT — the dormant engine's data and the exact POSTGRES_PASSWORD survive the round trip (edge 2, live-proven). Topology stamp `[topology] <t>` in work.env; `work up` boots the stamped engine; backup is topology-aware (pg_dump vs volume tar). Pod UI: topology selector on the work card. Retires the "photocopy of the host" doc fiction — the work pod's identity plane was always DB-backed while the host runs JSON files.
- **Fixed (latent, found by pinning the grounds path)**: create tool ports off by two (databasemcp collided with central); registry port-conflict grep never matched pretty-printed JSON; `create -b` silently ignored the bundle; TursoVectorStore.get_collection swallowed missing-table (memory_core never created its collection on fresh files); libsql-server healthcheck used curl (absent in the image — could never pass); podman-compose won't recreate a definition-changed service over its old container.
- **Tests**: 1071 passed, 0 failed (was 1023 before this batch); 11 tier-2 probes green live on g1-pg and gwork-pg in BOTH topologies. Design: plans/testing-grounds-and-work-topology-2026-10-05.md (DRAFT 3 FINAL + MiniMax counter-review).

### 2026-09-29 — fix(ragmcp): dense search works on collections that record their model
- **Fixed**: two independent defects, both found by indexing this repo and retesting against a live server. (1) `TursoVectorStore.query_dense()` aborted the whole query on a row with no vector — the indexer's `__collection_metadata__` point is stored with `vector=[]`, and `vector_distance_cos` raises `vectors must have the same length: 0 != 1024`, so any collection carrying that point returned zero results. Such rows are unmatchable anyway and are now excluded from the scan and the payload-filter path (Qdrant never showed this — it keeps empty-vector points out of the default vector space). (2) Model-name vocabulary mismatch: the indexer stores a full HuggingFace id (`BAAI/bge-base-en-v1.5`) while `LOCAL_EMBEDDING_MODELS` is keyed by short alias (`base`, `bge-m3`), so lookups missed and fell back to bge-m3 (1024d), rejecting a 768d index as a dimension mismatch. `resolve_local_model_alias()` now accepts either form and the model cache is keyed on the resolved alias.
- **Tests**: 1027 passed. Two new `TursoVectorStore` contract tests cover the empty-vector skip (both fail without the fix).
- **Note**: takes effect for the `mcp-work` pod only after its image is rebuilt — compose bakes code into `mcp-node:latest` and bind-mounts only `node.env`.

### 2026-09-29 — feat(pod-ui): Phase B — cluster registry CLI (create/rm/generic verbs)
- **Added**: `create <name> [--nodes 1|2] [--topology pg|turso] [--tools ...] [--port-base N] [-b bundle]` with fail-fast validation (name regex, built-in/registry collision, unknown tools, port freedom incl. registry overlap, bundle validity); writes `deploy/clusters.d/<name>.json` + `deploy/<name>.env`. `create` does NOT boot — cheap and reversible.
- **Added**: generic verbs for custom clusters — `up` (compose regenerated via `deploy/gen_custom_compose.py` + port re-check + health wait), `start/stop/status/logs/backup/restore/scrape/export-data/clean`. The `work`/`test` specialized paths are untouched; customs are bridged, own their ports, and are structurally unable to collide with the daily driver.
- **Added**: `rm <name>` — refuses while running, typed-`<name>` confirm, removes containers/volumes/registry/env/compose.
- CLI only; the UI half lands later on this branch. Verified live end to end. Suite 1023.

### 2026-09-29 — feat(pod-ui): v3.9 workflow restructure + CLI gaps (Phase A)
- **Changed**: lifecycle restructure (Operations → Lifecycle) with per-cluster Rebuild (topology-aware for TEST) / Start / Stop / Clean (typed wipe); duplicated backup/scrape/export buttons removed (they live on Backups & data and Bundles); guided migration walkthrough moved to Bundles.
- **Changed**: TEST topology selection in the Boot (absent) and Rebuild paths only — never beside Start all. Bundle mode offers all four topologies and boot re-passes the bundle path so `node.env` regenerates for the new topology (avoids the stale-env broken-cluster class); host mode pins the select to the `node.env`-matching topology with the reason in the tooltip. The turso `STREAM_EXPIRED` caveat is carried inline.
- **Added**: "Start MCP UI" on the BASIC card (`work ui` verb) noting the `:8400` conflict with host `startui`; harvest dry-run Preview button (writes nothing, renders the `--list` copy table, `rc=2` as "cannot run"); Container Inspect in the node drawer (curated image/digest, created/started, restart count, health-probe log, mounts, ports + collapsible raw `podman inspect` JSON, with KEY/SECRET/TOKEN/PASSWORD/DSN env values masked on screen).
- **Added**: preflight signals — a work-card badge when the host stack holds the canonical ports (boot will fail), and a test-card badge.

### 2026-09-29 — fix(ragmcp): repair the code-indexing pipeline
- **Fixed**: five bugs found by indexing this repo; every one was hit live while building the `supreme-mcp-code` collection. (1) the spawned indexer crashed on import (`ModuleNotFoundError: shared` — it put `tools/ragmcp` on `sys.path` but imports `shared.*`, which lives in `tools/shared`), and the parent reported "started successfully" regardless of the child's exit; `tools/` is now added explicitly. (2) collections were never created on Turso: `get_collection()` never raises for a missing collection (returns an empty `CollectionInfo`), so the indexer's Qdrant-style "Not found" check took the exists branch and every write failed with `no such table: vec_<name>`; existence is now detected via `list_collections()`, which also self-heals stale map rows. (3) worker stampede — `MAX_CONCURRENT` was `min(20, files)` with each worker loading its own embedding model, 25 GB RSS on a 22 GB host, swap-thrash, ~4 chunks/min; now `RAGMCP_INDEX_CONCURRENCY` (default 2). (4) dense search embedded queries with the server-global model, rejecting a 768d index as a dimension mismatch; `_do_dense_search` now prefers the model recorded in the collection's `__collection_metadata__`. (5) `get_collection()` sampled the first non-null embedding for the collection dimension, and the `__collection_metadata__` point (written with an empty vector) sorts first, so dimension came back unknown and even dimension-matching queries were rejected; `''` and `'[]'` are now skipped when sampling.

### 2026-09-29 — fix(deploy): bare re-up of a bundle-managed env keeps its compose override
- **Fixed**: a bare `./startcluster work` (or `test`) after a `-b` boot regenerated the compose chain **without** `compose-*.bundle.yml`, so the node came up on the image's baked sanitized tool configs and every authenticated call 401'd (system keys were `image-sanitized`). The restore path already preserved the override; the up paths now do too (work + test). Found live during the 2026-09-29 canonical-ports bundle test.

### 2026-09-29 — fix(deps): google_search_api needs google-search-results, not the serpapi PyPI pkg
- **Fixed**: the host venv's `serpapi` module comes from `google-search-results` 2.4.2, but the image never had it, so `google_search_api` died with `No module named serpapi`. The first attempt installed PyPI `serpapi` 1.1.2, which imports but dropped `GoogleSearch`; the real provider is pinned instead.

### 2026-09-29 — docs(deploy): drop duplicated half of automation.md
- **Fixed**: a bad fold in `a42492b` left `deploy/automation.md` with a duplicated half.

### 2026-09-08 — E3: multi-user identity layer (feature/e3-multiuser)
- **Added**: `MCP_AUTH_MODE=mono|multi` (default `mono` = exactly today). In `multi`, each tool's verifier is backed by the **user store** (`~/.config/supreme-mcp-tools/users.json`): `{system key → admin username (current mono user)} ∪ {per-user mcp_keys → usernames}`, gated by `enabled` + per-user `servers` reachability; `IdentityGateMiddleware` filters `tools/list` and rejects `tools/call` with "Unknown tool" for per-user `masked_functions`. Store edits hot-apply (mtime cache) — no restarts. Kill-switch: keep `mono` (or `MCP_USERS_STORE_DISABLE=1`).
- **Added**: central `/api/users` CRUD (create with once-only `mcp_key`, rotate, set password/servers/masked-functions, enable/disable, delete with last-admin guard) — list responses never contain secrets.
- **Added**: mcp_ui multi-user login (username/password via pbkdf2 store; legacy ENV admin pair bootstraps the store admin once, key shown once); admin-only **Users tab** (server checkboxes + per-server function-mask grid, rotate key, enable/disable, delete); Memory Explorer calls memorymcp **as the acting user**.
- **Changed**: `mcp.access` lines now append `user=<client_id>` (positional, `-` when unknown); central 8200 requires `MCP_MANAGEMENT_API_KEY` (P0 — it was open before); 81xx action routes + `/admin/flush-sessions` require the admin role (system key or admin user key).
- **Tests**: `test_management_auth.py` (4), `test_identity_middleware.py` (12), `test_users_store.py` (19), `test_e3_integration.py` (7) — 838 total.

### 2026-09-07 — E2: Memory Explorer (feature/e2-memory-explorer)
- **Added**: memorymcp `listMemories(limit, offset, tag, sort="recent")` — paged browse over the vector store's scroll contract (bounded scan, newest-first paging, tag filter; no embedding, no usage-count side effects). 15 → **16** memorymcp tools. `tests/test_memory_list.py` (7, contract-faithful fake).
- **Added**: mcp_ui **Memory tab** (memorymcp only) — browse cards (preview, type/tags/sensitivity badges, usage, date) with Prev/Next + tag filter, semantic search via `queryMemory`, detail view (`getMemory` + `auditTrail`), delete with confirm (`deleteMemory`). The UI consumes memorymcp **through its MCP surface only** (`mcp_ui/memory_client.py`, fastmcp Client + tool-config key) — no backend credentials in the UI, and every Explorer action is something an MCP client could do.

### 2026-09-07 — fix: pinned mgmt port no longer kills a tool's run (databasemcp first-run failure)
- **Fixed**: the recurring "first `./startlauncher` run starts without databasemcp, second run is fine" — databasemcp is the only tool with a **pinned mgmt port** (8110); every other tool self-heals a stale holder by auto-allocating the next free mgmt port. A pinned mgmt port whose holder outlives the 20s busy-retry window now **degrades to corridor auto-allocation** with a loud WARNING instead of failing the tool; consumers follow the service registry, so nothing breaks. Pinned **MCP** ports still raise (clients hardcode them). `tests/test_port_fallback.py` (3) reproduces the failure with a real socket holder on a synthetic range.

### 2026-09-07 — E1: runtime Function-Mask toggling (feature/e1-runtime-masks)
- **Added**: mask changes now apply to the RUNNING tool server — no restart. The central disable/enable endpoints (and `PUT /api/disabled-tools/{server}`, diff-pushed) persist **file-first**, then push to the tool's FastMCP instance; the per-tool 81xx mgmt server gains `POST /admin/function-masks` (`{"tool", "masked"}`, same API-key auth). Wiring: `server_manager` hands each tool's registry its `mcp_instance` (`_wire_mcp_instance`); responses carry `runtime_applied`/`runtime_note` so the mcp_ui Functions tab can say honestly whether the push happened (unreachable/pre-E1 server = file still wins at next start). All transports covered at once (disable is instance-level).
- **Tests**: `tests/test_e1_runtime_masks.py` (9): runtime helper disable/enable round trip, masked-call rejection, 81xx endpoint persist+apply over TestClient with auth check, central push, wiring, and a boot-sync regression (`validate_and_cleanup_config` prunes only stale mask names, never live ones).

### 2026-09-07 — databasemcp transaction management (E4, feature/db-transactions)
- **Added**: interactive transactions — `begin_transaction` (returns `tx_id`; one per connection; preset bypass applies), `commit_transaction`/`rollback_transaction`; `query`/`execute_sql` accept `tx_id` to run inside the open transaction with **no implicit commit**. Each transaction pins a **dedicated** connection (libSQL: second connection to the same DB; Postgres: standalone connection outside the pool; Oracle: held session from the pool) — ordinary traffic never blocks (15 → **18** tools).
- **Added**: atomic batches — `execute_sql statements=[...]` (max 50, mutually exclusive with `sql`) runs all statements in one all-or-nothing transaction; failures report the failing index and the undone rowcounts.
- **Added**: idle-transaction reaper — sweeps every 60s, rolls back transactions idle longer than `DB_TX_IDLE_TIMEOUT` (default 300s; `0` disables with a warning); starts lazily on the first `begin_transaction`. `disconnect_database` refuses while a transaction is open; `reset_connections` force-rolls-back; Oracle DDL inside a transaction warns about the implicit commit.
- **Tests**: `tests/test_db_transactions.py` (30: scripted-fake registry lifecycle, real libsql + real PG round trips incl. cross-connection isolation, batch, guards, reaper) + 4 gated live tests in `test_databasemcp_live.py`.
- **Plan**: `plans/databasemcp-transactions-2026-09-07.md` (P0 probes run before coding; deviations logged in its Deviations section).

### 2026-09-07 — databasemcp presets in mcp_ui (panel + mgmt actions)
- **Added**: mcp_ui databasemcp Overview tab now renders a **DB Connection Presets** panel — numbered `.env` presets with dialect badge, `_DESC` label, masked URL, live connected state, and per-preset **Connect / Disconnect** buttons (`mcp_ui/components/presets_panel.py`; data fetched fresh from the `connection_presets` data source at render time).
- **Added**: two databasemcp mgmt action extensions backing the buttons — `connect_preset` (by number/NAME alias, idempotent when already connected, secret-masking on failure) and `disconnect_connection`; `tests/test_db_ui_actions.py`.
- **Fixed**: mcp_ui `query_extension`/`execute_extension` sent params BARE while the central routes expect `{"params": ...}` (`QueryRequest`/`ExecuteRequest`) — every parameterized extension call silently reached handlers as empty params (found 2026-09-07: connect_preset answered "Provide 'preset'"); params now ride inside the key, and the panel unwraps the proxy's double envelope (`data`/`result`).
- **Doc**: transaction-management plan (`plans/databasemcp-transactions-2026-09-07.md`, E4) + native transaction-testing procedures in `.agents/skills/mcp-live-tool-test/SKILL.md` (libsql cross-call BEGIN/ROLLBACK verified; PG BEGIN-discarded verified).

### 2026-09-06 — databasemcp: oraclemcp renamed + multi-database overhaul (feature/databasemcp)
- **Renamed**: `tools/oraclemcp` -> `tools/databasemcp` — a database workbench for Oracle, Postgres, and libSQL (local file / Turso). Ports kept (MCP 8000; mgmt pinned 8110 via `databasemcp_mgmt` — 8100 collides with the auto-allocation floor); `databasemcp` joins `startlauncher`.
- **Added**: connection registry with named connections per dialect (`connect_database`, `disconnect_database`, `list_connections`, `use_database`), per-entry locks (parallel queries on different connections), active-connection switching with graceful fallback, and a lazy env-default Oracle connection (`USERID`/`DB_HOST` env as before) deactivatable with `DB_AUTOCONNECT=0`.
- **Added**: dialect layer — Oracle (oracledb thin + real `create_pool`, call timeouts), Postgres (psycopg 3 + psycopg_pool, statement timeout), libSQL (local `file:`/Turso; zero-infra test backend). Oracle-only tools guarded (`get_valid_languages`).
- **Changed**: `query` enforces read-only + real row limiting via fetchmany (old `max_rows` truncated after a full fetch); `execute_sql` invalidates the schema cache; `get_schemas` finally works (the old cache-gate bug made it unable to ever succeed) and includes foreign keys; `list_user_tables_with_descriptions` renamed `list_tables`; `optimize_sql_with_ai` uses `AI_BASE_URL`/`AI_MODEL` env, runs off the event loop, and gets live schema context; rules files (`optimization.json`, `proc_rules.md`) are user-local (`~/.config/supreme-mcp-tools/databasemcp/`, templates in `examples/`).
- **Fixed**: missing `global connection` in the schema-error path; connections/cursors never closed (registry owns close); no timeouts anywhere; dead `lifespan`/`table_columns_cache`/fake pool config removed; secrets never logged (masked params).
- **Added**: connection presets in `.env` — numbered `DB_PRESET_<NN>=oracle://user:pass@host:port/service` (or `postgres://`, `file:`/`libsql://`), optional `_DESC` label and `_NAME` alias, `DB_PRESET_AUTOCONNECT` for tolerant eager connect at tool startup; `list_presets`/`connect_preset` tools; **bypass**: any tool's `connection` parameter accepts an unconnected preset number/alias and connects it lazily; FEF `connection_presets` data source.
- **Removed**: `tests/test_oracle_thread_safety.py` (superseded by registry/concurrency suites).

### 2026-09-05 — launcher shutdown hardening (C7) + Function Mask enforcement (D4)
- **Added**: server-side Function Mask enforcement — each tool server disables its masked tools (`disabled_tools` in `~/.config/supreme-mcp-tools/tools_config.json`) at startup via `tools/shared/function_masks.py`, so MCP clients can no longer see or call them (`tools/list` hides them, calls return "Unknown tool"). Previously the masks were UI-cosmetic only — zero serving-path readers. Effective at next launcher start; `tests/test_function_masks.py` proves list+call+transport behavior
- **Added**: `tools/shared/atomic_io.py` — one atomic JSON/text write helper (flock on a stable `.lock` sidecar + tmp + `os.replace`); converted the three unsafe truncate-writers (`tools_config.py`, `distributed_registry.py`, `management_server.py` auth) and the mcp_ui writer now delegates to it (D2). Lock-sidecar design note: flock on the replaced file itself is inode-unsafe (reproduced live)
- **Removed**: the never-run FEF V3 persistence layer (D1) — `SQLitePersistence`, `FileConfigPersistence`, `ConfigManager`, `EventStore`, `DeadLetterQueue`, `AuditLogger`, `HAConfig`/`DistributedConfig`, the `fef_v3.json` writer (migration phase 3 now a documented no-op) and the unread `fefV3` block (incl. CORS `"*"`) in `launcher_config.json`; re-export hubs slimmed
- **Fixed**: restart race (third variant) — launcher teardown is now deadline-bounded (`timeout_graceful_shutdown=5` per uvicorn server, per-server concurrent `wait_for` in `stop_all_servers`, which also stops the per-tool management servers it previously leaked) with a 15s forcible-exit watchdog in `launchmcp.py` and a 5s post-run exit guard, so a dying launcher can no longer hold tool ports past the next launcher's 20s retry window; `tests/test_launcher_child_shutdown.py`
- **Docs**: persistence-unification design pass (`plans/launcher-persistence-design-2026-09-05.md`) — full store map, dead-layer inventory (D1), unsafe-writer list (D2), retention/secrets findings (D3)

### 2026-09-04 — codebase improvement batch (plans/codebase-improvement-plan-2026-09-04.md)
- **Added**: memory texts >8KB offloaded to ArtifactStore with transparent rehydration on all read paths and blob cleanup on delete/decay/merge (C2) · shared-cache test suite (15 tests) · memory pipeline e2e test incl. the redact→store leg (C3) · deterministic live-server tests (skip with reason when the launcher is down, C5)
- **Changed**: one cache implementation — `TTLCache` core absorbs `CacheManager`'s LRU and webmcp's `SimpleCache` (C1) · `mergeDuplicates` summary reports cosine/jaccard breakdown (C10) · management UI login no longer uses `password_toggle_button` (NiceGUI 3.16.0 event-wiring bug, bisected; C6) · `oauth_fix.py` marked superseded by the FastMCP 4 auth redesign (C9)
- **Fixed**: upsertMemory crashed on every medium/high-sensitivity text (tuple-unpack of `redact_sensitive_text` — caught by the new e2e test) · management pre-flight read a nonexistent ports.json key (`reserved.central_management` now authoritative, C4) · review-found search_text trgm param-order bug (wrong results with filters, `6ec75fe`)
- **Removed**: dead `tools/shared/pg_store.py` (superseded by `impls/postgres_sql.py`), local-only `old/` tree and empty `launcher/streamable_http/` husk (C8)

### Added
- **Monitoring System**: Built-in metrics collection with Prometheus exporter
  - Request/response metrics (count, duration, status codes)
  - Tool execution metrics (success/error rates, duration)
  - Server health monitoring
  - New `monitoring/` package with collector, exporters, middleware, and config
  - New `config/monitoring_config.json` and `config/monitoring_config.example.json`
  - Metrics endpoints: `/metrics`, `/metrics/health`, `/metrics/stats`
- **memorymcp auto-use policy**: reflex-use policy files plus `getMemoryAutousePolicy()` / `getMemoryCheatsheet()` tools; two-level analysis and priority classification (8eb00e6)
- **Backend abstraction (Phases 0–5)**: `SqlStore` / `VectorStore` ABCs with interchangeable Postgres/Turso/Qdrant impls (incl. pgvector and libSQL vector); `python -m tools.shared.migrate_store` CLI; architecture in `tools/memorymcp/BACKENDS.md` (245d494, 12b9db8, 06d205c)
- **Session management**: `POST /admin/flush-sessions` and `MCP_SESSION_IDLE_TIMEOUT` idle-TTL eviction on every tool (06d205c)
- **Per-request access logging**: `mcp.access` middleware records client dialect, tool, duration, and failure markers (68abeed)
- **mcp_ui overhaul**: NiceGUI 3.16 upgrade, drawer+tabs UX, Function Masks dialog and Functions tab, audit repairs, header actions into the drawer (ac3c189..a22b6d0)
- **Versatile multi-transport default**: every tool serves `/mcp` (stateful) + `/mcp-stateless` + `/sse`+`/messages` from one FastMCP instance — clients select by URL (adade68; launcher default-unpin 7801773; regression test c628db6)
- **FastMCP 4**: ported to fastmcp 4.0.0b3 and rolled out to main 2026-08-24 (old main preserved as branch+tag `fastmcp-3-final`); stable `fastmcp==4.0.0` 2026-09-02 (d7f1051)

### Changed
- Updated all tool implementations to support the unified launcher
- Updated `launchmcp.py` with improved tool discovery and error handling
- Per-tool single-transport selection replaced by multi-transport default; `--transport` remains as escape hatch
- README refreshed: primary entry point, working tool table, multi-transport endpoints, port map (3ff94c9)
- BUG_REPORT Fix-Progress tracker re-verified against 2026-09-02 code — 0 unfixed remain (18c100b)

### Deprecated
- Superseded plan/architecture docs carry dated deprecation banners; AGENTS.md is the canonical guide (7d5d9f2)

### Removed
- `tools/convertermcp/convertermcp.py` legacy server file in the Phase -1 transport purge (a411947)

### Fixed
- Port allocation conflict detection
- Tool discovery path resolution
- Busy manual ports retried instead of failing; parallel port-busy wait + fail-fast when another launcher is live (f9e33da, 7f4f8f3)
- webmcp: dropped manual Accept-Encoding — no more raw compressed bytes on brotli-less hosts (eec887d)
- ragmcp: sparse retrieval revived with loud guards, OR-joined FTS5 terms, repair-sparse CLI (28deb77)
- memorymcp: removed eager `setup_fef_v3()` call that starved launcher registration (9150261)
- CacheManager FIFO → true LRU (MED-16) (17089aa)
- `ArtifactStoreError` distinguishes storage failures from not-found (LOW-8) (f7f983b)
- Turso vector store: hardened float coercion + format-agnostic parser for legacy spaced rows (e3bb2e5)

---

## [1.0.0] - 2026-02-24

### Added
- Initial release of MCP Launcher
- Unified launcher for multiple MCP tools in single process
- Support for simplemcp, webmcp, oraclemcp, convertermcp, ragmcp
- Memory-efficient asyncio-based architecture (~50-65% savings)
- CLI with --list-tools, --verbose, --dry-run options
- Configuration via config.json with manual/auto port allocation
