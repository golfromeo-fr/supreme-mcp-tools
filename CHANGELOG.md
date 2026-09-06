# Changelog

All notable changes to the MCP Launcher will be documented in this file.

## [Unreleased]

### 2026-09-06 — databasemcp: oraclemcp renamed + multi-database overhaul (feature/databasemcp)
- **Renamed**: `tools/oraclemcp` -> `tools/databasemcp` — a database workbench for Oracle, Postgres, and libSQL (local file / Turso). Ports kept (8000/8100); `databasemcp` joins `startlauncher`.
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
