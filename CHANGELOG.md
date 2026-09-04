# Changelog

All notable changes to the MCP Launcher will be documented in this file.

## [Unreleased]

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
