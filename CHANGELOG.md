# Changelog

All notable changes to the MCP Launcher will be documented in this file.

## [Unreleased]

### Added
- **Monitoring System**: Built-in metrics collection with Prometheus exporter
  - Request/response metrics (count, duration, status codes)
  - Tool execution metrics (success/error rates, duration)
  - Server health monitoring
  - New `monitoring/` package with collector, exporters, middleware, and config
  - New `config/monitoring_config.json` and `config/monitoring_config.example.json`
  - Metrics endpoints: `/metrics`, `/metrics/health`, `/metrics/stats`
- **Streamable HTTP Transport**: All tools now support both SSE and Streamable HTTP transports
  - `*_streamable.py` files added for each tool
  - JSON-RPC framing support

### Changed
- Updated all tool implementations to support the unified launcher
- Updated `launchmcp.py` with improved tool discovery and error handling

### Fixed
- Port allocation conflict detection
- Tool discovery path resolution

---

## [1.0.0] - 2026-02-24

### Added
- Initial release of MCP Launcher
- Unified launcher for multiple MCP tools in single process
- Support for simplemcp8, webmcp, oraclemcp, convertermcp, ragmcp
- Memory-efficient asyncio-based architecture (~50-65% savings)
- CLI with --list-tools, --verbose, --dry-run options
- Configuration via config.json with manual/auto port allocation
