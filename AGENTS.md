# AGENTS.md

## Commands

```bash
pip install -r requirements.txt          # install dependencies

python -m launcher --list-tools          # show discovered tools
python -m launcher --tools webmcp,simplemcp   # start specific tools
python -m launcher --management-port 8200     # start with mgmt API
python -m launcher --transport sse            # use SSE transport for all tools
python -m launcher --dry-run webmcp     # preview without starting

python -m pytest                         # run all tests
python -m pytest tests/test_memory_text.py -v  # single test file

python -m mcp_ui                         # NiceGUI management UI (port 8400)
```

There is no linter, formatter, or typecheck configured — no lint/typecheck step needed.

## Architecture

Unified launcher runs multiple MCP tools in one Python process. Each tool is a FastMCP server mounted on its own port.

```
launcher/          # orchestration: discovery, port mgmt, server lifecycle, FEF V3
tools/<name>/      # individual MCP tools (each is a self-contained server)
  <name>_fastmcp.py  # PRIMARY entry point — must export `app` from mcp.streamable_http_app()
  config.json        # tool config — MUST wrap api_key under "auth": {"api_key": "..."}
  support modules    # helpers imported by the primary (never scanned by discovery)
tools/shared/      # cross-tool libraries (pg_store, pii_redactor, relevance_scorer, etc.)
config/            # ports.json (port ranges+assignments), launcher_config.json, monitoring_config.json
tests/             # pytest suite
```

### Port ranges (config/ports.json)

| Range | Purpose |
|-------|---------|
| 8000-8099 | MCP tool endpoints |
| 8100-8199 | Tool management servers |
| 8200-8299 | System (central_management: 8200) |
| 8300-8399 | Metrics |
| 8400-8499 | UI (management_ui: 8400) |

## Key conventions

### Tool discovery (launcher/tool_discovery.py)

- Non-recursive glob on `tools/<name>/` — only top-level `.py` files
- Filename `<name>_fastmcp.py` is canonical; `_fastmcp` suffix is stripped to derive tool name
- Auto-excluded: `*_sse.py`, `*_streamable.py`, `test_*`, `__init__.py`, `migrate_*`
- When both `_fastmcp.py` and bare `.py` exist, `_fastmcp` wins

### Transport switching

All tools support both SSE and streamable-http via FastMCP. Controlled centrally by the launcher:

| Method | Priority | Example |
|--------|----------|---------|
| `--transport` CLI flag | Highest | `python -m launcher --transport sse` |
| `MCP_TRANSPORT` env var | Medium | `MCP_TRANSPORT=sse python -m launcher` |
| `launcher_config.json` `"transport"` | Lowest | `{"transport": "sse"}` |
| Default | — | `streamable-http` |

The launcher sets `MCP_TRANSPORT` env var before importing tool modules. Each `_fastmcp.py` reads it at import time to select `mcp.sse_app()` or `mcp.streamable_http_app()`. Per-tool override is possible by setting `MCP_TRANSPORT` in the tool's `.env` — this takes precedence over the launcher's setting since the tool reads the env var directly.

### config.json auth structure

The `api_key` MUST be nested under `"auth"` — `load_auth_config()` reads `config.get("auth", {}).get("api_key")`:

```json
{
  "auth": {
    "api_key": "tool-test-key-xxxx"
  }
}
```

A bare top-level `"api_key"` is invisible to the auth system.

### FastMCP tool registration

Tools register via `@mcp.tool()` decorators at import time. Submodules register their tools as a side effect of being imported. The entry point (`<name>_fastmcp.py`) imports submodules, then calls `mcp.streamable_http_app()`.

### Side-effect imports

`memory_core.py` and `pg_store.py` initialize external connections (Qdrant, PostgreSQL) at import time. Tests that don't need these should import from dependency-free modules like `text_utils.py` instead.

### memorymcp module structure

The monolith was split into focused modules — all share one `FastMCP` instance from `memory_core.py`:
- `memory_core.py` — config, clients, `FastMCP` instance, utility functions
- `memory_tools.py` — CRUD tools (upsert, query, delete, etc.)
- `memory_graph.py` — graph tools (edges, export, visualization)
- `memory_text.py` — text processing (textToGraph, textToSmartGraph)
- `text_utils.py` — pure functions with no FastMCP/Qdrant deps (safe for test imports)

## Testing

- `python -m pytest` runs everything (~222 tests across old/ launcher/ tests/)
- `python -m pytest tests/` runs only the project test suite (not launcher internals)
- `tests/test_memory_text.py` covers text_utils and textToGraph integration (56 tests)
- `tests/test_fastmcp_critical_fixes.py` validates tool safety invariants
- Tests for shared modules: `test_pg_store.py`, `test_pii_redactor.py`, `test_relevance_scorer.py`
- Tests in `old/` are legacy NiceGUI tests — safe to ignore

## Environment

- `.env` at project root is loaded at startup by all tools (via `dotenv`)
- Auth defaults: `MCP_UI_USERNAME`/`MCP_UI_PASSWORD` (default admin/admin)
- `memorymcp` requires: `QDRANT_HOST`, `QDRANT_PORT` (defaults: qdrant:6333)
- `memorymcp` optional: PostgreSQL via `DATABASE_URL` for metadata/dedup
- `ragmcp` embedding config in `tools/ragmcp/config.json` under `embedding`
- Port conflicts detected via socket binding, not just config

## Common pitfalls

- Don't use `localhost` in server URLs — use `127.0.0.1` to avoid IPv6 `::1` resolution
- `pg_store.py:search_text` passes `query` param twice (WHERE similarity + SELECT similarity) — intentional, don't "fix"
- `tools/<name>/config.json` is machine-written by the management server API — preserve its JSON structure
- `ScoringWeights` constructor: `recency_half_life_days=0` and `max_usage=0` are valid (uses `is not None` checks, not truthiness)
