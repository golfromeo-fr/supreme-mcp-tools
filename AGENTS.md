# AGENTS.md

## Commands

```bash
pip install -r requirements.txt

# Primary entry point (new style)
python -m launcher --tools webmcp,simplemcp   # start specific tools
python -m launcher --management-port 8200    # start with mgmt API
python -m launcher --transport sse           # override transport
python -m launcher --no-management           # skip central mgmt server
python -m launcher --debug                   # DEBUG logging

# Legacy entry point — has extra flags (--list-tools, --dry-run, --verbose,
# --no-sync-tools, --health-check-logs, --health-check-interval) that the
# `python -m launcher` CLI doesn't expose. Use it when you need a dry-run
# or a tool listing.
python launchmcp.py --list-tools             # show discovered tools
python launchmcp.py --dry-run webmcp         # preview without starting

# UI
python -m mcp_ui                             # NiceGUI management UI (port 8400)

# Tests
python -m pytest                             # everything (~458 tests)
python -m pytest tests/                      # project test suite only (~415)
python -m pytest tests/test_memory_text.py -v
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
tools/shared/      # cross-tool libraries (pg_store, pii_redactor, relevance_scorer,
                   #   server_factory, artifact_store, cache, html_utils, memory_models, oauth_fix, utils)
config/            # ports.json (port ranges+assignments), launcher_config.json, monitoring_config.json
plans/FLEXIBLE_EXTENSIBILITY_FRAMEWORK_V3/   # FEF V3 design doc
tests/             # pytest suite (project tests)
launcher/streamable_http/tests/  # streamable transport tests
launcher/test_simplemcp_client.py
tests/fef_v3/      # FEF V3 test fixtures
```

### Port ranges (config/ports.json)

| Range | Purpose |
|-------|---------|
| 8000-8099 | MCP tool endpoints (mcp) |
| 8100-8199 | Tool management servers (mgmt) |
| 8200-8299 | System (central_management: 8200) |
| 8300-8399 | Metrics (metrics_server: 8300) |
| 8400-8499 | UI (management_ui: 8400) |

Port conflicts are detected via socket binding, not just config — a free config slot is not enough.

## Key conventions

### Tool discovery (launcher/tool_discovery.py)

- Non-recursive glob on `tools/<name>/` — only top-level `.py` files
- Filename `<name>_fastmcp.py` is canonical; `_fastmcp` suffix is stripped to derive tool name
- Auto-excluded: `*_sse.py`, `*_streamable.py`, `migrate_*`, `copilot_context_injector`, `test_*`, `__init__.py`
- When both `_fastmcp.py` and bare `.py` exist, `_fastmcp` wins
- The launcher auto-detects container dirs (no `*_fastmcp.py`) vs tool dirs and descends one level
- See `tools/AGENTS.md` for the full tool-directory convention

### Transport switching

All tools support both SSE and streamable-http via FastMCP. Controlled centrally by the launcher:

| Method | Priority | Example |
|--------|----------|---------|
| `--transport` CLI flag | Highest | `python -m launcher --transport sse` |
| `MCP_TRANSPORT` env var | Medium | `MCP_TRANSPORT=sse python -m launcher` |
| `launcher_config.json` `"transport"` | Lowest | `{"transport": "sse"}` |
| Default | — | `streamable-http` |

The launcher sets `MCP_TRANSPORT` env var before importing tool modules. Each `_fastmcp.py` reads it at import time to select `mcp.sse_app()` or `mcp.streamable_http_app()`. Per-tool override: set `MCP_TRANSPORT` in the tool's `.env` — this takes precedence over the launcher's setting since the tool reads the env var directly.

### config.json auth structure

The `api_key` MUST be nested under `"auth"` — `load_auth_config()` reads `config.get("auth", {}).get("api_key")`:

```json
{
  "auth": {
    "api_key": "tool-test-key-xxxx"
  }
}
```

A bare top-level `"api_key"` is invisible to the auth system. `tools/<name>/config.json` is machine-written by the management server API — preserve its JSON structure when editing.

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
- `memory_autouse.py` — registers `getMemoryAutousePolicy()` and `getMemoryCheatsheet()` tools, and patches `getMemorySystemPrompt()` to prepend an auto-use policy pointer

### memorymcp auto-use policy (reflex prompt)

memorymcp ships repo-local policy files that push the LLM to call memory tools by habit:

| File | Content |
|---|---|
| `tools/memorymcp/auto_use_policy.md` | Full reflex-use policy |
| `tools/memorymcp/auto_use_cheatsheet.md` | Short summary for tight context budgets |

The `getMemoryAutousePolicy()` and `getMemoryCheatsheet()` MCP tools read these files. The always-injected `getMemorySystemPrompt()` tool is patched at import time to prepend a pointer so the LLM discovers these tools on first session. If the files are unreadable, hard-coded inline fallbacks are returned.

Harness/client skills (e.g. `~/.agents/skills/memorymcp-autouse/`) are managed by the user and the harness, not by this MCP server. They can reference the MCP tools but the server does not depend on them.

To test: `python -m pytest tests/test_memory_autouse.py -v`

## Environment

- `.env` at project root is loaded at startup by all tools (via `dotenv`)
- Auth defaults: `MCP_UI_USERNAME`/`MCP_UI_PASSWORD` (default admin/admin)
- `memorymcp` requires: `QDRANT_HOST`, `QDRANT_PORT` (defaults: qdrant:6333)
- `memorymcp` optional: PostgreSQL via `DATABASE_URL` for metadata/dedup
- `ragmcp` embedding config in `tools/ragmcp/config.json` under `embedding`

## Common pitfalls

- Don't use `localhost` in server URLs — use `127.0.0.1` to avoid IPv6 `::1` resolution
- `pg_store.py:search_text` passes `query` param twice (WHERE similarity + SELECT similarity) — intentional, don't "fix"
- `ScoringWeights` constructor: `recency_half_life_days=0` and `max_usage=0` are valid (uses `is not None` checks, not truthiness)
- The legacy `launchmcp.py` and the new `python -m launcher` have **different** flag sets. Use the legacy entry point when you need `--list-tools` / `--dry-run` / `--verbose`
- The root `README.md` is partially stale (still describes the legacy `launchmcp.py` interface and lists some tools as "NOT WORKING" that are now functional) — trust `python -m launcher --help` and `python launchmcp.py --help` over the README
