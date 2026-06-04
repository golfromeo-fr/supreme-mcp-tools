# AGENTS.md — tools/

## Tool Directory Convention

Each MCP tool lives in `tools/<name>/` and must follow this structure:

```
tools/<name>/
├── <name>_fastmcp.py   # PRIMARY — FastMCP-based server (required)
├── <name>_sse.py       # LEGACY — SSE transport (auto-excluded)
├── <name>_streamable.py # LEGACY — Streamable HTTP (auto-excluded)
├── requirements.txt    # Python dependencies
├── config.json         # Tool-specific config
└── support modules     # Helpers only imported by the primary server
```

### What the launcher discovers

The launcher (`launcher/tool_discovery.py`) scans `tools/<name>/` using **non-recursive glob** — only top-level `.py` files in each tool directory. Subdirectories are never scanned.

A valid MCP tool module must export **one of**:

| Transport | Required exports |
|-----------|-----------------|
| **FastMCP** (preferred) | `app` — result of `mcp.streamable_http_app()` |
| **SSE** (legacy) | `server`, `app`, `sse_transport` |

### Files that are NOT tools

These patterns are auto-excluded from discovery:
- `*_sse.py` — legacy SSE variants
- `*_streamable.py` — legacy streamable variants
- `copilot_context_injector.py` — helper module
- `migrate_*` — migration scripts
- `test_*` — test files
- `__init__.py` — package markers
- Files in subdirectories (`indexer/`, `shared/`, etc.) — support modules, never scanned

### Subdirectories are support modules

Subdirectories within a tool directory contain **support code only**:

- `indexer/` — indexing utilities (ragmcp)
- `shared/` — cross-tool shared libraries

They are **never** scanned by the tool discovery system. Importing from them works via normal Python imports.

### Naming convention for `_fastmcp.py` files

The canonical tool server filename is `<name>_fastmcp.py`. The discovery system strips the `_fastmcp` suffix to derive the tool name (e.g. `webmcp_fastmcp.py` → tool name `webmcp`).

When both `_fastmcp.py` and a non-suffixed `.py` exist, the `_fastmcp` variant wins.

## Auto-use policy integrations

A tool can ship an auto-use policy as repo-local Markdown files in its directory. The support module (named `<tool>_autouse.py`) should:

1. Register MCP shortcut tools on the shared `FastMCP` instance that read repo-local `.md` files.
2. Patch the always-injected prompt tool (if any) to prepend a pointer that teaches the LLM the names of the new tools.
3. Have hard-coded inline fallbacks in case the files are unreadable so the LLM's flow is never broken.

For memorymcp:
- `tools/memorymcp/auto_use_policy.md` — full policy
- `tools/memorymcp/auto_use_cheatsheet.md` — short cheatsheet
