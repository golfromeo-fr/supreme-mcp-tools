# MCP Management UI

NiceGUI web interface for managing the MCP tools launched by the unified
launcher. Talks to the launcher's central management API.

**Run:**

```bash
python -m mcp_ui
```

**Ports** (from `config/ports.json`):

| What | Port |
|---|---|
| Management UI (this app) | `reserved.management_ui` = **8400** |
| Management API (talked to) | `reserved.central_management` = **8200** |

## What you get

- **Drawer navigation** with the live tool list and status badges.
- **Per-tool tabs**: Overview (status, endpoints, tool-specific panels like
  ragmcp collections), Extensions (data sources with auto-charts and actions),
  Env Vars (inline editing with masked secrets), Auth (per-tool API key).
- **Live status**: the tool list and connection badge poll the management API
  every 10 seconds without rebuilding open panels.
- **Dark mode** toggle in the header (default from `MCP_UI_THEME` or
  `ports.json` → `ui.theme`).

## Environment variables

| Variable | Effect | Default |
|---|---|---|
| `MCP_UI_PORT` | Override the UI port | `ports.json` → 8400 |
| `MCP_UI_HOST` | Bind host | `127.0.0.1` |
| `MCP_UI_THEME` | `dark` / `light` initial theme | `ports.json` → dark |
| `MCP_UI_USERNAME` / `MCP_UI_PASSWORD` | Login credentials (each independently overridable) | `admin` / `admin` |
| `MCP_UI_SECRET` | Session-cookie signing secret. **Set this for persistent logins** — without it an ephemeral per-process secret is generated and every restart logs everyone out. | ephemeral |
| `MCP_API_URL` | Management API base URL | `ports.json` → `http://127.0.0.1:8200` |
| `MCP_API_TIMEOUT` | API request timeout in seconds | `ports.json` → 30 |
| `MCP_API_KEY` | Bearer token sent to the management API (needed if the API runs with auth enabled) | none |

`MCP_UI_SECRET` may also live in `~/.mcp_ui/secrets` as `MCP_UI_SECRET=<value>`.

A warning is logged at startup when default admin/admin credentials are active.

## Security notes

- The login session cookie is signed with `MCP_UI_SECRET`. There is no
  hardcoded fallback secret; an unconfigured one is random per process.
- Credential comparison is constant-time; post-login redirects are restricted
  to same-site paths.
- The UI is a *client* of the management API. Per-tool API keys it writes go
  through `PUT /api/tools/{name}/auth` (server-side, preserving the nested
  `auth.api_key` structure of `tools/<name>/config.json`).

## Architecture

```
mcp_ui/
  __main__.py          # python -m mcp_ui → run_ui()
  management_ui.py     # pages, layout, handlers, polling
  api_client.py        # async aiohttp client for the management API
  state.py             # AppState (tools, selection, caches)
  models.py            # pydantic models mirroring API payloads
  logging_config.py    # trace-ID logging
  components/          # ToolList, ToolOverview, DataSourcesBox, ActionsBox,
                       # EnvVarEditor, AuthBox, tool settings dialogs, panels
```

All API access is async (`aiohttp`); the UI never blocks the event loop on
network calls. State changes refresh only the affected regions.

## Development

```bash
python -m mcp_ui            # start the UI (http://127.0.0.1:8400)
python -m pytest tests/     # repo test suite
```

Audit history: `docs/mcp_ui_audit_2026-08-29.md` (repairs + overhaul landed
the same day).
