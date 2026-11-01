# scripts/ — mainline entry-point scripts

Parameterizable replacements for the old gitignored root one-liners, driven by
the repo `.env` (placeholders template: [`../.env.example`](../.env.example)).

**Precedence everywhere: CLI flag > process env > `.env` > default.**
Unrecognized arguments are forwarded verbatim to the underlying engine, so
future engine flags work without editing these scripts.

## Fresh-machine recipe

```bash
git clone <repo> && cd supreme-mcp-tools
pip install -r requirements.txt
cp .env.example .env          # fill in the CHANGE-ME placeholders
./startlauncher               # or: scripts/startlauncher.sh
```

On a fresh clone the gitignored root wrappers don't exist — call the scripts
directly (`scripts/startlauncher.sh`) or recreate the 2-line wrappers (bottom
of this file).

## startlauncher.sh

Host launcher. Default command is byte-identical to the historical
`startlauncher`: `launchmcp.py --health-check-logs disable simplemcp ragmcp
webmcp memorymcp databasemcp convertermcp`.

| Flag | Effect |
|------|--------|
| `-t, --tools LIST` | tools to launch (space/comma separated); default `$STARTLAUNCHER_TOOLS` or the six above |
| `--all` | launch ALL discovered tools (no positional filter) |
| `--engine legacy\|modern` | `legacy` = `launchmcp.py` (default; only entry with `--dry-run`/`--list-tools`/`--health-check-*`/`--no-sync-tools`); `modern` = `python -m launcher` |
| `--verbose` | legacy: `--verbose` · modern: `--debug` |
| anything else | forwarded verbatim (`--dry-run`, `--list-tools`, `--no-management`, `--no-sync-tools`, `--log-level`, `--host`, `--management-port`, `--transport`, `--health-check-logs`, `--health-check-interval`, `--debug`, …) |

Env knobs: `STARTLAUNCHER_TOOLS`, `STARTLAUNCHER_ENGINE`,
`STARTLAUNCHER_HEALTH_CHECK_LOGS`, `STARTLAUNCHER_VERBOSE`.

## startui.sh / startpodui.sh

| Flag | Effect |
|------|--------|
| `--port N` | exports `MCP_UI_PORT` / `MCP_PODUI_PORT` (defaults 8400 / 8401) |
| `--host H` | exports `MCP_UI_HOST` / `MCP_PODUI_HOST` (default 127.0.0.1) |

Both `cd` to the repo root before exec so the apps' own `.env` loading works
regardless of caller cwd. Everything else (credentials, API URL/key) comes
from `.env` exactly as before.

## setup_opencode.sh

opencode MCP client bootstrap (root wrapper: `./setup_opencode_mcp.sh`).

| Flag | Effect |
|------|--------|
| `--tools LIST` | MCP tools to expose (default `$OPENCODE_MCP_TOOLS` else webmcp simplemcp ragmcp memorymcp) |
| `--config PATH` | opencode config path (default `~/.config/opencode/opencode.jsonc`) |
| `--no-restart` | only (re)write the config; don't touch a running launcher |
| `--no-keys` | keyless MCP entries (default attaches `X-API-Key` from `tools/<name>/config.json` when real) |
| `--force` | replace an existing unparseable config instead of aborting |
| `--dry-run` | print the exact JSONC (keys redacted); zero side effects |

Behavior: ports derived from `config/ports.json`; timestamped `.bak-<ts>` of
the config before any write; MERGE (existing non-mcp settings preserved;
unparseable existing config → abort with `--force` escape hatch); launcher
restart uses narrowed pkill patterns (`python -m launcher`, `launchmcp\.py`),
then a bounded 8200 `/health` wait; written config chmod 600.

## Root wrappers (optional, gitignored)

```bash
#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/startlauncher.sh" "$@"
```

(Same 2-line shape for `startui` → `scripts/startui.sh`, `startpodui` →
`scripts/startpodui.sh`, `setup_opencode_mcp.sh` → `scripts/setup_opencode.sh`.)
