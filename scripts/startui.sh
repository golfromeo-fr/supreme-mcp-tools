#!/usr/bin/env bash
# startui.sh — management UI entry point (NiceGUI, default port 8400).
#
# Real logic lives here; the root `startui` is a thin wrapper.
# The UI app reads MCP_UI_PORT / MCP_UI_HOST (and MCP_API_URL / MCP_API_KEY)
# from the environment or repo .env (which it loads itself, cwd-relative —
# so we cd to the repo root first).

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PORT="${MCP_UI_PORT:-}"
HOST="${MCP_UI_HOST:-}"

usage() {
    cat <<'USAGE'
startui.sh — management UI entry point (default port 8400)

usage: scripts/startui.sh [options]

  --port N   bind port          → exports MCP_UI_PORT   (default: ports.json → 8400)
  --host H   bind host          → exports MCP_UI_HOST   (default: 127.0.0.1)
  -h, --help this help

Env/config: MCP_UI_PORT, MCP_UI_HOST, MCP_UI_USERNAME/PASSWORD, MCP_UI_SECRET,
MCP_API_URL, MCP_API_KEY — flags beat process env, process env beats .env.
Single-instance rule: only one mcp_ui process at a time (AGENTS.md).
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --port)
            [ $# -ge 2 ] || smc_die "--port needs a value"
            PORT="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --host)
            [ $# -ge 2 ] || smc_die "--host needs a value"
            HOST="$2"; shift 2 ;;
        --host=*) HOST="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) smc_die "unknown option '$1' (the UI takes no other CLI args; set env vars or .env keys instead)" ;;
    esac
done

if [ -n "$PORT" ]; then
    export MCP_UI_PORT="$PORT"
fi
if [ -n "$HOST" ]; then
    export MCP_UI_HOST="$HOST"
fi

cd "$SMC_ROOT"
echo "+ MCP_UI_PORT=${MCP_UI_PORT:-default} MCP_UI_HOST=${MCP_UI_HOST:-default} python3 -m mcp_ui.management_ui" >&2
exec python3 -m mcp_ui.management_ui
