#!/usr/bin/env bash
# startpodui.sh — pod-UI entry point (default port 8401).
#
# Real logic lives here; the root `startpodui` is a thin wrapper.
# pod_ui reads MCP_PODUI_PORT / MCP_PODUI_HOST (default 127.0.0.1) and the
# MCP_UI_* keys from the repo .env (which it loads itself).

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PORT="${MCP_PODUI_PORT:-}"
HOST="${MCP_PODUI_HOST:-}"

usage() {
    cat <<'USAGE'
startpodui.sh — pod-UI entry point (default port 8401)

usage: scripts/startpodui.sh [options]

  --port N   bind port          → exports MCP_PODUI_PORT (default: 8401)
  --host H   bind host          → exports MCP_PODUI_HOST (default: 127.0.0.1)
  -h, --help this help

Env/config: MCP_PODUI_PORT, MCP_PODUI_HOST, MCP_UI_* keys via the repo .env
(pod_ui loads it itself). Flags beat process env, process env beats .env.
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
        *) smc_die "unknown option '$1' (pod_ui takes no other CLI args; set env vars or .env keys instead)" ;;
    esac
done

if [ -n "$PORT" ]; then
    export MCP_PODUI_PORT="$PORT"
fi
if [ -n "$HOST" ]; then
    export MCP_PODUI_HOST="$HOST"
fi

cd "$SMC_ROOT"
echo "+ MCP_PODUI_PORT=${MCP_PODUI_PORT:-default} MCP_PODUI_HOST=${MCP_PODUI_HOST:-default} python3 -m pod_ui" >&2
exec python3 -m pod_ui
