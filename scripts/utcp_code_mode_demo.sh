#!/usr/bin/env bash
# utcp_code_mode_demo.sh — live smoke/demo of @utcp/code-mode-cli against the
# running supreme-mcp-tools launcher. Demonstrates the Code-Mode payoff over
# MCP: chain several tools in ONE sandboxed TypeScript execution, with data
# flowing through code rather than the agent's context. Proven 2026-11-18;
# see docs/UTCP_CLIENTS.md §2.
#
# Requires: node >=18 + npx, the launcher UP, and a generated client config
# (scripts/setup_utcp_client.sh). The first run fetches the CLI + isolated-vm
# via npx (network); later runs use the npx cache. The sandbox exposes only the
# tool namespaces (no process/require/fetch), so the chain can only call tools.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${HOME}/.config/supreme-mcp-tools/utcp-client.json"
KEEP=0

usage() {
    cat <<'USAGE'
utcp_code_mode_demo.sh — code-mode-cli live smoke (validate + search + chain)

usage: scripts/utcp_code_mode_demo.sh [options]

  --config PATH  client config to use
                 (default ~/.config/supreme-mcp-tools/utcp-client.json)
  --keep         keep the tmp workdir (prints its path) instead of cleaning up
  -h, --help     this help

Prereqs: launcher up + scripts/setup_utcp_client.sh run once + node/npx.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config) [ $# -ge 2 ] || smc_die "--config needs a value"; CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --keep) KEEP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) smc_die "unknown flag: $1 (see --help)" ;;
    esac
done

command -v npx >/dev/null 2>&1 || smc_die "npx not found (need node >=18)"
[ -f "$CONFIG" ] || smc_die "config not found: $CONFIG — run scripts/setup_utcp_client.sh first"

WORK="$(mktemp -d)"
if [ "$KEEP" = 1 ]; then
    echo "# workdir kept: $WORK"
else
    trap 'rm -rf "$WORK"' EXIT
fi
# the generated config is secret-free (placeholders); keys stay in its sibling
# .env via the absolute load_variables_from path, so a copy works from any cwd
cp "$CONFIG" "$WORK/.utcp_config.json"
cd "$WORK"

echo "=== validate (register all manuals) ==="
npx -y @utcp/code-mode-cli validate 2>/dev/null

echo "=== search 'square' (generated TS interface) ==="
npx -y @utcp/code-mode-cli search "square" --limit 1 2>/dev/null

echo "=== run: 3-tool chain in one sandboxed execution ==="
RUN_OUT="$(npx -y @utcp/code-mode-cli run 2>/dev/null <<'EOF'
const a = await simplemcp.square({ body: { value: 7 } });
const n = parseFloat(a.result);
const b = await simplemcp.double({ body: { value: n } });
const g = await simplemcp.greet({ body: { name: "CodeMode" } });
return { squareOf7: a.result, parsed: n, doubled: b.result, greeting: g.result, chainedCalls: 3 };
EOF
)"
echo "$RUN_OUT"
echo "$RUN_OUT" | grep -q '"success":true' || smc_die "chain did not succeed (launcher up? key correct?): $RUN_OUT"

echo "=== done: code-mode chain OK ==="
