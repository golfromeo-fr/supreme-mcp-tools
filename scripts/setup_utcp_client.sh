#!/usr/bin/env bash
# setup_utcp_client.sh — generate a UTCP client config for the running
# supreme-mcp-tools launcher, so ANY harness can consume the tools over the
# UTCP dialect (2026-11-04):
#
#   - MCP-ONLY harnesses (Kilo Code, Claude Desktop-style mcpServers JSON):
#     register @utcp/mcp-bridge as ONE stdio MCP server with
#     UTCP_CONFIG_FILE=<generated json> — the bridge turns the 6 manuals
#     into search_tools/list_tools/call_tool. Testing the non-MCP path from
#     an MCP-only client, no per-tool server entries.
#   - Shell-capable agents: npx @utcp/code-mode-cli with the same config.
#   - Python clients: UtcpClient.create(config=<generated json>).
#
# Secrets stay OUT of the config: templates reference ${UTCP_<NAME>_API_KEY}
# (what our /utcp manuals advertise); the real keys land in a sibling .env
# consumed via the config's load_variables_from (dotenv loader). Both plain
# and provider-namespaced variable names are written — reference clients
# resolve provider-scoped placeholders under a "<manual>_" prefix.
#
# Conventions match setup_opencode.sh: ports from config/ports.json (no
# hardcodes), keys pulled from tools/<name>/config.json, timestamped backup
# before overwrite, --dry-run prints the exact JSON with keys redacted.

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# --- knobs --------------------------------------------------------------------
CONFIG="${HOME}/.config/supreme-mcp-tools/utcp-client.json"
ENVFILE="${HOME}/.config/supreme-mcp-tools/utcp-client.env"
DRY_RUN=0
INCLUDE_KEYS=1
TOOLS=""

usage() {
    cat <<'USAGE'
setup_utcp_client.sh — UTCP client config generator (bridge / CLI / python)

usage: scripts/setup_utcp_client.sh [options]

  --tools LIST     tool names, space/comma separated (default: ALL mcp
                   assignments in config/ports.json)
  --config PATH    UTCP client config path
                   (default ~/.config/supreme-mcp-tools/utcp-client.json)
  --env PATH       variables file the config loads via dotenv
                   (default ~/.config/supreme-mcp-tools/utcp-client.env)
  --no-keys        write keyless env (default: attach keys from
                   tools/<name>/config.json when a real key is present)
  --dry-run        print the exact JSON + env (keys redacted); no writes
  -h, --help       this help

After generation, register the bridge in an MCP-only harness:
  {"mcpServers": {"utcp": {"command": "npx",
    "args": ["-y", "@utcp/mcp-bridge"],
    "env": {"UTCP_CONFIG_FILE": "<config path>"}}}}
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tools) [ $# -ge 2 ] || smc_die "--tools needs a value"; TOOLS="$2"; shift 2 ;;
        --tools=*) TOOLS="${1#*=}"; shift ;;
        --config) [ $# -ge 2 ] || smc_die "--config needs a value"; CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --env) [ $# -ge 2 ] || smc_die "--env needs a value"; ENVFILE="$2"; shift 2 ;;
        --env=*) ENVFILE="${1#*=}"; shift ;;
        --no-keys) INCLUDE_KEYS=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) smc_die "unknown flag: $1 (see --help)" ;;
    esac
done

# --- generation (python does the JSON; bash does paths + backup) ---------------
# Dry-run writes NOTHING — not even directories (setup_opencode.sh convention).
if [ "$DRY_RUN" != 1 ]; then
    mkdir -p "$(dirname "$CONFIG")" "$(dirname "$ENVFILE")"
    if [ -f "$CONFIG" ]; then
        cp -p "$CONFIG" "$CONFIG.bak-$(date +%Y%m%d-%H%M%S)"
    fi
    if [ -f "$ENVFILE" ]; then
        cp -p "$ENVFILE" "$ENVFILE.bak-$(date +%Y%m%d-%H%M%S)"
    fi
fi

python3 - "$SMC_ROOT" "$CONFIG" "$ENVFILE" "$INCLUDE_KEYS" "$DRY_RUN" "$TOOLS" <<'GENPY'
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
config_path, env_path = Path(sys.argv[2]), Path(sys.argv[3])
include_keys, dry_run = sys.argv[4] == "1", sys.argv[5] == "1"
tools_filter = [t for t in sys.argv[6].replace(",", " ").split() if t]

ports = json.loads((root / "config" / "ports.json").read_text())["assignments"]["mcp"]
names = tools_filter or sorted(ports)

templates, env_lines, skipped = [], [], []
for name in names:
    if name not in ports:
        skipped.append(name)
        continue
    key = ""
    cfg = root / "tools" / name / "config.json"
    if include_keys and cfg.is_file():
        try:
            key = json.loads(cfg.read_text()).get("auth", {}).get("api_key", "")
        except (json.JSONDecodeError, OSError):
            key = ""
        if not key or key.startswith("CHANGE-ME"):
            key = ""
    var = "UTCP_" + name.upper().replace("-", "_") + "_API_KEY"
    templates.append({
        "name": name,
        "call_template_type": "http",
        "http_method": "GET",
        "url": f"http://127.0.0.1:{ports[name]}/utcp",
        "content_type": "application/json",
        # auth for the manual FETCH itself (the /utcp endpoint is auth-gated
        # like every dialect); placeholder resolved from the dotenv file
        "auth": {
            "auth_type": "api_key",
            "api_key": "${" + var + "}",
            "var_name": "X-API-Key",
            "location": "header",
        },
    })
    if key:
        # plain (RFC variable name) + provider-namespaced (reference clients
        # resolve manual placeholders under "<manual>_" scope)
        env_lines.append(f"{var}={key}")
        env_lines.append(f"{name}_{var}={key}")

config = {
    "load_variables_from": [
        {"variable_loader_type": "dotenv", "env_file_path": str(env_path)}
    ],
    "manual_call_templates": templates,
    "tool_repository": {"tool_repository_type": "in_memory"},
    "tool_search_strategy": {
        "tool_search_strategy_type": "tag_and_description_word_match"
    },
}

if skipped:
    print(f"# skipped (no port assignment): {', '.join(skipped)}", file=sys.stderr)
payload = json.dumps(config, indent=2) + "\n"
env_payload = ("# UTCP client variables (setup_utcp_client.sh) — NOT a shell source list\n"
               + "\n".join(env_lines) + "\n" if env_lines else "")

if dry_run:
    import re
    print(payload)
    print("# ── env ──")
    print(re.sub(r"=..*", "=<redacted>", env_payload, flags=re.M))
else:
    config_path.write_text(payload, encoding="utf-8")
    env_path.write_text(env_payload, encoding="utf-8")
    print(f"wrote {config_path} ({len(templates)} manuals)")
    print(f"wrote {env_path} ({len(env_lines)} variable lines)")
    print("# MCP-only harness registration:")
    print(json.dumps({"mcpServers": {"utcp": {
        "command": "npx", "args": ["-y", "@utcp/mcp-bridge"],
        "env": {"UTCP_CONFIG_FILE": str(config_path)}}}}, indent=2))
GENPY

# The env file carries real API keys — sibling setup_opencode.sh convention:
# keep it (and its backups) owner-only.
if [ "$DRY_RUN" != 1 ] && [ -f "$ENVFILE" ]; then
    chmod 600 "$ENVFILE" "$ENVFILE".bak-* 2>/dev/null || chmod 600 "$ENVFILE"
fi
