#!/usr/bin/env bash
# setup_opencode.sh — bootstrap the opencode MCP client config (+ optional
# launcher restart), replacing the old root setup_opencode_mcp.sh.
#
# Differences from the old script: ports derived from config/ports.json (no
# hardcodes), narrowed pkill patterns, MERGE (never silently destroy existing
# non-MCP settings), timestamped backup before any write, keys pulled from
# tools/<name>/config.json into opencode headers, bounded /health wait,
# --dry-run prints the exact JSONC with keys redacted.

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# --- knobs --------------------------------------------------------------------
eval "$(smc_env_export OPENCODE_MCP_TOOLS)"

TOOLS="${OPENCODE_MCP_TOOLS:-webmcp simplemcp ragmcp memorymcp}"
CONFIG="${HOME}/.config/opencode/opencode.jsonc"
RESTART=1
DRY_RUN=0
INCLUDE_KEYS=1
FORCE=0

usage() {
    cat <<'USAGE'
setup_opencode.sh — opencode MCP client config bootstrap

usage: scripts/setup_opencode.sh [options]

  --tools LIST     MCP tool names, space/comma separated
                   (default: $OPENCODE_MCP_TOOLS, else webmcp simplemcp ragmcp memorymcp)
  --config PATH    opencode config path (default ~/.config/opencode/opencode.jsonc)
  --no-restart     only (re)write the config; don't touch the running launcher
  --restart        restart the launcher first (this is the default, kept for clarity)
  --no-keys        write keyless MCP entries (default: attach X-API-Key from
                   tools/<name>/config.json when a real key is present)
  --force          overwrite an existing but unparseable config instead of aborting
  --dry-run        print the exact JSONC that would be written (keys redacted);
                   no pkill, no launcher start, no file writes
  -h, --help       this help

Precedence: flag > process env > .env (OPENCODE_MCP_TOOLS) > default.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tools)
            [ $# -ge 2 ] || smc_die "--tools needs a value"
            TOOLS="$2"; shift 2 ;;
        --tools=*) TOOLS="${1#*=}"; shift ;;
        --config)
            [ $# -ge 2 ] || smc_die "--config needs a value"
            CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --no-restart) RESTART=0; shift ;;
        --restart) RESTART=1; shift ;;
        --no-keys) INCLUDE_KEYS=0; shift ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) smc_die "unknown option '$1'" ;;
    esac
done

# --- derive ports + keys, build the mcp block via python3 ----------------------
gen_config() {
    python3 - "$SMC_ROOT" "$TOOLS" "$INCLUDE_KEYS" "$DRY_RUN" <<'SMC_OC_PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tool_names = [t for t in sys.argv[2].replace(",", " ").split() if t]
include_keys = sys.argv[3] == "1"
dry_run = sys.argv[4] == "1"

ports = json.loads((root / "config" / "ports.json").read_text(encoding="utf-8"))["assignments"]["mcp"]

mcp = {}
skipped = []
for name in tool_names:
    if name not in ports:
        skipped.append(name)
        continue
    entry = {"type": "remote", "url": f"http://127.0.0.1:{ports[name]}/mcp", "enabled": True}
    if include_keys:
        cfg_path = root / "tools" / name / "config.json"
        key = ""
        if cfg_path.is_file():
            try:
                key = json.loads(cfg_path.read_text(encoding="utf-8")).get("auth", {}).get("api_key", "")
            except (json.JSONDecodeError, OSError):
                key = ""
        if key and not key.startswith("CHANGE-ME"):
            if dry_run:
                entry["headers"] = {"X-API-Key": f"<redacted len={len(key)}>"}
            else:
                entry["headers"] = {"X-API-Key": key}
    mcp[name] = entry

if skipped:
    print(f"# skipped (no port assignment in config/ports.json): {', '.join(skipped)}", file=sys.stderr)

print(json.dumps({"$schema": "https://opencode.ai/config.json", "mcp": mcp}, indent=2))
SMC_OC_PY
}

# --- restart helper -------------------------------------------------------------
restart_launcher() {
    echo "=== Stopping any running MCP launchers ==="
    pkill -f 'python -m launcher' 2>/dev/null || true
    pkill -f 'launchmcp\.py' 2>/dev/null || true
    sleep 1

    TOOLS_CSV="${TOOLS// /,}"
    echo "=== Starting MCP servers: python3 -m launcher --tools $TOOLS_CSV ==="
    cd "$SMC_ROOT"
    mkdir -p logs
    nohup python3 -m launcher --tools "$TOOLS_CSV" >> logs/opencode-launcher.log 2>&1 &
    echo "launcher PID $! (log: logs/opencode-launcher.log)"

    up=0
    for _ in $(seq 1 30); do
        if curl -fsS -o /dev/null http://127.0.0.1:8200/health 2>/dev/null; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" = 0 ]; then
        echo "WARNING: central API (8200 /health) not healthy after 30s — continuing with the config write" >&2
    fi
}

# --- main -----------------------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] config that would be written to $CONFIG (keys redacted):"
    gen_config
    echo "[dry-run] no launcher restart, no file writes."
    exit 0
fi

mkdir -p "$(dirname "$CONFIG")"

# Timestamped backup of whatever exists before any write.
if [ -f "$CONFIG" ]; then
    TS="$(date +%Y%m%d-%H%M%S)"
    cp -p "$CONFIG" "${CONFIG}.bak-${TS}"
    echo "Backup: ${CONFIG}.bak-${TS}"
fi

if [ "$RESTART" = 1 ]; then
    restart_launcher
else
    echo "=== --no-restart: leaving any running launcher alone ==="
fi

# Merge: load the existing config (pure JSON), set the mcp block, keep the rest.
merge_and_write() {
    python3 - "$CONFIG" "$FORCE" <<'SMC_OC_MERGE_PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
force = sys.argv[2] == "1"

existing = {}
if config_path.is_file():
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if force != "1":
            sys.stderr.write(
                f"ERROR: existing config {config_path} is not plain JSON ({exc}).\n"
                "It may be hand-edited JSONC. A timestamped backup was already taken.\n"
                "Reconcile manually or re-run with --force to replace it wholesale.\n"
            )
            sys.exit(3)
        existing = {}

existing["mcp"] = json.loads(sys.stdin.read())
config_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
print(f"Config written: {config_path} (merge: existing non-mcp keys preserved)")
SMC_OC_MERGE_PY
}

gen_config | merge_and_write
chmod 600 "$CONFIG"
echo "Done. Run 'opencode mcp list' to verify servers are connected."
