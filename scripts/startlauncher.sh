#!/usr/bin/env bash
# startlauncher.sh — parameterizable host launcher entry point.
#
# Real logic lives here; the root `startlauncher` is a thin wrapper
# (startcluster precedent: wrapper at root, logic in a subdir, no drift).
#
# Default reproduces the historical root script byte-for-byte:
#   ./launchmcp.py --health-check-logs disable simplemcp ragmcp webmcp memorymcp databasemcp convertermcp
#
# Precedence: CLI flag > process env > .env > default.
# Unrecognized args are forwarded verbatim to the engine.

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# --- knobs: process env beats .env, flags beat both --------------------------
eval "$(smc_env_export STARTLAUNCHER_TOOLS STARTLAUNCHER_ENGINE \
                          STARTLAUNCHER_HEALTH_CHECK_LOGS STARTLAUNCHER_VERBOSE)"

ENGINE="${STARTLAUNCHER_ENGINE:-legacy}"
TOOLS="${STARTLAUNCHER_TOOLS:-simplemcp ragmcp webmcp memorymcp databasemcp convertermcp}"
HEALTH_LOGS="${STARTLAUNCHER_HEALTH_CHECK_LOGS:-disable}"
VERBOSE=0
case "${STARTLAUNCHER_VERBOSE:-0}" in
    1|true|yes) VERBOSE=1 ;;
esac

PASS=()
SAW_HEALTH_LOGS=0

usage() {
    cat <<'USAGE'
startlauncher.sh — host launcher entry point

usage: scripts/startlauncher.sh [options]

First-class options:
  -t, --tools LIST        tools to launch, space/comma separated
                          (default: $STARTLAUNCHER_TOOLS, else
                          simplemcp ragmcp webmcp memorymcp databasemcp convertermcp)
  --all                   launch ALL discovered tools (no positional filter)
  --engine legacy|modern  legacy = ./launchmcp.py (default; the only entry with
                          --dry-run/--list-tools/--health-check-*/--no-sync-tools)
                          modern = python -m launcher
  --verbose               legacy: --verbose · modern: --debug
  --quiet                 silence the per-request access log entirely
                          (MCP_DISABLE_REQUEST_LOGS=1). GET /health 404s are
                          already skipped by default — see the
                          MCP_ACCESS_LOG_SKIP_404_PATHS note in AGENTS.md.
  -h, --help              this help

Everything else is forwarded verbatim to the engine, e.g.:
  --dry-run, --list-tools, --no-management, --no-sync-tools
  --log-level L, --host H, --management-port N, --transport {streamable-http,sse}
  --health-check-logs {enable,disable,errors-only}, --health-check-interval S

Precedence: flag > process env > .env (STARTLAUNCHER_* keys) > default.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        -t|--tools)
            [ $# -ge 2 ] || smc_die "--tools needs a value"
            TOOLS="$2"; shift 2 ;;
        --tools=*) TOOLS="${1#*=}"; shift ;;
        --all) TOOLS=""; shift ;;
        --engine)
            [ $# -ge 2 ] || smc_die "--engine needs legacy|modern"
            ENGINE="$2"; shift 2 ;;
        --engine=*) ENGINE="${1#*=}"; shift ;;
        --verbose) VERBOSE=1; shift ;;
        --quiet) export MCP_DISABLE_REQUEST_LOGS=1; shift ;;
        --health-check-logs)
            [ $# -ge 2 ] || smc_die "--health-check-logs needs a value"
            PASS+=(--health-check-logs "$2"); SAW_HEALTH_LOGS=1; shift 2 ;;
        --health-check-logs=*) PASS+=("$1"); SAW_HEALTH_LOGS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PASS+=("$@"); break ;;
        *) PASS+=("$1"); shift ;;
    esac
done

case "$ENGINE" in
    legacy)
        ENGINE_CMD=("$SMC_ROOT/launchmcp.py")
        if [ "$VERBOSE" = 1 ]; then
            PASS=(--verbose "${PASS[@]}")
        fi
        if [ "$SAW_HEALTH_LOGS" = 0 ]; then
            # Prepend, never append: argparse can't parse positionals that
            # follow optionals (intermix limitation) — a forwarded bare word
            # before the flags used to die with "unrecognized arguments".
            # HEALTH_LOGS already honors the STARTLAUNCHER_HEALTH_CHECK_LOGS
            # knob (env/.env > default 'disable').
            PASS=(--health-check-logs "$HEALTH_LOGS" "${PASS[@]}")
        fi
        ;;
    modern)
        ENGINE_CMD=(python3 -m launcher)
        if [ "$VERBOSE" = 1 ]; then
            PASS+=(--debug)
        fi
        ;;
    *) smc_die "unknown engine '$ENGINE' (legacy|modern)" ;;
esac

cd "$SMC_ROOT"

TOOL_ARGS=()
if [ -n "$TOOLS" ]; then
    if [ "$ENGINE" = "modern" ]; then
        # R6 (review 2026-11-11): the modern entry takes tools ONLY via
        # --tools (comma-separated string), not as positionals — forwarding
        # the bare names used to die with argparse "unrecognized arguments".
        TOOL_ARGS=(--tools "${TOOLS// /,}")
    else
        read -ra TOOL_ARGS <<< "${TOOLS//,/ }"
    fi
fi

echo "+ ${ENGINE_CMD[*]} ${PASS[*]} ${TOOL_ARGS[*]}" >&2

exec "${ENGINE_CMD[@]}" "${PASS[@]}" "${TOOL_ARGS[@]}"
