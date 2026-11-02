#!/usr/bin/env bash
# Thin wrapper — real logic in scripts/setup_opencode.sh (startcluster
# precedent: wrapper at root stays tiny, logic lives in scripts/; the old
# script is archived verbatim in scripts/legacy/2026-10-31/).
exec "$(cd "$(dirname "$0")" && pwd)/scripts/setup_opencode.sh" "$@"
