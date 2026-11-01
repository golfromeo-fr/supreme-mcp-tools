#!/usr/bin/env bash
# scripts/common.sh — shared helpers for the scripts/*.sh entry points.
# Sourced, never executed. cwd-independent repo-root resolution + exact-
# semantics .env reads (python-dotenv, eval-safe quoting).
#
# Precedence: CLI flag > process env > .env value > default

# Repo root from this file's own location: scripts/.. (parent of scripts/)
SMC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

smc_die() { echo "ERROR: $*" >&2; exit 1; }

# smc_env_export KEY... — emit `export KEY='value'` lines for keys found in
# $SMC_ROOT/.env that are NOT already set in the process environment (real
# env beats .env). shlex-quoted, eval-safe. Uses python-dotenv (a project
# dependency) so parsing matches what the apps see.
smc_env_export() {
    python3 - "$SMC_ROOT" "$@" <<'SMC_ENV_PY'
import os
import shlex
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys = sys.argv[2:]
env_path = root / ".env"

if not env_path.is_file():
    sys.exit(0)

from dotenv import dotenv_values

values = {k: v for k, v in dotenv_values(str(env_path)).items() if v is not None}

for key in keys:
    if key not in os.environ and key in values:
        print(f"export {key}={shlex.quote(values[key])}")
SMC_ENV_PY
}
