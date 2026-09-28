#!/usr/bin/env python3
"""env_strip — inverse of bundle_node_env: remove startcluster-GENERATED
content from a pod env file, leaving the user-space env (identity plane,
tool keys, presets, kept data planes).

Used by `startcluster <env> scrape` (via harvest-config.sh --from-pod
--env-file): the scraped env becomes a bundle env again, so env-specific
state-plane credentials (POSTGRES_PASSWORD, POSTGRES_HOST=127.0.0.1/db,
S3 block) never round-trip into a bundle.

Removal rules (order-independent, applied to every line):
  - `# [bundle] ...` annotation lines
  - `# ---- state plane ...` / `# ---- shared S3 ...` block markers (both the
    bundle generator's and the legacy host-bootstrap wording)
  - KEY= lines whose KEY the generator owns (state-plane backends + POSTGRES_*
    + S3_*)
  - TURSO_DATABASE_URL=http://db:8080 (the generator's canonical embedded DSN)

usage: env_strip.py <env_file>   (cleaned env on stdout)
"""

import re
import sys

KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

GENERATED_KEYS = {
    "MCP_USERS_BACKEND", "MCP_STATE_BACKEND",
    "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER",
    "POSTGRES_PASSWORD", "POSTGRES_DB",
    "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY",
}
GENERATED_MARKERS = ("# [bundle]", "# ---- state plane", "# ---- shared S3")
GENERATED_TURSO = "TURSO_DATABASE_URL=http://db:8080"


def strip(lines):
    out = []
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if any(stripped.startswith(m) for m in GENERATED_MARKERS):
            continue
        m = KEY_RE.match(stripped)
        if m and m.group(1) in GENERATED_KEYS:
            continue
        if stripped == GENERATED_TURSO:
            continue
        out.append(line)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: env_strip.py <env_file>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in strip(f.readlines()):
            print(line)


if __name__ == "__main__":
    main()
