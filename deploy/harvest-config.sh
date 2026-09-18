#!/usr/bin/env bash
# harvest-config.sh — snapshot the working tree's live config into a portable BUNDLE.
#
# A bundle is a plain folder (tweak it, then `startcluster work|test --config-bundle <path>`)
# with one copy of every config surface the pod environments consume:
#
#   .env                                   -> env/.env                     (identity plane, API keys, presets)
#   ~/.config/supreme-mcp-tools/users.json -> identity/users.json          (identity seed)
#   ~/.config/supreme-mcp-tools/tools_config.json -> identity/tools_config.json (masks + inventory)
#   config/*.json                          -> config/                      (ports, launcher, monitoring)
#   tools/*/config.json                    -> tools/<name>/config.json     (per-tool runtime config)
#
# Data planes (turso dir, HF cache) are NOT copied — their host paths are recorded
# in manifest.json. History (.env~), mutation logs, locks and backups are never harvested.
#
# usage:
#   deploy/harvest-config.sh [options] [BUNDLE_PATH]
#
# BUNDLE_PATH: a path as-is, or a BARE NAME -> <cwd>/my-bundles/<name>.
# Default: <cwd>/my-bundles/bundle-<UTC YYYYMMDD-HHMM>
#
# options:
#   --zip            also write <BUNDLE_PATH>.zip (chmod 600) — transport copy
#   --redact         blank out secret values (__REDACTED__) — for sharing; default OFF
#   --from-work-db   harvest users + masks from the LIVE work-cluster DB when reachable
#                    (the DB is the source of truth once MCP_USERS_BACKEND=db ran;
#                    falls back to the json seeds with a warning)
#   --list           dry-run: show what would be collected, write nothing
#
# exit codes: 0 ok (warnings allowed) · 1 usage/fatal · 2 required file missing (.env)
set -euo pipefail
shopt -s nullglob
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG_DIR="${HOME}/.config/supreme-mcp-tools"
# bare bundle names resolve against the INVOKING directory (startcluster cd's
# to the repo root before exec'ing us, so it hands us its ORIG_PWD)
BASE_PWD="${HARVEST_INVOCATION_PWD:-$PWD}"

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

LIST=0; ZIP=0; REDACT=0; FROM_DB=0
BUNDLE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --zip) ZIP=1 ;;
    --redact) REDACT=1 ;;
    --from-work-db) FROM_DB=1 ;;
    --list) LIST=1 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "[harvest] unknown option '$1'"; usage; exit 1 ;;
    *) [ -z "$BUNDLE" ] || { echo "[harvest] unexpected extra argument '$1'"; exit 1; }; BUNDLE="$1" ;;
  esac
  shift
done
if [ -z "$BUNDLE" ]; then
  BUNDLE="$BASE_PWD/my-bundles/bundle-$(date -u +%Y%m%d-%H%M)"
else
  case "$BUNDLE" in
    */*) : ;;                            # explicit path (absolute or relative) — as given
    *)  BUNDLE="$BASE_PWD/my-bundles/$BUNDLE" ;;   # bare name -> <cwd>/my-bundles/<name>
  esac
fi

# ---------- the copy list (source, bundle-relative destination, required?) ----------
SRCS=(); DSTS=(); REQS=()
add_pair() { SRCS+=("$1"); DSTS+=("$2"); REQS+=("${3:-optional}"); }
add_pair "$ROOT/.env" "env/.env" required
add_pair "$CFG_DIR/users.json" "identity/users.json"
add_pair "$CFG_DIR/tools_config.json" "identity/tools_config.json"
for f in "$ROOT"/config/*.json; do
  case "$(basename "$f")" in *.example.json) continue ;; esac
  add_pair "$f" "config/$(basename "$f")"
done
for f in "$ROOT"/tools/*/config.json; do
  add_pair "$f" "tools/$(basename "$(dirname "$f")")/config.json"
done

WARNINGS=()
warn() { WARNINGS+=("$1"); echo "[harvest] WARNING: $1" >&2; }

# ---------- dry-run ----------
if [ "$LIST" = 1 ]; then
  echo "bundle would be: $BUNDLE"
  for i in "${!SRCS[@]}"; do
    if [ -f "${SRCS[$i]}" ]; then mark="ok     "; else mark="MISSING"; fi
    printf "  [%s] %s -> %s\n" "$mark" "${SRCS[$i]}" "${DSTS[$i]}"
    [ "${REQS[$i]}" = required ] && printf "        (required — harvest fails without it)\n"
  done
  [ "$FROM_DB" = 1 ] && echo "  [--from-work-db] users/masks would be pulled from the mcp-work DB if reachable"
  [ "$REDACT" = 1 ] && echo "  [--redact] secret values would be replaced with __REDACTED__"
  [ -f "$ROOT/.env" ] || { echo "FATAL: $ROOT/.env missing"; exit 2; }
  exit 0
fi

# ---------- real harvest ----------
[ -f "$ROOT/.env" ] || { echo "FATAL: $ROOT/.env missing — nothing to harvest"; exit 2; }
mkdir -p "$BUNDLE/env" "$BUNDLE/identity" "$BUNDLE/config" "$BUNDLE/tools"
chmod 700 "$BUNDLE"

: > "$BUNDLE/.sources.tsv"
for i in "${!SRCS[@]}"; do
  src="${SRCS[$i]}"; dst="${DSTS[$i]}"
  if [ -f "$src" ]; then
    mkdir -p "$BUNDLE/$(dirname "$dst")"
    cp -p "$src" "$BUNDLE/$dst"
    printf '%s\t%s\n' "$dst" "$src" >> "$BUNDLE/.sources.tsv"
  else
    if [ "${REQS[$i]}" = required ]; then
      echo "FATAL: required file vanished: $src"; exit 2
    fi
    warn "not found, skipped: $src"
  fi
done

# ---------- --from-work-db: live identity/docs beat the (possibly stale) json seeds ----------
db_doc() { # sql, dst, label
  local tmp="$BUNDLE/$2.tmp"
  if podman exec mcp-work_db_1 psql -U mcp -d mcp -Atc "$1" > "$tmp" 2>/dev/null \
       && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$tmp" 2>/dev/null; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); json.dump(d,open(sys.argv[2],'w'),indent=2,ensure_ascii=False)" \
      "$tmp" "$BUNDLE/$2"
    rm -f "$tmp"; echo "[harvest] $3 pulled live from the work-cluster DB"
  else
    rm -f "$tmp"; warn "$3 not available from the work DB — keeping the json seed"
  fi
}
if [ "$FROM_DB" = 1 ]; then
  if podman exec mcp-work_db_1 pg_isready -q -U mcp -d mcp 2>/dev/null; then
    db_doc "SELECT data FROM mcp_users_store WHERE id = 1" "identity/users.json" "users"
    db_doc "SELECT data FROM mcp_state_docs WHERE name = 'tools_config'" "identity/tools_config.json" "tools_config"
  else
    warn "work-cluster DB not reachable — identity/docs stay as the json seeds"
  fi
fi

# ---------- --redact ----------
if [ "$REDACT" = 1 ]; then
  python3 - "$BUNDLE/env/.env" <<'PY'
import re, sys
p = sys.argv[1]
secret_suffix = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|DSN)$")
out = []
for line in open(p, encoding="utf-8"):
    s = line.rstrip("\n")
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", s)
    if m and m.group(2) != "":
        k = m.group(1)
        if secret_suffix.search(k) or (k.startswith("DB_PRESET_") and not k.endswith("_DESC")):
            s = k + "=__REDACTED__"
    out.append(s)
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
  python3 - "$BUNDLE/identity/users.json" <<'PY'
import sys
p = sys.argv[1]
try:
    import json
    d = json.load(open(p, encoding="utf-8"))
    for u in d.get("users", {}).values():
        if isinstance(u, dict):
            if "mcp_key" in u: u["mcp_key"] = "__REDACTED__"
            if "password_hash" in u: u["password_hash"] = "__REDACTED__"
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
except Exception as e:
    print(f"[harvest] WARNING: could not redact users.json: {e}", file=sys.stderr)
PY
  for f in "$BUNDLE"/tools/*/config.json; do
    python3 - "$f" <<'PY'
import sys
p = sys.argv[1]
try:
    import json
    d = json.load(open(p, encoding="utf-8"))
    if "api_key" in d.get("auth", {}):
        d["auth"]["api_key"] = "__REDACTED__"
        json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
except Exception:
    pass
PY
  done
  echo "[harvest] redacted — this bundle is for SHARING, pods booted with it have dead secrets"
fi

# ---------- permissions: env + identity + tool configs hold secrets ----------
chmod 600 "$BUNDLE/env/.env" 2>/dev/null || true
chmod 600 "$BUNDLE"/identity/*.json 2>/dev/null || true
chmod 600 "$BUNDLE"/tools/*/config.json 2>/dev/null || true

# ---------- manifest.json ----------
# flush warnings first: re-runs overwrite (E14) and the manifest below reads this file
printf '%s\n' "${WARNINGS[@]+"${WARNINGS[@]}"}" > "$BUNDLE/.warnings.txt"
TURSO_DIR=$(sed -n 's/^TURSO_DATABASE_URL=file:\([^ ]*\).*/\1/p' "$BUNDLE/env/.env" | xargs dirname 2>/dev/null || true)
GIT_SHA=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DESCRIBE=$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo unknown)
GIT_BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
python3 - "$BUNDLE" "$TURSO_DIR" "$GIT_SHA" "$GIT_DESCRIBE" "$GIT_BRANCH" <<'PY'
import hashlib, json, os, socket, sys
from datetime import datetime, timezone

bundle, turso_dir, sha, describe, branch = sys.argv[1:6]
sources = {}
for line in open(os.path.join(bundle, ".sources.tsv"), encoding="utf-8"):
    dst, src = line.rstrip("\n").split("\t", 1)
    sources[dst] = src
warnings = []
wf = os.path.join(bundle, ".warnings.txt")
if os.path.exists(wf):
    warnings = [l.rstrip("\n") for l in open(wf, encoding="utf-8") if l.strip()]

files = []
for dirpath, _, names in os.walk(bundle):
    for n in sorted(names):
        if n in (".sources.tsv", ".warnings.txt"):   # harvest bookkeeping, not payload
            continue
        p = os.path.join(dirpath, n)
        rel = os.path.relpath(p, bundle)
        data = open(p, "rb").read()
        files.append({"path": rel, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "source": sources.get(rel)})
manifest = {
    "bundle_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "git": {"sha": sha, "describe": describe, "branch": branch},
    "host_data_paths": {"turso_dir": turso_dir or None,
                        "hf_cache": os.path.expanduser("~/.cache/huggingface")},
    "redacted": os.path.exists(os.path.join(bundle, "env", ".env"))
                and "__REDACTED__" in open(os.path.join(bundle, "env", ".env")).read(),
    "files": files,
    "warnings": warnings,
}
with open(os.path.join(bundle, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

# ---------- generated README ----------
cat > "$BUNDLE/README.md" <<EOF
# config bundle — $(basename "$BUNDLE")

Harvested $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(hostname) (git $(cd "$ROOT" && git describe --tags --always --dirty 2>/dev/null || echo '?')).

| Path | What it is |
|---|---|
| \`env/.env\` | the working tree's live environment (identity plane, tool API keys, presets) |
| \`identity/users.json\` | identity seed — imported once when a pod's user DB is empty |
| \`identity/tools_config.json\` | function masks + tool inventory seed |
| \`config/*.json\` | ports / launcher / monitoring config (mounted over the image's baked copies) |
| \`tools/<name>/config.json\` | per-tool runtime config (mounted over the baked copies) |
| \`manifest.json\` | provenance: git sha, per-file sha256, warnings |

Data planes are NOT in the bundle (recorded in manifest.json \`host_data_paths\`).

## Use it

    ./startcluster work --config-bundle "$BUNDLE"
    ./startcluster test pg --config-bundle "$BUNDLE"

Tweak any file here, re-run the command — the next \`up\` re-reads the bundle.
Redacted bundles boot with dead secrets (see manifest.json \`redacted\`).
EOF

# ---------- optional zip ----------
if [ "$ZIP" = 1 ]; then
  ( cd "$(dirname "$BUNDLE")" && python3 -m zipfile -c "$(basename "$BUNDLE").zip" "$(basename "$BUNDLE")" )
  chmod 600 "$BUNDLE.zip"
  echo "[harvest] zip -> $BUNDLE.zip"
fi

echo "[harvest] bundle -> $BUNDLE ($(grep -c . "$BUNDLE/.sources.tsv" || true) files)"
exit 0
