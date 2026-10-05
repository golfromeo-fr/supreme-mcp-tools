#!/usr/bin/env bash
# startcluster — manage the two podman environments of the project.
#
#  ENV: work  — the pod-based DAILY DRIVER (replaces startlauncher):
#       one node, all six tools, CANONICAL ports (8000-8005 + central 8200),
#       host networking, own pgvector state plane (own volume).
#       First boot imports the HOST users.json: same accounts, same keys.
#       Data planes come from the host .env (real memories, real presets).
#
#  ENV: test  — the M4 TESTING ground (two light nodes on 18xxx/19xxx,
#       shared identity plane). Never touched by your working clients.
#
#  The two environments are fully isolated: separate containers, volumes
#  and env files. Only one of {work cluster, host startlauncher} may run
#  at a time (same canonical ports). Automated tests target the canonical
#  ports: stop the work cluster first, or the suite tests the work env.
#
#  usage:
#    startcluster work [up|stop|start|status|clean|logs [svc]] [-b|--config-bundle <path>]
#    startcluster test [pg|turso|external-pg|external-turso|stop|start|status|clean|logs [svc]] [-b|--config-bundle <path>]
set -euo pipefail
umask 077                           # generated env files carry secrets — keep them private
PW_TMPFILE=""                       # temp env file carrying a preserved POSTGRES_PASSWORD
trap 'rm -f "${PW_TMPFILE:-}"' EXIT
ORIG_PWD="$PWD"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."                 # repo root (compose calls do `cd deploy` from here)
REPO_ROOT="$PWD"

TEST_PROJECT="mcp-multihost"
WORK_PROJECT="mcp-work"
CONFIG_DIR="${HOME}/.config/supreme-mcp-tools"

# ---------- shared helpers ----------
usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; cat <<'HELP'

  usage: startcluster work [up|stop|start|status|backup|restore <file>|clean|logs [svc]|service <start|stop|restart> <name>|scrape [bundle]|export-data [out]|import-data <tar.gz>] [-b|--config-bundle <path>]
         startcluster test [pg|turso|external-pg|external-turso|status|stop|start|clean|logs [svc]|service <start|stop|restart> <name>|backup|restore <file>|scrape [bundle]|export-data [dir]|import-data <dir>] [-b|--config-bundle <path>]
         startcluster harvest [--zip|--redact|--from-work-db|--list] [BUNDLE_PATH]

  work:  daily-driver pod (canonical ports 8000-8005/8200; UI: startcluster work ui)
  test:  M4 two-node bench (node centrals 18200/19200)
  service: podman start/stop/restart ONE container (bare service name like db,
      node1, lb, minio — or a full container name)
  backup/restore: pg_dump of the state plane <-> ~/supreme-mcp-tools-backups/

  migration (pod -> bundle -> new pod):
  scrape [bundle]      reverse-harvest a RUNNING pod's config plane into the
      bundle (users/masks/state docs from the pod DB via exec + the pod's
      current env, generator content stripped). Default bundle = the one
      recorded in the env's [bundle] header.
  export-data/import-data
      POD-OWNED data stores only: work = artifacts volume (tar);
      test = MinIO objects (mc mirror). External data planes (host turso
      dir, LAN Postgres, Qdrant) are never scraped; the pg dump covers
      pg-topology data-plane tables.

  -b <path> | --config-bundle <path|host>  seed the environment from a harvested
      config bundle (made by: startcluster harvest). 'host' (the default) keeps
      today's behavior: derive from the live host files. In test mode, file:
      data-plane lines are stripped for safety (--keep-dataplanes overrides).
      Honored by the boot actions only (work up / test <topology>); every
      other action accepts and IGNORES it (E10).
  bundle-node-env <bundle> <work|pg|turso|external-pg|external-turso>
      print the env file a bundle would generate (debug/tests, no podman).
  harvest [BUNDLE_PATH]  harvest into BUNDLE_PATH; a bare NAME goes to
      <cwd>/my-bundles/<NAME>; default: <cwd>/my-bundles/bundle-<UTC ts>
HELP
}
wait_http() { # url, tries
  for _ in $(seq 1 "${2:-40}"); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep "${SC_WAIT_SLEEP:-3}"; done
  return 1
}

# ---------- config-bundle support (plans/pod-config-bundle-2026-09-17.md) ----------
BUNDLE_RAW="host"   # --config-bundle value, extracted from the args in the dispatch
BUNDLE_DIR=""       # resolved ABSOLUTE bundle folder; "" = host mode (today's behavior)
KEEP_DATAPLANES=0

resolve_bundle() { # $BUNDLE_RAW -> $BUNDLE_DIR ("" = host mode)
  [ "$BUNDLE_RAW" = "host" ] && return 0
  case "$BUNDLE_RAW" in
    /*) p="$BUNDLE_RAW" ;;
    *)  p="$ORIG_PWD/$BUNDLE_RAW" ;;
  esac
  if [ -f "$p" ] && [ "${p##*.}" = "zip" ]; then
    tmp="$REPO_ROOT/deploy/.bundle-tmp/$(basename "$p" .zip)"
    rm -rf "$tmp"; mkdir -p "$tmp"
    python3 -m zipfile -e "$p" "$tmp" || { echo "FATAL: cannot extract $p"; exit 1; }
    # tolerate zips that carry the bundle folder as a top-level prefix
    if [ -f "$tmp/env/.env" ]; then BUNDLE_DIR="$tmp"
    elif [ -f "$tmp/$(basename "$p" .zip)/env/.env" ]; then BUNDLE_DIR="$tmp/$(basename "$p" .zip)"
    else echo "FATAL: $p contains no env/.env — not a config bundle"; exit 1; fi
  elif [ -d "$p" ]; then
    [ -f "$p/env/.env" ] || { echo "FATAL: $p is not a config bundle (missing env/.env)"; exit 1; }
    BUNDLE_DIR="$(cd "$p" && pwd)"
  else
    echo "FATAL: bundle not found: $p"; exit 1
  fi
  echo "[bundle] using $BUNDLE_DIR" >&2
}

bundle_node_env() { # bundle_dir, mode, existing_env, minio_pw, keep_dp — env to stdout
  python3 "$REPO_ROOT/deploy/bundle_node_env.py" "$1" "$2" "${3:-}" "${4:-}" "${5:-0}"
}

minio_password() { # test-helper override via env, else deploy/.env
  if [ -n "${BUNDLE_MINIO_PW+x}" ]; then printf '%s' "$BUNDLE_MINIO_PW"; return; fi
  [ -f deploy/.env ] && sed -n 's/^MINIO_ROOT_PASSWORD=//p' deploy/.env | head -1 || true
}

pw_tmpfile_from() { # env_file — stash its POSTGRES_PASSWORD in a 0600 temp env
  # file and set PW_TMPFILE to that path ("" when there is none to preserve).
  # bundle_node_env takes an env-file PATH as its third arg, never a value:
  # passing the raw password made open() fail and generated a FRESH random
  # password, bricking the already-initialized pgdata on re-up.
  # Also matches the DORMANT form ('# [topology:pg-dormant] POSTGRES_PASSWORD=')
  # written by a turso-era switch — the pg password must survive topology
  # round trips even while parked (grounds plan edge 2).
  PW_TMPFILE=""
  local envf="$1" pw=""
  [ -f "$envf" ] || return 0
  pw=$(sed -n 's/^POSTGRES_PASSWORD=//p' "$envf" | head -1)
  [ -n "$pw" ] || pw=$(sed -n 's/^# \[topology:pg-dormant\] POSTGRES_PASSWORD=//p' "$envf" | head -1)
  [ -n "$pw" ] || return 0
  PW_TMPFILE="$(mktemp "${TMPDIR:-/tmp}/startcluster-pw.XXXXXX")"
  printf 'POSTGRES_PASSWORD=%s\n' "$pw" > "$PW_TMPFILE"
  chmod 600 "$PW_TMPFILE"
}

work_stamp() { # echo the topology stamp from work.env ("pg" when unset)
  sed -n 's/^\[topology\] //p' "${1:-$REPO_ROOT/deploy/work.env}" 2>/dev/null | head -1
}

write_bundle_compose() { # outfile, then specs:
  #   svc|hostpath|containerpath[:ro|rw]   -> volumes mount (existing form)
  #   env|svc|KEY=VAL                      -> environment entry (grounds: the
  #       ephemeral in-container data plane — file: turso inside the container)
  out="$1"; shift
  python3 - "$out" "$@" <<'PY'
import re, sys
out, specs = sys.argv[1], sys.argv[2:]
def yval(s):  # YAML-quote a mount spec only when the host path needs it (E9)
    if re.match(r"^[A-Za-z0-9_./@%+=,:-]+$", s):
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
vols: dict[str, list[str]] = {}
envs: dict[str, list[str]] = {}
for s in specs:
    parts = s.split("|")
    if parts[0] == "env":
        _, svc, kv = parts
        envs.setdefault(svc, []).append(f"      - {yval(kv)}")
        continue
    svc, host, cont, *rest = parts
    mode = "" if (rest or ["ro"])[0] == "rw" else ":ro"
    vols.setdefault(svc, []).append(f"      - {yval(f'{host}:{cont}{mode}')}")
services: dict[str, list[str]] = {}
for svc in sorted(set(vols) | set(envs)):
    services[svc] = []
    if svc in vols:
        services[svc] += ["    volumes:"] + vols[svc]
    if svc in envs:
        services[svc] += ["    environment:"] + envs[svc]
lines = ["# [bundle] GENERATED by startcluster --config-bundle — do not hand-edit.",
         "# Mounts the bundle's config over the image's baked copies (tweaks need no rebuild).",
         "services: {}" if not services else "services:"]
for svc, body in services.items():
    lines += [f"  {svc}:"] + body
open(out, "w").write("\n".join(lines) + "\n")
PY
}

# ---------- WORK environment ----------
work_env() { action="${1:-up}"; shift || true
  PROJECT="$WORK_PROJECT"
  # D7: effective identity-plane topology — explicit `work pg|turso` beats
  # the stamp in work.env; the stamp defaults to pg. Backups/branches read it.
  WORK_TOPO="${WORK_TOPO:-$(work_stamp)}"
  WORK_TOPO="${WORK_TOPO:-pg}"
  if [ "$action" = backup ]; then
    ts=$(date +%Y%m%d-%H%M%S)
    if [ "$WORK_TOPO" = "turso" ]; then
      # D10: turso identity backup — sqld has no pg_dump; v1 = stop-free
      # volume export of work_sqldata (restore = startgrounds/work clean + import)
      out="${HOME}/supreme-mcp-tools-backups/work-turso-$ts.tar"
      mkdir -p "$(dirname "$out")"
      podman volume export "work_sqldata" --output "$out" 2>/dev/null \
        || podman volume export "${PROJECT}_sqldata" --output "$out" \
        && echo "[work] backup (turso, volume tar) -> $out ($(wc -c < "$out") bytes)" \
        || { echo "[work] backup FAILED — is the work env running?"; exit 1; }
      exit 0
    fi
    out="${HOME}/supreme-mcp-tools-backups/work-pg-$ts.sql"
    mkdir -p "$(dirname "$out")"
    podman exec "${PROJECT}_db_1" pg_dump -U mcp -d mcp > "$out" \
      && echo "[work] backup -> $out ($(wc -c < "$out") bytes)" \
      || { echo "[work] backup FAILED — is the work env running?"; exit 1; }
    exit 0
  fi
  if [ "$action" = restore ]; then
    f="${1:-}"   # post-shift: the file is the first remaining arg
    [ -f "$f" ] || { echo "usage: $0 work restore <backup.sql>"; exit 1; }
    echo "[work] RESTORE wipes the current work data volume (users/masks) and imports $f — type 'wipe' to confirm:"
    read -r ans; [ "$ans" = "wipe" ] || { echo "aborted"; exit 1; }
    ( cd deploy && podman-compose -p "$PROJECT" down ) 2>/dev/null || true
    podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
    podman volume rm -f "${PROJECT}_work_pgdata" 2>/dev/null || true
    # keep the bundle's config/tool mounts across a restore (fix-band b):
    # compose-work.bundle.yml exists iff the last up was bundle-managed
    BUNDLE_F=()
    [ -f deploy/compose-work.bundle.yml ] && BUNDLE_F=(-f compose-work.bundle.yml)
    ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml \
        ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} up -d db ) \
      || ( cd deploy && podman-compose -p "$PROJECT" up -d db )
    sleep 8
    podman exec -i "${PROJECT}_db_1" psql -U mcp -d mcp < "$f" >/dev/null \
      && echo "[work] restore imported — starting the node..." \
      || { echo "[work] restore import FAILED"; exit 1; }
    ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml \
        ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} up -d work ) \
      || ( cd deploy && podman-compose -p "$PROJECT" up -d work )
    if wait_http http://127.0.0.1:8200/health 50; then
      echo "[work] restored and healthy"
    else
      echo "[work] node not healthy yet — check podman logs ${PROJECT}_work_1"; exit 1
    fi
    exit 0
  fi
  # `clean` is handled by the case below: it WIPES containers AND the state
  # volume after an explicit 'wipe' confirmation.

  # scrape: reverse-harvest a RUNNING pod back into its bundle (Phase A).
  # Config plane only: users/masks/state docs from the pod DB + the pod's
  # current env (generator content stripped). Data planes are NOT scraped
  # here — see export-data / backup.
  if [ "$action" = scrape ]; then
    B="${1:-}"
    if [ -z "$B" ] && [ -f deploy/work.env ]; then
      src=$(sed -n 's/^# \[bundle\] GENERATED by startcluster from //p' deploy/work.env | head -1)
      [ -n "$src" ] && B="$(dirname "$(dirname "$src")")"
    fi
    [ -n "$B" ] || { echo "[work] FATAL: no bundle given and work.env has no [bundle] header (host mode) — pass the bundle path"; exit 1; }
    case "$B" in /*) ;; *) B="$PWD/$B" ;; esac
    [ -f "$B/env/.env" ] || { echo "[work] FATAL: not a bundle (missing env/.env): $B"; exit 1; }
    echo "[work] scraping pod state into bundle: $B"
    HARVEST_INVOCATION_PWD="$ORIG_PWD" "$REPO_ROOT/deploy/harvest-config.sh" \
      --from-pod "${PROJECT}_db_1" --env-file "$PWD/deploy/work.env" "$B"
    echo "[work] scraped — the bundle is now the source of truth; next 'work -b $B' re-seeds a pod from it"
    exit 0
  fi

  # export-data / import-data: the POD-OWNED data stores (Phase C).
  # The work pod's memory data planes are HOST resources (turso dir mount,
  # LAN Postgres) — deliberately NOT scraped; only the artifacts volume is.
  if [ "$action" = export-data ]; then
    out="${1:-$HOME/supreme-mcp-tools-backups/work-artifacts-$(date +%Y%m%d-%H%M%S).tar.gz}"
    mkdir -p "$(dirname "$out")"
    vp=$(podman volume inspect "${PROJECT}_work_artifacts" --format '{{.Mountpoint}}' 2>/dev/null) \
      || { echo "[work] FATAL: volume ${PROJECT}_work_artifacts not found"; exit 1; }
    tar -C "$(dirname "$vp")" -czf "$out" "$(basename "$vp")" \
      && echo "[work] artifacts -> $out ($(wc -c < "$out") bytes)" \
      || { echo "[work] FATAL: artifacts export failed (tar error)"; exit 1; }
    cat <<'OWN'
[work] ownership map:
  pod-owned: state plane pgdata (users/masks/docs) -> `work backup`
  pod-owned: work_artifacts volume (large-memory blobs) -> exported above
  external:  turso file store, LAN Postgres, HF cache -> NOT scraped (host resources)
OWN
    exit 0
  fi
  if [ "$action" = import-data ]; then
    f="${1:-}"
    [ -f "$f" ] || { echo "usage: $0 work import-data <artifacts.tar.gz>"; exit 1; }
    if podman ps --format '{{.Names}}' | grep -q "^${PROJECT}_work_1$"; then
      echo "[work] FATAL: the env is running — 'work stop' first (the volume must not be in use)"; exit 1
    fi
    vp=$(podman volume inspect "${PROJECT}_work_artifacts" --format '{{.Mountpoint}}')
    tar -C "$(dirname "$vp")" -xzf "$f" \
      && echo "[work] artifacts imported — 'work start' to bring the env back" \
      || { echo "[work] FATAL: artifacts import failed (tar error)"; exit 1; }
    exit 0
  fi

  # non-boot actions live entirely off the env-generation below (E10: a -b
  # bundle must neither break them nor make them mutate work.env)
  case "$action" in
    # stop/start must cover the WHOLE cluster: the ui service is profile-gated
    # (compose stop ignores it without --profile) — stop it explicitly; sweep
    # the project prefix so nothing survives as a "partial" stop
    stop)
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.yml stop ) 2>/dev/null || true
      for c in $(podman ps --filter "name=${PROJECT}_" -q); do podman stop -t 10 "$c" >/dev/null 2>&1 || true; done
      exit 0 ;;
    start)
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.yml start ) 2>/dev/null || true
      [ -n "$(podman ps -a --filter "name=${PROJECT}_ui_1" -q)" ] && podman start "${PROJECT}_ui_1" >/dev/null 2>&1 || true
      # bounded health wait — a bare sleep made 'start ok' optimistic
      if wait_http http://127.0.0.1:8200/health 20; then
        echo "[work] central healthy"
      else
        echo "[work] WARNING: central not healthy yet — check: podman logs ${PROJECT}_work_1"
      fi ;;
    clean)
      echo "[work] this removes containers AND the state volume (users/masks) — type 'wipe' to confirm:"
      read -r ans; [ "$ans" = "wipe" ] || { echo "aborted"; exit 1; }
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.yml down -v ) 2>/dev/null || true
      podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
      for c in $(podman ps -a --filter "name=${PROJECT}_" -q); do podman rm -f "$c" >/dev/null 2>&1 || true; done
      echo "[work] fully wiped"; exit 0 ;;
    logs)  podman logs --tail "${2:-40}" "${PROJECT}_${1:-work}_1"; exit 0 ;;
    service)
      # per-container verb (plain podman — the same direct-podman pattern the
      # logs arm uses; podman-compose 1.3.0 per-service control is unreliable).
      # <name> takes a bare service name (db, work, ui) or a full container name.
      sub="${1:-}"; name="${2:-}"
      case "$sub" in start|stop|restart) ;; *)
        echo "usage: $0 work service <start|stop|restart> <name>"; exit 1 ;; esac
      [ -n "$name" ] || { echo "usage: $0 work service <start|stop|restart> <name>"; exit 1; }
      if podman ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
        c="$name"
      else
        c="${PROJECT}_${name}_1"
        podman ps -a --format '{{.Names}}' | grep -Fxq "$c" \
          || { echo "[work] FATAL: no container '$name' (tried '$c') — services: db, work, ui"; exit 1; }
      fi
      podman "$sub" "$c" >/dev/null \
        && echo "[work] $sub ok: $c" \
        || { echo "[work] FATAL: podman $sub $c failed"; exit 1; }
      exit 0 ;;
    ui)
      # work env must be up; bring up only the ui service on top of the existing pod
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.yml --profile ui up -d ui )
      printf "UI: "; curl -sf http://127.0.0.1:8400/ -o /dev/null -w '%{http_code}\n' || echo "not answering"
      exit 0 ;;
    status)
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.yml ps ) || true
      printf "central: "; curl -sf http://127.0.0.1:8200/health || echo "not answering"
      echo; exit 0 ;;
  esac

  # bundle mode: work.env is REGENERATED from the bundle on every up (the bundle
  # is the source of truth — edit the bundle, re-run `startcluster work`).
  # host mode keeps the one-shot bootstrap below.
  if [ -n "$BUNDLE_DIR" ]; then
    # FIX (P1): bundle_node_env's third arg is an env-file PATH. Stash the
    # running env's POSTGRES_PASSWORD in a 0600 temp file so the regenerated
    # work.env keeps the already-initialized pgdata volume reachable.
    pw_tmpfile_from deploy/work.env
    [ -f deploy/work.env ] && cp -p deploy/work.env "deploy/work.env.bak-$(date +%Y%m%d-%H%M%S)"
    bundle_node_env "$BUNDLE_DIR" work "$PW_TMPFILE" "" 0 > deploy/work.env
    chmod 600 deploy/work.env
    [ -n "$PW_TMPFILE" ] && rm -f "$PW_TMPFILE"
    echo "[work] work.env regenerated from the bundle (previous copy backed up)"
  elif [ ! -f deploy/work.env ]; then
    [ -f .env ] || { echo "[work] FATAL: no host .env to derive work.env from"; exit 1; }
    cp .env deploy/work.env
    {
      echo ""
      echo "# ---- state plane overrides (added by startcluster work) ----"
      echo "MCP_USERS_BACKEND=db"
      echo "MCP_STATE_BACKEND=db"
      echo "POSTGRES_HOST=127.0.0.1"
      echo "POSTGRES_PORT=5433"
      echo "POSTGRES_USER=mcp"
      # Generated ONCE on first boot; rotation = edit deploy/work.env, then
      # `startcluster work clean` (wipes state) + `up` to re-bootstrap.
      echo "POSTGRES_PASSWORD=work-$(python3 -c 'import secrets; print(secrets.token_urlsafe(16))')"
      echo "POSTGRES_DB=mcp"
    } >> deploy/work.env
    chmod 600 deploy/work.env
    echo "[work] deploy/work.env created from the host .env (data planes identical; state plane -> own pgvector)"
  fi

  # D7/D8 topology: stamp the chosen identity-plane engine; on turso, park
  # the pg block dormant (password survives the return trip — edge 2) and
  # activate the sqld state plane via the PUBLISHED loopback port (the node
  # is host-networked and cannot resolve podman DNS names).
  PREV_TOPO="$(work_stamp)"
  WORK_TOPO="${WORK_TOPO:-$(work_stamp)}"
  WORK_TOPO="${WORK_TOPO:-pg}"
  if [ ! -f deploy/work.env ]; then
    echo "[work] FATAL: work.env missing after env setup"; exit 1
  fi
  sed -i '/^\[topology\] /d' deploy/work.env
  echo "[topology] $WORK_TOPO" >> deploy/work.env
  if [ "$WORK_TOPO" = "turso" ]; then
    sed -i -E 's^(POSTGRES_[A-Z_]+=.*)^# [topology:pg-dormant] \1^' deploy/work.env
    sed -i '/^TURSO_DATABASE_URL=http:/d' deploy/work.env
    { echo ""; echo "# ---- state plane (turso era; pg block parked dormant above) ----";
      echo "TURSO_DATABASE_URL=http://127.0.0.1:9080"; } >> deploy/work.env
    echo "[work] identity plane -> turso (pg block parked dormant; pgdata volume kept)"
  fi

  # concrete compose (host paths inlined — no interpolation dependency)
  # POSIX sed (no GNU grep -oP — busybox/BSD-safe); falls back to /nonexistent
  # bundle mode: the freshly generated work.env carries the bundle's data-plane paths
  TURSO_SRC="$([ -n "$BUNDLE_DIR" ] && echo deploy/work.env || echo .env)"
  # dirname of the file: DSN via parameter expansion — E9: paths with spaces
  # must survive (the old `sed | xargs dirname` truncated at the first space)
  HOST_TURSO_DIR=""
  if [ -f "$TURSO_SRC" ]; then
    while IFS= read -r _dsn_line; do
      if [[ "$_dsn_line" =~ ^TURSO_DATABASE_URL=file:(.+)$ ]]; then
        case "${BASH_REMATCH[1]}" in
          */*) HOST_TURSO_DIR="${BASH_REMATCH[1]%/*}" ;;
          *)   HOST_TURSO_DIR="." ;;
        esac
        break
      fi
    done < "$TURSO_SRC"
  fi
  if [ -n "$BUNDLE_DIR" ]; then
    [ -f "$BUNDLE_DIR/identity/users.json" ] \
      || { echo "[work] FATAL: bundle missing identity/users.json (the daily driver needs its users)"; exit 1; }
    export HOST_CONFIG_DIR="$BUNDLE_DIR/identity"
  else
    export HOST_CONFIG_DIR="$CONFIG_DIR"
  fi
  export HOST_TURSO_DIR="${HOST_TURSO_DIR:-/nonexistent}"
  export HOST_HF_CACHE="${HOME}/.cache/huggingface"
  WORK_BASE=deploy/compose-work.yml
  [ "$WORK_TOPO" = "turso" ] && WORK_BASE=deploy/compose-work.turso.yml
  sed -e "s|\${HOST_CONFIG_DIR}|$HOST_CONFIG_DIR|g" \
      -e "s|\${HOST_TURSO_DIR}|$HOST_TURSO_DIR|g" \
      -e "s|\${HOST_HF_CACHE}|$HOST_HF_CACHE|g" \
      "$WORK_BASE" > deploy/compose-work.run.yml

  # bundle mode: mounts the bundle's config over the image's baked copies
  # (override file comes LAST in the -f chain — podman-compose unions volumes
  # and lets later same-target mounts win; probe-verified 1.3.0 rec_merge_one)
  BUNDLE_F=()
  if [ -n "$BUNDLE_DIR" ]; then
    specs=()
    # scoped nullglob: a missing/empty bundle subdir must produce NO mount
    # (a literal '*.json' host path would poison the compose file)
    shopt -s nullglob
    for f in "$BUNDLE_DIR"/config/*.json; do specs+=("work|$f|/app/config/$(basename "$f")|ro"); done
    for f in "$BUNDLE_DIR"/tools/*/config.json; do
      specs+=("work|$f|/app/tools/$(basename "$(dirname "$f")")/config.json|ro")
    done
    shopt -u nullglob
    write_bundle_compose deploy/compose-work.bundle.yml ${specs[@]+"${specs[@]}"}
    BUNDLE_F=(-f compose-work.bundle.yml)
  elif [ -f deploy/compose-work.bundle.yml ]; then
    # bare re-up of a bundle-managed env: re-apply the override recorded by
    # the last -b boot, or the node comes up on the image's baked sanitized
    # configs — tool ports shuffle and every authenticated call 401s
    # (found live 2026-09-29, canonical-ports test)
    BUNDLE_F=(-f compose-work.bundle.yml)
  fi

  # default action: up (rebuild) — stop the running work env first.
  # NOTE: the down is project-wide and removes the ui container too, so the
  # up below must bring ui back (8400 is part of the daily surface).
  echo "[work] stopping the running work env (volumes kept)..."
  ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml down ) 2>/dev/null || true
  podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
  # topology switch: remove the old engine's containers (podman-compose will
  # not recreate a definition-changed service over its old container — found
  # live on the gwork round trip); volumes are KEPT, the dormant engine's
  # data waits for the return trip
  if [ -n "$PREV_TOPO" ] && [ "$PREV_TOPO" != "$WORK_TOPO" ]; then
    echo "[work] topology switch $PREV_TOPO -> $WORK_TOPO: removing old engine containers (volumes kept)"
    podman rm -f "${PROJECT}_db_1" "${PROJECT}_work_1" >/dev/null 2>&1 || true
  fi
  DBPORT=5433; [ "$WORK_TOPO" = "turso" ] && DBPORT=9080
  echo "[work] preflight: canonical ports (8000-8005/8200), UI 8400 and db $DBPORT must be free (host launcher down?)..."
  BUSY=$(ss -ltn | grep -cE ":(8000|8001|8002|8003|8004|8005|8200|8400|$DBPORT)\b" || true)
  if [ "${BUSY:-0}" -ne 0 ]; then
    echo "[work] FATAL: canonical ports busy — stop the host launcher (startlauncher) first."; exit 1
  fi

  echo "[work] grabbing latest code..."
  git pull --ff-only || echo "[work] WARNING: pull failed — deploying local code"

  echo "[work] (re)building node image + starting work env..."
  # stamp the image with the real version (podman-compose passes env through
  # to build.args; without this the labels stay 'dev / unknown')
  export IMAGE_VERSION="${IMAGE_VERSION:-$(git describe --tags --always --dirty)}"
  export GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD)}"
  # compose reuses the tagged image if present — drop it so the build runs
  podman rmi -f localhost/mcp-node:latest localhost/mcp-work_work:latest \
              localhost/mcp-work-work_work:latest >/dev/null 2>&1 || true
  ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml \
      ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} --profile ui up -d db work ui )

  echo "[work] waiting for the node central on :8200..."
  if ! wait_http http://127.0.0.1:8200/health 50; then
    echo "[work] node not healthy — check: podman logs ${PROJECT}_work_1"; exit 1
  fi

  # one-time masks/inventory adoption (same hardened seeding as the test env:
  # load /app/.env or the exec runs json-mode and silently no-ops; merge the
  # bundle masks if the doc lacks them; restart the node when anything changed)
  seed_out=$(podman exec "${PROJECT}_work_1" python3 -c "
import sys, json
sys.path.insert(0, '/app'); sys.path.insert(0, '/app/tools')
from dotenv import load_dotenv; load_dotenv('/app/.env')
from tools.shared import state_docs, users_store
users_store.central_tokens()
src = None
try:
    src = json.load(open('/root/.config/supreme-mcp-tools/tools_config.json'))
except Exception:
    pass
doc = state_docs.load_doc(state_docs.DOC_TOOLS_CONFIG)
if src and (doc is None or not doc.get('disabled_tools')):
    if doc is None:
        state_docs.save_doc(state_docs.DOC_TOOLS_CONFIG, src)
        print('seeded')
    else:
        doc['disabled_tools'] = src.get('disabled_tools', {})
        state_docs.save_doc(state_docs.DOC_TOOLS_CONFIG, doc)
        print('merged')
else:
    print('present')
" 2>/dev/null || true)
    echo "[work] state-plane seed: ${seed_out:-exec failed}"
    case "$seed_out" in
      seeded|merged)
        echo "[work] restarting the node to apply the seeded masks..."
        ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml \
            ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} restart work )
        wait_http http://127.0.0.1:8200/health 40 || echo "[work] WARNING: node still settling"
        ;;
    esac

  echo "[work] UP — canonical surface:"
  curl -s http://127.0.0.1:8200/health; echo
  echo "  tools: 8000-8005 | central: 8200 | UI (opt.): startcluster work ui -> 8400"
}

# ---------- TEST environment ----------
test_env() { action="${1:-pg}"; shift || true
  PROJECT="$TEST_PROJECT"
  case "$action" in
    # podman-compose 1.3.0 doesn't auto-discover the per-topology files (it
    # only looks for compose.yml/compose.yaml/docker-compose.yml); the explicit
    # -f chain below mirrors the `up` line so stop/start/clean/status/logs
    # work the same way they always did
    # the cluster spans FOUR compose files (common + topology + lb + artifacts)
    # — stopping with only compose-common.yml left db/lb/minio running forever
    # ("partial" stops). Full chain + prefix sweep; start: db first, then all.
    stop)
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml \
          -f compose-lb.yml -f compose-artifacts.yml stop ) 2>/dev/null || true
      for c in $(podman ps --filter "name=${PROJECT}_" -q); do podman stop -t 10 "$c" >/dev/null 2>&1 || true; done
      exit 0 ;;
    start)
      [ -n "$(podman ps -a --filter "name=${PROJECT}_db_1" -q)" ] && podman start "${PROJECT}_db_1" >/dev/null 2>&1 || true
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml \
          -f compose-lb.yml -f compose-artifacts.yml start ) 2>/dev/null || true
      if wait_http http://127.0.0.1:18200/health 20; then
        echo "[test] node1 healthy"
      else
        echo "[test] WARNING: node1 not healthy yet — check: podman logs ${PROJECT}_node1_1"
      fi
      exit 0 ;;
    backup)
      ts=$(date +%Y%m%d-%H%M%S)
      out="${HOME}/supreme-mcp-tools-backups/test-pg-$ts.sql"
      mkdir -p "$(dirname "$out")"
      podman exec "${PROJECT}_db_1" pg_dump -U mcp -d mcp > "$out" \
        && echo "[test] backup -> $out ($(wc -c < "$out") bytes)" \
        || { echo "[test] backup FAILED — is the pg-topology env running?"; exit 1; }
      exit 0 ;;
    restore)
      f="${1:-}"
      [ -f "$f" ] || { echo "usage: $0 test restore <backup.sql>"; exit 1; }
      echo "[test] RESTORE wipes the pgdata volume and imports $f — type 'wipe' to confirm:"
      read -r ans; [ "$ans" = "wipe" ] || { echo "aborted"; exit 1; }
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml down ) 2>/dev/null || true
      podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
      podman rm -f "${PROJECT}_node1_1" "${PROJECT}_node2_1" "${PROJECT}_db_1" "${PROJECT}_lb_1" 2>/dev/null || true
      podman volume rm -f "${PROJECT}_pgdata" 2>/dev/null || true
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml up -d db )
      sleep 8
      podman exec -i "${PROJECT}_db_1" psql -U mcp -d mcp < "$f" >/dev/null \
        && echo "[test] restore imported — starting nodes..." \
        || { echo "[test] restore import FAILED"; exit 1; }
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml \
          -f compose-lb.yml up -d node1 node2 lb )
      if wait_http http://127.0.0.1:18200/health 40; then
        echo "[test] restored and healthy"
      else
        echo "[test] nodes not healthy — podman logs ${PROJECT}_node1_1"; exit 1
      fi
      exit 0 ;;
    scrape)
      B="${1:-}"
      if [ -z "$B" ] && [ -f deploy/node.env ]; then
        src=$(sed -n 's/^# \[bundle\] GENERATED by startcluster from //p' deploy/node.env | head -1)
        [ -n "$src" ] && B="$(dirname "$(dirname "$src")")"
      fi
      [ -n "$B" ] || { echo "[test] FATAL: no bundle given and node.env has no [bundle] header (hand-written) — pass the bundle path"; exit 1; }
      case "$B" in /*) ;; *) B="$PWD/$B" ;; esac
      [ -f "$B/env/.env" ] || { echo "[test] FATAL: not a bundle (missing env/.env): $B"; exit 1; }
      echo "[test] scraping pod state into bundle: $B"
      HARVEST_INVOCATION_PWD="$ORIG_PWD" "$REPO_ROOT/deploy/harvest-config.sh" \
        --from-pod "${PROJECT}_db_1" --env-file "$PWD/deploy/node.env" "$B"
      exit 0 ;;
    export-data)
      out="${1:-$HOME/supreme-mcp-tools-backups/test-minio-$(date +%Y%m%d-%H%M%S)}"
      mkdir -p "$out"
      [ -f deploy/.env ] && . ./deploy/.env
      podman exec -e MC_HOST_local="http://mcp-artifacts:${MINIO_ROOT_PASSWORD}@127.0.0.1:9000" \
        "${PROJECT}_minio_1" sh -c 'rm -rf /tmp/export && mc mirror --overwrite local/memory-artifacts /tmp/export' \
        || { echo "[test] FATAL: MinIO not running or bucket missing"; exit 1; }
      podman cp "${PROJECT}_minio_1:/tmp/export/." "$out/" \
        || { echo "[test] FATAL: copying objects out of MinIO failed"; exit 1; }
      echo "[test] MinIO objects -> $out"
      cat <<'OWN'
[test] ownership map:
  pod-owned: pgdata (state plane + any data-plane tables) -> `test backup`
  pod-owned: miniodata (shared artifacts)  -> exported above
  pod-owned: sqldata (turso topology)      -> advanced: tar the volume (not automated)
  external:  anything node.env points at beyond the compose services -> NOT scraped
OWN
      exit 0 ;;
    import-data)
      d="${1:-}"
      [ -d "$d" ] || { echo "usage: $0 test import-data <minio-export-dir>"; exit 1; }
      [ -f deploy/.env ] && . ./deploy/.env
      podman cp "$d/." "${PROJECT}_minio_1:/tmp/import" \
        && podman exec -e MC_HOST_local="http://mcp-artifacts:${MINIO_ROOT_PASSWORD}@127.0.0.1:9000" \
             "${PROJECT}_minio_1" sh -c 'mc mirror --overwrite /tmp/import local/memory-artifacts && rm -rf /tmp/import' \
        && echo "[test] MinIO objects imported" \
        || { echo "[test] FATAL: MinIO import failed"; exit 1; }
      exit 0 ;;
    clean)
      ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml down ) 2>/dev/null || true
      podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
      podman rm -f "${PROJECT}_node1_1" "${PROJECT}_node2_1" "${PROJECT}_db_1" 2>/dev/null || true
      echo "[test] removed (volumes pgdata/sqldata kept)"; exit 0 ;;
    logs)  podman logs --tail "${2:-40}" "${PROJECT}_${1:-node1}_1"; exit 0 ;;
    service)
      # per-container verb (plain podman, consistent with the logs arm;
      # podman-compose 1.3.0 per-service control is unreliable). <name> takes
      # a bare service name (db, node1, node2, lb, minio) or a full container name.
      sub="${1:-}"; name="${2:-}"
      case "$sub" in start|stop|restart) ;; *)
        echo "usage: $0 test service <start|stop|restart> <name>"; exit 1 ;; esac
      [ -n "$name" ] || { echo "usage: $0 test service <start|stop|restart> <name>"; exit 1; }
      if podman ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
        c="$name"
      else
        c="${PROJECT}_${name}_1"
        podman ps -a --format '{{.Names}}' | grep -Fxq "$c" \
          || { echo "[test] FATAL: no container '$name' (tried '$c') — services: db, node1, node2, lb, minio"; exit 1; }
      fi
      podman "$sub" "$c" >/dev/null \
        && echo "[test] $sub ok: $c" \
        || { echo "[test] FATAL: podman $sub $c failed"; exit 1; }
      exit 0 ;;
    status)
      podman pod ls | grep "$PROJECT" || echo "[test] not created"
      printf "node1: "; curl -sf http://127.0.0.1:18200/health || echo "down"
      echo; printf "node2: "; curl -sf http://127.0.0.1:19200/health || echo "down"
      echo; exit 0 ;;
  esac
  TOPOLOGY="$action"
  case "$TOPOLOGY" in
    pg)          TOPO_FILE="compose-pg.yml";     WITH_DB=1 ;;
    turso)       TOPO_FILE="compose-turso.yml";  WITH_DB=1 ;;
    external-pg) TOPO_FILE="";                   WITH_DB=0 ;;
    external-turso) TOPO_FILE="";                WITH_DB=0 ;;
    *) echo "unknown test topology '$TOPOLOGY'"; exit 1 ;;
  esac

  # bundle mode: node.env is GENERATED from the bundle per topology (identity
  # plane copied, state plane rewritten, file: data-plane lines stripped unless
  # --keep-dataplanes). users.json mounts into node1 ONLY: its one-time import
  # seeds the shared DB; node2 reads the same DB (no concurrent-import race).
  if [ -n "$BUNDLE_DIR" ]; then
    # FIX (P1): bundle_node_env's third arg is an env-file PATH, not a value —
    # stash the current POSTGRES_PASSWORD in a 0600 temp env file so the pg
    # topology keeps its already-initialized pgdata volume reachable.
    pw_tmpfile_from deploy/node.env
    [ -f deploy/node.env ] && cp -p deploy/node.env "deploy/node.env.bak-$(date +%Y%m%d-%H%M%S)"
    bundle_node_env "$BUNDLE_DIR" "$TOPOLOGY" "$PW_TMPFILE" "$(minio_password)" "$KEEP_DATAPLANES" > deploy/node.env
    chmod 600 deploy/node.env
    [ -n "$PW_TMPFILE" ] && rm -f "$PW_TMPFILE"
    echo "[test] node.env generated from the bundle (previous copy backed up)"
    specs=()
    [ -f "$BUNDLE_DIR/identity/users.json" ] \
      && specs+=("node1|$BUNDLE_DIR/identity/users.json|/root/.config/supreme-mcp-tools/users.json|ro")
    [ -f "$BUNDLE_DIR/identity/tools_config.json" ] \
      && specs+=("node1|$BUNDLE_DIR/identity/tools_config.json|/root/.config/supreme-mcp-tools/tools_config.json|ro")
    # scoped nullglob: a missing/empty bundle subdir must produce NO mount
    # (a literal '*.json' host path would poison the compose file)
    shopt -s nullglob
    for f in "$BUNDLE_DIR"/config/*.json; do
      specs+=("node1|$f|/app/config/$(basename "$f")|ro" "node2|$f|/app/config/$(basename "$f")|ro")
    done
    for f in "$BUNDLE_DIR"/tools/*/config.json; do
      t="$(basename "$(dirname "$f")")"
      specs+=("node1|$f|/app/tools/$t/config.json|ro" "node2|$f|/app/tools/$t/config.json|ro")
    done
    shopt -u nullglob
    write_bundle_compose deploy/compose-test.bundle.yml ${specs[@]+"${specs[@]}"}
    [ -f "$BUNDLE_DIR/identity/users.json" ] || \
      echo "[test] WARNING: bundle has no identity/users.json — cluster starts without users (break-glass central key only)"
  fi

  echo "[test:$TOPOLOGY] grabbing latest code..."
  git pull --ff-only || true
  cleanup_cluster() {
    ( cd deploy && podman-compose -p "$PROJECT" down ) 2>/dev/null || true
    podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
    podman rm -f ${PROJECT}_node1_1 ${PROJECT}_node2_1 ${PROJECT}_db_1 2>/dev/null || true
  }
  echo "[test] tearing down old cluster (volumes kept)..."
  cleanup_cluster
  cd deploy
  export IMAGE_VERSION="${IMAGE_VERSION:-$(git describe --tags --always --dirty)}"
  export GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD)}"
  # compose-common.yml interpolates ${HOST_HF_CACHE} (local embedding model
  # cache — plan F4 fix); unset would render an empty mount source
  export HOST_HF_CACHE="${HOST_HF_CACHE:-${HOME}/.cache/huggingface}"
  podman rmi -f localhost/${PROJECT}_node1:latest localhost/${PROJECT}_node2:latest 2>/dev/null || true
  echo "[test] starting..."
  # MinIO credentials live in deploy/.env (gitignored) — compose reads the
  # same file for ${MINIO_ROOT_PASSWORD} interpolation in compose-artifacts.yml.
  BUNDLE_F=()
  if [ -n "$BUNDLE_DIR" ]; then BUNDLE_F=(-f compose-test.bundle.yml)
  elif [ -f compose-test.bundle.yml ]; then
    # bare re-up of a bundle-managed bench: keep the recorded bundle mounts
    # (same failure class as the work env, found live 2026-09-29)
    BUNDLE_F=(-f compose-test.bundle.yml)
  fi
  if [ "$WITH_DB" = 1 ]; then
    # NOTE: cwd is deploy/ here (the flow cd'd in for the compose calls) —
    # reference the env file absolutely or the check self-fails (latent bug
    # that never fired while the stale root wrapper was in use)
    if [ ! -f "$REPO_ROOT/deploy/.env" ] || ! grep -q '^MINIO_ROOT_PASSWORD=..' "$REPO_ROOT/deploy/.env"; then
      echo "[test] FATAL: deploy/.env missing or MINIO_ROOT_PASSWORD unset —"
      echo "       add 'MINIO_ROOT_PASSWORD=<generate-one>' to deploy/.env (see deploy/README.md)"
      exit 1
    fi
    . "$REPO_ROOT/deploy/.env"
    podman-compose -p "$PROJECT" -f compose-common.yml -f "$TOPO_FILE" \
      -f compose-lb.yml -f compose-artifacts.yml ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} up -d db node1 node2 lb minio
  else
    # external-* topologies: no embedded state plane AND no shared artifacts
    podman-compose -p "$PROJECT" -f compose-common.yml \
      -f compose-lb.yml ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} up -d node1 node2 lb
  fi
  ok=0; wait_http http://127.0.0.1:18200/health 40 && ok=1
  [ "$ok" = 1 ] && echo "[test] UP: node1 :18200, node2 :19200, LB :18080 ($TOPOLOGY)" \
                || { echo "[test] nodes not healthy — podman logs ${PROJECT}_node1_1"; exit 1; }

  # bundle mode: seed the shared state plane AFTER the nodes are healthy.
  # The exec must load /app/.env itself — a bare `podman exec` shell has no
  # dotenv env and would silently run in json mode (proven live). If the
  # nodes' boot sync created the doc WITHOUT the bundle's masks (race) or the
  # doc is missing, seed/merge + restart the nodes so masks apply immediately.
  if [ -n "$BUNDLE_DIR" ] && [ -f "$BUNDLE_DIR/identity/tools_config.json" ]; then
    seed_out=$(podman exec "${PROJECT}_node1_1" python3 -c "
import sys, json
sys.path.insert(0, '/app'); sys.path.insert(0, '/app/tools')
from dotenv import load_dotenv; load_dotenv('/app/.env')
from tools.shared import state_docs, users_store
users_store.central_tokens()
src = None
try:
    src = json.load(open('/root/.config/supreme-mcp-tools/tools_config.json'))
except Exception:
    pass
doc = state_docs.load_doc(state_docs.DOC_TOOLS_CONFIG)
if src and (doc is None or not doc.get('disabled_tools')):
    if doc is None:
        state_docs.save_doc(state_docs.DOC_TOOLS_CONFIG, src)
        print('seeded')
    else:
        doc['disabled_tools'] = src.get('disabled_tools', {})
        state_docs.save_doc(state_docs.DOC_TOOLS_CONFIG, doc)
        print('merged')
else:
    print('present')
" 2>/dev/null || true)
    echo "[test] state-plane seed: ${seed_out:-exec failed}"
    case "$seed_out" in
      seeded|merged)
        echo "[test] restarting nodes to apply the seeded masks..."
        ( cd deploy && podman-compose -p "$PROJECT" -f compose-common.yml -f compose-pg.yml \
            -f compose-lb.yml ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} restart node1 node2 )
        wait_http http://127.0.0.1:18200/health 40 || echo "[test] WARNING: nodes still settling"
        ;;
    esac
  fi

  # MinIO artifacts bucket (credentials from deploy/.env — NOT hardcoded).
  # MC_HOST_* env form avoids -p on the command line; stat-or-create is one
  # idempotent shot (no unreliable `mc ls`-to-detect-absence dance).
  if [ "$WITH_DB" = 1 ]; then
    wait_http http://127.0.0.1:19000/minio/health/live 10 \
      || echo "[test] WARNING: MinIO health endpoint not answering yet"
    podman exec -e MC_HOST_local="http://mcp-artifacts:${MINIO_ROOT_PASSWORD}@127.0.0.1:9000" \
      ${PROJECT}_minio_1 sh -c 'mc stat local/memory-artifacts >/dev/null 2>&1 || mc mb local/memory-artifacts' \
      && echo "[test] MinIO bucket 'memory-artifacts' ready" \
      || echo "[test] WARNING: MinIO bucket bootstrap failed — check MINIO_ROOT_PASSWORD in deploy/.env"
  fi
}

# ---------- cluster registry (custom clusters, plans V4) ----------
REGISTRY_DIR="$REPO_ROOT/deploy/clusters.d"

registry_names() {
  [ -d "$REGISTRY_DIR" ] || return 0
  for f in "$REGISTRY_DIR"/*.json; do [ -f "$f" ] && basename "$f" .json; done
  return 0
}

registry_field() { # field name -> value ("" when absent)
  [ -f "$REGISTRY_DIR/$2.json" ] || return 0
  python3 -c "import json,sys; print(json.load(open(sys.argv[2])).get(sys.argv[1]) or '')" \
    "$1" "$REGISTRY_DIR/$2.json"
}

registry_is_known() { [ -f "$REGISTRY_DIR/$1.json" ]; }

valid_cluster_name() {
  echo "$1" | grep -qE '^[a-z][a-z0-9-]{2,15}$'
}


# ---------- custom clusters (registry, plans V4) ----------
CANONICAL_TOOL_PORT() { # tool -> container port (config/ports.json assignments)
  case "$1" in
    databasemcp) echo 8000 ;; webmcp) echo 8001 ;; simplemcp) echo 8002 ;;
    convertermcp) echo 8003 ;; ragmcp) echo 8004 ;; memorymcp) echo 8005 ;;
    *) echo 0 ;;
  esac
}

create_cluster() { # name [flags]: --nodes 1|2 --topology pg|turso --tools "..." --port-base N --layout custom|work -b <bundle>
  local name="" nodes=1 topology="pg" tools="simplemcp convertermcp webmcp" port_base="" bundle="" layout="custom"
  while [ $# -gt 0 ]; do
    case "$1" in
      --nodes) nodes="$2"; shift 2 ;;
      --topology) topology="$2"; shift 2 ;;
      --tools) tools="$2"; shift 2 ;;
      --port-base) port_base="$2"; shift 2 ;;
      --layout) layout="$2"; shift 2 ;;
      -b) bundle="$2"; shift 2 ;;
      *) name="$1"; shift ;;
    esac
  done
  [ -n "$name" ] || { echo "FATAL: create needs a cluster name"; exit 1; }
  echo "$name" | grep -qE '^[a-z][a-z0-9-]{2,15}$' \
    || { echo "FATAL: invalid name '$name' — lowercase letters/digits/-, 3-16 chars, starts with a letter"; exit 1; }
  [ "$name" = "work" ] || [ "$name" = "test" ] \
    && { echo "FATAL: '$name' is a built-in cluster"; exit 1; }
  registry_is_known "$name" && { echo "FATAL: cluster '$name' already exists in the registry"; exit 1; }
  case "$nodes" in 1|2) ;; *) echo "FATAL: --nodes must be 1 or 2"; exit 1 ;; esac
  case "$topology" in pg|turso) ;; *) echo "FATAL: create supports pg|turso topologies (external via -b bundle)"; exit 1 ;; esac
  case "$layout" in custom|work) ;; *) echo "FATAL: --layout must be custom or work"; exit 1 ;; esac
  if [ "$layout" = "work" ]; then
    # work-shaped ground (grounds plan §4 D7/D8): single node on the
    # CANONICAL ports with host networking — the three-way mutual exclusion
    # (host launcher XOR mcp-work XOR gwork) is enforced here, loudly.
    [ "$nodes" = "1" ] || { echo "FATAL: work layout is single-node (--nodes 1)"; exit 1; }
    [ -n "$bundle" ] || { echo "FATAL: work layout requires a config bundle (-b)"; exit 1; }
    local dbport=5433; [ "$topology" = "turso" ] && dbport=9080
    local cp
    for cp in 8000 8001 8002 8003 8004 8005 8200 "$dbport"; do
      ss -ltn | grep -q ":$cp\b" \
        && { echo "FATAL: canonical port $cp is held — work-layout grounds need the host launcher, mcp-work and other gwork grounds DOWN (mutual exclusion)"; exit 1; }
    done
  fi
  for t in $tools; do
    [ "$(CANONICAL_TOOL_PORT "$t")" = "0" ] && { echo "FATAL: unknown tool '$t'"; exit 1; }
  done
  [ -n "$bundle" ] && [ ! -f "$bundle/env/.env" ] \
    && { echo "FATAL: bundle missing env/.env: $bundle"; exit 1; }

  # port_base: given, or auto-pick the lowest free base in 20000-20999.
  # work layout skips the window entirely — its ports are the canonical ones,
  # already checked free above.
  if [ "$layout" = "work" ]; then port_base=""; fi
  if [ "$layout" != "work" ]; then
    if [ -z "$port_base" ]; then
      local base busy p
      for base in $(seq 20000 100 20900); do
        busy=0
        local win2=""
        [ "$nodes" = "2" ] && win2=$(seq "$((base + 1002))" "$((base + 1007))")
        for p in "$base" "$((base + 1000))" $(seq "$((base + 2))" "$((base + 7))") $win2; do
          ss -ltn | grep -q ":$p\b" && busy=1 && break
          for other in $(registry_names); do
            grep -qE ": ?$p\b" "$REGISTRY_DIR/$other.json" 2>/dev/null && busy=1 && break
          done
          [ "$busy" = 1 ] && break
        done
        [ "$busy" = 0 ] && { port_base=$base; break; }
      done
      [ -n "$port_base" ] || { echo "FATAL: no free port base in 20000-20999"; exit 1; }
    fi
    local p win2=""
    [ "$nodes" = "2" ] && win2=$(seq "$((port_base + 1002))" "$((port_base + 1007))")
    for p in "$port_base" "$((port_base + 1000))" $(seq "$((port_base + 2))" "$((port_base + 7))") $win2; do
      ss -ltn | grep -q ":$p\b" && { echo "FATAL: port $p is already in use"; exit 1; }
      for other in $(registry_names); do
        [ "$other" = "$name" ] && continue
        grep -qE ": ?$p\b" "$REGISTRY_DIR/$other.json" 2>/dev/null \
          && { echo "FATAL: port $p already mapped by cluster '$other'"; exit 1; }
      done
    done
  fi

  local host_ports=""
  if [ "$layout" != "work" ]; then
    for t in $tools; do
      host_ports="$host_ports $t:$((port_base + 2 + $(CANONICAL_TOOL_PORT "$t") - 8000))"
    done
  fi

  # registry + env (create does NOT boot — cheap and reversible)
  mkdir -p "$REGISTRY_DIR"
  python3 - "$name" "$nodes" "$topology" "$tools" "$port_base" "$bundle" "$layout" <<'PY'
import json, sys, datetime
name, nodes, topology, tools, port_base, bundle, layout = sys.argv[1:8]
tool_list = tools.split()
PORT = {"databasemcp": 8000, "webmcp": 8001, "simplemcp": 8002,
        "convertermcp": 8003, "ragmcp": 8004, "memorymcp": 8005}
if layout == "work":
    # canonical ports; recorded for attribution/probes (host networking)
    ports = {"central": 8200, "central2": None, "db": 5433 if topology == "pg" else 9080,
             "tools": {t: PORT[t] for t in tool_list}}
else:
    ports = {"central": int(port_base),
             "central2": int(port_base) + 1000 if nodes == "2" else None,
             # +2: tools occupy the RESERVED base+2..base+7 window (central owns
             # base). Without the offset databasemcp (container 8000) collides
             # with central — never hit before grounds (clusters.d was empty).
             "tools": {t: int(port_base) + 2 + PORT[t] - 8000 for t in tool_list}}
    if nodes == "2":
        # node2 = node1 + 1000 (mirrors central2) — its tools must not
        # republish node1's ports
        ports["tools2"] = {t: int(port_base) + 1002 + PORT[t] - 8000
                           for t in tool_list}
entry = {"name": name, "nodes": int(nodes), "topology": topology,
         "layout": layout,
         "tools": tool_list,
         **({"port_base": int(port_base)} if port_base else {}),
         "ports": ports,
         "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
if bundle and bundle != "none":
    entry["bundle"] = bundle
with open(f"deploy/clusters.d/{name}.json", "w") as f:
    json.dump(entry, f, indent=2)
print(f"[create] registry -> deploy/clusters.d/{name}.json")
PY
  if [ "$bundle" != "none" ] && [ -n "$bundle" ]; then
    python3 "$REPO_ROOT/deploy/bundle_node_env.py" "$bundle" "$topology" > "deploy/$name.env"
  else
    cp deploy/node.env "deploy/$name.env"
    {
      echo ""
      echo "# ---- state plane (custom cluster, $topology topology) ----"
      echo "MCP_USERS_BACKEND=db"
      echo "MCP_STATE_BACKEND=db"
    } >> "deploy/$name.env"
  fi
  echo "[create] env -> deploy/$name.env"
  if [ "$layout" = "work" ]; then
    # work layout (grounds plan §4 D8): the node shares the HOST netns and
    # cannot resolve podman DNS names — rewrite the state-plane addresses
    # from bridge form (db:5432 / db:8080) to the published loopback ports.
    # The publish in the generated compose is LOAD-BEARING (see the D8 trap:
    # do not copy compose-turso.yml's no-publish shape).
    sed -i -e 's|^POSTGRES_HOST=db$|POSTGRES_HOST=127.0.0.1|' \
           -e 's|^TURSO_DATABASE_URL=http://db:8080$|TURSO_DATABASE_URL=http://127.0.0.1:9080|' \
           -e '/^POSTGRES_PORT=/d' "deploy/$name.env"
    if [ "$topology" = "pg" ]; then
      echo "POSTGRES_PORT=5433" >> "deploy/$name.env"
    fi
    echo "[create] state plane rewritten for host networking (127.0.0.1)"
  fi
  echo "[create] cluster '$name' ready: $nodes node(s), $topology topology, ports:$host_ports"
  echo "[create] boot it with: startcluster $name up"
}

rm_cluster() { # name
  local name="$1"
  registry_is_known "$name" || { echo "FATAL: cluster '$name' is not in the registry"; exit 1; }
  if podman ps --format "{{.Names}}" | grep -q "^${name}_"; then
    echo "FATAL: cluster '$name' is running — stop it first"; exit 1
  fi
  echo "rm cluster $name — removes containers, volumes, registry entry and env file — type '$name' to confirm:"
  read -r ans; [ "$ans" = "$name" ] || { echo "aborted"; exit 1; }
  ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" down -v ) 2>/dev/null || true
  podman pod rm -f "pod_$name" >/dev/null 2>&1 || true
  for c in $(podman ps -a --filter "name=${name}_" -q); do podman rm -f "$c" >/dev/null 2>&1 || true; done
  for v in $(podman volume ls --format "{{.Name}}" | grep "^${name}_"); do podman volume rm -f "$v" >/dev/null 2>&1 || true; done
  rm -f "$REGISTRY_DIR/$name.json" "deploy/$name.env" "deploy/compose-$name.run.yml"
  echo "[rm] cluster '$name' fully removed"
}

custom_up() { # name
  local name="$1"
  python3 "$REPO_ROOT/deploy/gen_custom_compose.py" "$REGISTRY_DIR/$name.json" \
    > "deploy/compose-$name.run.yml" 2>/dev/null \
    || { echo "[custom] FATAL: cannot generate compose — registry entry broken"; exit 1; }
  # the generated compose interpolates ${HOST_HF_CACHE} (embedding model
  # cache — plan F4/C2 fix); unset would render an empty mount source
  export HOST_HF_CACHE="${HOST_HF_CACHE:-${HOME}/.cache/huggingface}"
  # C3 (grounds plan): chain the bundle identity/config override so the
  # one-time users.json import actually runs — `create -b` only writes the
  # env file. Regenerated on EVERY up from the recorded bundle path, so a
  # bare re-up keeps the chain (same failure class as the work env's bare
  # re-up, fixed 138b471).
  local bundle nodes f t topology
  bundle=$(python3 -c "import json;print(json.load(open('$REGISTRY_DIR/$name.json')).get('bundle',''))")
  nodes=$(python3 -c "import json;print(json.load(open('$REGISTRY_DIR/$name.json'))['nodes'])")
  topology=$(python3 -c "import json;print(json.load(open('$REGISTRY_DIR/$name.json'))['topology'])")
  BUNDLE_F=()
  if [ -n "$bundle" ]; then
    local specs=()
    [ -f "$bundle/identity/users.json" ] \
      && specs+=("node1|$bundle/identity/users.json|/root/.config/supreme-mcp-tools/users.json|ro")
    [ -f "$bundle/identity/tools_config.json" ] \
      && specs+=("node1|$bundle/identity/tools_config.json|/root/.config/supreme-mcp-tools/tools_config.json|ro")
    # EPHEMERAL DATA PLANE (grounds design §3): the bundle strips file: data
    # lines, which would otherwise leave memorymcp/ragmcp on NullVectorStore
    # (silent no-ops — found live on the first g1-pg boot). A container-local
    # turso file gives them a REAL vector/SQL plane that dies with the ground:
    # zero host mounts, zero real-data risk.
    # pg topology ONLY: in the turso topology TURSO_DATABASE_URL already
    # points at the sqld state plane, which then also serves the data plane.
    [ "$topology" = "pg" ] \
      && specs+=("env|node1|TURSO_DATABASE_URL=file:/app/ground-memory.db")
    shopt -s nullglob
    for f in "$bundle"/config/*.json; do
      specs+=("node1|$f|/app/config/$(basename "$f")|ro")
      [ "$nodes" = "2" ] && specs+=("node2|$f|/app/config/$(basename "$f")|ro")
    done
    for f in "$bundle"/tools/*/config.json; do
      t="$(basename "$(dirname "$f")")"
      specs+=("node1|$f|/app/tools/$t/config.json|ro")
      [ "$nodes" = "2" ] && specs+=("node2|$f|/app/tools/$t/config.json|ro")
    done
    shopt -u nullglob
    if [ ${#specs[@]} -gt 0 ]; then
      write_bundle_compose "deploy/compose-$name.bundle.yml" ${specs[@]+"${specs[@]}"}
      BUNDLE_F=(-f "compose-$name.bundle.yml")
      echo "[custom] $name: bundle override chained ($bundle)"
    else
      echo "[custom] WARNING: bundle has no identity/config files — cluster starts without users (break-glass central key only)"
    fi
  fi
  local p
  for p in $(grep -oE '^\s*- [0-9]+:' "deploy/compose-$name.run.yml" | grep -oE '[0-9]+' | sort -nu); do
    ss -ltn | grep -q ":$p\b" && { echo "FATAL: port $p taken since create — remove the holder or recreate the cluster"; exit 1; }
  done
  ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" ${BUNDLE_F[@]+"${BUNDLE_F[@]}"} up -d )
  local central
  central=$(python3 -c "import json;print(json.load(open('$REGISTRY_DIR/$name.json'))['ports']['central'])")
  wait_http "http://127.0.0.1:$central/health" 40 \
    && echo "[custom] $name up — central on :$central" \
    || echo "[custom] WARNING: central not healthy yet — check: podman logs ${name}_node1_1"
}

custom_verb() { # name action [args]
  local name="$1" action="$2"; shift 2
  case "$action" in
    up)    custom_up "$name" ;;
    start) ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" start ) 2>/dev/null || true ;;
    stop)  ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" stop ) 2>/dev/null || true
           for c in $(podman ps --filter "name=${name}_" -q); do podman stop -t 10 "$c" >/dev/null 2>&1 || true; done ;;
    status) podman ps -a --filter "name=${name}_" --format "{{.Names}}: {{.Status}}" ;;
    logs)  podman logs --tail "${3:-40}" "${name}_${2:-node1}_1" ;;
    backup)
      ts=$(date +%Y%m%d-%H%M%S)
      out="${HOME}/supreme-mcp-tools-backups/$name-pg-$ts.sql"
      mkdir -p "$(dirname "$out")"
      podman exec "${name}_db_1" pg_dump -U mcp -d mcp > "$out" \
        && echo "[custom] backup -> $out ($(wc -c < "$out") bytes)" \
        || { echo "[custom] backup FAILED — is the cluster running?"; exit 1; } ;;
    restore)
      f="${3:-}"; [ -f "$f" ] || { echo "usage: startcluster $name restore <dump.sql>"; exit 1; }
      echo "[custom] RESTORE wipes the pgdata volume and imports $f — type 'wipe' to confirm:"
      read -r ans; [ "$ans" = "wipe" ] || { echo "aborted"; exit 1; }
      ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" stop ) 2>/dev/null || true
      podman volume rm -f "${name}_pgdata" 2>/dev/null || true
      ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" up -d db )
      sleep 8
      podman exec -i "${name}_db_1" psql -U mcp -d mcp < "$f" >/dev/null \
        && echo "[custom] restore imported" \
        || { echo "[custom] restore import FAILED"; exit 1; }
      ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" up -d ) ;;
    scrape)
      bundle="${3:-}"; [ -n "$bundle" ] \
        || { echo "[custom] scrape needs a bundle path (custom envs carry no [bundle] header when created without -b)"; exit 1; }
      "$REPO_ROOT/deploy/harvest-config.sh" --from-pod "${name}_db_1" --env-file "$PWD/deploy/$name.env" "$bundle" ;;
    export-data) echo "[custom] no pod-owned stores (v1 customs) — data planes are external" ;;
    clean)
      ( cd deploy && podman-compose -p "$name" -f "compose-$name.run.yml" down ) 2>/dev/null || true
      podman pod rm -f "pod_$name" >/dev/null 2>&1 || true
      for c in $(podman ps -a --filter "name=${name}_" -q); do podman rm -f "$c" >/dev/null 2>&1 || true; done
      echo "[custom] $name cleaned (volumes kept)" ;;
    *) echo "unknown custom verb '$action'"; exit 1 ;;
  esac
}

# ---------- dispatch ----------
# ---------- dispatch ----------
# --config-bundle / --keep-dataplanes may appear anywhere; everything else is positional
positional=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config-bundle)
      [ $# -ge 2 ] || { echo "FATAL: --config-bundle needs a bundle path (or 'host')"; exit 1; }
      BUNDLE_RAW="$2"; shift 2 ;;
    --config-bundle=*) BUNDLE_RAW="${1#*=}"; shift ;;
    -b)
      [ $# -ge 2 ] || { echo "FATAL: -b needs a bundle path (or 'host')"; exit 1; }
      BUNDLE_RAW="$2"; shift 2 ;;
    --keep-dataplanes) KEEP_DATAPLANES=1; shift ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- ${positional[@]+"${positional[@]}"}

case "${1:-}" in
  "")    usage; exit 0 ;;
work)  shift
       # E10: the bundle flag is honored by the BOOT path only — resolving a
       # missing bundle must not break stop/status/logs/..., and a valid one
       # must not make them rewrite work.env. (Default action = up.)
       # D7: `work pg|turso` = topology boot (identity-plane engine, grounds
       # plan §4); plain `work up` boots with the stamped topology (default pg).
       WORK_TOPO=""
       case "${1:-}" in
         pg|turso) WORK_TOPO="$1"; resolve_bundle; shift ;;
         up|"")    resolve_bundle ;;
       esac
       work_env "$@" ;;
  test)  shift
         # same E10 rule: only the topology boots consume the bundle
         # (no topology arg = default `pg` = a boot action)
         case "${1:-}" in ""|pg|turso|external-pg|external-turso) resolve_bundle ;; esac
         test_env "$@" ;;
  pg|turso|external-pg|external-turso) resolve_bundle; test_env "$@" ;;   # legacy shorthand = test
  harvest) shift; HARVEST_INVOCATION_PWD="$ORIG_PWD" exec "$REPO_ROOT/deploy/harvest-config.sh" "$@" ;;
  bundle-node-env)   # hidden: print the env a bundle generates (tests/debug, no podman)
    shift
    # the bundle may come as a positional OR via -b/--config-bundle
    case "${1:-}" in
      "")
        echo "usage: startcluster bundle-node-env [<bundle>|(-b <bundle>)] <work|pg|turso|external-pg|external-turso> [existing_env_file]"; exit 1 ;;
      work|pg|turso|external-pg|external-turso)
        [ "$BUNDLE_RAW" != "host" ] || { echo "FATAL: no bundle given — pass <bundle> or -b/--config-bundle <path>"; exit 1; }
        MODE="$1"; EXISTING="${2:-}" ;;
      *)
        BUNDLE_RAW="$1"; MODE="${2:-}"; EXISTING="${3:-}"
        [ -n "$MODE" ] || { echo "FATAL: bundle-node-env needs a mode (work|pg|turso|external-pg|external-turso)"; exit 1; } ;;
    esac
    resolve_bundle
    bundle_node_env "$BUNDLE_DIR" "$MODE" "$EXISTING" "$(minio_password)" "$KEEP_DATAPLANES"
    exit 0 ;;
  bundle-compose)    # hidden: write a bundle override compose file (tests/debug)
    shift
    [ $# -ge 1 ] || { echo "usage: startcluster bundle-compose <outfile> [svc|host|cont[:ro|rw]]..."; exit 1; }
    out="$1"; shift
    write_bundle_compose "$out" "$@"
    exit 0 ;;
  create) shift
          # the global pre-parser consumes -b/--config-bundle before create
          # sees it — hand a real bundle path through (Phase-1 fix: `create
          # -b X` used to silently ignore the bundle; 'host' stays default-off)
          [ -n "${BUNDLE_RAW:-}" ] && [ "$BUNDLE_RAW" != "host" ] \
            && set -- "$@" -b "$BUNDLE_RAW"
          create_cluster "$@" ;;
  rm)     shift; [ -n "${1:-}" ] || { echo "usage: startcluster rm <cluster>"; exit 1; }; rm_cluster "$1" ;;
  *)
    # custom cluster verbs: name in the registry -> generic handler
    if registry_is_known "$1"; then
      name="$1"; action="${2:-status}"; shift 2
      custom_verb "$name" "$action" "$@"
    else
      usage; echo; echo "unknown command '$1'"; exit 1
    fi ;;
esac
