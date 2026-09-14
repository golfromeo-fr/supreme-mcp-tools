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
#    startcluster work [up|stop|start|status|clean|logs [svc]]
#    startcluster test [pg|turso|external-pg|external-turso|stop|start|status|clean|logs [svc]]
set -euo pipefail
cd "$(dirname "$0")"

TEST_PROJECT="mcp-multihost"
WORK_PROJECT="mcp-work"
CONFIG_DIR="${HOME}/.config/supreme-mcp-tools"

# ---------- shared helpers ----------
usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; cat <<'HELP'

  usage: startcluster work [up|stop|start|status|backup|restore <file>|clean|logs [svc]]
         startcluster test [pg|turso|external-pg|external-turso|status|stop|start|clean|logs [svc]]

  work:  daily-driver pod (canonical ports 8000-8005/8200; UI: startcluster work ui)
  test:  M4 two-node bench (node centrals 18200/19200)
  backup/restore: pg_dump of the work state plane <-> ~/supreme-mcp-tools-backups/
HELP
}
wait_http() { # url, tries
  for _ in $(seq 1 "${2:-40}"); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 3; done
  return 1
}

# ---------- WORK environment ----------
work_env() { action="${1:-up}"; shift || true
  PROJECT="$WORK_PROJECT"
  if [ "$action" = backup ]; then
    ts=$(date +%Y%m%d-%H%M%S)
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
    ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml up -d db ) \
      || ( cd deploy && podman-compose -p "$PROJECT" up -d db )
    sleep 8
    podman exec -i "${PROJECT}_db_1" psql -U mcp -d mcp < "$f" >/dev/null \
      && echo "[work] restore imported — starting the node..." \
      || { echo "[work] restore import FAILED"; exit 1; }
    ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml up -d work ) \
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

  # first-run bootstrap: work.env generated from the host .env
  if [ ! -f deploy/work.env ]; then
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
    echo "[work] deploy/work.env created from the host .env (data planes identical; state plane -> own pgvector)"
  fi

  # concrete compose (host paths inlined — no interpolation dependency)
  # POSIX sed (no GNU grep -oP — busybox/BSD-safe); falls back to /nonexistent
  HOST_TURSO_DIR=$(sed -n 's/^TURSO_DATABASE_URL=file:\([^ ]*\).*/\1/p' .env | xargs dirname 2>/dev/null || true)
  export HOST_CONFIG_DIR="$CONFIG_DIR"
  export HOST_TURSO_DIR="${HOST_TURSO_DIR:-/nonexistent}"
  export HOST_HF_CACHE="${HOME}/.cache/huggingface"
  sed -e "s|\${HOST_CONFIG_DIR}|$HOST_CONFIG_DIR|g" \
      -e "s|\${HOST_TURSO_DIR}|$HOST_TURSO_DIR|g" \
      -e "s|\${HOST_HF_CACHE}|$HOST_HF_CACHE|g" \
      deploy/compose-work.yml > deploy/compose-work.run.yml

  case "$action" in
    stop)  ( cd deploy && podman-compose -p "$PROJECT" stop ) ; exit 0 ;;
    start) ( cd deploy && podman-compose -p "$PROJECT" start ) ; sleep 5 ;;
    clean)
      echo "[work] this removes containers AND the state volume (users/masks) — type 'wipe' to confirm:"
      read -r ans; [ "$ans" = "wipe" ] || { echo "aborted"; exit 1; }
      ( cd deploy && podman-compose -p "$PROJECT" down -v ) 2>/dev/null || true
      podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
      echo "[work] fully wiped"; exit 0 ;;
    logs)  podman logs --tail "${3:-40}" "${PROJECT}_${2:-work}_1"; exit 0 ;;
    status)
      ( cd deploy && podman-compose -p "$PROJECT" ps ) || true
      printf "central: "; curl -sf http://127.0.0.1:8200/health || echo "not answering"
      echo; exit 0 ;;
  esac

  # default action: up (rebuild) — stop the running work env first
  echo "[work] stopping the running work env (volumes kept)..."
  ( cd deploy && podman-compose -p "$PROJECT" down ) 2>/dev/null || true
  podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
  echo "[work] preflight: canonical ports must be free (host launcher down?)..."
  BUSY=$(ss -ltn | grep -cE ":(8000|8001|8002|8003|8004|8005|8200)\b" || true)
  if [ "${BUSY:-0}" -ne 0 ]; then
    echo "[work] FATAL: canonical ports busy — stop the host launcher (startlauncher) first."; exit 1
  fi

  echo "[work] grabbing latest code..."
  git pull --ff-only || echo "[work] WARNING: pull failed — deploying local code"

  echo "[work] (re)building node image + starting work env..."
  # compose reuses the tagged image if present — drop it so the build runs
  podman rmi -f localhost/mcp-node:latest localhost/mcp-work_work:latest \
              localhost/mcp-work-work_work:latest >/dev/null 2>&1 || true
  ( cd deploy && podman-compose -p "$PROJECT" -f compose-work.run.yml up -d db work )

  echo "[work] waiting for the node central on :8200..."
  if ! wait_http http://127.0.0.1:8200/health 50; then
    echo "[work] node not healthy — check: podman logs ${PROJECT}_work_1"; exit 1
  fi

  # one-time masks/inventory adoption (users.json is auto-imported by the store)
  podman exec "${PROJECT}_work_1" python3 -c "
import sys, json
sys.path.insert(0, '/app'); sys.path.insert(0, '/app/tools')
from tools.shared import state_docs
if state_docs.load_doc(state_docs.DOC_TOOLS_CONFIG) is None:
    src = json.load(open('/root/.config/supreme-mcp-tools/tools_config.json'))
    state_docs.save_doc(state_docs.DOC_TOOLS_CONFIG, src)
    print('[work] masks+inventory imported from the host tools_config.json')
else:
    print('[work] state docs already present')
" 2>/dev/null || echo "[work] (mask seeding skipped — run again later if needed)"

  echo "[work] UP — canonical surface:"
  curl -s http://127.0.0.1:8200/health; echo
  echo "  tools: 8000-8005 | central: 8200 | UI (opt.): startcluster work ui -> 8400"
}

# ---------- TEST environment ----------
test_env() { action="${1:-pg}"; shift || true
  PROJECT="$TEST_PROJECT"
  case "$action" in
    stop)  ( cd deploy && podman-compose -p "$PROJECT" stop ) ; exit 0 ;;
    start) ( cd deploy && podman-compose -p "$PROJECT" start ) ; sleep 5 ; exit 0 ;;
    clean)
      ( cd deploy && podman-compose -p "$PROJECT" down ) 2>/dev/null || true
      podman pod rm -f "pod_$PROJECT" >/dev/null 2>&1 || true
      podman rm -f "${PROJECT}_node1_1" "${PROJECT}_node2_1" "${PROJECT}_db_1" 2>/dev/null || true
      echo "[test] removed (volumes pgdata/sqldata kept)"; exit 0 ;;
    logs)  podman logs --tail "${3:-40}" "${PROJECT}_${2:-node1}_1"; exit 0 ;;
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
  podman rmi -f localhost/${PROJECT}_node1:latest localhost/${PROJECT}_node2:latest 2>/dev/null || true
  echo "[test] starting..."
  # MinIO credentials live in deploy/.env (gitignored) — compose reads the
  # same file for ${MINIO_ROOT_PASSWORD} interpolation in compose-artifacts.yml.
  if [ "$WITH_DB" = 1 ]; then
    if [ ! -f deploy/.env ] || ! grep -q '^MINIO_ROOT_PASSWORD=..' deploy/.env; then
      echo "[test] FATAL: deploy/.env missing or MINIO_ROOT_PASSWORD unset —"
      echo "       add 'MINIO_ROOT_PASSWORD=<generate-one>' to deploy/.env (see deploy/README.md)"
      exit 1
    fi
    . ./deploy/.env
    podman-compose -p "$PROJECT" -f compose-common.yml -f "$TOPO_FILE" \
      -f compose-lb.yml -f compose-artifacts.yml up -d db node1 node2 lb minio
  else
    # external-* topologies: no embedded state plane AND no shared artifacts
    podman-compose -p "$PROJECT" -f compose-common.yml \
      -f compose-lb.yml up -d node1 node2 lb
  fi
  ok=0; wait_http http://127.0.0.1:18200/health 40 && ok=1
  [ "$ok" = 1 ] && echo "[test] UP: node1 :18200, node2 :19200, LB :18080 ($TOPOLOGY)" \
                || { echo "[test] nodes not healthy — podman logs ${PROJECT}_node1_1"; exit 1; }

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

# ---------- dispatch ----------
case "${1:-}" in
  "")    usage; exit 0 ;;
  work)  shift; work_env "$@" ;;
  test)  shift; test_env "$@" ;;
  pg|turso|external-pg|external-turso) test_env "$@" ;;   # legacy shorthand = test
  *)     usage; echo; echo "unknown command '$1'"; exit 1 ;;
esac
