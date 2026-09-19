# Pod UI — a second management UI for the podman environments (plan for GLM-5.3-flash)

**Date:** 2026-09-19 · **Branch target:** `feature/m4-multihost` · **Status: PLANNED (awaiting go; dev by flash)**
**Tracker:** TODO.md "Open — pod UI". **Companion docs:** deploy/README.md (verbs), plans/pod-config-bundle-2026-09-17.md (bundles).

## 0. Probe facts (live-verified 2026-09-19 — trust these over assumptions)

| # | Fact | Evidence |
|---|------|----------|
| F1 | Port **8401 is free** and inside the ui range (8400-8499, config/ports.json) | `ss -ltn` empty for :8401 |
| F2 | `podman ps -a --format "{{.Names}}\|{{.Status}}\|{{.Ports}}"` yields pipe-separated rows; an **Exited container can still carry "(healthy)"** in its status string (`Exited (137) About an hour ago (healthy)`) → parse state from the FIRST WORD (`Up`/`Exited`/`Created`), never from the `(healthy)` suffix | live output incl. mcp-multihost_node1_1 |
| F3 | Image version labels exist but are `dev / unknown` on script-built images — `podman-compose build:` passes no args; only publish-image.sh does. `podman inspect localhost/mcp-node:latest --format '{{index .Labels "org.opencontainers.image.version"}} / {{index .Labels "org.opencontainers.image.revision"}}'` | live: `dev / unknown` |
| F4 | `podman volume ls --format "{{.Name}}"` + `podman volume inspect <v> --format '{{.Mountpoint}}'` work; but `du` as user `gr` **under-reads root-owned volume dirs** (pgdata shows 4.0K) → volume sizes must come from `podman system df -v`, container RW sizes from `podman ps -as --format "... {{.Size}}"` | live |
| F5 | startcluster confirmation prompts (stdin `read -r ans`, expects `wipe`): **work restore (~L141), work clean (~L278), test restore (~L384)**. `test clean` does NOT prompt. Non-interactive subprocess must pipe `"wipe\n"` to stdin AFTER the UI collected its own typed confirmation | grep |
| F6 | mcp_ui reference patterns: standalone `ui.run(**run_kwargs)` entry; `MCP_UI_USERNAME`/`MCP_UI_PASSWORD` (default admin/admin, `hmac.compare_digest`); `MCP_UI_SECRET` signs `app.storage.user` (unset ⇒ ephemeral warning); login gate = `storage.user.get("authenticated")` | mcp_ui/management_ui.py |
| F7 | NiceGUI pinned `nicegui==3.16.0` | requirements.txt |
| F8 | Bundles live in `<repo>/my-bundles/<name>/` (dir 700, `manifest.json` + `env/ identity/ config/ tools/`); backups in `~/supreme-mcp-tools-backups/` as `work-pg-*.sql`, `test-pg-*.sql`, `work-artifacts-*.tar.gz`. **0-byte backups exist** (nightly crontab fired while env down) → flag, don't hide | ls |
| F9 | All pod operations are already verbs: `deploy/startcluster.sh` `work|test` × `up(default)|start|stop|status|clean|backup|restore|scrape|export-data|import-data|logs|ui` + `harvest`. UI MUST shell out; never reimplement | the script itself |
| F10 | test-env containers from a stopped cluster still appear in `podman ps -a` (Exited) — the dashboard must show them as stopped, not hide them | live |

## 1. Goal / non-goals

**Goal:** a dedicated host-side web UI ("pod UI") that manages the podman environments — detect state, run lifecycle ops with live streamed output, manage bundles/backups/data (harvest/scrape/backup/restore/export/import) — so pod work stops being CLI-only. User: "I am a bit lazy on command-line nowadays."

**Non-goals (veto-able):** no reimplementing startcluster logic (shell out only); no multi-user store (host-admin tool, single login); no in-container deployment of THIS UI (it manages the host's podman — host-only app, NOT copied into the node image); no editing of bundle file contents in-UI v1 (list/inspect/delete only — edit stays in the editor).

## 2. Stack & fixed decisions

| Decision | Choice | Why / alternative rejected |
|---|---|---|
| Stack | **NiceGUI 3.16.0** (pure Python; Vue/Quasar underneath — same as mcp_ui) | user asked "python + VueJS?" — NiceGUI IS Vue-backed with zero node toolchain. Hand-written Vue SPA rejected: second codebase + build step for ~12 buttons. Escape hatch if ever needed: FastAPI + Vue-via-CDN |
| Port | **8401** (`MCP_PODUI_PORT` override) | F1; keeps 8400 = mcp_ui. Second instance naturally fails to bind = single-instance guard |
| Package | new `pod_ui/` at repo root, launched `python -m pod_ui` (+ root wrapper `startpodui`, local-only/gitignored like startui/startcluster) | mirrors mcp_ui layout conventions |
| Auth | ALWAYS require login; reuse `MCP_UI_USERNAME`/`MCP_UI_PASSWORD`/`MCP_UI_SECRET` from `.env` (no new keys to set) | this UI can erase pods; user already has the creds |
| Execution | `asyncio.create_subprocess_exec("bash", "<repo>/deploy/startcluster.sh", *args)` with `cwd=REPO_ROOT`, stdout+stderr merged, streamed line-by-line; global op lock (one op at a time); Kill button (terminate) | F9 |
| Log viewer controls | copy the mcp_ui Logs tab control set verbatim (tail select ≤10000, font/cols/rows number+steppers, Max out, filter, Auto 5s) reading the same `MCP_UI_LOG_*` env defaults | user-tuned UX, zero new env keys |

## 3. Package layout (create exactly these)

```
pod_ui/
  __init__.py        # version string only
  __main__.py        # from .app import main; main()  (python -m pod_ui)
  app.py             # NiceGUI app assembly, ui.run(port=8401), secret warning (F6 pattern)
  auth.py            # login page + storage.user gate (mono, always-on)
  state.py           # PURE: podman/startcluster output parsing, listing, status derivation (testable)
  ops.py             # PURE-ish: command builders + the OpRunner (lock/stream/history)
  pages/
    __init__.py
    dashboard.py     # environments overview (P1)
    logs.py          # container log viewer (P1)
    operations.py    # lifecycle actions + streamed output pane (P2)
    bundles.py       # bundles list/harvest/manifest/boot-from-bundle/scrape/delete (P3)
    backups.py       # backups list/backup/restore + export/import-data (P3)
tests/test_pod_ui_helpers.py   # state.py + ops.py pure functions ONLY (no NiceGUI import)
startpodui                    # root wrapper, gitignored (local-only like startcluster)
```

Repo-root `REPO_ROOT = Path(__file__).resolve().parents[1]` in state.py; every subprocess runs with `cwd=REPO_ROOT` so relative `my-bundles/...` paths match the CLI behavior.

## 4. Config table (verbatim)

| Key | Default | Used for |
|---|---|---|
| `MCP_PODUI_PORT` | `8401` | listen port |
| `MCP_UI_USERNAME` / `MCP_UI_PASSWORD` | `admin` / `admin` | login (shared with mcp_ui) |
| `MCP_UI_SECRET` | (unset ⇒ warning, ephemeral sessions) | cookie signing (shared) |
| `MCP_UI_LOG_TAIL/FONT/COLS/ROWS/MAX` | 500 / 12 / 160 / 30 / 0 | log-viewer defaults (shared with mcp_ui Logs tab) |

## 5. state.py — data layer (pure functions, verbatim command table)

| Datum | Command (verbatim) | Parse rule |
|---|---|---|
| containers | `podman ps -a --format "{{.Names}}\|{{.Status}}\|{{.Size}}\|{{.Ports}}"` (note `-as`) | split on `\|`; state = first word of Status (`Up`→running, `Exited`/`Created`/other→stopped); NEVER trust the `(healthy)` suffix on exited rows (F2); group by prefix `mcp-work_`→work env, `mcp-multihost_`→test env |
| container sizes | same row, `.Size` field (`=size on disk`) | display string as-is |
| volume reclaim | `podman system df -v` | section "Local Volumes"; map name→SIZE+RECLAIMABLE columns |
| image version | `podman inspect localhost/mcp-node:latest --format '{{index .Labels "org.opencontainers.image.version"}} {{index .Labels "org.opencontainers.image.revision"}}'` | show raw; `dev unknown` shown honestly + tooltip why (F3) |
| health: work | `curl -sf http://127.0.0.1:8200/health` | 200 ⇒ healthy (JSON has tools_count) |
| health: test | `curl -sf http://127.0.0.1:18200/health` and `:19200` | per-node badges |
| bundles | `sorted((REPO_ROOT/"my-bundles").glob("*/manifest.json"))` | dir must exist; manifest fields: created_utc, git.describe, redacted, scraped (may be absent) |
| backups | `sorted((~/supreme-mcp-tools-backups).glob("*.sql") + glob("*.tar.gz"))` desc | size==0 ⇒ badge "empty (env down when taken?)" (F8); kind by suffix |

Derived model (dataclass): `Environment(key, title, project_prefix, containers[list], health, default_actions)`. Two instances: work, test. A container whose name matches neither prefix goes to an "other" group (displayed, no actions).

Polling: one `ui.timer(5.0)` refresh; NO central-API calls, so no audit-spam concern (the 585e518 lesson doesn't apply, but polling must stay cheap: the two podman calls + two curls only).

## 6. ops.py — the operation runner

Command map (VERBATIM — the single source of truth for every button):

| UI action | argv (after `bash deploy/startcluster.sh`) | Typed confirmation? | stdin |
|---|---|---|---|
| work up (rebuild) | `work` or `work -b <bundle>` | no (script stops itself safely; preflight FATALs stream to pane) | – |
| work start / stop | `work start` / `work stop` | no | – |
| work clean | `work clean` | **yes: type `wipe`** | pipe `wipe\n` |
| work backup | `work backup` | no | – |
| work restore | `work restore <file>` | **yes: type `wipe`** | pipe `wipe\n` |
| work scrape | `work scrape [<bundle>]` (no arg = script resolves from [bundle] header) | no | – |
| work export-data | `work export-data` (fixed default out path) | no | – |
| work import-data | `work import-data <tar.gz>` | no (script refuses if env running — FATAL streams) | – |
| work logs / ui | not buttons (logs = viewer page; `work ui` = small "start UI" action, no confirm) | – | – |
| test up | `test pg` or `test pg -b <bundle>` | no | – |
| test start / stop | `test start` / `test stop` | no | – |
| test clean | `test clean` | no (script does NOT prompt — volumes kept; the pane shows what was kept) | – |
| test backup / restore | `test backup` / `test restore <file>` | restore: **type `wipe`** | pipe `wipe\n` |
| test scrape / export-data / import-data | `test scrape [<bundle>]` / `test export-data` / `test import-data <dir>` | no | – |
| harvest | `harvest <name> [--zip] [--redact] [--from-work-db]` | no | – |

Runner contract:
1. One global `asyncio.Lock` — a second click while an op runs gets a toast "operation in progress", never a queue.
2. `create_subprocess_exec("bash", str(SCRIPT), *argv, stdout=PIPE, stderr=STDOUT, stdin=PIPE, cwd=REPO_ROOT)`; if confirmed-wipe: `stdin.write(b"wipe\n")`, drain, close stdin (F5).
3. Stream each decoded line (`errors="replace"`) into the output pane AND append to a ring-buffer history (last 200 ops: argv, start/end ISO, exit code, last 50 lines).
4. Exit code + duration in a result line (`✓ rc=0 in 94s` / `✗ rc=1`); FATAL lines in the script output are highlighted (match `FATAL`).
5. Kill button → `proc.terminate()`; op ends with rc=-15 marker.
6. Closing the UI kills a running op (acceptable; document in the pane header while running: "keep this tab open").

## 7. Pages (element-level spec — NOT full code)

**Login (`/login`)** — username/password + Log in; `hmac.compare_digest` vs env pair; on success `storage.user = {"authenticated": True}`; **do NOT set `password_toggle_button=True`** (NiceGUI 3.16.0 event-wiring bug — mcp_ui removed it in 4c422c3). Every other page redirects here when not authenticated.

**Dashboard (`/`)** — one card per environment: title (work/test), big state chip (Running if any container Up + health OK; Stopped if all Exited/absent; Degraded if containers Up but health fails), health badges (work: central 8200; test: node1 18200, node2 19200), image version line (F3), per-env volume sizes (from system df), and the container table (name, state, status raw, size, ports). "Other containers" collapsed section. Auto-refresh 5s. Buttons per env link to `/ops?env=work|test`.

**Logs (`/logs`)** — container picker (union of both envs' expected containers that exist in podman ps -a: work → db/work/ui; test → node1/node2/db/lb/minio) + the mcp_ui Logs control set (§2) polling `podman logs --tail N <name>` every 5s when Auto. Filter is client-side substring (post-tail) — different from mcp_ui's server-side grep; keep it simple, label it "filter (shown lines)".

**Operations (`/ops`)** — env tabs; action buttons per the §6 map; typed-confirmation dialog for the three `wipe` actions (input must equal `wipe`, else disabled); the streamed output pane (`ui.log`, monospace, FATAL highlight) + result line + Kill; op history accordion below (last 10, expandable).

**Bundles (`/bundles`)** — table of `my-bundles/*`: name, created, git describe, redacted?, scraped-from?, per-row actions: **Boot work from this** / **Boot test pg from this** (confirm dialog: names the env + that up stops the running one; canonical-port warning for work), Scrape into (env picker), Delete (typed confirmation = bundle name). Top bar: Harvest form (name input + `--zip`/`--redact`/`--from-work-db` checkboxes) — runs `harvest <name> [flags]` via the runner. Manifest drawer: pretty JSON of manifest.json.

**Backups & data (`/backups`)** — backups table (file, kind icon sql/tar, size, 0-byte flag F8) with per-row Restore (typed `wipe`, env inferred from prefix `work-pg-`/`test-pg-`); per-env action rows: Backup now, Export data (out path fixed default, shown), Import data (picker listing matching `work-artifacts-*.tar.gz` / test: export dirs under backups dir; empty-state hint with the CLI equivalent). Ownership-map note rendered after export ops (script prints it — just let it stream, then ALSO show it as a static info box).

## 8. Edge-case catalogue (implement + cover in tests where pure)

| # | Case | Behavior |
|---|---|---|
| E1 | container row `Exited (137) ... (healthy)` | state=stopped (F2); raw status shown in tooltip |
| E2 | image label `dev unknown` | show as-is + tooltip "script-built image; publish-image.sh stamps versions" |
| E3 | 0-byte backup file | badge "empty — env probably down when taken"; restore allowed but warns |
| E4 | op clicked while another runs | toast + ignored (lock) |
| E5 | `work clean` typed confirmation wrong | dialog stays, button disabled until input == `wipe` |
| E6 | script FATAL (ports busy / MINIO unset / bundle missing) | rc≠0, FATAL line highlighted, no retry loop |
| E7 | work up while host launcher holds 8000-8005 | script preflight FATAL streams — expected, surfaced |
| E8 | `work import-data` while env running | script refuses (streams FATAL) — surfaced, no pre-check in UI |
| E9 | harvest name with spaces | reject in form (regex `[A-Za-z0-9._-]+`) |
| E10 | bundle deleted while listed | row action fails with script's FATAL; refresh after every op |
| E11 | podman absent/broken | dashboard shows connection-error banner, actions disabled |
| E12 | UI restarted mid-op | op dies with UI; history note "interrupted"; environments re-probed |
| E13 | `my-bundles/` missing | empty state with the harvest hint |
| E14 | MCP_UI_SECRET unset | startup warning (F6 wording), login still works, sessions ephemeral |
| E15 | containers from BOTH envs + strays | grouped work/test/other; strays read-only |
| E16 | scrape with no [bundle] header (host-mode env) | script FATAL streams — surfaced with the hint to pick a bundle explicitly |
| E17 | restore picker file outside backups dir | not offered (list-based picker only, no free text) |
| E18 | test clean while nodes running | script downs+removes (no prompt by design); pane shows "volumes kept" |
| E19 | du under-reading volumes (F4) | volume sizes ONLY from `podman system df -v`, never du |
| E20 | second UI instance | port bind fails at startup = intended single-instance guard |

## 9. Failure playbook (symptom → cause → fix)

| Symptom | Likely cause | Fix |
|---|---|---|
| 500 page on any pod_ui page | a select default not in its option keys (ui.select raises at build; the 64cb666 lesson) | every `ui.select(value=X)` must have X among options — grep before shipping each page |
| 500 with `TypeError: not all arguments converted` | ui.number `format=".0f"` (needs printf `"%.0f"`; the f9dab07 lesson) | use `format="%.0f"` |
| login button dead | `password_toggle_button=True` used (3.16.0 wiring bug) | remove the prop (4c422c3 precedent) |
| duplicated elements after refresh | `@ui.refreshable` misuse | container.clear() + rebuild (mcp_ui precedent) |
| UI dies at startup, port error | 8401 taken / second instance | intended; check `ss -ltn :8401` |
| op output empty but rc=0 | stderr not merged | stderr=STDOUT in the runner (§6.2) |
| wipe-op hangs at confirm | stdin pipe missing | pipe `wipe\n` then close stdin (F5) |
| dashboard shows env stopped but health OK (or inverse) | state parsed from `(healthy)` suffix | first-word rule (F2) |
| volume sizes absurdly small | du permission under-read | system df -v only (F4/E19) |
| browser-verified page broken though curl 200 | NiceGUI builds pages on socket connect | verify pages with a REAL browser render (temp admin recipe from the 64cb666 session) before declaring done |

## 10. Phases (each ends green; suite gate per commit)

| Phase | Content | Gate |
|---|---|---|
| P1 | package skeleton + auth + state.py (+helpers tests) + dashboard + logs viewer | browser-verified dashboard/logs; `python -m pytest tests/test_pod_ui_helpers.py` green; full suite green |
| P2 | ops.py runner + operations page (all §6 verbs, confirmations, streaming, history) | browser-verified: run `test status`-class safe op + a `work stop`/`start` round trip; FATAL highlighting proven with a deliberately-wrong bundle |
| P3 | bundles page + backups/data page (harvest/boot/scrape/backup/restore/export/import) | browser-verified harvest→boot(test)→scrape round trip against the real bundle; restore keeps its typed confirmation |
| P4 (optional) | migration wizard (scrape→backup→export→stop→boot→restore→import walkthrough), op-history persistence, `podman system df` overview | user call |

Tests (flash): `tests/test_pod_ui_helpers.py` — pure only: ps-row parsing incl. E1, env grouping incl. E15, bundle/backup listing with tmp_path incl. E3/E13, command-builder table (every §6 row → exact argv + stdin flag), confirmation map (exactly 3 wipe actions), backup-kind inference. NO NiceGUI imports in tests.

Docs: deploy/README.md gains a "pod UI" section (start via `startpodui`, port 8401); root AGENTS.md Commands block one line. `startpodui` wrapper is gitignored (local-only, like startcluster/startlauncher).

## 11. Small fixes to ride along (veto-able)

| Fix | Where | Why |
|---|---|---|
| Version badge real data | startcluster exports `IMAGE_VERSION=$(git describe --tags --always --dirty)` + `GIT_SHA` before compose up; compose-work.yml/compose-common.yml `build.args` pass them (`${IMAGE_VERSION:-dev}`) | F3: badge otherwise always `dev/unknown`; the Containerfile comment already claims startcluster stamps (stale for the compose path) |

## 12. Open decisions (defaults chosen, veto-able)

| # | Decision | Default |
|---|---|---|
| D1 | stack | NiceGUI (no node toolchain) |
| D2 | port | 8401, `MCP_PODUI_PORT` override |
| D3 | auth | shared MCP_UI_* env pair, always-login, no user store |
| D4 | log viewer env keys | shared MCP_UI_LOG_* (user's tuned defaults apply to both UIs) |
| D5 | ports.json `reserved.pod_management_ui: 8401` | add (informational; harmless to port_manager) |
| D6 | bundle content editing in-UI | NOT in v1 (inspect/delete only) |
| D7 | op history persistence (across UI restarts) | P4, not v1 |
