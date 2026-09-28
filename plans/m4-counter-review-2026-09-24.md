# Counter-review of plans/m4-code-review-2026-09-24.md

**Date:** 2026-09-24
**Subject review:** M4 multi-host branch code review (branch `feature/m4-multihost`, HEAD `835ded5`)
**Method:** every finding independently re-verified against the code by three parallel verification agents (deploy scripts / launcher+state / pod UI), plus main-thread checks of the git-level and credential claims. The subject review's own verification claims were reproduced: `pytest -q` → **981 passed** (launcher live), the same-second CAS probe re-run with the same result, the `set -e` + `&&` + `exit 0` shell behavior re-confirmed, merge base `3bcbb5e` and HEAD `835ded5` confirmed.

## Verdict

**21 of 23 findings confirmed, 2 partial (one with its consequence refuted), 0 fabricated.** Line citations are accurate or off by one throughout; no wrong-file citations. The review is high quality and its merge recommendation stands. The corrections below re-scope one P0, one consequence, and one severity; nine additional defects the review missed are listed at the end.

| # | Finding | Verdict | Correction |
|---|---------|---------|------------|
| P0 | Credentials baked into image | PARTIAL | Real keys confirmed in all six tracked configs, not excluded by `.containerignore` — but exposure pre-exists on public `main`; see below |
| P1 | `python -m launcher` bypasses bootstrap/mgmt auth | CONFIRMED | `ManagementServer` has no env fallback (`management_server.py:279`, `:322-323` → `"open"`); pods unaffected (Containerfile CMD uses `launchmcp.py`) |
| P1 | Mirror write failures reported as success | CONFIRMED | exact |
| P1 | Env deletion not propagating | CONFIRMED | arguably **understated** — see correction 3 |
| P1 | CAS one-second resolution | CONFIRMED | probe independently re-run; both writes `True`, second payload wins |
| P1 | Pod UI `0.0.0.0` + admin/admin | CONFIRMED | mcp_ui really does default loopback + warn (`management_ui.py:1230-1236`); on this host `.env` creds apply, so only the bind is live |
| P2 | v3 Popen bypasses OpRunner | CONFIRMED | genuine capability gap: `startcluster` has no single-container verb |
| P2 | Secret env files 0644 | CONFIRMED | files are mode 664 on this host right now |
| P2 | Preflight omits 8400 | CONFIRMED | slightly worse: `wait_http` polls only 8200, so a crash-looping UI can still report UP |
| P2 | `./startcluster` untracked | CONFIRMED | `.gitignore:261` |
| P2 | Cleanup omits LB/MinIO layers | PARTIAL | facts right, **consequence refuted** — see correction 2 |
| JC | Repeated action mappings | CONFIRMED | **understated** — 7 sites, not 3 |
| P1 | Boot-from-bundle positional ignored | CONFIRMED | strongest finding; see correction 5 |
| P1 | Re-seed passes password as filename | CONFIRMED | green suite explained: the test passes a real *file* (`test_config_bundle.py:258-265`) while both real call sites pass the value |
| P1 | external-pg drops POSTGRES_* | CONFIRMED | pass-through at `bundle_node_env.py:163-166` is dead code; mode unusable |
| P1 | external-turso drops TURSO_AUTH_TOKEN | CONFIRMED | truth table below |
| P1 | Export/import `exit 0` swallows failure | CONFIRMED | borderline P1/P2; same-family unchecked `podman cp` also at `:447` |
| P2 | `-b` not ignored for non-up actions | CONFIRMED | plan §3.3/§3.4/E10 quotes verified |
| P2 | Masked discovery seeds inventory | CONFIRMED | union protection lives only in the CLI `sync_tools_config` path |
| P2 | nullglob/quoting/xargs | CONFIRMED | latent in practice — harvest mkdirs all subdirs; current Turso path has no spaces |
| P2 | Harvest manifest history (E14) | CONFIRMED | `scrape` re-harvests into the same bundle, destroying history every run |
| P2 | Redaction misses POSTGRES_* | PARTIAL | spec deviation real (plan §3.1 requires it) but current data unaffected — see correction 6 |
| P2 | v3 direct podman vs F9 | CONFIRMED | F9's verb list is env-level, but "never reimplement" + ops.py's own golden rule cover it |

## Corrections

### 1. The P0 is mis-scoped — the exposure pre-exists on public `main`

All factual elements check out: `deploy/Containerfile:35` is exactly `COPY tools/ ./tools/`; all six tracked `tools/*/config.json` carry non-placeholder `auth.api_key` values (lengths 20–26); `.containerignore` excludes `.env` but not the tool configs; `publish-image.sh` will push them. However:

- The repo is public (`github.com/golfromeo-fr/supreme-mcp-tools`, referenced by the image's own `org.opencontainers.image.source` label), and the branch diff for `tools/*/config.json` is **empty** — the identical keys sit on `main`. Publishing the image adds zero incremental exposure over the already-public git tree.
- "This contradicts the image's own comment" is inaccurate: the comment (`Containerfile:52-53`) scopes itself to `POSTGRES_*` and identity/management secrets, which genuinely are not baked.

Correct framing: the branch-introduced defect is a missing `.containerignore` entry (hardening, P2-class); the P0-class issue is *real credentials tracked in a public repo*, which pre-dates M4 and lives on `main`. "Rotate any credentials that may already have been published" is still right advice — but for the repo, not the image.

### 2. LB/MinIO cleanup: consequence refuted

The compose-file and `podman rm` omissions are real, but `podman pod rm -f "pod_$PROJECT"` (`startcluster.sh:468`, `:518`) removes every container in the project pod, and live inspection shows all five containers (node1, node2, db, lb, minio) in `pod_mcp-multihost` — podman-compose's default in-pod placement means lb/minio **are** removed in practice. The observation stands as fragile coupling (it depends on in-pod placement, and `restore` at `:412` does name `lb` but omit `minio`); the claimed user-visible impact does not occur today.

### 3. Env deletion is worse than stated

Beyond siblings keeping the stale value: in pods both nodes mount the same read-only `node.env` (`compose-common.yml:34,61`), so the local `.env` comment-out fails (`env_manager.py:419-426`), and **the node that processed the delete re-absorbs the revoked credential from its own mounted `.env` on next boot** — the "shared doc is authoritative" comment is exactly what fails, since the doc entry was popped. Revocation fails cluster-wide including the handling node.

### 4. Repeated mappings: 7 sites, not 3

Beyond the three bundle pages: `pod_ui/pages_v2/home.py:21,39` and `pod_ui/pages_v3/home.py:178,184`. No shared helper exists anywhere in pod_ui.

### 5. Boot-from-bundle: slightly narrower than stated, and always broken

"The script only consumes `--config-bundle`/`-b`" is overbroad: `scrape` **does** consume a positional bundle (`startcluster.sh:168`) and is the only way to pass one — so the scrape-into-bundle buttons work; only `up`/`pg` boots are broken. The stray positional is silently ignored (never an error, never a tool name): `startcluster work up /b/bundle` runs a normal **host-mode** rebuild, prints `[work] UP`, exits 0. `git log -S 'config-bundle' -- pod_ui/` is empty — the UI never passed `-b` in any commit. `tests/test_pod_ui_helpers.py:80-81` asserts the broken shape (lines 78-79, scrape, are actually correct). The plan prescribed the right shape all along (`pod-ui-plan-2026-09-19.md:93,102`).

### 6. Redaction: deviation real, exposure theoretical

Plan §3.1 does require `POSTGRES_*` redaction, so it is a genuine spec deviation. But today's host `.env` contains only `POSTGRES_TEST_DSN` among `POSTGRES_*` (caught by the `DSN` suffix rule), and scrape-mode bundles strip `POSTGRES_*` entirely via `env_strip.py:29-30`. Exposure requires a user-added external-pg block. P2 is overstated for current data.

### 7. Line-number drift (minor)

Off-by-one in several citations: `launcher/__main__.py` 34-37/274-279 (not 33-36/273-279), `cluster.py:150-165` (not 149-164), `pod_ui/app.py:17` (not 16), `pages_v3/home.py:94-95` (not 91-95), `pages/bundles.py:30` (not 28-31). All point at the right code.

### 8. bundle_node_env topology truth table (for findings external-pg / external-turso)

| Topology | TURSO_DATABASE_URL | TURSO_AUTH_TOKEN | POSTGRES_* |
|----------|--------------------|------------------|------------|
| work | kept | kept | kept |
| pg / turso (embedded) | dropped | dropped | kept (fresh random password — see A4) |
| external-pg | dropped | kept (by design — turso is a data plane) | **dropped** (bug, finding stands) |
| external-turso | kept | **dropped** (bug, finding stands) | n/a |

## Additional defects the review missed

1. **P1-class — v1/v2 boot-from-bundle buttons have no confirmation at all.** `confirm_or_run(factory, title)` without `word` fires immediately (`pod_ui/confirm.py:29-31`); `pages/bundles.py:73-80` and `pages_v2/bundles.py:96-104` pass only a title, so "Boot work from this" — which stops the running environment — executes instantly. The plan requires a confirm dialog (`pod-ui-plan-2026-09-19.md:127`). Only v3 has real dialogs.
2. **P1-class — shared-state plane can die permanently on a transient backend hiccup.** `state_docs.py:101-107` sets `_init_done = True` *before* the connection attempt; if the backend is briefly unreachable at first use, `_conn_singleton` stays `None` for the process lifetime — all loads `None`, all saves `False`, which finding "mirror booleans ignored" then hides completely.
3. **P2 — fan-out auth failure poisons the node registry.** A node lacking `MCP_MANAGEMENT_API_KEY` sends unauthenticated mask-push (`management_server.py:92-93`); the sibling's 401 is treated as unreachability (`:103`, `:108-112`) → `mark_unreachable` drops a healthy sibling from all future fan-outs until reboot.
4. **P2 — pod UI Functions inventory can never populate in-container.** Discovery builds `http://127.0.0.1:{port}/mcp` (`mcp_ui/management_ui.py:541`) but the ui service runs in its own container (`compose-common.yml:65-82`); discovery always returns `[]`, and `_save_tools_config` writes the UI container's local file, not the node's.
5. **P2 — `logs` action off-by-one on the service name.** After `shift`, the service sits in `$1` but the arms read `${2:-work}` / `${2:-node1}` (`startcluster.sh:292`, `:471`): `startcluster test logs node2` tails node1's logs.
6. **P2 — dead S3 endpoint in external-* bundle mode.** `bundle_node_env.py:207-215` emits `S3_ENDPOINT=http://minio:9000` for all non-work modes, but external up starts no minio (`startcluster.sh:544-547`).
7. **P2 — preflight also misses 5433.** `compose-work.yml:27` publishes `127.0.0.1:5433:5432` and the work node's health gate depends on it; the `:311` check covers only 8000-8005/8200. Same partial-startup class as the 8400 finding.
8. **P2 — silent busy-refusal + false success notifies.** `ops.runner.start()` returns `None` when an op is already running (`ops.py:92-94`); bundle/backup handlers ignore it and notify success ("boot finished", "harvest finished") or navigate away as if booted (`pages/bundles.py:31-32,118-119`; v2/v3 equivalents; `pages/backups.py:20,25,30,35`). `ops.fire()` exists to refuse loudly and is unused here.
9. **LOW — the v3 Popen path DEVNULLs output, never checks the return code, and still notifies "sent".** `pages_v3/home.py:94-96`.

## Assessment of the review itself

- **Methodology:** sound and honest — the exclusion scope, the live probes, and the suite result all reproduce. The remark about `docs/agents/issue-tracker.md` is accurate (the file doesn't exist) but the `/setup-matt-pocock-skills` recommendation is boilerplate noise.
- **Severity calibration:** mostly fair. One overstatement at P0 (mis-scoped, see correction 1), one refuted consequence (LB/MinIO cleanup), one overstated-for-today redaction impact, two understated findings (env deletion in pods, mapping duplication count).
- **Coverage gaps:** the nine defects above, of which the v1/v2 no-confirm destructive buttons and the one-shot state-plane init are the most significant.
- **Merge recommendation:** **agree — not merge-ready.** With the P0 re-scoped, the blocking set is the P1 bundle/UI/startup cluster: boot-from-bundle silently starting host mode, the password-vs-filename re-seed bug (bricks existing pgdata on re-up), external-pg/external-turso bundle modes being unusable, the unauthenticated `python -m launcher` management API, and the shared-state silent-failure trio (mirror booleans, deletion propagation, CAS resolution).

## Execution status (2026-09-25)

All confirmed defects and missed defects were fixed in three bands on `feature/m4-multihost`, suite **1012 passed** (was 981; +31 new tests):

| Band | Commit | Contents |
|------|--------|----------|
| D | `7c5c91c` | deploy scripts + bundle generator + harvest + Containerfile multi-stage key sanitize + new `service` verb |
| L | `9e96154` | launcher M4 bootstrap parity, mirror-outcome visibility, env-deletion tombstones, µs CAS + self-bump, retrying state init, fan-out 401 fix, discovery union + `MCP_DISCOVERY_HOST` |
| U | `52232aa` | `-b` boot shape (v1/v2/v3), boot confirms, honest busy-refusal, v3 via OpRunner, `boot_action()` dedup, loopback default + cred warning |

**Deliberately NOT done:** rotating the six tool API keys already public on `main` (user decision, still open — the image no longer bakes them, but git history does); `_save_tools_config` in the pod-UI container still writes a container-local file (flagged, deeper redesign); the three deploy-band agent observations (bundle env first-occurrence/last-wins edge, `restore`/`start` arms without bundle mounts, `start` without health wait) are recorded as known issues, not fixed. Agent-noticed defects are logged in the band agents' reports; none block merge.

Deployment note: launcher/state fixes activate at the next launcher restart; deploy-script and image fixes at the next cluster rebuild; pod-UI fixes at the next `startpodui`.
