# Config bundle — harvest the working tree's config, inject it into pod environments

**Date:** 2026-09-17 · **Branch:** `feature/m4-multihost` · **Status:** PLANNED (awaiting go)

Problem: the pod environments (`startcluster work`, `startcluster test`) each
assemble their config from a different ad-hoc place — `work` copies the host
`.env` once at first boot and mounts two live host files; `test` uses a
hand-written `node.env` with an **empty identity plane**. There is no way to
say "give this pod exactly the config my working tree has right now", and no
portable snapshot of that config.

Goal, in the user's words: *a script that scrapes all the useful config in my
working tree into a zip/folder that can be tweaked later*, and an **optional
parameter on the deployment scripts** (`startcluster work|test ...`) that
adds all the necessary config to the new pod, so testing the pods feels like
the usual `startlauncher` (same users, same keys, same masks, same tool
configs, same presets).

---

## 0. Probe facts (verified in the working tree, 2026-09-17)

These are the ground truths the design builds on — re-verify none of them
changed before implementing.

### 0.1 Config surfaces that exist today

| # | Surface | Path | Consumed by pods today | In bundle? |
|---|---------|------|------------------------|------------|
| 1 | Root env (30+ keys: identity, tool API keys, SIMPLEMCP_SECRET, MCP_UI_SECRET, MCP_MANAGEMENT_API_KEY, embedding, cache, DB_PRESET_02/03(+DESC), POSTGRES_TEST_DSN, tester bootstrap) | `.env` | work: copied once into `deploy/work.env` at first boot · test: NOT used (hand-written `node.env`) | YES → `env/.env` |
| 2 | Identity seed (3 users: admin/tester/testuser, each mcp_key + password_hash + masked_functions + servers + db_presets + rag_collections) | `~/.config/supreme-mcp-tools/users.json` | work: mounted RO, one-time auto-import when the DB store is empty (`users_store.py` `MCP_USERS_BACKEND=db`) · test: NOT mounted → test cluster starts with ZERO users | YES → `identity/users.json` |
| 3 | Function masks + tool inventory (`disabled_tools`, `tools`{6 tools→function lists}, `version`) | `~/.config/supreme-mcp-tools/tools_config.json` | work: mounted RO + one-time `state_docs` adoption in `startcluster.sh:154-164` · test: absent (empty masks) | YES → `identity/tools_config.json` |
| 4 | Per-tool runtime config (`auth.api_key`, storage blocks, ragmcp `embedding`, convertermcp `allowed_roots`) | `tools/<name>/config.json` ×6 | BAKED into the image at build (`Containerfile: COPY tools/`) — a tweak needs a rebuild | YES → `tools/<name>/config.json` |
| 5 | Launcher config (port ranges/assignments, logging, monitoring) | `config/ports.json`, `config/launcher_config.json`, `config/monitoring_config.json` | BAKED into the image (`COPY config/`) | YES → `config/…` |
| 6 | Turso DATA dir | `/home/gr/turso_data/` (from `TURSO_DATABASE_URL=file:…`) | work: dir mounted at the same path (`HOST_TURSO_DIR`) · test: impossible (file: URLs cannot cross containers) | NO (data, not config) — path recorded in manifest |
| 7 | HF embedding cache | `~/.cache/huggingface` | mounted (`HOST_HF_CACHE`) | NO (cache) |
| 8 | Mutation logs, locks, backups | `~/.config/supreme-mcp-tools/{simplemcp,webmcp}.json`, `*.lock`, `backups/` | not consumed as config | NO (D3 decision: logs for debugging, not config) |
| 9 | `.env` history | `.env~`, `.env.20260623` | nothing | NO (only live `.env`) |

### 0.2 How the two environments consume config (startcluster.sh, deployed copy)

| Aspect | `work` (single-pod daily driver) | `test` (two-node M4 bench) |
|---|---|---|
| compose | `compose-work.yml` → sed-rendered `compose-work.run.yml` (`${HOST_CONFIG_DIR}`, `${HOST_TURSO_DIR}`, `${HOST_HF_CACHE}` inlined) | `compose-common.yml` + topology file (+ lb + artifacts) |
| env file | `deploy/work.env` — generated ONCE (first boot): `cp .env` + appended state-plane block (`MCP_USERS_BACKEND=db`, `MCP_STATE_BACKEND=db`, `POSTGRES_HOST=127.0.0.1`, `POSTGRES_PORT=5433`, `POSTGRES_USER=mcp`, `POSTGRES_PASSWORD=work-<random>`, `POSTGRES_DB=mcp`) | `deploy/node.env` — hand-written from `node.env.template`, mounted at `/app/.env` on BOTH nodes |
| identity plane | host `users.json` + `tools_config.json` mounted RO into `/root/.config/supreme-mcp-tools/` | **nothing mounted — empty DB identity plane** |
| state plane | own pgvector container, `127.0.0.1:5433` | embedded pg (`db` service, topology `pg`) / sqld / external |
| data planes | host `.env`'s (real turso dir same-path mount) | whatever `node.env` says — template keeps them commented |
| tool set | all six, canonical ports 8000-8005 + central 8200, host networking | `simplemcp convertermcp webmcp` per node; centrals 18200/19200; tools 18002/18003, 19002/19003; LB 18080 |

### 0.3 Mechanics the design relies on (probe-verified)

- **users.json auto-import:** `tools/shared/users_store.py` — with
  `MCP_USERS_BACKEND=db`, the whole-document `mcp_users_store` row is
  one-time imported from the json path (`MCP_USERS_STORE` or
  `~/.config/supreme-mcp-tools/users.json`) when the store is empty.
  Import is idempotent-guarded; seeding via a RO mount is exactly what the
  work env already does.
- **Masks adoption:** `tools/shared/state_docs.py` (`DOC_TOOLS_CONFIG =
  "tools_config"`, `load_doc`/`save_doc`) — `startcluster.sh` seeds it from
  the mounted json on first boot (lines 154-164). Same snippet works against
  a test node (`podman exec …_node1_1`).
- **`load_dotenv` does not override pre-set container env** — compose
  `environment:` (TOOLS_LIST, MCP_NODE_NAME…) always wins over the mounted
  env file. This is why node.env generation only needs to carry the keys the
  compose file doesn't set.
- **`MCP_AUTH_MODE=multi` is live in the host `.env`** — bundles therefore
  carry multi-user identity by default.
- **gitignore:** `deploy/node.env`, `deploy/work.env`, `deploy/.env`,
  `deploy/compose-*.run.yml` already ignored. New generated files (§3.4)
  must be added.
- **STALE WRAPPER (found during planning):** the root `./startcluster` is an
  OLD COPY of `deploy/startcluster.sh` — it still hardcodes the MinIO
  placeholder secret (`change-me-minio-secret`) instead of reading
  `deploy/.env`, and greps `localhost` instead of `127.0.0.1`. Anyone running
  `./startcluster test pg` from the repo root gets the buggy variant. Fix
  included in this plan (§3.6).

---

## 1. Design in one paragraph

`deploy/harvest-config.sh` snapshots every row-YES surface of §0.1 into a
**bundle folder** (default `~/supreme-mcp-tools-bundles/<name>/`, optional
`--zip`), with a `manifest.json` (git sha/dirty, timestamps, sha256 per file,
warnings) and a generated `README.md`. `startcluster work|test` grow an
optional `--config-bundle <path|host>` flag: with a bundle path, the env is
seeded from the bundle instead of the ad-hoc host files — work regenerates
`work.env` from the bundle env and mounts the bundle's identity copies; test
**generates `node.env` from the bundle** (identity plane copied, state plane
written per topology, data planes stripped for safety) and mounts
`identity/users.json` into node1 for the one-time import. No flag (`host`,
the default) = today's behavior, unchanged. Tweaking = edit the bundle
folder, re-run `startcluster ... up`.

**Bundle folder is canonical; zip is a transport add-on** (a folder is
directly editable "tweak later"; a zip is a frozen snapshot that gets
unzipped once).

---

## 2. CLI contracts (verbatim)

### 2.1 harvest-config.sh

```
deploy/harvest-config.sh [options] [BUNDLE_PATH]

  BUNDLE_PATH            bundle folder to create/update
                         (default: ~/supreme-mcp-tools-bundles/config-bundle-<UTC YYYYMMDD-HHMM>)

options:
  --zip                  additionally write <BUNDLE_PATH>.zip (chmod 600)
  --redact               replace secret values with "__REDACTED__" (for sharing;
                         default OFF = real secrets, the bundle is private)
  --from-work-db         when the work cluster DB is reachable, harvest users +
                         state docs from the LIVE db (mcp_users_store,
                         mcp_state_docs) instead of the (possibly stale)
                         ~/.config json seeds; falls back to json + loud warning
  --list                 dry-run: print the copy table + warnings, write nothing
  -h, --help             usage
```

Exit codes: 0 ok (warnings allowed) · 1 usage/fatal · 2 partial harvest
(≥1 required file missing: `.env`).

### 2.2 startcluster flag

```
./startcluster work  [up|stop|start|status|backup|restore <f>|clean|logs [svc]] [--config-bundle <path>|host]
./startcluster test  [pg|turso|external-pg|external-turso|stop|start|status|clean|logs [svc]] [--config-bundle <path>|host]

  --config-bundle <path>   seed this env from a bundle folder or .zip
  --config-bundle host     (default) today's behavior: derive from live host files
```

The flag is parsed by scanning `"$@"` and removing flag+value before the
existing positional dispatch — no change to existing invocations. A bundle
path ending in `.zip` is unzipped to `deploy/.bundle-tmp/<name>/` first.
`startcluster harvest …` is added as an alias subcommand that execs
`deploy/harvest-config.sh "$@"` (single entry point for discoverability).

> **Amendment (2026-09-17, post-build):** `-b <path>` is accepted everywhere
> as the short form of `--config-bundle <path>` (both may also feed the
> hidden `bundle-node-env` subcommand). `harvest`'s BUNDLE_PATH positional
> is optional. **Amendment 2 (2026-09-18, user preference):** the default
> location is no longer `~/supreme-mcp-tools-bundles/` — a bare NAME lands
> in `<invocation cwd>/my-bundles/<NAME>` and no argument defaults to
> `<cwd>/my-bundles/bundle-<UTC ts>` (prefix "bundle" chosen over the
> user's "bunconfig"/"config" suggestions; one-line switch in
> harvest-config.sh). `my-bundles/` is gitignored (bundles hold real
> secrets); startcluster passes its ORIG_PWD via HARVEST_INVOCATION_PWD so
> "cwd" means where the user invoked ./startcluster.

---

## 3. Implementation spec

### 3.1 Bundle layout (folder)

```
<bundle>/
  manifest.json            generated: {bundle_version:1, created_utc, host,
                           git:{sha, describe, dirty, branch},
                           files:[{path, source, sha256, bytes}...],
                           warnings:[...]}
  README.md                generated: what each file is, how to tweak, warnings
  env/.env                 verbatim copy of root .env (chmod 600)
  identity/users.json      verbatim copy (chmod 600)
  identity/tools_config.json
  config/ports.json        verbatim copies
  config/launcher_config.json
  config/monitoring_config.json
  tools/<name>/config.json ×6   (convertermcp databasemcp memorymcp ragmcp simplemcp webmcp)
```

Rules:
- Missing non-fatal sources (identity/*.json, any tools/config.json): skip,
  record in `manifest.warnings`, exit 0. Missing `.env`: fatal (exit 2).
- `--from-work-db`: `podman exec mcp-work_db_1 psql -U mcp -d mcp -Atc
  "SELECT doc FROM mcp_users_store"` → pretty-print to
  `identity/users.json`; same for `mcp_state_docs` row `tools_config` →
  `identity/tools_config.json`. Unreachable container ⇒ fallback to json
  files + warning line + manifest entry.
- `--redact`: env lines whose KEY matches
  `(KEY|SECRET|TOKEN|PASSWORD|DSN)` or is `DB_PRESET_*`/`POSTGRES_*` → value
  becomes `__REDACTED__`; `users.json` → blank `mcp_key`/`password_hash`;
  tool `config.json` → `auth.api_key = "__REDACTED__"`. Best-effort, stated
  as such in the generated README.
- Every copy is `cp -p`; bundle dir `chmod 700`.

### 3.2 Shared env-derivation helper (new function in startcluster.sh)

`bundle_node_env <bundle> <topology>` — writes **`deploy/node.env`**
(test env) to stdout; also used (with `work` mode) to build `work.env`.
Deterministic, side-effect-free, exported as a hidden subcommand
(`./startcluster bundle-node-env <bundle> <topology>`) so pytest can call it
without podman.

Transformation of the bundle's `env/.env`:

1. **Keep verbatim:** identity/auth plane (`MCP_AUTH_MODE`, `MCP_UI_*`,
   `MCP_UI_SECRET`, `MCP_MANAGEMENT_API_KEY`), tool keys
   (`BRAVE_*`, `SERPAPI_*`, `AI_API_KEY`, `SMARTGRAPH_*`, `SIMPLEMCP_*`),
   embedding/cache vars, tester bootstrap vars.
2. **State plane: REWRITTEN per topology** (never copied):
   - `pg` → `MCP_USERS_BACKEND=db`, `MCP_STATE_BACKEND=db`,
     `POSTGRES_HOST=db`, `POSTGRES_PORT=5432`, `POSTGRES_USER=mcp`,
     `POSTGRES_PASSWORD=<from existing node.env if present, else
     cluster-<random>>`, `POSTGRES_DB=mcp`
   - `turso` → backends db + `TURSO_DATABASE_URL=http://db:8080`
   - `external-*` → keep backends db; copy host `POSTGRES_*`/
     `TURSO_DATABASE_URL` values from the bundle but warn that DSNs must be
     network-reachable from inside the containers
3. **Data planes: STRIPPED with a warning** (this is the safety property —
   the test bench must never write the real memory store): drop any line
   whose value starts with `file:` (TURSO_DATABASE_URL, DB_PRESET_03) and
   drop `POSTGRES_TEST_DSN`. Keep network DSNs (`DB_PRESET_02`
   postgresql://…) with a warning comment. `--keep-dataplanes` escape hatch
   skips the stripping.
4. **MinIO/S3:** if `deploy/.env` has `MINIO_ROOT_PASSWORD`, emit
   `S3_ENDPOINT=http://minio:9000`, `S3_ACCESS_KEY=mcp-artifacts`,
   `S3_SECRET_KEY=<that password>`; else strip `S3_*` + warning.
5. Every transformation emits a `# [bundle] <what/why>` comment line so the
   generated file is self-explaining.

### 3.3 work env with `--config-bundle`

In `work_env()`, when a bundle is given (action `up` only; stop/start/
status/logs/backup/restore ignore the flag):

1. FATAL if `<bundle>/env/.env` missing.
2. Regenerate `deploy/work.env` = bundle env + appended state-plane block
   (existing logic) — but **preserve the running DB's password**: if the old
   `work.env` exists, reuse its `POSTGRES_PASSWORD` (else the existing
   pgdata volume becomes unreachable); back the old file up to
   `work.env.bak-<ts>`. Bundle is the source of truth for everything else —
   re-`up` applies bundle tweaks (that's the "tweak later" loop).
3. `HOST_TURSO_DIR` / `HOST_HF_CACHE` still derived from the **bundle**
   env / host home (data stays on the host — manifest records the path).
4. `HOST_CONFIG_DIR=<bundle>/identity` — the existing compose mounts
   (`${HOST_CONFIG_DIR}/users.json`, `.../tools_config.json`) then mount the
   bundle copies with zero template changes. FATAL if `identity/users.json`
   missing (the daily driver needs its users).
5. New generated override file `deploy/compose-work.bundle.yml`
   (gitignored): mounts each existing bundle file RO over its baked path —
   `<bundle>/config/*.json → /app/config/*.json` and
   `<bundle>/tools/<name>/config.json → /app/tools/<name>/config.json` —
   so tool-config tweaks need **no image rebuild**. Composed via
   `-f compose-work.run.yml -f compose-work.bundle.yml`.
6. Everything else (port preflight, pull, build, health wait, mask-seeding
   exec) unchanged.

### 3.4 test env with `--config-bundle`

In `test_env()`, for the four topology actions (stop/start/status/clean
ignore the flag):

1. FATAL if `<bundle>/env/.env` missing.
2. Generate `deploy/node.env` via `bundle_node_env <bundle> <topology>`
   (§3.2). Back up any existing node.env first. (One file for both nodes —
   unchanged contract.)
3. **Identity seeding:** new generated override `deploy/compose-test.bundle.yml`
   (gitignored) that mounts `<bundle>/identity/users.json` RO into
   **node1 only** (`/root/.config/supreme-mcp-tools/users.json`) — node1
   boots, imports into the (empty) DB; node2 reads the shared DB, so
   mounting into both would only risk the concurrent-import race.
   If `<bundle>/identity/tools_config.json` exists, also mount it into
   node1; after node1 is healthy, run the existing state_docs seeding exec
   against `…_node1_1` (same snippet as work, lines 154-164).
4. Tool/config overrides: same body as §3.5 but targeting both nodes
   (mounts over baked `/app/config` + `/app/tools/*/config.json`).
5. Everything else (teardown, image rm, MinIO bucket bootstrap, LB) unchanged.

### 3.5 Generated override compose files — construction

Both are emitted by startcluster right before `podman-compose up`, from a
loop over the bundle's `config/*.json` + `tools/*/config.json` entries
(volumes list, `:ro`, host paths absolute). If the bundle carries none, the
override file is still written (empty volumes) so the `-f` chain is
constant. Regenerated on every `up` — never hand-edited (same contract as
`compose-work.run.yml`).

### 3.6 Root wrapper fix (bug found while planning)

Replace the 250-line stale copy at repo-root `startcluster` with:

```bash
#!/usr/bin/env bash
exec "$(dirname "$0")/deploy/startcluster.sh" "$@"
```

(Root copy predates the `deploy/.env` MinIO-credentials fix and would create
the MinIO bucket with the placeholder secret.)

### 3.7 gitignore additions

```
deploy/compose-work.bundle.yml
deploy/compose-test.bundle.yml
deploy/.bundle-tmp/
```

(bundles default to `~/supreme-mcp-tools-bundles/` — outside the repo, like
`~/supreme-mcp-tools-backups/`.)

---

## 4. Edge-case catalogue (implement + test each)

| # | Case | Behavior |
|---|---|---|
| E1 | bundle path doesn't exist | FATAL exit 1, message names the path |
| E2 | `.zip` given | unzip to `deploy/.bundle-tmp/<name>/`, use that; keep the zip untouched |
| E3 | bundle missing `env/.env` | FATAL (both envs) |
| E4 | bundle missing `identity/users.json` | work: FATAL · test: skip mount + warning (cluster usable via break-glass central key) |
| E5 | work.env already exists | preserve its POSTGRES_PASSWORD; back up old file |
| E6 | node.env already exists | back up; password reuse per §3.2 (2) keeps the existing pgdata/sqldata usable |
| E7 | `file:` data-plane lines in test mode | stripped + `# [bundle]` comment (safety); `--keep-dataplanes` overrides |
| E8 | `MINIO_ROOT_PASSWORD` absent from deploy/.env | S3_* stripped + warning |
| E9 | paths with spaces (turso dir) | everything quoted; sed uses `\|` delimiter (already the pattern in work_env) |
| E10 | `--config-bundle` with `stop/start/status/logs/backup/restore/clean` | flag accepted and ignored (documented) |
| E11 | `--redact` then apply | apply does NOT refuse, but prints a loud "this bundle is REDACTED — pods will boot with dead secrets" warning |
| E12 | users.json mounted into node1 while node2 boots | import runs only in node1 (mount only node1) — race impossible by construction |
| E13 | harvest with work cluster DOWN + `--from-work-db` | fallback to json files + warning, exit 0 |
| E14 | harvest re-run onto an existing bundle | overwrite in place, merge manifest history (keep `previous` one-level-deep summary) |
| E15 | `.env~`/`.env.*` history files | never harvested |

---

## 5. Tests (pytest, no podman needed)

New `tests/test_config_bundle.py` (~15 cases):

- **harvest (run the real bash script in tmp_path):** `--list` writes
  nothing; happy path creates the exact §3.1 tree + manifest sha256s match;
  missing users.json → warning + exit 0; missing .env → exit 2; `--redact`
  blanks `mcp_key`/`password_hash`/`auth.api_key`/`KEY|SECRET|TOKEN|PASSWORD|DSN`
  env values; re-run overwrites (E14); history files not harvested (E15).
- **bundle_node_env (hidden subcommand):** pg topology ⇒ identity keys
  verbatim, `POSTGRES_HOST=db`, no `file:` lines, no `POSTGRES_TEST_DSN`;
  turso topology ⇒ `TURSO_DATABASE_URL=http://db:8080`; network preset kept
  + file preset dropped (E7); `--keep-dataplanes` keeps them; S3 stripped
  when MINIO password absent (E8); password reuse from an existing node.env
  (E6); generated file starts with a `# [bundle]` header.
- **flag parsing:** `--config-bundle` removed from `$@` before positional
  dispatch (sourced-function test or a `--dry-run`-style echo subcommand);
  `.zip` path triggers the tmp-unzip branch (fake zip fixture).
- Existing suites untouched; gate: full `python -m pytest` green
  (~903 + ~15).

## 6. Live verification (after tests, before done)

1. `deploy/harvest-config.sh --list` → table matches §0.1.
2. `deploy/harvest-config.sh` → bundle created 0600; manifest sha256 spot-check.
3. Host launcher STOPPED → `./startcluster work --config-bundle <b>` →
   central `/health` ok; `tools/list` with the bundle admin mcp_key works;
   masked `brave_search_web` absent from `tools/list` (mask seeding via
   bundle identity); one real memory query (data plane via same-path mount).
4. `./startcluster test pg --config-bundle <b>` → node1/node2 healthy;
   `GET :19200/api/users` with bundle central key lists the bundle users
   (import worked via node1); masked tool absent on node2; LB :18080 still
   alternates.
5. Re-`up` after tweaking the bundle folder (change a tool api_key) →
   visible in the container without rebuild (mounted override proof).

## 7. Docs

- `deploy/README.md`: new "Config bundles" section (CLI table §2, layout
  §3.1, safety stripping §3.2(3)); Quickstart gains the bundle variant.
- `deploy/PODS-EXPLAINED.md`: 3-sentence plain-language add ("a box with
  photocopies of your real settings").
- Coordinate with the user's IN-FLIGHT uncommitted edits to these two files
  (they document the test-cluster doors) — rebase the doc edits on top, don't
  clobber.
- Root `AGENTS.md`: one line under Commands (`deploy/harvest-config.sh`).

## 8. Phasing (each ends green, in order)

| Phase | Content | Gate |
|---|---|---|
| P1 | `harvest-config.sh` + `tests/test_config_bundle.py` (harvest half) | script tests green |
| P2 | startcluster flag parsing + `bundle_node_env` + hidden subcommand + tests (env half) | unit green; `bundle-node-env` output eyeballed against node.env.template |
| P3 | work-env bundle branch + `compose-work.bundle.yml` generation | live verification step 3 |
| P4 | test-env bundle branch + identity seeding + `compose-test.bundle.yml` | live verification step 4 |
| P5 | root wrapper fix, gitignore, docs | full suite green; docs diff shown to user |

Estimate: P1 ~200 lines bash + ~120 pytest; P2–P4 ~+150 lines in
startcluster.sh + ~80 pytest; P5 trivial.

## 9. Open decisions for the user (veto-able defaults chosen)

| # | Decision | Default chosen | Alternative |
|---|---|---|---|
| D-1 | bundle format | folder (zip optional via `--zip`) | zip-first |
| D-2 | secrets in bundle | real secrets, dir 700/file 600 (private machine) | `--redact` default |
| D-3 | test-env data planes | stripped (protect the real memory store) | mounted through (`--keep-dataplanes`) |
| D-4 | identity seed into test cluster | users.json → node1 one-time import | pre-seed SQL dump |
| D-5 | flag name | `--config-bundle` | `--seed` / `--from-bundle` |
| D-6 | bundle location | `~/supreme-mcp-tools-bundles/` | in-repo `deploy/.bundles/` |
