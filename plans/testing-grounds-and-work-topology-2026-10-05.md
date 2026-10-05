# Testing grounds + BASIC topology switch — design

**Status:** DRAFT 3 · FINAL, 2026-10-05 — self-check (§0.1) + MiniMax counter-review adjudicated (§0.2); all open questions resolved by veto-able default (§7). Execution checklist lives in `TODO.md` ("Open — testing grounds + work topology"). Ready for implementation handoff.
**Supersedes:** nothing. Complements `plans/pod-config-bundle-2026-09-17.md` (bundle machinery reused, not changed).
**Tracker:** pointer in `TODO.md` (Open — design).

---

## 0. Summary

Three decisions come out of the 2026-10-05 session:

1. **Three-tier test shape** — hermetic logic tests stay; a new small real-pod
   smoke tier runs against *recreatable testing grounds*; manual live sweeps stay manual.
2. **Testing grounds** — disposable pods booted from pinned bundles via the
   EXISTING `startcluster create` registry, ensured/recreated before test runs
   ("pre-testing recreation"), never conflated with the real clusters.
3. **BASIC topology switch** — the work cluster's identity/state-plane db
   becomes selectable (pg | turso) via CLI + pod UI, retiring the false
   "photocopy of the host" contract.

Order: grounds first (they are the safety net the topology switch needs).

### 0.1 Self-check pass (2026-10-05, before external review) — 5 defects fixed

| # | Defect in DRAFT 1 | Fix |
|---|-------------------|-----|
| C1 | D3a claimed grounds can boot "with the work port layout" via `create` — **no such capability exists**: `gen_custom_compose.py` always emits port-shifted published ports + podman-DNS central URL, never host networking | D3 rewritten: v1 grounds are port-shifted custom clusters (coexist with everything); a work-layout ground is an explicit Phase-2 extension (`--layout work`) needed to validate D8 |
| C2 | Phase 0 HF-cache fix targeted only the test-node compose — the **custom-cluster generator has the same hole** (no mounts at all in `node_service`), so all grounds would fail the embedding probe | D4 rewritten: fix lands in `compose-common.yml` AND `gen_custom_compose.py` AND the Phase-2 layout-work generator |
| C3 | Probe 7 (identity import) assumed bundle grounds seed users.json — **they don't**: `create -b` generates only the env file; no override compose mounts the bundle's `identity/*.json` into the node, so identity import never runs on custom clusters | D5 amended: extend `custom_up` to chain a bundle-override compose (reuse `write_bundle_compose`), mirroring the work E10 lesson (138b471) |
| C4 | Edge case 2 proposed a NEW password sidecar — the mechanism **already exists**: `pw_tmpfile_from` stashes POSTGRES_PASSWORD across work.env regeneration in bundle mode (startcluster.sh:309-317); host mode generates once only when the file is missing | Edge 2 rewritten: reuse `pw_tmpfile_from`, add a pg→turso→pg round-trip regression test instead |
| C5 | F7 cited wrong line numbers and missed that the custom-cluster path has **zero test coverage and zero live usage** (clusters.d is empty; no test greps `gen_custom_compose`) | F7 corrected; new F13/F14/F15/F16; Phase 1 starts by unit-pinning `gen_custom_compose` (pure function, cheap) |

### 0.2 Counter-review (MiniMax M3.1 flash, 2026-10-05) — adjudicated, fixes applied

Authoritative artifact: `plans/testing-grounds-and-work-topology-counter-review-2026-10-05.md`
(its §2.1 retracted the stronger D8 framing that the relayed summary contained; author adjudication
in its §6, incl. two corrections TO the review). Accepted and folded into this draft:

- **R1** D8: the `compose-turso.yml` shape-citation trap — its db has NO `ports:` publish
  (bridge `http://db:8080`); copying its shape yields a node that boots healthy and fails on first
  sqld contact. Fix: D8 rewritten — publish + `127.0.0.1` DSN marked LOAD-BEARING, citation
  disambiguated (image/healthcheck from compose-turso.yml; reachability from compose-work.yml).
- **R2** `ensure` recreates on drift (refusal taxed the disposable-ground premise;
  evidence-preservation belongs to `probe`) — D5 + edge 3; `--no-recreate` escape hatch.
- **R3** probe 3 floor oracle (exact counts rot; `fef_v3/test_runner.py:348` `>=` precedent) — §3.5.
- **R4** Phase 2 split 2a/2b with a green gate — §5.
- **R5** tier-2 rot named; manual-by-default v1, scheduled-run decision point — §7 Q10.
- **R6/R7** F14 line cites corrected (`:112` def, `:311`/`:610` calls); F15 reframed as
  convention-to-re-verify, not a dated `ss` fact.
- **R8** users_store facts: db path EXISTS (`users_store.py:101`, `state_docs.py:35`) but is
  stub-tested only (`_FakeConn`, `test_users_store.py:288`) — Phase 2 = configuration + first
  integration run, not greenfield — edge 5.
- **R9** (author correction to the review, strengthening it): `users_store.py` does NOT raise on
  db-init failure — `:124-125` and `:154-155` BOTH warn and stay on users.json. Probe 7 therefore
  requires a positive db-path assertion (data lands in SQL / no "staying on users.json" in the log),
  or it can pass while silently testing the JSON fallback — §3.5 probe 7 amended.
- **R10** (live-proven): podman-compose APPENDS service volume lists across `-f` files —
  `podman inspect mcp-work_work_1` (last bundle-managed boot) shows the base mounts (HF cache,
  turso dir, work.env, users.json, artifacts, logs) all survived the bundle override. BASIC was
  never the F4 victim; the C3 override chain for grounds is safe by the same semantics — F17.

---

## 1. Verified context (all facts checked in-repo 2026-10-05)

| # | Finding | Evidence |
|---|---------|----------|
| F1 | The pytest suite never touches a real pod. Pod-shaped tests are hermetic: bundle tests run in a sandbox ("no podman needed"), pod-UI/scrape tests parse canned `podman ps` strings. | `tests/test_config_bundle.py:5`, `tests/test_pod_ui_helpers.py:21-23`, `tests/test_pod_scrape.py:35` |
| F2 | Live tool tests hit whatever listens on the canonical ports — host launcher OR the work pod, indistinguishably; they skip when nothing listens. | `tools/simplemcp/test_tools.py:41-60` |
| F3 | The TEST (multihost) cluster receives zero automated traffic; its only automated touch is a string assertion on compose text. | `tests/test_deploy_configs.py:46` |
| F4 | Concrete bug class the fake tier cannot see: the work compose mounts the host HF cache, the TEST composes mount it NOWHERE → local BGE-M3 embeddings fail inside TEST nodes. Found via the 2026-10-03/04 embedding-model work. | `deploy/compose-work.yml:57` vs grep of `compose-test.bundle.yml`, `compose-common.yml` (no HF mount) |
| F5 | "Photocopy of the host" is fiction: host runs JSON identity + turso file data (`POSTGRES_HOST` commented out "using Turso", `QDRANT_HOST` commented, `TURSO_DATABASE_URL=file:` active); BASIC runs DB-backed identity (pgvector) + the same turso file bind-mounted. | `.env:36,80-87`; `deploy/ARCHITECTURE.md:73-75,255` |
| F6 | BASIC's shape is load-bearing for suite assumptions (topology coupling the user wants removed). | `tests/test_config_bundle.py:264` asserts `POSTGRES_PORT=5433`; `deploy/compose-work.yml:16-19` mutual exclusion |
| F7 | A custom-cluster generator with a registry ALREADY exists: `startcluster create <name> --nodes 1\|2 --topology pg\|turso --tools "..." --port-base N -b <bundle>`; custom clusters get their own podman project (= name) and verbs (up/stop/start/status/logs/backup/restore via `rm`). `create` does NOT boot (cheap + reversible); auto port-base picks the lowest free base in 20000-20999 (reserves base=central, base+1000=central2, base+2..+7=tools). Registry entry `deploy/clusters.d/<name>.json` is key-free (name, nodes, topology, tools, port_base, ports, bundle path, created_utc). **Caveat (C5): zero test coverage and zero live usage today** — `clusters.d/` is empty and no test file references the generator. | `deploy/startcluster.sh:765` (`create_cluster`), `:874` (`custom_up`), `:891` (`custom_verb`), `:989` (dispatch); registry write :819-837 |
| F13 | **Custom clusters are NOT work-shaped** (`gen_custom_compose.py`): nodes reach central via podman DNS (`MCP_NODE_CENTRAL_URL: http://node1:8200`), all ports published SHIFTED (never canonical, never host networking); pg db is `postgres:16-alpine` (work uses `pgvector/pg16`); `depends_on` is `service_started` (work: `service_healthy`); node services have **NO volume mounts** — no HF cache (embedding probe would fail, C2), no users.json RO (identity import impossible, C3), no host turso dir, no artifacts/logs volumes. Upside: custom grounds are data-plane isolated **by construction** — they physically cannot touch the real memory store. | `deploy/gen_custom_compose.py:21-64` vs `deploy/compose-work.yml:22-62` |
| F14 | POSTGRES_PASSWORD stability across work.env regeneration is ALREADY solved in bundle mode: `pw_tmpfile_from` stashes the running env's password into the regenerated file ("keeps the already-initialized pgdata volume reachable"); host mode writes the state-plane block only when work.env is missing. | `deploy/startcluster.sh:112` (def), `:311` + `:610` (calls), `:308-318` (bundle regen block), `:320-330` (one-shot host bootstrap) |
| F15 | sqld publish port 9080 chosen as the convention (outside all corridors, F12). NOT a verified machine fact — verify free at execution; a taken port fails LOUDLY at publish time (podman publish error), and the grounds `ensure` port preflight re-checks. | convention; re-verify at boot (counter-review R7: a dated `ss` reading is transient state, not a repo fact) |
| F16 | `compose-test.bundle.yml` is generated + gitignored (cited in F4 as if static); the static test-node services live in `compose-common.yml`. | `git check-ignore` 2026-10-05 |
| F17 | podman-compose APPENDS service volume lists across `-f` override files (does not replace). Live-proven on the actual last bundle-managed boot: `podman inspect mcp-work_work_1` (exited) shows the base mounts — HF cache, `~/turso_data`, `work.env`, `users.json`, `work_artifacts`, `work_logs` — all present ALONGSIDE the 9 bundle config mounts from `compose-work.bundle.yml`. Grounds the C3 override chain (users.json RO addition) and closes the "does the bundle override drop base mounts?" question. | `podman inspect` 2026-10-05 |
| F8 | TEST topology is already a CLI action: `startcluster test <topology>` (pg/turso/external-*), honored by boot actions only. | `deploy/startcluster.sh:593-599`, :62 |
| F9 | The pod UI builds commands as `[script, env_key, action]` — the test topology selector value is passed verbatim as the action. `boot_action()` returns "up" for work, "pg" for test. | `pod_ui/ops.py:27-43` |
| F10 | Two state-plane env writers exist: work (`# ---- state plane overrides (added by startcluster work) ----` → POSTGRES_* hardcoded) and per-topology for test/custom (`# ---- state plane (custom cluster, $topology topology) ----`). | `deploy/startcluster.sh:322-329`, :849-853 |
| F11 | `my-bundles/` is gitignored; work.env carries real API keys. Ground bundles cannot be committed. | `.gitignore:273`; `deploy/work.env` |
| F12 | Port corridors: mcp 8000-8099, mgmt 8100-8199, system 8200-8299, metrics 8300-8399, ui 8400-8499. 5433 (pg publish) sits outside; 9080 (proposed sqld publish) outside. Custom-cluster backup is pg-only (`pg_dump`); the docs already note turso volumes are "advanced: tar the volume". | `config/ports.json`; `deploy/startcluster.sh:911-918`, :548 |

---

## 2. Tier model (D1 — agreed with user)

| Tier | What | Ground | Owner |
|------|------|--------|-------|
| 1 | ~630 hermetic logic tests (parsing, env rewriting, UI helpers) | host, no pods | `python -m pytest` (unchanged) |
| 2 | NEW: real-pod smoke probes (~15), run against recreatable grounds | real containers, pinned bundles | `./startgrounds ensure` + pytest fixture, **opt-in** |
| 3 | Full-surface live MCP sweeps against the REAL clusters | real clusters | user (manual, `.agents/skills/mcp-live-tool-test`) |

**Claim rule (the argument for the split):** a test belongs on tier 2 iff its
claim mentions a container-boundary fact (image contents, mounts, published
ports, engine reachability, identity import). Everything else stays tier 1.

**Hygiene (Phase 0):** tier-1 pod-* test docstrings must say they never touch
pods (F1) so the F1→F4 category confusion cannot recur.

---

## 3. Testing grounds

### D2 — Grounds are custom clusters, not a new stack

A ground = `startcluster create <name> --topology <t> --nodes N --tools <list>
-b <bundle>` (F7). The grounds manager is a thin policy layer (ensure / match /
recreate / probe) over the existing registry verbs. No new compose stack for
grounds — but see C1/C3: two generator gaps must be closed first (HF mount,
identity seeding) and one extension is deferred to Phase 2 (work layout).

**Ground taxonomy (corrected by the check pass):**

| Ground | Shape | Phase | podman project |
|--------|-------|-------|----------------|
| `g1-pg` | 1 node, all six tools, pg state plane, bundle env | 1 | `g1-pg` |
| `g1-turso` | same, turso state plane | 2 | `g1-turso` |
| `g2-pg` / `g2-turso` | 2 nodes (multihost invariants), pg/turso | 3 | `g2-*` |
| `gwork-*` | WORK-LAYOUT ground (host networking, canonical ports, work mounts) | 2, via new `--layout work` | `gwork-*` |

v1 grounds (Phase 1) are honest about what they are: **single-node all-tools
clusters in the custom-cluster shape** — they exercise the container boundary
(image contents, mounts, engine reachability, identity seeding) but NOT the
work layout itself (host-network port behavior, canonical-port collisions,
real-data-plane mounts). That residue is exactly what `gwork-*` (Phase 2) adds,
and it is required to validate the D8 topology switch safely. No namespace
collision with `mcp-work` / `mcp-multihost` (F7: project = name).

### D3 — Ports (RESOLVED by check pass; was Q1)

- **v1 grounds: port-shifted, by construction.** `gen_custom_compose.py`
  always publishes shifted ports (auto base 20000-20999, F7/F13), never
  canonical. Grounds therefore COEXIST with the host launcher, BASIC and TEST
  simultaneously — no three-way exclusion, no preflight refusals in v1. The
  manager and probes read the actual ports from the registry JSON
  (`deploy/clusters.d/<name>.json` → `ports.central`, `ports.tools`).
- **`gwork-*` grounds (Phase 2): canonical ports + exclusion.** Built via the
  new `--layout work` extension (Q1-narrowed: implement inside `create` as a
  generator branch, or parameterize the work path's project name — the
  implementation choice stays open). These are the only grounds that need the
  mutual-exclusion preflight (reuse `pod_ui/state.py:foreign_port_holders`).

### D4 — Ground env source, secrets, and the mount fixes

- Grounds boot from a **pinned bundle** (`-b`), harvested once per ground from
  the host stack (`./startcluster harvest <ground>-seed`). Bundles stay in
  gitignored `my-bundles/` (F11) — nothing secret is committed.
- What IS committed: `deploy/grounds/<name>.spec` — key-free spec: create
  flags (topology, nodes, tools), bundle *expectation* (path + sha256), probe
  list name. The manager applies the spec via `create`, so the runtime
  registry (`deploy/clusters.d/`, machine-local) stays distinct from the
  committed intent; hash mismatch → drift flag (D5).
- **Embedding mounts (C2):** the HF-cache bind mount must be added in THREE
  places — `deploy/compose-common.yml` (test nodes, the F4 bug),
  `deploy/gen_custom_compose.py` `node_service` (grounds), and the Phase-2
  `--layout work` generator. A ground that cannot load BGE-M3 is exactly the
  failure tier 2 exists to catch; the §3.5 probe turns it red instead of silent.

### D5 — Ensure / match / recreate policy ("pre-testing recreation")

`./startgrounds <verb> [name]` (new top-level script, matching the
`startcluster` / `startpodui` convention; alternative: startcluster
subcommand — Q5):

| Verb | Behavior |
|------|----------|
| `ensure` | detect (podman ps by project) → absent: apply spec via `create` + `up`; present + spec/bundle hash matches: no-op; present + drifted: **recreate** (stop + `rm` + `create` + `up`), printing what it did — grounds are disposable by definition and `ensure` runs pre-probe, so there is no evidence to preserve at this point (counter-review R2). Escape hatch `--no-recreate` for inspecting a drifted ground |
| `recreate` | stop + `rm` + `create` + `up` from the spec (typed confirm, reuses the WIPE_ACTIONS stdin pattern) |
| `status` | machine-readable JSON: per ground — exists/running, topology, registry ports, bundle hash, health |
| `probe` | run the tier-2 probe list against an ensured ground; non-zero exit on any failure; NEVER recreates on failure (preserve evidence for logs/inspect) |
| `stop` / `clean` | thin wrappers over registry verbs |

**Identity seeding for bundle grounds (C3, Phase-1 deliverable):** `create -b`
today generates only the env file — nothing mounts the bundle's
`identity/*.json` into the node, so the one-time users.json import never runs
on custom clusters and probe 7 is impossible. Fix: `custom_up` chains a
generated override compose (reuse the existing `write_bundle_compose` /
hidden `bundle-compose` verb) mounting `<bundle>/identity/users.json` RO into
node1, and the `-f` chain must survive bare re-`up` — the exact lesson of the
work E10 fix (138b471). Regression test required (tier 1).

**Data-plane isolation (safety property, keep):** v1 custom grounds mount NO
host data (F13) — they physically cannot write the real turso store or real
users.json. The Phase-2 `gwork-*` layout mounts work-shaped volumes and MUST
mount **copies** (ground-local users.json + ground-local turso dir), never the
real files read-write — a work-layout ground writing real memories would
defeat the point of a disposable ground.

Never recreate automatically on probe failure. Recreate triggers: explicit
command, or topology/bundle-hash mismatch confirmed by the caller.

### D6 — pytest integration

- Session fixture `pod_ground(name)` in `tests/conftest.py`, gated by env
  `POD_GROUNDS=<name>[,<name>]`. Unset → tier-2 tests **skip** (same pattern as
  the `pg_dsn` reachability fixture, F-adjacent precedent). The fixture shells
  `./startgrounds ensure <name>` and reads `status` JSON for ports.
- Tier-2 probes live in `tests/test_pod_smoke.py` (skipif no `POD_GROUNDS`).
- Default `python -m pytest` stays hermetic and fast — grounds runs are an
  explicit act (`POD_GROUNDS=g1-pg python -m pytest tests/test_pod_smoke.py`).

### 3.5 — Tier-2 probe list (v1: `g1-*` grounds; `g2-*` invariants Phase 3; `gwork-*` Phase 2)

1. every ground container `Up (healthy)` (podman ps)
2. central 8200: unauthenticated request rejected (P0 posture), keyed request OK
3. per tool server (6×): `tools/list` — **oracle is a floor, not an exact
   count** (counter-review R3): assert `len(functions) >= <minimum>` AND the
   server's canonical function is present (e.g. `echo` for simplemcp). Exact
   counts rot on every tool addition; precedent for the `>=` pattern:
   `tests/fef_v3/test_runner.py:348` (asserts extension counts from a config
   table with `>=`). Optional stronger check: diff the ground's function-name
   set against the host launcher's own `tools/list` when both are reachable.
4. per tool server (6×): one safe `tools/call` (safe-args table from
   `.agents/skills/mcp-live-tool-test`), result not `isError`
5. **in-container embedding probe** (the F4 catcher): `podman exec <work>
   python3 -c "load BAAI/bge-m3; embed('probe')"` → returns dim 1024
6. state-plane roundtrip: write a tagged state doc via central API, read it back
7. identity import: admin `mcp_key` from the bundle's users.json authenticates **AND the db path
   was actually taken** — positive assertion (a tagged state doc/user row lands in SQL, read back
   via a direct SQL query) + absence of the "staying on users.json" warning in the container log.
   Rationale (R9): `users_store.py:124-125` and `:154-155` BOTH degrade to JSON on db-init failure
   with only a log warning — a successful login alone can pass while silently testing the fallback.
8. mask enforcement: a masked function call returns "Unknown tool" semantics
9. memory roundtrip: upsert + query a `ground-smoke`-tagged collection, delete
   (self-cleaning, dry-run-first recipe)
10. probe runner verifies it is talking to the GROUND (container id from
    podman vs the server's reported node name) — attribution, closing the F2 blind spot

TEST grounds add (Phase 3): lb :18080 door routes to both nodes; **cross-node
state visibility** (write via node1, read via node2 — the multihost invariant);
MinIO artifact roundtrip; mask push propagates through lb. `gwork-*` grounds
add (Phase 2): canonical-port probes verbatim (the existing live-test port
conventions), the D8 turso identity plane on 9080, backup/restore branch, and
the topology round trip of edge 2.

---

## 4. BASIC topology switch (identity/state-plane db: pg | turso)

### D7 — CLI: follow the TEST pattern

`startcluster work <topology>` boots/rebuilds with that topology (mirror of
`test <topology>`, F8); `work up` = boot with the last-used topology (default
pg), stamp recorded as a `[topology] <t>` header line in `deploy/work.env`
(precedent: the `[bundle] GENERATED` header, F11-adjacent). This keeps the pod
UI plumbing unchanged: `boot_action("work")` returns the stored topology and
`fire("work", <value>)` flows through `build_command` verbatim (F9).

### D8 — Compose + env writer

- New `deploy/compose-work.turso.yml` override: `db` = `libsql-server:latest`,
  volume `work_sqldata`, healthcheck `curl -sf http://127.0.0.1:8080/health`
  (healthcheck pattern from `compose-turso.yml` — safe to copy, it runs INSIDE
  the db container), **published `127.0.0.1:9080:8080`** (outside all
  corridors, F12). pg path unchanged (`compose-work.yml` db today).
- The work state-plane writer (startcluster.sh:322) branches per topology:
  pg → the current POSTGRES_* block (unchanged); turso → `MCP_USERS_BACKEND=db`,
  `MCP_STATE_BACKEND=db`, `TURSO_DATABASE_URL=http://127.0.0.1:9080` (no auth
  token — open inside the podman net only, same contract as `compose-turso.yml`).
- **NETWORK MODEL — deliberately different from the file being copied
  (counter-review R1).** `compose-turso.yml` serves its db to nodes over the
  podman bridge (`http://db:8080`, podman DNS) and publishes NO ports. The
  work node runs `network_mode: host` — it shares the HOST netns and
  **cannot resolve `db`**. The node→db path exists ONLY through the published
  loopback port (exactly how pg works today: `POSTGRES_HOST=127.0.0.1`,
  `POSTGRES_PORT=5433` + the publish at `compose-work.yml:27`). Mirroring
  `compose-turso.yml` verbatim — the obvious move its citation invites —
  produces a db with no publish and a node pointing at an unresolvable
  `db:8080`: silent connection refusal at boot. The publish on 9080 and the
  `127.0.0.1` DSN are therefore LOAD-BEARING, not incidental.

### D9 — UI

- The work card gets the same topology selector as the test card (values:
  `pg (pgvector)`, `turso (sqld)`; **no external-\* options for work in v1** —
  external identity would break the users.json import story).
- Engine-CHANGE rebuild confirm must name the cost: "Rebuild cluster work
  (turso)? Drops the pgvector identity volume; users re-import from users.json;
  masks/state re-seed. Memory data (turso file store) is untouched."
- Drop the "fixed by node.env" tooltip asymmetry; BASIC and TEST cards now read
  the same way.

### D10 — Backup / restore

- pg: unchanged (`pg_dump` via 127.0.0.1:5433).
- turso: v1 = stop + `podman volume export` tar of `work_sqldata` (precedent:
  startcluster.sh:548 already calls this the advanced path). The custom-cluster
  backup verb is pg-only today (F12) — the topology switch must extend or gate
  it, or `startcluster work backup` on a turso ground/cluster fails misleadingly.

### D11 — Data-plane boundary (veto-able, user-agreed reading)

The selector covers the **identity/state plane only**. BASIC's data planes stay
inherited from the host `.env`/bundle (turso file store bind-mounted at the
identical path). Making memory data a boot option invites wiping real memories
with a misclick.

### D12 — Docs debt cleared in the same change

Retire "photocopy of the host": `deploy/PODS-EXPLAINED.md:91`, the
`compose-work.yml:11-13` header comment, and rewrite `deploy/ARCHITECTURE.md`
§pod_mcp-work (db section becomes topology-parameterized). Replace with the
true contract: "identity/state = DB-backed per topology; data planes = verbatim
host .env/bundle."

---

## 5. Phasing

| Phase | Content | Why this order |
|-------|---------|----------------|
| 0 (hours) | Tier-1 doc clarity (F1); HF-cache mount in `compose-common.yml` **and `gen_custom_compose.py`** (F4 + C2); ground spec dir skeleton | unblocks real verification; fixes a real bug class |
| 1 | **Unit-pin `gen_custom_compose` + create flag parsing (C5 — the path has zero coverage today)**; `custom_up` bundle-override chain for identity seeding (C3, with tier-1 regression test); `./startgrounds` + `g1-pg` + `tests/test_pod_smoke.py` (probes 1-10) + `POD_GROUNDS` fixture | grounds stand on tested foundation before anything boots |
| 2a | `--layout work` extension → `gwork-pg`/`gwork-turso` grounds (copies of identity/data, never the real files); gwork probe set incl. **turso identity import** (edge 5) and the pg→turso→pg round trip (edge 2) | build the safety net BEFORE the surgery it guards |
| 2b — **GATE: ships only if 2a probes are green** | work topology switch (D7-D12); tier-1 tests de-hardcoded (`POSTGRES_PORT=5433` → topology-aware); docs (D12) | validation and shipping are separate gates, not one cell (counter-review R4) |
| 3 | `g2-*` two-node grounds + multihost invariant probes | extends coverage; no blockers |

---

## 6. Edge cases & failure playbook (prescribed catalogue)

1. **Port exclusion — `gwork-*` grounds only (Phase 2)**: ground ensure while
   the host launcher or BASIC holds the canonical ports → refuse, name the
   holder (existing preflight prints holder PIDs); never kill automatically.
   v1 `g1-*`/`g2-*` grounds are port-shifted (D3) and need no exclusion.
2. **pg → turso → pg round trip (RESOLVED design, was Q3)**: password
   stability is NOT a new problem — `pw_tmpfile_from` already carries
   POSTGRES_PASSWORD across work.env regeneration in bundle mode (F14), and
   host mode never rewrites an existing work.env. The D8 topology writer must
   reuse `pw_tmpfile_from` verbatim; REQUIRED regression: switch pg → turso →
   pg on a `gwork-*` ground and assert the same password + the old pgdata
   volume reattaches (proves no silent wipe-or-auth-fail). The topology STAMP
   (last-used topology) still needs persisting — add a `[topology] <t>` header
   line in work.env (D7), not a sidecar.
3. **Bundle drift**: recorded sha256 ≠ bundle on disk → `ensure` RECREATES by
   default (both hashes printed in the log; `--no-recreate` to inspect);
   `recreate` always proceeds after typed confirm. Only `probe` never
   recreates — it is the evidence-preserving verb.
4. **Probe failure**: keep the ground running (evidence), exit 1, print
   `./startgrounds status` + container log hints; never auto-wipe.
5. **One-time identity import**: every fresh identity volume re-imports the
   bundle users.json (mounted RO, via the C3 override chain). Turso identity
   import has never run anywhere. Sizing facts (verified, counter-review R8):
   the db backend EXISTS — `tools/shared/users_store.py:101`
   (`MCP_USERS_BACKEND`) and `tools/shared/state_docs.py:35`
   (`MCP_STATE_BACKEND`) — but `tests/test_users_store.py` db-path tests run
   against a FAKE DB-API double ("standing in for libsql/psycopg",
   test_users_store.py:288,317-343): neither libsql nor Postgres is exercised
   at the unit tier, so Phase 2 is configuration + live validation, not
   greenfield — but also not unit-covered. If the import path assumes
   pg_dump/psql anywhere it needs a turso path (cf. custom_verb restore,
   F12). Exercise on `gwork-turso` (Phase 2a) before touching BASIC.
6. **MinIO**: `g2-*` specs must carry `MINIO_ROOT_PASSWORD` (pod UI already
   warns when missing, `pod_ui/pages_v3/home.py:235`).
7. **Tier-2 timeout**: bounded ensure (healthcheck retries × overall ~5 min
   ceiling) → fail loud, not hang. `custom_up` already waits on central
   `/health` (40 polls) — extend the same pattern, don't replace it.
8. **Never touch real-cluster volumes**: v1 grounds are isolated by
   construction (F13); `gwork-*` grounds must mount COPIES of users.json and
   the turso data dir — assert in the generator (not just convention) that no
   `gwork-*` mount resolves to the real paths. Probes assert the container
   project prefix before touching anything (attribution, probe 10).
9. **Untested foundation (C5)**: the whole custom-cluster path ships with zero
   test coverage — Phase 1 starts by unit-pinning `gen_custom_compose.py`
   (pure function of registry JSON) and the create flag parser before the
   first ground boots.

---

## 7. Questions — resolved by the self-check vs still open

**Resolved 2026-10-05 (verified in code, not assumptions):**

- ~~Q1~~ — **v1 grounds port-shifted** (generator does it by construction, D3);
  canonical ports + exclusion only for Phase-2 `gwork-*`. NARROWED: the
  `--layout work` implementation choice (create flag vs parameterized work
  path) remains OPEN.
- ~~Q2~~ — **Answered, and the answer broke DRAFT 1**: `create` does NOT
  reproduce the work layout (F13) and drops bundle identity seeding (C3).
  D2/D3/D4/D5 rewritten accordingly.
- ~~Q3~~ — **Reuse `pw_tmpfile_from`** (F14); regression test replaces the
  proposed sidecar (edge 2).
- ~~Q4~~ — **9080 free** (F15); acceptable convention.
- ~~Q7~~ — **Committed thin spec (`deploy/grounds/<name>.spec`, key-free,
  bundle sha256) APPLIED via `create`**; the machine-local registry stays
  runtime state. The registry entry itself is also key-free (F7) but contains
  runtime-resolved ports that differ per machine — wrong thing to commit.

**Still open for the external review pass — RESOLVED BY DEFAULT 2026-10-05 (veto-able recommendations; nothing blocks execution):**

- **Q5 → `./startgrounds` as a separate top-level script** (matches the
  `startcluster` / `startpodui` convention; a grounds manager is an
  orchestration surface, not a cluster verb).
- **Q6 → TEST-node embedding probe happens at the next TEST boot** (Phase 0's
  mount fix is in; the probe is one `podman exec` — fold into Phase 3 setup
  or any incidental TEST reboot, whichever comes first).
- **Q8 → `--layout work` as a generator branch** in
  `gen_custom_compose.py` (one source of truth for shapes, registry-driven
  ports; the work-path-parameterization alternative forks env/mount paths).
- **Q9 → keep `postgres:16-alpine` for `g1-*`/`g2-*`** (identity tables don't
  need pgvector; work parity arrives with the Phase-2a `gwork-*` layout,
  which reuses `compose-work.yml`'s pgvector image).
- **Q10 → tier 2 is human-invoked in v1** (say so plainly in the docs);
  revisit a weekly cron at Phase-1 completion with real run-cost numbers
  (safe unattended: v1 grounds are port-shifted, D3).

---

## 8. Cost estimate

| Item | Cost |
|------|------|
| Phase 0 | hours |
| Phase 1 (manager + probes + fixture) | one build; per run: boot ≈ 1-3 min (cached image) + probes ≈ 1-2 min |
| Disk per ground | ~0.5-1 GB (db volume + logs) |
| Default `pytest` | unchanged (tier-2 skips without `POD_GROUNDS`) |
| Phase 2 | compose variant + env-writer branch + UI selector + test de-hardcoding + docs |
