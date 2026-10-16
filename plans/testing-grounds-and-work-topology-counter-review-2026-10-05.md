# Counter-review — testing grounds + BASIC topology switch

**Reviewer:** adversarial pass against `plans/testing-grounds-and-work-topology-2026-10-05.md` (DRAFT 2)
**Date:** 2026-10-05
**Scope:** §1 "Verified context" table (F1–F16) and the load-bearing assumptions in §3–§4 (D3–D12)
**Method:** independent reads of the tree; the plan's own citations re-derived, not trusted. Verdict is
REFUTED / PARTIALLY TRUE / CONFIRMED, never "looks right". No file was modified by this review.
**Branch reviewed:** `feature/pod-ui-gaps`, tip `193fca1`

**Headline:** the plan survives the attack. The §1 evidence table is unusually honest — its
own C1/C5 self-corrections check out. Two findings survive re-scoping, both as clarifications rather
than refutations: **D8's `compose-turso.yml` citation is a real trap** — that file's db has no
`ports:` publish, so the shape cannot be copied literally — and the plan's sizing of the
`MCP_USERS_BACKEND=db` work is **wrong: the path already exists, and neither backend has ever been
integration-tested.**

---

## 1. Verdict table

| # | Claim under attack | Verdict | Evidence observed |
|---|---|---|---|
| **F4** | TEST compose has no HF mount; `compose-work.yml:57` does | **CONFIRMED** | `deploy/compose-work.yml:57` → `- ${HOST_HF_CACHE}:/root/.cache/huggingface`. Grep `huggingface\|HF_HOME\|hf_cache` over `compose-common.yml`, `compose-test.bundle.yml`, `compose-work.bundle.yml`, `gen_custom_compose.py`, `Containerfile` → **zero hits** (not even a baked cache in the image). F4 is a real bug, and it also holds for the work *bundle* compose. |
| **F16** | `compose-test.bundle.yml` is generated + gitignored | **CONFIRMED** | `git check-ignore -v` → `.gitignore:269:deploy/compose-test.bundle.yml`; `.gitignore:268:deploy/compose-work.bundle.yml`. `git ls-files deploy/` returns only `bundle_node_env.py` — no `compose-*.bundle.yml` is tracked. Citing it as static in F4 would indeed have been wrong. |
| **F7 / C5** | No test references `gen_custom_compose` or `create`; `deploy/clusters.d/` empty | **CONFIRMED** | `ls -A deploy/clusters.d/` → empty. `grep -rln "gen_custom_compose\|clusters\.d\|create_cluster" tests/` → **no matches**. The custom-cluster path genuinely has zero coverage and zero live usage. C5's self-correction is accurate. |
| **F13-a** | `gen_custom_compose.py` node services have NO volume mounts | **CONFIRMED** | `gen_custom_compose.py:44-64` — `node_service()` builds exactly: `build`, `env_file`, `environment`, `ports`, `healthcheck`, `restart`, and conditionally `depends_on`. No `volumes` key. C2 is right that this is a second instance of the F4 hole. |
| **F13-b** | db image is `postgres:16-alpine` | **CONFIRMED** | `gen_custom_compose.py:34` → `"image": "docker.io/library/postgres:16-alpine"`. |
| **F13-c** | `depends_on` is `service_started` | **CONFIRMED** | `gen_custom_compose.py:92` → `{"db": {"condition": "service_started"}}`; `:99` → `{"node1": {"condition": "service_started"}}`. |
| **F13-d** | `MCP_NODE_CENTRAL_URL` uses podman DNS | **CONFIRMED** | `gen_custom_compose.py:53` → `"MCP_NODE_CENTRAL_URL": f"http://{node}:8200"`. |
| **F14** | `pw_tmpfile_from` exists and stashes POSTGRES_PASSWORD across work.env regeneration | **CONFIRMED** (citation off by ~3 lines) | Function body at `deploy/startcluster.sh:112-124`; the bundle-mode call is at **`:311`**, inside the `if [ -n "$BUNDLE_DIR" ]` branch. Plan cites `:308-320`; body is at `:112-124`. Same mechanism, imprecise pointer. Mechanism verified: it sed-extracts `POSTGRES_PASSWORD=`, writes a `chmod 600` temp env, and passes the **path** to `bundle_node_env` — the comment at `:114-117` records that passing the raw value made `open()` fail and generate a fresh password, bricking initialized pgdata. |
| **D8 (clarification)** | "The work node already runs `network_mode: host`" | **CONFIRMED, and the plan states the publish it depends on** | `compose-work.yml:44` → `network_mode: host` on service `work` (declared `:36`), comment `# canonical ports + host-side DSNs/presets`; `compose-work.yml:78` → same on service `ui`. Host netns means the DSN resolves against the **host's** loopback, which is what the `ports:` publish produces. The plan already states the publish (`127.0.0.1:9080:8080`) and gives it as the reason, so nothing is missing; service `db` (`:22`) publishes normally today (`127.0.0.1:5433:5432` at `:27`), confirming the shape. The residual risk is that the cite to `compose-turso.yml` is copied as a shape. See §2.1. |
| **D8** | `compose-turso.yml` is the pattern source: `libsql-server:latest` + `/health` on 8080 | **CONFIRMED for image + healthcheck; REACHABILITY PATTERN IS A TRAP** | `compose-turso.yml:11` → `image: ghcr.io/tursodatabase/libsql-server:latest`; `:16` → `test: curl -sf http://127.0.0.1:8080/health \|\| exit 1`. The healthcheck is real, but this file's db has **no `ports:` publish** — its siblings reach it over **podman DNS** at `http://db:8080` (`compose-turso.yml:6` comment). Lift the image and healthcheck; do **not** lift the service shape, or the host-networked work node boots healthy and fails on first sqld contact. Reachability comes from `compose-work.yml`'s published db. See §2.1. |
| **Probe 3 oracle** | No existing literal expected tool/function count to conflict with | **PARTIALLY TRUE** | No MCP tool-surface count oracle exists. Nearest neighbours: `tests/test_function_masks.py:98` → `assert len(_run(mcp.list_tools())) == 2` (a 2-tool synthetic fixture, not a real server contract) and `tests/fef_v3/test_runner.py:348` → `expected_count = len(self.TOOLS[tool_name]["extensions"])` (computed from a fixture table). No hard conflict, but probe 3 will partially duplicate the FEF V3 counting idea. There is also no authoritative count to validate *against* — probe 3's oracle would itself be hand-maintained, so it will drift silently unless it derives from source. |
| **F12** | 5433 is the work pg publish port, outside the corridors; 9080 also outside | **CONFIRMED** | `compose-work.yml:27` → `- "127.0.0.1:5433:5432"`. Programmatic check of all five ranges in `config/ports.json` (mcp 8000-8099, mgmt 8100-8199, system 8200-8299, metrics 8300-8399, ui 8400-8499): **`5433 in corridors: False`**, **`9080 in corridors: False`**. Also confirmed: custom-cluster backup is pg-only, and the turso volume path is separately flagged advanced (`startcluster.sh:548`). |

---

## 2. Findings that actually change the design

### 2.1 D8 — the citation to `compose-turso.yml` is a real trap (clarification, not refutation)

The plan does state the publish, and it names it as the reason. Both halves of D8:

> `db` = `libsql-server:latest`, volume `work_sqldata`, healthcheck `curl -sf
> http://127.0.0.1:8080/health` (pattern lifted from `compose-turso.yml`), published
> `127.0.0.1:9080:8080` (outside all corridors, F12).
>
> The work node already runs `network_mode: host`, so it reaches the published loopback port — no
> DNS dependency (this is why the publish port exists at all).

An earlier draft of this review claimed the plan "never says the turso db must be published at all".
**That premise was wrong and is retracted.** The `ports:` publish is in D8 bullet 1, and
"(this is why the publish port exists at all)" in bullet 3 *is* the reachability rationale. The line
is not missing and host-networking is not being offered as sufficient on its own.

What survives is narrower, and it is a citation problem rather than a design problem. The same
bullet marks the whole service as "pattern lifted from `compose-turso.yml`" — and that file's db has
**no `ports:` publish** at all, because its siblings reach it over podman DNS at `http://db:8080`
(`compose-turso.yml:6` comment). Compare the two existing shapes:

- `compose-work.yml` db — published to the host (`127.0.0.1:5433:5432` at `:27`), reached by the
  host-networked work node over loopback.
- `compose-turso.yml` db — **no publish**; reached by sibling services over podman DNS at
  `http://db:8080`.

The `network_mode: host` fact is **true** (`compose-work.yml:44`). Host networking *removes* the
container from the podman bridge network — it shares the host's network namespace, so
`TURSO_DATABASE_URL=http://127.0.0.1:9080` resolves against the **host's** loopback, not against a
sibling service. That is exactly what a `ports:` publish produces, which is why the arithmetic works.

So the trap is not a wrong design; it is a **shape-citation** an implementer can read literally. Copy
the *shape* of `compose-turso.yml`'s service block — named volume, healthcheck, no `ports:` — instead
of just its image and healthcheck, and you get a service that is **silently unreachable** from the
host-networked work node. Because host-networking also removes compose's port-publishing on the *node*
side, the failure is a connection refusal at first use, not a startup error: the node comes up
healthy, the healthcheck passes (it curls its own loopback), and the identity plane fails later when
it first talks to sqld.

**Recommended edit:** no design change — keep the publish exactly as written, and make it explicitly
**load-bearing** rather than incidental, so no one "simplifies" it away. Disambiguate the
source-of-truth citation: the "pattern lifted from `compose-turso.yml`" covers the *image and
healthcheck* only; **reachability** comes from `compose-work.yml`'s published db. That one
disambiguation is what closes the trap.

### 2.2 The `MCP_USERS_BACKEND=db` path already exists — Phase 2 is re-framed, not blocked

The plan does not treat the turso identity plane as work to be built. Its edge case 5 makes the
narrower claim, and that claim is accurate: **"Turso identity import has never run anywhere — the
import tooling is PG-tested only."** That is a statement about *validation*, not about
implementation. The path it depends on already exists:

- `tools/shared/users_store.py:101` — backend selection,
  `os.environ.get("MCP_USERS_BACKEND", "json").strip().lower()`.
- `tools/shared/users_store.py:76` — the `db` section header.
- `tools/shared/users_store.py:124` — raises when `MCP_USERS_BACKEND=db` but no SQL backend is
  configured (fail-loud, correct).
- `tools/shared/users_store.py:154` — init failure **falls back to JSON with a warning** rather than
  crashing (matches the README's documented behaviour).
- `MCP_STATE_BACKEND=db` is equally live: `tools/shared/state_docs.py:35` (selection),
  `launcher/tools_config.py:28` (M4/H2 routes the tools_config document to the shared store),
  `tools/shared/function_masks.py:46` (mask state gate), `tools/shared/cluster.py:44` (single-place
  gating).
- **Tests exist for both**: `tests/test_users_store.py` and `tests/test_state_docs.py`.

**But "PG-tested only" overstates coverage in the other direction too.** The tests do not run a real
libsql or Postgres instance: `tests/test_users_store.py:288` defines
`_FakeConn` — `"""Minimal DB-API connection standing in for libsql/psycopg."""` — a hand-rolled
DB-API stand-in that pattern-matches SQL strings. So the accurate statement is stronger and more
honest than the plan's: **the selection and fallback logic is stub-tested, while *neither* backend
has ever been integration-tested.** The real risk is therefore "never integration-tested", not
"unimplemented" — and "PG-tested only" should not be read as a passing end-to-end result.

Consequence: D8's env-writer work is **configuration, not greenfield**, and probe 7 (identity import)
has a real code path to exercise rather than a stub to build. The honest phrasing is "wire the existing
db backend to the turso state plane and prove it with a ground", not "add a db identity plane".
This *reduces* Phase 2 risk and cost; it does not block it. The residual work is a first-ever
integration run, which is a ground to stand up, not a subsystem to write.

One knock-on, and it is the strongest warning in this section: because `users_store.py:154`
**degrades to JSON on init failure**, probe 7 must assert the *db* path was actually taken (e.g. no
fallback warning, state doc lands in SQL) — otherwise the probe passes while silently testing the
JSON fallback, which is precisely the identity bug C3 is worried about. Given that the only existing
"db" coverage is `_FakeConn`, probe 7 is the first thing that can catch the stub standing in for a
real driver.

---

## 3. Design questions

### Q1 — Would tier-2 pod tests ever run automatically? **No. There is no automation.**

No `.github/`, no `.gitlab-ci.yml`, no `.circleci`, and no workflow YAML anywhere in the tree. The only
skill is `.agents/skills/mcp-live-tool-test/`, a manual procedure (its `scripts/sweep_all.py` and
`probe_eras.py` are run by hand, and `AGENTS.md` requires the agent to make native
`mcp__<server>__<function>` calls itself rather than shell out).

Combined with D6's `POD_GROUNDS` opt-in gate, tier 2 stays manual by construction. That matches the
plan's stated intent, so it is not a defect — but the plan should say it plainly instead of leaving
tier 2 looking like an automated tier. If automatic execution is ever wanted, a runner has to be added
and it is not in the phasing table.

### Q2 — Does a disposable-environment primitive already exist? **No. `./startgrounds` duplicates nothing.**

Grep for `scratch|sandbox|ephemeral|disposable` across `tools/`, `deploy/`, `tests/` hits only four
files, all **pytest fixture** sandboxes (`tests/test_config_bundle.py`, `tests/test_shared_cache.py`,
`tests/test_regression.py`, `tests/test_mcp_ui_repairs.py`). None manage a container lifecycle.

The nearest existing machinery is the registry verbs D5 already reuses — `custom_up` /
`custom_verb` (`startcluster.sh:874`, `:891`) and the typed-`WIPE_ACTIONS` stdin confirm pattern. D5's
re-derivation of those is correct, and "thin policy layer over existing verbs" is the right shape.

### Q3 — Turso identity import: implemented, not absent. See §2.2.

---

## 4. Could not verify

| Item | Why |
|---|---|
| **F15** ("port 9080 free; 5433 free while cluster is down") | Transient runtime state from a 2026-10-05 `ss -ltn` run; not reproducible from the tree. Treat as advisory only — commit `138b471` in this very branch exists because port collisions here are a recurring real failure. Re-check at execution time. |
| **F5** (`.env:36,80-87` — POSTGRES/QDRANT commented out, turso `file:` active) | `.env` is secret-bearing and excluded from read tooling. Corroborated indirectly by `deploy/ARCHITECTURE.md` and `graphify-out/`, consistent with the rename history, but not verified directly. |
| **F1, F3** line citations | Only spot-checked **F2** this pass: `tools/simplemcp/test_tools.py:41-60` does contain a `socket.create_connection` probe gate and a module-level `pytestmark = pytest.mark.skipif(...)`, consistent with "skips when nothing listens". F1/F3 (`test_config_bundle.py:5`, `test_pod_ui_helpers.py:21-23`, `test_pod_scrape.py:35`, `test_deploy_configs.py:46`) not independently re-read. |
| **F9** (`pod_ui/ops.py:27-43` `[script, env_key, action]` construction) | Not re-read this pass. It is load-bearing for D7's "plumbing unchanged" claim, so it deserves a check before Phase 2 sizing. |
| Whether `compose-work.bundle.yml` carries an HF mount | Grepped as part of F4: no. Consistent with the finding, so it extends rather than contradicts F4. |

---

## 5. Bottom line

- **§1 evidence table: trustworthy.** F4, F7, F12, F13, F14, F16 all re-derived and confirmed. The
  self-check pass caught its own C1/C5 errors correctly, and the F4 bug it found is real and *broader
  than stated* (it also affects the custom-cluster generator and the work bundle compose).
- **One citation needs disambiguating before implementation:** D8's "`compose-turso.yml`" pattern
  reference (§2.1). The design and the publish are both already correct; the hazard is that the cited
  file's db has no `ports:` and is reached by podman DNS, so an implementer who copies its *shape*
  instead of its image + healthcheck gets a node that boots healthy and fails on first sqld contact.
  Make the publish explicitly load-bearing and cite `compose-work.yml`'s published db for reachability.
- **One sizing correction:** Phase 2 is cheaper than planned — the db-backed identity/state path is
  already implemented (§2.2). But it is **stub-tested, not integration-tested**: `test_users_store.py:288`
  `_FakeConn` stands in for libsql/psycopg, so *neither* backend has ever run end-to-end. The residual
  work is a first integration run, not a subsystem. Probe 7 needs a stronger oracle so it cannot pass
  on the JSON fallback.
- **No blocking defects found.** Nothing here says "do not proceed".

---

## 6. Author adjudication (ZCode plan author, 2026-10-05, same day)

This §6 verifies THIS document's own claims (not just the summary the user relayed — note the
relayed message contained a *stronger* D8 framing, "justification is wrong / non-sequitur", that
§2.1 of this file had already retracted; the file is the authoritative artifact).

**Confirmed by independent re-read:** `users_store.py:76` (db section header), `:124-125` and
`:154-155` (warn-and-fallback sites), `launcher/tools_config.py:26-30` (`_state_backend_active`,
M4/H2), `tools/shared/function_masks.py:44-46`, `tools/shared/cluster.py:42-46` (`is_db_mode`
single-gate), `.gitignore:268-269` (both bundle composes ignored), `tests/test_function_masks.py:98`
(exact `== 2` on a synthetic 2-tool fixture), and F9 (`pod_ui/ops.py:27-43` — verified earlier the
same session: `boot_action` returns `up`/`pg`, `build_command` emits `[script, env_key, action]`).
The §4 "could not verify" item on F9 is thereby closed.

**Two corrections to this document:**

1. **§2.2 bullet 3 is wrong on mechanism: `users_store.py:124` does NOT raise.** Verified
   `:124-125` and `:154-155`: BOTH db-init failure paths (no SQL backend configured; backend init
   exception) `logger.warning(...)` and **stay on users.json**. The only `raise`s in the module are
   validation errors (`:173`, `:286`, `:347`, `:351`). This makes §2.2's "strongest warning"
   STRONGER than stated: *every* db-path failure silently degrades to JSON with only a log line.
   The probe-7 oracle (assert the db path was actually taken) is therefore not optional hardening —
   it is the only thing standing between "ground green" and "identity plane silently on JSON".
   Applied to the plan: probe 7 now requires a positive db-path assertion (data lands in SQL / no
   "staying on users.json" in the container log), not merely a successful login.
2. **§5's F4 broadening ("also affects ... the work bundle compose") is neutralized by live
   evidence.** `deploy/compose-work.bundle.yml` indeed has no HF mount (only the 9 bundle config
   mounts), but `podman inspect mcp-work_work_1` (the actual last bundle-managed boot, exited,
   inspected 2026-10-05) shows the full base mount set SURVIVED the `-f` override:
   `~/.cache/huggingface → /root/.cache/huggingface`, `~/turso_data → /home/gr/turso_data`,
   `work.env → /app/.env`, `users.json`, `work_artifacts`, `work_logs` — all present alongside the
   bundle mounts. podman-compose **appends** service volume lists across `-f` files; it does not
   replace. So BASIC was never the embedding victim (TEST nodes remain the F4 hole), and the same
   append semantics is what makes the C3 users.json override chain safe for grounds. Recorded in
   the plan as F17.

**Disposition of this document's recommendations (all applied in plan DRAFT 3):**

| Recommendation | Where applied |
|---|---|
| §2.1 make the publish load-bearing + disambiguate the `compose-turso.yml` cite (image/healthcheck only; reachability from `compose-work.yml`) | plan D8 rewritten — "NETWORK MODEL" block |
| §2.2 reframe Phase 2 as configuration + first integration run | plan edge 5 rewritten |
| §2.2 probe 7 must assert the db path was taken | plan §3.5 probe 7 amended (per correction 1 above) |
| §3 Q1 say plainly that tier 2 is manual | plan §7 Q10 (decision point + recommendation) |
| §4 F15 advisory-only, re-verify at execution | plan F15 reframed (counter-review R7) |
| §4 F9 check before Phase 2 sizing | closed — verified (above) |

**Verdict on this document:** trustworthy — every re-derivable claim re-derived and confirmed
except the two corrections above, one of which (no raise at :124) it published against itself as a
warning that turns out to understate the hazard. Its self-retraction discipline (§2.1) is noted as
exemplary. Nothing withdrawn; plan proceeded to DRAFT 3 on the strength of it.
