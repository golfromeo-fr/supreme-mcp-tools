# Code review — standalone host deployment (mono), 2026-11-11

**Scope:** the base code at HEAD (`2fea496`) as used by the STANDALONE host launcher
(`./startlauncher`, `MCP_AUTH_MODE=mono` — the user's primary usage), NOT the pod path.
Two disjoint surfaces: (1) launcher orchestration + entry chain + management plane,
(2) shared serving/auth stack + store impls + tool entry points.

**Method:** two parallel general-purpose review agents (whole-tree reads, not a diff),
per the 2026-10-18 runbook; every medium+ finding re-verified inline by the main session
(code reads + behavioral probes). Agents ran 224 targeted tests + in-process probes on
scratch state only; no live process or data was touched. Baseline: prior full review
2026-10-18 + fix batch merged 2026-10-27 (`e4e7441`); delta emphasis on UTCP
(`416239c`/`0213430`), quiet-404 logging (`eadafac`), databasemcp preset/libsql fixes
(`140f704`/`ffd88cd`/`902ba31`), mainlined scripts (`f492f20`/`9a3bfb1`), turso hardening
(`eed7996`), log-handler fix (`50fbf1f`).

**Verdict:** no critical findings. The mono auth/mask/session core — including the new
UTCP dialect — is solid. The real defects cluster in (a) FTS5 sync gaps in the store
layer, one of which silently corrupts a core memorymcp operation on the LIVE turso
backend, and (b) launcher startup/entry-chain edge paths. Fix order at the bottom.

Full suite at HEAD (run in parallel with the review): **1296 passed, 0 failed, 18 skipped**.

---

## Findings (impact-ordered)

Verification legend: **V-inline** = main session re-read the code / re-ran the probe;
**V-agent** = review agent traced the path or ran a behavioral probe (not re-run inline).

### Data integrity (store layer)

**R1 — HIGH — `upsertMemory(memory_id=…)` with changed text silently diverges SQL from vector (LIVE turso backend).** V-inline + agent probe.
`tools/shared/impls/turso_sql.py:202-248` — `upsert_memory` dedups ONLY by
`(text_hash, memory_type)`; there is no by-id lookup. Updating an existing id with new
text: hash SELECT misses → INSERT hits the `id` PRIMARY KEY → `UNIQUE constraint failed:
memories.id` (the `ON CONFLICT(text_hash, memory_type)` clause does not cover the id PK)
→ raise (S6) → `memory_tools.py:312-316` catches and continues vector-only → the tool
returns SUCCESS while the SQL row keeps the OLD text/hash/tags/last_accessed forever.
Consequences: keyword lane (`search_text`) serves stale text and stale dedup hash;
`migrate_store` export propagates the stale SQL text; SQL-side decay criteria drift.
`postgres_sql.py:297-308` has the identical shape (PG combo). `memory_id` update is a
documented core operation (the tool's own parameter doc). No contract test covers
update-with-existing-id.
**Fix:** resolve by-id first and UPDATE that row (with the S3 FTS old-text replay), or
add an `ON CONFLICT(id)` branch; add a contract test on both impls.

**R2 — MEDIUM — `decay_memories` deletes SQL rows without FTS5 sync → phantom keyword-lane hits (LIVE store).** V-inline + agent probe.
`tools/shared/impls/turso_sql.py:413-441` — bare `DELETE FROM memories`; triggers are
deliberately never created, and unlike `delete_memory`/`_update_existing` there is no
FTS `'delete'` command. Probe: decay a row, insert a new one (SQLite reuses the rowid)
→ `search_text` for the decayed term returns the NEW unrelated memory. Orphan doclists
also grow unbounded; nothing rebuilds `memories_fts`. Live caller: `decayOrExpire` →
`memory_tools.py:983`.
**Fix:** select (rowid, text) of the doomed rows and issue FTS `'delete'` per row before
the DELETE (same pattern as `delete_memory`).

**R3 — MEDIUM — turso_vector FTS sidecar drifts on re-upsert AND delete (ragmcp sparse/hybrid).** V-inline + agent probe.
`tools/shared/impls/turso_vector.py:299-308` — upsert INSERTs into the external-content
FTS without deleting the old entry: old terms still match after a text-changing
re-upsert. (The prior review's "CLEARED by live probe" verdict for this area is refuted
— that probe evidently used identical text.) `:350-365` — `delete()` never syncs FTS:
a deleted point's terms match a NEW point after rowid reuse. Affects ragmcp re-indexing
and stale-file removal. memorymcp's vector collection is `sparse=False`, so its vector
side is unaffected (its SQL side is R2).
**Fix:** same family as R2 — replay old text via FTS `'delete'` before update/delete,
or schedule `rebuild_sparse`.

**R4 — MEDIUM (latent, PG+pgvector combo) — `PostgresVectorStore.scroll` fails on every call.** V-inline.
`tools/shared/impls/postgres_vector.py:329,333` — `params = params + [limit]` but LIMIT
is f-string-interpolated, leaving one extra bind parameter → psycopg placeholder-count
ProgrammingError. The S2 fix corrected the three query methods but missed `scroll`.
Breaks `listMemories` (`memory_core.py:210` scroll_all) and ragmcp `get_indexed_files`
on the documented PG combo. Contract tests cover Turso scroll only.
**Fix:** drop the appended `limit` (or bind LIMIT as `%s`); extend the contract test.

### Launcher lifecycle / entry chain

**R5 — MEDIUM — Orphaned MCP server task when `start_server` fails after task creation.** V-inline + agent behavioral probe.
`launcher/server_manager.py:193-230` — the MCP uvicorn task is created at :193, but
`self.servers`/`self.tasks` are populated only at :212-213 after `mgmt_server.start()`.
The cleanup block (:220-225) only cancels tasks found in `self.tasks`, so if
`mgmt_server.start()` raises — or the 30s `SERVER_STARTUP_TIMEOUT` in `launchmcp.py:355`
cancels the coroutine (CancelledError bypasses `except Exception`) — the MCP task is
orphaned: it keeps serving the MCP port, untracked. Proven in-process: with
`ExtensionHTTPServer.start` patched to raise, the MCP port remained bound while
`servers`/`tasks` stayed empty. Consequences: tool reported FAILED while actually
serving; `launchmcp.py:415` releases the port reservation (re-allocation → bind
conflict); `stop_all_servers` never stops the orphan. The :210 comment shows the intent
("avoid zombie state") — the fix missed the already-created task.
**Fix:** register the task in `self.tasks` immediately after creation, or cancel the
local `task` in the except block and on CancelledError (try/finally).

**R6 — MEDIUM — `./startlauncher --engine modern` is broken with `-t/--all` (and with the default tool list).** V-inline (behavioral).
`scripts/startlauncher.sh:106-113` forwards tool names as POSITIONAL args for both
engines; `launcher/__main__.py:parse_args` defines no positionals (only `--tools`).
Probe: `parse_args(['webmcp'])` → `error: unrecognized arguments: webmcp`. So
`./startlauncher --engine modern` dies immediately on the default 6-tool list, and
`-t webmcp` / `--all` likewise. The script advertises these as engine-agnostic.
**Fix:** in the modern branch translate `TOOLS` to `--tools a,b,c` (omit for `--all`).

**R7 — LOW-MED — No start path verifies the bind; a dead MCP port still reports "healthy".** V-agent (code path traced; trigger = port race, inferred).
`server_manager.py:247` sets `status="running"` before `serve()` binds;
`ExtensionHTTPServer.start` (`http_server.py:391-394`) and `ManagementServer.start`
(`management_server.py:1086-1089`) create the serve task, `sleep(0.5)`, log success
unconditionally; `launchmcp.py:683/:584` swallow serve errors into a log line. If a port
is grabbed between the availability probe and the bind, the MCP server dies silently
while its 81xx mgmt server stays up — and registry health polls only mgmt `/health`, so
central `/api/tools` shows the tool healthy with a dead MCP endpoint.
**Fix:** await `server.started` (or probe the bound socket) before declaring success;
mark the instance errored when the serve task exits early.

### Auth / consistency (mostly latent or multi-mode)

**R8 — LOW-MED (multi-mode only) — two databasemcp entry points escaped the F1 gate sweep.** V-inline (first site) + V-agent.
`tools/databasemcp/db_tools.py:805` — `get_valid_languages` rides `_with_entry(None, …)`
with no `_assert_preset_grant` (can trigger the legacy env-default lazy connect).
`:1084-1099` — `optimize_sql_with_ai`'s schema-context block calls `REGISTRY.get(None)`
+ `describe_table` for any table named in the SQL: a non-admin can extract
column/comment metadata of the active connection (and it is sent to the external AI
endpoint). All other entry points are correctly gated. Mono unaffected (admin).
**Fix:** gate both with `_assert_preset_grant(None)`.

**R9 — LOW-MED (libsql `file::memory:` only) — transactions and batches write to a different database.** V-agent probe.
`tools/databasemcp/dialects/libsql.py:163-175` — `open_tx` opens a SECOND
`libsql.connect(url)`; each `file::memory:` connect is a private empty DB. So
`begin_transaction` / `execute_sql(statements=[…])` on an in-memory connection report
success while every write lands in a throwaway DB. File/`libsql://` URLs are fine.
No `file::memory:` preset exists in the host `.env` today (checked) — latent.
**Fix:** reject tx/batch on `:memory:` URLs loudly, or use a shared-cache memory URL.

**R10 — LOW — `MCP_HEALTH_CHECK_LOGS` env/.env value is always clobbered (precedence bug).** V-inline.
`launchmcp.py:239-244` — `--health-check-logs` default `"enable"` makes
`if args.health_check_logs:` (:741-743) always true → the env var is overwritten even
when no flag was passed. `startlauncher.sh:23,91-93` separately forces its own default
`disable` as an explicit flag, so the documented `.env` knob never wins on either path.
Violates flag > env > .env > default.
**Fix:** argparse `default=None`, set env only when the flag was actually passed; in the
script pass the flag only when `STARTLAUNCHER_HEALTH_CHECK_LOGS`/flag is set.

**R11 — LOW — Boot-time tools-config sync REPLACES the inventory with the masked list.** V-inline + agent fastmcp probe.
`launchmcp.py:893-922` — builds `discovered` from in-process `mcp.list_tools()` (which
excludes disabled tools — probed on fastmcp 4.0.0) and ASSIGNS
`config_data["tools"][server] = tool_names`, bypassing `merge_discovered_tools`'s UNION
guard written for the 07bdb1a "mask self-destruct" lesson (`tools_config.py:356-368`).
Every boot shrinks the inventory by exactly the masked functions; any `list_tools`
exception writes `[]`, wiping a server's whole inventory. Masks themselves survive
(no-prune guard intact) and the UI compensates (`tool_settings.py:273` unions
inventory+disabled), so impact = inventory drift.
**Fix:** route the boot sync through `merge_discovered_tools`; skip empty results.

**R12 — LOW — Residual unlocked read-modify-write on tools_config (F2 class).** V-agent.
Mask mutators run under `atomic_mutate_json` (good), but
`update_config_with_discovered_tools` (`tools_config.py:389-395`),
`validate_and_cleanup_config` (:410-425) and the launchmcp boot sync (:917-922) still
load→save outside the flock. A mask toggle racing the boot-sync window can be clobbered.
**Fix:** run these through `_mutate_config` too.

**R13 — LOW — `revoke_system_key` missed the S1 flock conversion.** V-inline.
`tools/shared/users_store.py:584-593` — raw `load_users → mutate → save_users`; every
other CRUD runs under `_mutate_users`' sidecar flock. A revoke racing another store
mutation can be silently lost (revocation doesn't persist).
**Fix:** route through `_mutate_users`.

**R14 — LOW — `.env` first-creation still 0644 (incomplete L3 fix).** V-agent.
`launcher/env_manager.py:460-465` — the create-if-missing branch uses `open("w")`
(umask → 0644); the 0o640 fix (:471) only covers the flock update branch.
**Fix:** create via `os.open(..., 0o640)` in that branch as well.

**R15 — LOW — Central→81xx proxy auth is config-only, not env-first.** V-agent.
`launcher/distributed_registry.py:655-660` — `_auth_headers` reads `load_auth_config`
only. With `<NAME>_API_KEY`/`MCP_API_KEY` env keys, the 81xx servers accept the env key
(`server_manager.py:124-125` uses `resolve_tool_api_key`) but central's proxy calls send
the config key → 401. Same class as the fixed `sync_tools_config` L3 item, different
site. Latent on this host (config keys are real).
**Fix:** use `resolve_tool_api_key`.

**R16 — LOW — `PUT /api/tools/{tool}/auth` accepts an empty key.** V-agent.
`launcher/management_server.py:229-231,1045` — `AuthUpdate.api_key` has no non-empty
constraint, while `put_tool_config` (:953-958) explicitly guards this exact lockout.
**Fix:** same non-empty validation.

### Latent / edge / legacy

**R17 — LOW (latent) — S6 fix deleted the BLOB-fallback DDL but kept its success message.** V-agent (diff-confirmed).
`tools/shared/impls/turso_vector.py:206-224` — on a build genuinely lacking VECTOR, the
except branch logs "created with BLOB embedding" but executes no CREATE TABLE → later
upserts fail "no such table". Live build has VECTOR.
**Fix:** restore the BLOB CREATE in that branch, or raise honestly.

**R18 — LOW (edge) — multi-statement guard desyncs on double-quoted identifiers containing a single quote.** V-agent analysis.
`tools/databasemcp/dialects/libsql.py:19-30` tracks only `'`; `SELECT "it's"; DROP …`
passes the guard and libsql silently runs only the first statement — the exact
partial-execution the guard exists to prevent. Bounded: caller already holds execute
rights; no escalation.
**Fix:** track `"` state too (reject on ambiguity).

**R19 — LOW (latent) — CORS allow-lists point at the wrong port.** V-agent.
`launcher/management_server.py:335-339` builds origins from `reserved.central_management`
(8200) with an `or 8400` fallback revealing the intent (the UI port); `http_server.py:135-139`
uses the tool's own mgmt port. Neither allows the actual UI origin (8400). No functional
impact today (mcp_ui's api_client is server-side aiohttp).
**Fix:** allow the `management_ui` reserved port.

**R20 — LOW (legacy CLI, never in the serving path) — `launcher.migration rollback` deletes the backup it restores from.** V-agent.
`launcher/migration.py:366-371` — rollback `rmtree`s the whole config dir (destroying
the backup under `backups/`, plus live masks and users.json) BEFORE
`copytree(backup_config, …)`, which then fails because the source is gone.
`_phase_backup` (:223-225) also copies the config dir into a destination inside itself.
Only bites if someone runs the V1→V3 utility.
**Fix:** restore from the backup before removing anything, or exclude `backups/`.

**R21 — LOW batch — small dead/misleading surfaces.** V-agent.
- `launcher/__main__.py:167-173` — `--log-level` parsed but never used (silent no-op).
- Both entries: `--transport` choices only `streamable-http|sse` while env/config also
  accept `multi` and `streamable-stateless` — CLI can't select what env can.
- `launcher/debug.py:256,275,303` — `circuit`/`cache` commands hit routes that don't
  exist (always 404); `logs` reads a directory that is never written.
- `launcher/service_registry.py:114-121` — `_should_log_health` dead; `errors-only`
  mode doesn't suppress registry failure warnings.
- Known-unfixed L3 leftovers (previously reported, listed for the record): central WS
  accepts only the system key; blocking `time.sleep` retry loops on the loop
  (`port_manager.py:299,350`); failed-start `_mgmt` reservation leak;
  `sync_tools_config.py:43` uses `localhost` (works via address fallback).

---

## What both passes agree is CLEAN (mono host deployment)

- **Mono auth core:** hmac-compare everywhere, keyless central = reads-open /
  mutations-403, 81xx admin gates fail closed, env-first key resolution on the serving
  path, traversal guards on config writes.
- **UTCP dialect (newest surface, `416239c`+`0213430`):** probed in mono AND multi with
  a scratch users.json — 401 without/wrong key; per-user masks dropped from the manual
  and 404 on call; server-reachability and disabled users refused; E1 native disables
  honored on both routes; flush gate 403 for role=user / 200 for admin. fastmcp's
  `RequestContextMiddleware` IS deduplicated onto the parent app, so
  `get_http_request()` works inside UTCP tool calls — the tool-internal E3.5 gates
  apply on UTCP too. (The module docstring's "ContextVar unset" claim is wrong, but the
  code is safe — doc fix only.)
- **Session management:** TTL, flush fail-closed under user-store verifier, quiet-404
  skip is 404-only.
- **Recent delta:** `eed7996` key sanitization, `902ba31` query-credential masking,
  `140f704` nickname resolution, `ffd88cd` build-agnostic autocommit (error text probed
  not to echo auth tokens), `50fbf1f` log-handler fix — all verified correct.

## Prior-review fix spot-checks (e4e7441 batch — all confirmed present)

- `is_user_store` fail-closed markers: `_memory_caller` (memory_tools.py:80-102),
  databasemcp connections.py:394-443, ragmcp_fastmcp.py:103-128 — all discriminate on
  the verifier marker, fail closed under user-store, fail open in mono.
- databasemcp `connection=None` gating resolves the active entry and refuses
  non-granted actives; explicit-name fail-open closed; batch/tx/metadata/use_database
  gated; connect_database admin-only. (Two entry points missed → R8.)
- S2 PG query param order fixed (scroll missed → R4); S3 dedup-update FTS old-text
  replay present (decay missed → R2); F3 PG list_collections reverse-map present;
  S6 upsert RAISE present (interaction with by-id updates → R1); F2/S1 whole-RMW flock
  present (residual paths → R12, revoke missed → R13); L1/L2 shutdown deadlines +
  watchdog present; L3 deletions present (.env 0640 partial → R14); C4 reserved-key
  read fixed.

## Coverage (honest, per area)

| Area | Coverage |
|---|---|
| launcher/*.py (all 18) + subdirs (config/events/monitoring/plugins/resilience/security/tool_extensions) | READ full (agent 1) |
| launchmcp.py, startlauncher, scripts/startlauncher.sh, common.sh, startui.sh, startpodui.sh | READ full (agent 1) |
| scripts/setup_opencode.sh, setup_utcp_client.sh | GREPPED-ONLY (not delegated by startlauncher) |
| scripts/legacy/ | UNREAD (archived) |
| config/*.json as consumed | READ (agent 1) |
| server_factory, identity, users_store, function_masks, atomic_io, utcp_endpoint, store_factory | READ full (agent 2) |
| impls: turso_sql, turso_vector, postgres_vector, qdrant_vector, libsql_reconnect | READ full (agent 2) |
| impls: postgres_sql | upsert/get/delete regions READ; remainder prior-reviewed, unchanged since (git-verified) |
| databasemcp: db_tools, connections, presets, dialects/libsql | READ full (agent 2) |
| databasemcp: dialects/{postgres,oracle,_base}, core, rules | GREPPED-ONLY this pass; prior full review, unchanged |
| memorymcp: memory_core full; memory_tools ~60% (caller/upsert/get/list/delete/decay/provenance) | READ; remainder prior-reviewed, unchanged |
| ragmcp_fastmcp | resolver + wiring READ; remainder prior-reviewed, unchanged |
| simplemcp / webmcp / convertermcp | GREPPED-ONLY (wiring + body-param check); prior full review clean, unchanged since e4e7441 |
| mcp_ui / pod_ui / deploy | OUT OF SCOPE (standalone serving path only; targeted cross-boundary reads for verification only) |

Main-session inline verification: R1, R2, R3, R4, R5, R6 (behavioral), R8 (first site),
R10, R11, R13 re-read/probed directly; remaining findings carry the agents' traced
evidence. Tests: agents ran 224 targeted tests (all green) + scratch-state probes;
full-suite result below.

## Recommended fix order

1. **R1** — silent corruption of a documented core operation on the live backend.
2. **R2 + R3** — same FTS-sync root cause family; one contract-test pattern covers both.
3. **R5 + R6** — launcher startup reliability + the broken modern entry path.
4. **R4** — before anyone uses the PG+pgvector combo.
5. **R8, R9, R10–R16** — small, mechanical; batch them.
6. **R17–R21** — latent/edge/legacy; fix when touched, or delete the dead surfaces.

---

## FIXED — execution record (2026-11-12, branch `feature/review-fixes-2026-11-11`)

Counter-check verdict first (user requested): **all 25 findings confirmed at HEAD**, with
two amendments — R21's `--log-level` is a no-op only in the MODERN entry (legacy uses
it); R1's naively-simple fix would leave `text_hash` stale and let the OLD text dedup
back onto the row (hash now moves with the text). 3 commits, suite gate per commit:

| Commit | Items | Tests |
|---|---|---|
| `e7a061c` | R1 (turso+pg by-id update, hash moves), R2 (decay FTS sync), R3 (turso_vector re-upsert/delete FTS replay), R4 (PG scroll params + first-page offset) | +5 behavioral (R1×2 turso, R2, R3×2), R1-PG mock pin, R4 real-PG scroll |
| `d844191` | R5 (orphan MCP task cancelled on failed start), R6 (modern `--tools` forwarding) | +`tests/test_startlauncher_scripts.py` (5, exec-shimmed) + R5 ordering pin |
| `774947c` | R7 (bind verification on both mgmt starts), R8 (2 escaped databasemcp gates), R9 (open_tx refuses `:memory:`), R10, R11 (boot-sync UNION), R12, R13, R14, R15, R16 (empty-key 422), R17 (real BLOB DDL), R18 (double-quote guard), R19 (UI-port CORS), R20 (backup/rollback staging), R21 (log-level wired, transport choices, dead debug commands, dead registry helper) + leftovers (port-manager waits loop-safe, failed-start port release, sync 127.0.0.1) | +5 dialect tests (R9/R18), R5/R8 pins |

**Bonus defects found & fixed while applying** (not in the original 25, all caught by the
new tests): PG `scroll` second bug — with the extra param removed, the first page still
died on `int(offset)` with `offset=None`; legacy engine intermix — a forwarded bare word
before `--health-check-logs` crashed argparse ("unrecognized arguments"); libsql DECAY
`rowcount` unreliable (count now `len(doomed)`).

**One item NOT changed — adjudicated by design:** central WebSocket system-key-only
auth. The central management API's action routes are admin-only by policy; the WS
extension-event stream belongs to the same admin surface, and accepting user keys there
would widen it in multi-mode. Left as-is, documented here rather than forced.

Full suite after all three commits: **1316 passed / 0 failed / 18 skipped**
(baseline 1296/0/18). Nothing merged or pushed — merge/push per user decision.

---

*Report generated 2026-11-11 by the two-agent review runbook + inline verification; fix record appended 2026-11-12.*