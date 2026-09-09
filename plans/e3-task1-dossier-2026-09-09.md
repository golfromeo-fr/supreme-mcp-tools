# E3 next steps + TASK 1 dossier (2026-09-09)

Written for the glm-5.3 handoff. Part A = the standing roadmap proposals.
Part B = everything known about TASK 1 (the E3.5 integration failures) so the
analysis can start from evidence, not archaeology.

---

## Part A — roadmap proposals (as of 2026-09-09)

All E3/E3.5/UI work lives on `feature/e3-multiuser` (latest: db6a53d).
Suite: 863 passed (unit); integration suite status = Part B. Main is frozen
by user decision until multi-user is certified.

1. **Merge gate (next milestone).** User acceptance pass on the Users admin
   UI is underway (picker, Cancel, placeholders all user-approved 2026-09-09).
   Once certified: merge `feature/e3-multiuser` → main. TASK 1 (Part B)
   should be resolved first so the merge is honest.
2. **E3 hardening options** (independent, small):
   - per-user *central* management tokens — the multi-admin increment;
   - `users_store` on a DB backend (Turso/PG via the store ABCs) instead of
     `users.json` — user wants JSON now, DB later;
   - filter the UI sidebar by each user's `servers` grant (non-admins
     currently see tools their key is rejected on — cosmetic, enforcement is
     already server-side via the per-tool token map).
3. **M4 multi-host builds** — design done (`plans/m4-multihost-2026-09-08.md`),
   H1–H4 not started, parked on `feature/m4-multihost` (explicitly not a
   priority per user).
4. **Packaging + graphify** — postponed by user (2026-09-06), still parked.
5. **Anytime cleanups:** sweep leftover `e35test` memories from the live
   store; keep watching the one-off `.env` truncation that ate `MCP_USER_*`
   lines once (writer never identified).

---

## Part B — TASK 1 dossier: E3.5 integration failures

### How to run

```bash
# live launcher required, multi mode (MCP_AUTH_MODE=multi in .env)
cd /home/gr/supreme-mcp-tools
timeout 280 python3 tests/test_e35_data_plane.py
```

The suite self-creates accounts `e35admin` / `e35alice` / `e35bob` /
`e35carol` in `~/.config/supreme-mcp-tools/users.json`, calls the real MCP
servers over HTTP (ports 8000–8004 from `config/ports.json`), and deletes the
accounts at the end (unless it crashes mid-way — see caveats).

### Environment (2026-09-09)

- Launcher: pid 1904116, started Sep 9 00:56, `./launchmcp.py` with simplemcp
  ragmcp webmcp memorymcp databasemcp; central mgmt 8200 keyed by
  `MCP_MANAGEMENT_API_KEY`; UI on 8400 (user's instance).
- Branch `feature/e3-multiuser` at c2227fc. `.env` has `MCP_AUTH_MODE=multi`,
  `MCP_MANAGEMENT_API_KEY`, `MCP_UI_USERNAME/PASSWORD`, `MCP_USER_TESTER_*`.
- Tool API keys in `tools/<name>/config.json` under `auth.api_key`.

### Mechanical test-file bugs — FIXED (c2227fc)

Two scope bugs crashed the suite before completion and polluted the failure
picture:

- `main()` re-imported `time` at 3 sites → UnboundLocalError at the rotation
  section; removed the local imports.
- `STORE_PATH` was referenced but never defined → NameError in the
  store-corruption section; defined with the canonical
  `MCP_USERS_STORE` / `~/.config/supreme-mcp-tools/users.json` resolution.

With those fixed the suite runs end-to-end: **34 PASS / 10 FAIL**.

### Current failures (run B, complete)

```
✗ alice query sees own
✗ alice cannot delete bob's memory
✗ bob query via 02 denied
✗ alice disconnect 02
✗ old key rejected after rotation
✗ new key works after rotation
✗ get_secret masked for admin
✗ get_secret masked for alice
✗ carol denied on memorymcp
✗ hot-reload user works (external edit picked up)
```

### KEY EVIDENCE — the failure set drifts between runs

Run A (same launcher, minutes earlier, before the mechanical fixes):
`alice query sees own` **PASSED**; the two `get_secret` mask checks and
`carol denied on memorymcp` were not observed failing (earlier sections
scrolled past, but the rotation-adjacent failures were identical).

Run B (after fixes, full): `alice query sees own` **FAILED**, plus the
`get_secret`/`carol`/`hot-reload` failures appeared.

Interpretation: the checks are **not deterministic against a long-lived
launcher** — cross-run state pollution in the tool child processes is the
prime suspect. Each run re-creates the e35* accounts with NEW mcp_keys while
the 5 tool processes keep serving; anything those processes memoize about
identities/masks from previous runs leaks into the next run's results.
This matches the original diagnosis "need a fresh session against the live
launcher". Suggested first experiment for the new session: restart the
launcher, run the suite immediately (fresh state) and record the failure
set; then run it a second time WITHOUT restarting and diff the sets.

### Failure clusters and hypotheses

1. **Rotation propagation (old key still accepted, new key rejected).**
   users_store has an st_mtime_ns-based per-process cache
   (`tools/shared/users_store.py`, `tokens_map_for_tool()` rebuilds per
   verify); the test sleeps 1.0 s after rotating (eedebf2). Still failing.
   Check: do the tool child processes call `tokens_map_for_tool` per request,
   or does `DualHeaderVerifier`/`UserStoreVerifier`
   (`tools/shared/server_factory.py`) memoize the map or the auth decision?
   Also check fastmcp-side auth caching.
2. **Owner-scoping (alice query sees own / cannot delete bob's memory /
   carol denied).** `memory_tools.py` resolves the caller via
   `_memory_caller()` (request headers → verifier token map). Drift across
   runs suggests identity resolution sometimes returns a stale/wrong
   client_id — consistent with a memoized token map inside the memorymcp
   child process.
3. **databasemcp preset deny (bob query via 02 / alice disconnect 02).**
   `_assert_preset_grant()` in `tools/databasemcp/db_tools.py` runs
   `_current_caller_grants()` on the event loop before `to_thread`.
   `alice query via preset 02` PASSES both runs while the deny-side checks
   fail — look at whether the grant check sees the right caller or defaults
   to admin when identity resolution fails.
4. **get_secret mask failures (admin + alice).** Function masks are
   server-enforced at import (`function_masks.apply_function_masks`) plus E1
   runtime push from the central API. `get_secret` is a user-masked function
   per `~/.config/supreme-mcp-tools/tools_config.json` history — check
   whether a previous E1 runtime push left simplemcp's live mask state
   diverged from disk (the drift symptom again).
5. **hot-reload user (external edit picked up).** The test writes
   users.json DIRECTLY (`STORE_PATH.write_text`) while the launcher's
   management API writes it via `atomic_io` (tmp + os.replace). Direct
   in-place writes race the atomic writer and may be clobbered or missed by
   mtime-since-replace caching. This check may be testing an unrealistic
   write path — consider rewriting it to mutate via the management API.

### "alice query via preset 02" hang (watch item)

Did NOT reproduce in either run today (passed both times). Earlier it hung
indefinitely while direct psycopg to the same PG was fine; suspected
`get_http_request()` ContextVar propagation into `asyncio.to_thread` or
connection pile-up. Keep the timeout guard (`mcp_call` has one) and treat a
reappearance as new information, not a regression of a known bug.

### Gotchas for the debugging session

- Never call `/admin/flush-sessions` on a tool while testing THROUGH ZCode —
  it kills ZCode's own sessions.
- Native `mcp__<server>__<tool>` calls survive a launcher restart (modern
  2026-07-28 dialect); the launcher itself should be restarted via the
  user's `./startlauncher`.
- The suite leaves e35* accounts behind if it crashes — `users.json` may
  need manual cleanup between experiments (or delete via the Users UI).
- Users.json is live state: testuser/admin/tester are REAL accounts
  (testuser carries a user-set mask `simplemcp: [greet]` — do not "restore"
  over it).
