# Follow-up report — 2026-09-06

Companion to `TODO.md` (the canonical tracker). Covers: what just closed, the full case for the D1 deletion (see §1 — written out because it deserves more than a commit line), the open follow-ups that need your decision, and the evolution candidates for the next phase.

State: suite **687 passed**; launcher live with Function-Mask enforcement verified (fresh clients cannot see `brave_search_web`/`get_secret`); exit-guard fix `e16a346` active since your latest restart. Unpushed at time of writing: `fb48478` (D2), `1f4554e` (D1 + doc hygiene), plus this report.

---

## 1. The D1 deletion — why "never-run" is the operative phrase, and how to undo it

**What was deleted** (commit `1f4554e`, −2,454 lines):

| Module | What it was designed for |
|---|---|
| `launcher/config/persistence.py` (`FileConfigPersistence`) | JSON config store per tool |
| `launcher/config/sqlite_persistence.py` (`SQLitePersistence`) | SQLite `config.db` — change log + current-state tables |
| `launcher/config/manager.py` (`ConfigManager`) | Flat key-value config writer (restore-at-startup) |
| `launcher/events/sourcing.py` (`EventStore`) | `events.db` event log with time-travel `replay()` |
| `launcher/resilience/dead_letter_queue.py` | `dlq/*.json` failed-operation queue |
| `launcher/security/audit.py` (`AuditLogger`) | `audit.log` JSONL with rotation |
| `launcher/config/ha.py` + `distributed.py` | High-availability / multi-machine deployment config |
| `fef_v3.json` writer (migration phase 3) + the `fefV3` block in `launcher_config.json` | The switch that was supposed to select between the backends above |

**Why this is not scary — four independent pieces of evidence, all verified before deletion:**

1. **No importer.** Tree-wide grep (launcher/, tools/, mcp_ui/, tests/, both entry points) found zero imports of any of these classes outside the modules themselves and the re-export hubs (`launcher/__init__.py` etc.). Dead code that nothing can reach is, by definition, not load-bearing.
2. **No data ever existed.** These stores create their files on first write: `config.db`, `events.db`, `audit.log`, `dlq/`. Your real state directory (`~/.config/supreme-mcp-tools/`) contains only `tools_config.json`, two per-tool mutation logs, and an empty `backups/`. If any of this layer had *ever* run, its file would be there. It never ran — not once, in months of daily launcher use.
3. **The activation switch was itself dead.** `fef_v3.json`'s `persistence.type` field (and the `fefV3` block in `launcher_config.json`) was written by the migration CLI but **read by zero loaders** — there was no code path that could ever have started these backends. The layer wasn't disabled; it was unreachable.
4. **Nothing changed after deletion.** Suite went 676 → 687 (11 new tests, 0 failures), the launcher restarted 4/4, and live client calls behave identically. If anything live had depended on this code, that's where it would have broken.

**What was deliberately KEPT:** the FEF V3 framework itself is alive and untouched — the extension registry, per-tool management servers (81xx), `tool_extensions/`, the `EventAggregator` (in-memory, live: feeds the management UI's websocket events), and the one mutation-log writer that actually runs (`distributed_registry.ConfigPersistence` — its fate is D3, below). "FEF V3" is not deleted; only its never-wired persistence sub-layer is.

**How to reverse it:** everything is one commit. `git revert 1f4554e` restores all 2,454 lines exactly; no data was lost because (per evidence #2) no data was ever written. If the design intent (pluggable persistence for launcher config) matters to a future evolution, it's better rebuilt small against today's real needs than resurrected from 2024-era assumptions — that judgment call is yours.

**Process note, honestly stated:** your "close things as much as possible" instruction plus the design pass's classification ("D1/D2 mechanical, D3 needs nods") is what I acted on; you never named D1 explicitly. I compressed the justification into one commit line — that was too thin for a deletion this size, hence this section.

---

## 2. Open follow-ups — decisions needed

| # | Item | Detail | My recommendation |
|---|------|--------|-------------------|
| F1 | **D3a — `.env` history** | Every value ever set via the mgmt API stays in `.env` as a commented `# VAR=old` line — forever, secrets included (`env_manager.py:477-503`). | Drop the history comments (the file is not a changelog; git/logs cover history needs). Minutes. |
| F2 | **D3b — mutation logs** | `~/.config/supreme-mcp-tools/{tool}.json` append every extension-mutation call — including **plaintext api_key params** (verified in `simplemcp.json`) — and nothing ever reads them back. | Delete the feature (writer + files). It's an audit log nobody audits, holding secrets. Alternative: redact-and-cap. ~1h. |
| F3 | **D3c — `.nicegui` session files** | One `storage-user-*.json` per browser session since March, never pruned. | Prune >30 days at UI startup. ~30m. |
| F4 | **NiceGUI issue draft** | Paste-ready upstream report for the `password_toggle_button` event-wiring bug at `zcode-nicegui-issue-draft.md` (3.16.0 is latest; no duplicate found 2026-09-05). | Post it (minutes, closes a limbo item), or tell me to archive it. |
| F5 | **Security-audit follow-ups** | 4 low-value items parked in TODO: oraclemcp `SessionPool` (0.5d, needs an Oracle env), `table_name` regex defense-in-depth (minutes), atomic write in `migration.py` codegen (1h), copytree-before-rmtree in rollback (minutes). | Do the two minutes-level ones; record the rest as won't-unless-Oracle. |
| F6 | **Scratch hygiene** | Gitignored clutter: `proc_skill*` ×9, `test.txt`, `requirements.merged.txt`, `.env.20260623`, `.env~`, `startlauncher~`, `testlocalmcp~`, `BUG_REPORT.html`. | Delete after your glance — especially the `.env` backups (secrets). |
| F7 | **Push** | `fb48478`, `1f4554e`, this report. | Push. |

Say e.g. "F1-F3 as recommended, F4 post it, F6 delete" and everything in this table closes today.

## 3. Evolution candidates (next phase — no decisions made)

1. **Runtime Function-Mask toggling** — masks currently apply at server start. Add a small endpoint on each tool's existing 81xx mgmt server calling fastmcp's `enable()`/`disable()` (both directions already test-proven), kept in sync with `tools_config.json` so restarts agree. The UI toggle becomes instant. ~half day + tests.
2. **Fresh graphify run** — validates the node-id namespacing fix (the 400 same-label collisions), measures how the C/D batches reshaped the graph, and produces the next evidence-based lead list. Token-costly (~10M input tokens last time); best done when you're ready to invest in the *next* improvement batch.
3. **Memory Explorer in the UI** (audit item F-5) — a real memory browser over memorymcp; needs a memorymcp extension first. Medium-sized, user-facing value.
4. **Packaging & ops** — pip-installable launcher entry point, a systemd unit (or documented supervisor) for `startlauncher`, `.env.example` templating. Makes the stack portable beyond this machine.
5. **Multi-host arc** — the `/mcp-stateless` endpoint was built for load-balanced/multi-process setups; actually deploying that (or splitting tools across machines) is the big-design-item tier.

**Suggested order:** F1–F7 (close-out, today) → runtime masks (small, high-visibility win) → graphify re-run → pick the next arc from its evidence.

*All file:line references verified 2026-09-06; suite green at 687.*

---

## 4. Decisions (2026-09-06, user)

| # | Decision | Consequence |
|---|----------|-------------|
| F1 | **Keep `.env` history as-is** — "very practical, you can swap configuration in a flash" | The commented old-value history is a deliberate feature (instant config swap-back), not an accident. D3a closed as won't-change. Secret-hygiene note stands on record but the user accepts the trade-off. |
| F2 | **Mutation logs: keep** — user wants the debugging value; asked for more info | Facts gathered: 2 files, 1.7KB total, 11 mutations, span 2026-03-24 → 04-04 (dormant since). Debug value = "which extension config changed, when, with which params" (cache_config, timeout_config). Real cost: secret-named param values (e.g. `key`) sit plaintext. Remaining option if ever wanted: redact-on-write (mask values of secret-named params, keep names/timing). Decision: keep as-is for now. |
| F3 | **Session files: prune at 1 month** — implemented | `mcp_ui.management_ui.prune_stale_session_files()` runs at UI startup, removes `storage-user-*.json` idle >30d (`MCP_UI_SESSION_PRUNE_DAYS`, 0 disables). 31 of 62 files eligible on first run. Tests: `tests/test_session_prune.py`. |
| F4 | **NiceGUI draft: intention clarified, decision pending** | The draft is an upstream bug report for zauberzeug/nicegui: `password_toggle_button=True` breaks click events for later elements on the page (proven by bisection). Intention: contribute the fix knowledge upstream so the trap is fixed for everyone; our workaround is already in place, so this is optional goodwill, not a need. Post or archive whenever. |
| F5 | **Security follow-ups: clarified, skip for now** | The 4 items are oraclemcp/migration-CLI-scoped hardening from the June audit. oraclemcp is not even in the live tool set, and the migration CLI never ran in production — none touch the serving path. Recorded in TODO; revisit only if oraclemcp goes live. |
| F6 | **Scratch files: let them live** | Total 440KB — under the user's "not big" bar. Closed. |
| F7 | **Push: done** | All commits on `origin/main`. |

**Evolutions:** E1 runtime masks + E2 Memory Explorer + E3 multi-user/multi-host → planned in `plans/evolutions-plan-2026-09-06.md`. Graphify re-run postponed (not enough structural change to justify the tokens). Packaging postponed ("too early").
