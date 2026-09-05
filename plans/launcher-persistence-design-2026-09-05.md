# Launcher persistence unification — design pass (2026-09-05)

Status: PROPOSAL — no code changed. Feeds the deferred planning item from `codebase-improvement-plan-2026-09-04.md` ("ConfigPersistence / SQLitePersistence / EventStore — needs its own design pass").
Evidence base: full read-only map of every persistence mechanism (2026-09-05, explorer agent + hand verification of the P0 claim).

---

## P0 finding (discovered by this pass — promoted out of persistence scope)

**Function Masks are not enforced anywhere in the serving path.**

- `disabled_tools` is written by the management API (`launcher/management_server.py:404,414` → `launcher/tools_config.py:79-123`) and by the UI (`mcp_ui/components/tool_settings.py`), displayed in the UI, and pruned by `validate_and_cleanup_config()`.
- **No code applies it**: `launchmcp.py`, `launcher/tool_discovery.py`, `tools/shared/*`, and every tool server were checked — zero readers. The MCP tools list and execution are unaffected by masks.
- Consequence: the user's live masks (`webmcp.brave_search_web`, `simplemcp.get_secret` per `~/.config/supreme-mcp-tools/tools_config.json`) are UI-cosmetic only. Any MCP client (Kilo Code, ZCode, Copilot) still sees and can call both tools.

**Options (needs user decision):**
- **A. Enforce server-side (recommended).** Each tool server reads `disabled_tools` at startup (shared helper in `tools/shared/`, mtime-cached) and excludes masked tools from registration; FastMCP tools support enable/disable so runtime toggles via the existing mgmt server (81xx) become possible later. Masks then mean what the UI implies.
- **B. Reposition honestly.** Rename the UI feature to "hidden in UI" so it no longer implies call-level blocking.
- Either way the UI stays (user directive: never prune or bury the mask UI).

---

## 1. Current state (verified 2026-09-05)

Ground truth on disk: `~/.config/supreme-mcp-tools/` holds only `tools_config.json`, `simplemcp.json`, `webmcp.json`, empty `backups/`. **No `config.db`, `events.db`, `audit.log`, `dlq/`, `migration.json`, `fef_v3.json` — most of the FEF V3 "persistence layer" has never run.**

### Live stores

| Store | Writer(s) | Safety | Read back? | Retention |
|---|---|---|---|---|
| `config/ports.json` | none (hand-edited) | n/a | yes — SoT | n/a |
| `config/launcher_config.json` | none (hand-edited) | n/a | yes (`launcher_config.py:271`) | n/a; legacy `portAllocation` block dead (overridden at `:249-269`) |
| `tools/{t}/config.json` auth block | mgmt API `PUT /auth` (`management_server.py:535-562`) | bare truncate, no lock | yes (auth) | n/a; plaintext key in a repo file |
| `.env` | `env_manager.py:395-474` | flock + rewrite ✓ | yes | old values kept as comments **forever** (secrets persist) |
| `~/.config/.../tools_config.json` | launcher `tools_config.py:49` (bare truncate) AND UI `tool_settings.py:76` (flock+atomic) | **two writers, two safety levels** | yes | pruned by `validate_and_cleanup_config` |
| `~/.config/.../{tool}.json` mutation logs | `distributed_registry.py:481-488` | bare truncate | **never applied** — audit only | grows forever; **stores plaintext `api_key` params** (verified in `simplemcp.json`) |
| `logs/launcher.log` | size+timed handler (`config_types.py:103-133`) | bounded ✓ | yes | pruned ✓ |
| `.nicegui/storage-user-*.json` | UI sessions | per-session | cookie round-trip | **never pruned** (dozens since March) |

### Dead / dormant (zero callers, files never created)

| Component | Location | Note |
|---|---|---|
| `FileConfigPersistence` | `launcher/config/persistence.py:16` | re-exported only |
| `ConfigManager` | `launcher/config/manager.py:16` | would write a *different schema to the same paths* as the live mutation logs — latent collision |
| `SQLitePersistence` | `launcher/config/sqlite_persistence.py:17` | never called; `config.db` absent |
| `EventStore` + replay | `launcher/events/sourcing.py:37` | never called; `events.db` absent; only store with pruning (1M cap) |
| DLQ | `launcher/resilience/dead_letter_queue.py:38` | never called |
| `AuditLogger` | `launcher/security/audit.py:55` | never called; 4th audit schema duplicating EventStore's job |
| `HAConfig` / `DistributedConfig` save/load | `launcher/config/ha.py:111`, `distributed.py:178` | no callers, no default paths |
| `fef_v3.json` `persistence.type` switch | written `migration.py:309-331` | **never read by any loader** — the pluggable-backend design was never wired |
| `MigrationStatus` file | `migration.py:31-99` | CLI-only; never run in prod |

## 2. Diagnosis

The 2024-era design anticipated four interchangeable persistence backends for one concept ("tool X, extension Y changed with params P at T by U") plus an event log; only one ad-hoc variant ever ran, and nothing replays it. The cost today is not the dead code itself but **three real defects hiding next to it**:

1. **Unsafe writers on live files** — `tools_config.json` already suffered a truncation bug (documented in `tool_settings.py:77-83` docstring; the UI got the flock+atomic fix, the launcher didn't). Same bare-truncate pattern in mutation-log writes (`distributed_registry.py:488`) and auth writes (`management_server.py:561`). The launcher can still destroy the file the UI carefully guards.
2. **Secrets sprawl** — plaintext in `.env` (with forever-history), `tools/{t}/config.json`, and mutation logs. No shared redaction rule.
3. **Unbounded growth** — mutation logs, `.env` history, `.nicegui` session files.

## 3. Recommendation

**Phase D1 — delete the dead persistence layer (subtract first).** Remove `FileConfigPersistence`, `ConfigManager`, `SQLitePersistence`, `EventStore`, DLQ, `AuditLogger`, `HAConfig`/`DistributedConfig` save-load, and the `fef_v3.json` persistence block writer. Update the two `launcher/__init__.py` re-exports and `launcher/config/__init__.py`. Keep `MigrationStatus` only if the migration CLI stays (it does — it's the backend-migration tool). Verify: grep zero importers, suite green. Cost ~1-2h; kills ~1,500 lines that will otherwise rot and mislead.

**Phase D2 — one safe-write helper, three call sites.** Extract `atomic_write_json(path, data)` (flock + `.tmp` + `os.replace`, the `tool_settings.py:76` pattern) into `tools/shared/` or `launcher/`; convert `tools_config.py:49`, `distributed_registry.py:481-488`, `management_server.py:561`; make `tool_settings.py` import it (kills the duplicated path constant too). Verify: unit test (concurrent writers, kill-mid-write leaves old file intact). Cost ~1h.

**Phase D3 — retention + secrets hygiene (needs user nod on each).**
- `.env` history comments: drop or cap (secrets persist forever by design today).
- Mutation logs (`{tool}.json`): decide — (a) delete the feature (nothing reads mutations back; mgmt API `/mutate` endpoints are the only writers), (b) keep as bounded audit with secret redaction. Leans (a).
- `.nicegui` session files: prune >30d at UI startup.
- `launcher_config.json`: drop the dead `portAllocation` block (overridden at load since forever).

**Phase D4 — Function Masks enforcement (the P0; design in §P0, user picks A/B).**

**Out of scope:** unifying SQLite-vs-JSON backends for the live set — with the dead layer deleted there is exactly one of each, and ports/launcher configs are hand-edited files that should stay that way.

## 4. Execution order

D2 → D1 (helper first, so D1's surviving writers move onto it immediately) → D3 → D4. D2/D1 are mechanical (suite-gated); D3 needs per-item nods; D4 is a feature decision. Nothing here requires a launcher-down window except activating D4 (and C-batch activation already needs one).
