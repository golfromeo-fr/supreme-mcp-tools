# Evolutions plan — 2026-09-06

Status: PROPOSAL for the next phase, per user direction ("let's plan a memory explorer", "expand evolution 1", "we could plan a bit on multi-host / multi-users"). Nothing here is built. Companion to `plans/follow-up-report-2026-09-06.md` and `TODO.md`.

Priorities set by the user 2026-09-06: **E1 runtime masks** (expand + build) → **E2 Memory Explorer** (plan now) → **E3 multi-user/multi-host** (plan now, build later). Graphify re-run postponed ("too soon"), packaging postponed ("too early").

---

## E1 — Runtime Function-Mask toggling (expanded)

**The problem.** Since D4 (`a9a6c41`), masks are enforced: `apply_function_masks()` runs once at tool-server import and calls `mcp.disable(names=...)` (fastmcp 4 native). Toggling in the UI persists to `tools_config.json` immediately, but the *running* server never re-reads the file — a mask change needs a launcher restart to take effect. The user experience lies by omission: the UI says "Mask saved. Changes take effect immediately." — true for the file, not yet for the serving path.

**Why it's cheap: fastmcp 4 already supports both directions at runtime.** Empirically verified 2026-09-05 (probe + `tests/test_function_masks.py`): `await mcp.disable(names={"x"})` hides the tool from `list_tools()` and makes calls raise `Unknown tool`; `mcp.enable(names={"x"})` restores it. We only need to reach the running FastMCP instance from a control endpoint.

**Design.**

1. **Wiring mcp into the mgmt server.** Each tool process already runs both the MCP server and its FEF V3 management server (`ExtensionHTTPServer`, port 81xx) in the same process. The launcher calls `tool.setup_extensions(registry=...)` at startup (`server_manager.py:147-155`). Extend that handoff: the registry gains a `mcp_instance` reference (or `setup_extensions` passes it explicitly), so the mgmt server can reach `mcp` without new globals. ~20 lines.
2. **New mgmt endpoint** on the per-tool server: `POST /admin/function-masks` body `{"tool": "<tool_name>", "masked": true|false}` — same API-key auth as the rest of the 81xx server. Handler: `mcp.disable(names={tool})` / `mcp.enable(names={tool})`, then persist the new `disabled_tools` list to `tools_config.json` via `tools/shared/atomic_io.atomic_write_json` (file first or file after — see edge cases). ~40 lines + error handling.
3. **Central API pass-through.** The UI talks to the central server (8200); add `POST /api/tools/{name}/masks/{tool_name}` that forwards to the tool's 81xx endpoint (the central server already knows each tool's mgmt URL from the service registry). The UI's existing `apply_function_mask` helper (`mcp_ui/components/tool_settings.py:38`) switches from writing config directly to calling this endpoint. ~30 lines.
4. **Restart stability** comes free: boot still reads `tools_config.json` (`function_masks.py`) — because live toggles persist to the same file, a restart converges to whatever the UI last set.

**Edge cases to handle in the implementation.**
- *File/runtime drift ordering:* persist to file first, then toggle runtime. If the toggle call fails, the UI shows the error and the file can be reverted — never the reverse (runtime toggled but file lost = a restart silently re-enables a tool the user thinks is masked).
- *Concurrent UI writes:* already serialized — all writers go through `atomic_write_json` (D2) with the `.lock` sidecar.
- *Launcher startup sync:* `launchmcp.py` rewrites the `tools` section of tools_config.json at boot but must not clobber `disabled_tools` — verify `validate_and_cleanup_config` only prunes entries for tools that no longer exist (it does; add a regression test).
- *Stateless/SSE parity:* none needed — disable happens at the FastMCP instance level, so all three transports (/mcp, /mcp-stateless, /sse) are covered at once.

**Verify:** integration test that toggles a tool over the 81xx endpoint and asserts `tools/list` hides it immediately (fastmcp Client against the live app); UI round-trip stays as today's mask-dialog flow. **Effort:** ~half day.

---

## E2 — Memory Explorer (plan)

A management-UI tab to browse, search, inspect, and manage memorymcp contents — the UI's first window into actual MCP data (everything today is config/masks/monitoring).

**Principle: the UI consumes the MCP tool surface, never the backend directly.** The UI process already holds every tool's API key (reads `tools/<name>/config.json`) and fastmcp is importable there — so the Explorer calls memorymcp's `/mcp` endpoint as a privileged MCP client. No backend credentials in the UI, no layering violation, and every action the Explorer can do is exactly what an MCP client could do.

**Server-side gap (small):** memorymcp has query/get/delete/metrics/audit tools but **no browse tool**. Add one MCP tool:

- `listMemories(limit=20, offset=0, tag=None, sort="recent")` → id, preview, tags, sensitivity, created/updated, access count. A thin paged wrapper over the existing sql/vector stores (the `iter_all` streaming contract already exists in both impls). ~1h + tests; it also benefits any future client.

**UI (new "Memory" tab, same pattern as the Functions tab — drawer entry + tab, never buried):**
1. **Search view** (v1): text box → `queryMemory`, results as cards (preview, tags, score, date); click → **detail view** via `getMemory` (full text, incl. ArtifactStore rehydration — the D2/C2 path gets a user-visible payoff), provenance/source, and the per-memory `auditTrail`.
2. **Browse view** (v1): `listMemories` paged table with tag filter chips.
3. **Actions** (v1): delete with confirm (`deleteMemory` — uses `_delete_artifact_key` cleanup automatically). No edit in v1 (editing memory text touches the redaction pipeline — defer).
4. **Dashboard strip** (v2): `getMemoryMetrics` (counts by type, backend, storage size) + `decayOrExpire` dry-run preview and run button (danger-gated).
5. **Graph view** (v3, optional): reuse `exportGraphAsMarkdown`/`createMemoryEdge` for a relationship pane.

**NiceGUI traps to respect** (from the mcp_ui audit, all documented): no `password_toggle_button`, no `@ui.refreshable` for the results container (use `container.clear()` rebuilds), no absolute-center positioning, header actions live in the drawer.

**Effort:** listMemories + tests ~1.5h; UI tab with search/browse/detail/delete ~3h; live sweep addition (Explorer actions are native calls, so the sweep-all rule covers them). Roughly a day.

---

## E3 — Multi-user / multi-host (plan — the big arc)

User goal: several humans/agents use the same deployment; **each has their own key, sees only their allowed tools, and everything is attributable**. The user's instinct is right about the building blocks: fastmcp 4's auth/middleware layer and our stateless endpoint are exactly what makes this tractable.

**What already supports it (verified):**
- `DualHeaderVerifier` takes a *map* of tokens → identities (`tokens={key: {"client_id": ..., "scopes": [...]}}`, see `tests/test_era_negotiation.py:56-58`) — multi-key auth is native, we just pass one key today.
- `/mcp-stateless` serves every request fresh with no session affinity — any node can answer any request, which is the prerequisite for load-balanced multi-host (and for Anthropic-style stateless clients).
- Backends (Turso/PG/Qdrant) are already network services, not local files — stateless nodes can share state.
- `mcp.access` logging logs every request per server — attribution needs only the identity, not new plumbing.
- Function masks are enforced server-side (D4) — the per-*server* half of visibility control exists.

**The two gaps, in build order:**

**M1 (spike) — ANSWERED 2026-09-07, see `plans/e3-m1-identity-spike-2026-09-07.md`:** yes, middleware sees the caller (get_http_request() raw token → client_id via the verifier's map; verified AccessToken is not exposed but resolution is exact post-auth); per-user list/call gating demonstrated live; decision = middleware filter path. Remaining M1 work = productionize (tools/shared/identity.py + factory hook + regression tests). Original spike text: can fastmcp 4 middleware see *who* is calling? Specifically: does the middleware `Context` (hooks `on_list_tools` / `on_call_tool`) expose the authenticated identity from `DualHeaderVerifier`? If yes → per-user visibility is a middleware filter (list: drop tools not allowed for this caller; call: reject). If no → per-user visibility needs per-identity server instances (fastmcp proxies/transforms — heavier). **This spike gates everything below; build nothing before it.**

**M2 — user store + keys (≈1 day):** a small purpose-built user store (users, per-user key hash, allowed-tools map, enabled flag) — deliberately minimal (the D1 lesson: no speculative pluggable backends). JSON file under `~/.config` with `atomic_write_json`, or single-file SQLite if lookup-per-request wants indexes. Central mgmt API (8200) gains `POST/GET/DELETE /api/users` + key rotation; mcp_ui gains a **Users tab** (create user, issue/copy key once, toggle allowed tools per server — reusing the Functions-tab mask matrix UI pattern). Keys reach tool servers the way masks would in E1: boot reads the store; a `POST /admin/identities` endpoint pushes the current token map to running servers (DualHeaderVerifier accepts a tokens dict — refresh = rebuild verifier or call an update method if fastmcp exposes one; spike M1 confirms).

**M3 — per-user visibility (≈1 day, after M1):** allowed-tools enforcement as middleware driven by the caller's identity: `tools/list` filters, `tools/call` rejects with "Unknown tool" (same UX as masks — no information leak). The existing global masks (D4) compose: a tool masked globally is invisible to everyone; user allow-lists intersect with it.

**M4 — multi-host (design doc, then build):** N nodes each running `launchmcp.py` with a tool subset; one shared backend combo (Turso+Turso or PG+Qdrant); LB in front (stateless endpoints make this trivially round-robin); node identity in `mcp.access` lines; the central mgmt API aggregates nodes (service registry already abstracts "where is tool X"). Session-bearing `/mcp` stays per-node (sticky) while stateless clients float.

**Honest constraints to record:** per-user *data* isolation in memorymcp (user A's memories vs user B's) is a separate, harder problem (tag-by-owner or partitioned collections) — out of scope until someone needs it; the free-plan quota and single-machine reality mean M4 is a design-first exercise.

**Suggested sequencing:** E1 (half day) → M1 spike (half day) → E2 Memory Explorer (a day, independent — can interleave) → M2 → M3 → M4 doc.

## E3 implementation plan (2026-09-07)

Full spec-grade plan: **`plans/e3-multiuser-overhaul-2026-09-07.md`** — whole-project scope per the user's clarification (mcp_ui logins, keys, roles; 12-surface inventory in the spike doc). Phases P0→M3 on `feature/e3-multiuser`; E4 tx_owner binding included (F7: identity visible inside tools).

## E4 — databasemcp transaction management (planned 2026-09-07)

Full plan: **`plans/databasemcp-transactions-2026-09-07.md`** (spec-grade, implementer probes included). Two tiers: atomic multi-statement batches (`execute_sql statements=[]`) and interactive transactions (`begin_transaction` → `tx_id` → `commit`/`rollback`, idle reaper mandatory). 15 → 18 tools. Load-bearing probe fact (2026-09-07, live): the libsql shared connection already carries explicit BEGIN/ROLLBACK across MCP calls — capability proven, hazard proven (no lock ⇒ other callers join the open tx) — so Tier 2 pins a dedicated per-tx connection on every dialect. **Sequencing: strictly after the `feature/databasemcp` merge**, as branch `feature/db-transactions` off merged main (~1–1.5 days). Note for E3: a `tx_id` is a bearer capability — when M2/M3 add identities, transactions should bind to the caller's identity.
