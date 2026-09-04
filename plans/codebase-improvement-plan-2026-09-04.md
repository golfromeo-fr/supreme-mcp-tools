# Codebase Improvement Plan — 2026-09-04

Derived from the graphify analysis (`graphify-out/GRAPH_REPORT.md`, 5,152 nodes / 7,925 edges / 377 communities) cross-checked against live code and this session's code review. Every item names exact files, the exact change, and how to verify it. No vague entries.

**Status:** pending · **Execution:** 2 waves of ≤3 parallel agents (ZAI plan limit) + main-thread tasks · **Gate:** full suite green after every task; one commit per task.

---

## Checklist

- [ ] **C1 — Cache unification** (2-3h)
- [ ] **C2 — ArtifactStore: wire in or delete** (decision + 1-2h)
- [ ] **C3 — Memory pipeline end-to-end test** (1h)
- [ ] **C4 — port_manager wrong-key read fix** (15m)
- [ ] **C5 — Deterministic test suite** (30m)
- [ ] **C6 — mcp_ui persistent secret + login-redirect fix** (1.5h)
- [ ] **C7 — Launcher child-shutdown fix** (1-2h, needs launcher-down window)
- [ ] **C8 — Delete dead modules** (30m)
- [ ] **C9 — oauth_fix documentation pointers** (5m)
- [ ] **C10 — text_utils micro-cleanups** (15m)

---

## C1 — Unify the three caches into one module

**Evidence:** graph flagged `SimpleCache`/`CacheManager`/`TTLCache` as semantically similar; three live implementations verified.
**Change:**
1. Extend `tools/shared/cache.py` into THE cache: TTL + size bound + cleanup (already there) + true LRU ordering (pattern from commit `17089aa`).
2. Replace `SimpleCache` in `tools/webmcp/webmcp_fastmcp.py:102-135` (~35 lines) with the shared cache; keep `MCP_CACHE_MAX_SIZE` env wiring.
3. Replace `CacheManager` in `launcher/distributed_registry.py:181-230` (~50 lines) with the shared cache; keep the async-lock semantics its callers rely on.
**Verify:** existing tests `TestMED16LRUCacheEviction`, `tests/test_bug_fixes_medium.py`, webmcp cache tests all green; grep confirms one `class.*Cache` in the tree.
**Deletes:** ~80 duplicated lines; future MED-16-class bugs become structurally impossible.

## C2 — ArtifactStore: wire it in or delete it

**Evidence:** god node #1 (66 edges) with **zero production callers** (verified during LOW-8); it consumed a bug fix + 19 tests for dead infrastructure.
**Decision needed from user:** wire-in (recommended) vs delete.
- **Wire-in:** `tools/memorymcp/memory_tools.py` `upsertMemory` stores text payloads > 8KB via `ArtifactStore.save`, keeping only the artifact key in the point payload; `getMemory`/`queryMemory` fetch-through on read. Honors its original design intent (memorymcp is the most-used server; payload bloat is real).
- **Delete:** remove `tools/shared/artifact_store.py` + `tests/test_artifact_store.py`; CHANGELOG "Removed" entry.
**Verify:** wire-in → new round-trip test (save >8KB memory → query → full text returned); delete → grep shows zero importers, suite green.

## C3 — Memory pipeline end-to-end test

**Evidence:** graph shows `upsertMemory()` is the only genuine 3-community bridge (Memory tools ↔ contract tests ↔ PII redaction); coverage is piecewise.
**Change:** new `tests/test_memory_e2e.py` (dependency-free, mocking at the store boundary): redact → upsert → query → decay asserts the full flow, including "sensitive text stored in redacted form".
**Verify:** the test passes and covers a path no existing test walks end-to-end.

## C4 — port_manager wrong-key read

**Evidence:** `launcher/port_manager.py:45` reads `cfg["assignments"]["system"]["central_management"]`, but ports.json keeps `central_management` under `reserved` — read always throws, hardcoded 8200 fallback masks it silently.
**Change:** read `reserved.central_management` with the assignments key as fallback; unit test asserts the real `config/ports.json` resolves to 8200 through the function (not the fallback).
**Verify:** new test fails on the old code, passes on the new.

## C5 — Deterministic test suite

**Evidence:** `tools/simplemcp/test_tools.py` (5 tests) fails with `ConnectionError` whenever the launcher is down — burned investigation time twice this week.
**Change:** session-scoped fixture probing 127.0.0.1:8002 once; `pytest.mark.skipif` on those 5 tests when the launcher is absent.
**Verify:** suite green with launcher up AND down; the skip is reported, not failed.

## C6 — mcp_ui persistent secret + login-redirect fix

**Evidence:** loop debug 2026-09-02 — `MCP_UI_SECRET` unset (ephemeral secret ⇒ every restart invalidates all sessions ⇒ stale-client reload dance); `try_login`'s client-side `ui.navigate.to("/")` races the Set-Cookie commit (login ping-pong suspected).
**Change:**
1. Generate + add `MCP_UI_SECRET` to `.env` (user runs `python -c "import secrets; print(secrets.token_hex(32))"`).
2. `mcp_ui/management_ui.py` `try_login` (lines ~223-230): replace `ui.navigate.to("/")` with a full-request redirect so the session cookie is committed before `main_page` reads it (or follow NiceGUI's official auth pattern).
**Verify:** login → main page on first try, no `/`↔`/login` bounce (watch for repeated `page: main_page loaded` lines); restart startui → existing session survives.

## C7 — Launcher child-shutdown fix

**Evidence:** TODO.md — third restart-race variant: tool children outlived the dying parent through the full 20s port window → 2/4 tools up.
**Change:** parent shutdown handler explicitly terminates tracked child PIDs (already tracked in `ServerManager`) before exiting, in addition to the existing retry window. **Needs:** launcher-down window + user nod on kill-children design.
**Verify:** simulated restart race test: parent killed → children gone within window → restart gets 4/4 ports first try.

## C8 — Delete dead modules

**Evidence:** `tools/shared/pg_store.py` has zero runtime importers (only legacy source-assertion tests — verified during HIGH-14); `old/mcp_ui_v2/` superseded twice over (graph twin-hub evidence).
**Change:** delete both; port the pg_store source assertions to target `tools/shared/impls/postgres_sql.py`; CHANGELOG "Removed" entries.
**Verify:** grep zero references; suite green.

## C9 — oauth_fix documentation pointers

**Evidence:** the graph's only 2 AMBIGUOUS edges are `apply_oauth_fix` ↔ servers — the suppression is applied dynamically in `server_factory`, invisible to static analysis.
**Change:** one comment at each server's creation site in `tools/webmcp/webmcp_fastmcp.py` + `tools/ragmcp/ragmcp_fastmcp.py`: "OAuth discovery suppression for Copilot: applied via tools/shared/oauth_fix at import (see server_factory)".
**Verify:** next graphify run shows those edges EXTRACTED, not AMBIGUOUS.

## C10 — text_utils micro-cleanups

**Evidence:** today's standards review.
**Change:** in `tools/memorymcp/text_utils.py` + `memory_tools.py`: use the `(score, method)` tag in mergeDuplicates' summary output ("12 cosine + 3 jaccard pairs" — better than dropping it); remove the dead `max(len(union), 1)` guard; drop the defensive `getattr(p, "vector", None)` on our own dataclass.
**Verify:** mergeDuplicates dry-run output shows the method breakdown; suite green.

---

## Execution order

| Wave | Tasks | Why |
|------|-------|-----|
| 1 | C1, C4, C5 | Independent files, unambiguous, kill the irritants first |
| 2 | C2 (wire-in), C6, C9 | Need user decision (C2) + user env action (C6 secret) |
| 3 | C3, C8, C10 | Ride along after wave 2 settles |
| 4 | C7 | Needs launcher-down window — schedule with user |

Rules: ≤3 parallel agents (ZAI limit); one commit per task (`fix:`/`refactor:`/`test:` prefixes); full suite gate before each commit; no TODO.md/BUG_REPORT.md edits by agents — main thread updates trackers.

## Out of scope (noted, not planned)

- Graphify node-id namespacing (`path::name`) to kill the 400 same-label collisions — upstream graphify improvement, not repo code.
- Launcher config persistence unification (ConfigPersistence/SQLitePersistence/EventStore) — real but needs its own design pass.
