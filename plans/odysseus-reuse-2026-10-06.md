# Odysseus reuse plan — owner-dedup, embedding lanes, hybrid recall, tidy fingerprints

**STATUS:** DRAFT for review — no implementation started.
**Date:** 2026-10-06
**Source:** comparison against `~/odysseus` (self-hosted AI workspace). Odysseus is **AGPL-3.0-or-later** and this repo is public: **patterns, algorithms, constants and test ideas only — zero code copied verbatim.**
**Branch:** `feature/odysseus-lessons` in worktree `../supreme-mcp-tools-odysseus`, cut from main `c5f7237` (NOT from `feature/pod-ui-gaps`, which is 16 commits ahead and unmerged — this work is independent of it).
**Sequencing:** P2 → P1 → P3 → P4 → P5. Phases are independently mergeable; each ends with suite green + a live native-tool sweep (`.agents/skills/mcp-live-tool-test`) before `--no-ff` merge.

---

## Verified facts this plan builds on (probed 2026-10-06)

| # | Fact | Where |
|---|------|-------|
| F1 | SQL dedup key is `UNIQUE(text_hash, memory_type)`; `text_hash = sha256(text.strip().lower())[:40]` — no owner dimension | `postgres_sql.py:120,295`, `turso_sql.py:204,498`, `shared/hashing.py` |
| F2 | E3.5 already stamps `payload["owner"]` from caller identity on Qdrant writes and owner-filters queries/gets/deletes | `memory_tools.py:49-75,244-273,370-374,562-569,642-647` |
| F3 | `sql_store.upsert_memory()` does NOT receive owner → in multi mode, user B's identical text+type upsert hits `ON CONFLICT` and remaps to user A's row (`actual_id` returned to B) | `memory_tools.py:299-309` + F1 |
| F4 | `ensure_collection` returns silently when a collection already exists — it never validates `dense_dim` against the existing config | `qdrant_vector.py:186-190` |
| F5 | Embedding dim is derived at process start from provider: local bge-m3=1024 / bge-base=768 / azure text-embedding-3-large=3072 | `incremental_indexer.py:85-94` |
| F6 | ragmcp collection name comes from `get_collection_config()["default_collection"]`, overridable per call via `params["collection"]` | `ragmcp_fastmcp.py:390` |
| F7 | Indexing metadata sidecar is per-workspace: `.indexing_metadata/indexing_metadata.json` | `indexer/metadata_config.py` |
| F8 | `SqlStore.search_text(query, limit, *, memory_type, tags, agent_id)` exists on the ABC; PG = feature-scoped pg_trgm with degrade-warn-once (HIGH-14); Turso = FTS5. `queryMemory` has zero FTS/trgm usage today | `sql_store.py:42-47`, `postgres_sql.py:165-217` |
| F9 | `mergeDuplicates` = full `iter_all(with_vectors=True)` + O(n²) pairwise per type group (cosine→Jaccard fallback), no skip-if-unchanged, no owner precondition | `memory_tools.py:1216-1300` |
| F10 | `decayOrExpire` = `scroll_all` + TTL/usage rules; an empty/degraded scroll is indistinguishable from a healthy empty store (no mass-delete guard) | `memory_tools.py:717-775` |
| F11 | `relevance_scorer` blends α·semantic + β·recency + γ·usage (0.5/0.3/0.2, env-tunable `MEMORY_ALPHA/BETA/GAMMA`, half-life 14d) | `shared/relevance_scorer.py` |
| F12 | dotenv is loaded at 3 sites without `encoding="utf-8-sig"` (a BOM saved by Windows editors poisons the first env key) | `launcher/__main__.py:37`, `memory_core.py:90`, `ragmcp_fastmcp.py:114` |
| F13 | upsertMemory contains two back-to-back identical E3.5 owner-stamping try/except blocks (duplication smell; P1 touches this code anyway) | `memory_tools.py:244-273` |

---

## P1 — Owner-scoped SQL dedup leg (memorymcp · E3 multi-user correctness)

**Problem (F1+F3):** Qdrant rows are owner-scoped but the SQL dedup key is not. In multi mode the SQL leg silently merges cross-tenant memories and remaps the caller's id to another user's row.

**Design — owner-mixed hash (chosen):**
- `shared/hashing.py`: `text_hash(text, owner=None)` — `None` ⇒ exactly today's hash (mono/legacy untouched); owner set ⇒ `sha256(f"{owner}\0{text.strip().lower()}")[:40]`. Mirrors the validated odysseus pattern (owner+`\0`+text, empty owner = legacy id).
- `upsert_memory(..., owner: str | None = None)` on the ABC + both impls; `thash = text_hash(text, owner)` at the top. No schema change, no migration, legacy rows keep their hashes.
- Call site: `upsertMemory` passes `_memory_caller()[0]` (F2 already resolves it).
- `mergeDuplicates`: owner-equality precondition on candidate pairs (both payloads' `owner` must match, `None == None` counts as equal) — cross-owner pairs are skipped and counted in the report.

**Rejected alternative (record for posterity):** add an `owner` column + `UNIQUE(text_hash, memory_type, owner)`. Costs schema migration on both impls, PG `NULL`-distinct semantics break mono dedup (needs a `__local__` sentinel), and `migrate_store` format changes. Revisit only if SQL-side owner filtering becomes a requirement (see P3 note).

**Edge-case catalogue:**
1. Mono mode → owner `None` → byte-identical hashes; zero behavior change (test: hash vectors before/after refactor).
2. Multi mode, two users, same text+type → two SQL rows, two Qdrant points, each caller gets their own id back.
3. Same user re-upsert → dedup-update path still fires (hash self-consistent per owner).
4. Legacy row (old hash) + owned row (mixed hash) with identical text → coexist; legacy stays admin-owned per F2 comment.
5. **Open question OQ1:** the F2 comment mentions admin passing an *explicit* owner for shared memories — verify whether `upsertMemory` exposes it; if yes, the hash must use the *effective* owner (override wins), so two admins writing the same shared memory still dedup.
6. `migrate_store` export/import: `text_hash` travels precomputed in `_sql` rows — no format change; run `verify` round-trip on the grounds store.
7. Concurrent same-(text,type,owner) upserts: `ON CONFLICT` race safety unchanged from today.
8. `verifyBackendParity`: unaffected (hash is internal to SQL leg).

**Failure playbook:** PG/Turso hash drift would surface in `verifyBackendParity` — the change lives only in `shared/hashing.py` so both impls inherit; never inline the hash computation.

**Drive-by (F13):** collapse the duplicated owner-stamping try/except into one block.

**Tests (~10):** hashing owner cases; PG+Turso two-owner upsert with **exact full param-list assertion** (sql-param-order blindspot: mock tests can't see placeholder-order swaps); mergeDuplicates cross-owner skip; multi-mode stamping; migrate round-trip on grounds.

---

## P2 — Embedding lanes (ragmcp · unblocks the OpenRouter free-embeddings test)

**Problem (F4+F5):** switching embedding model/provider against an existing collection fails only at write time (dim mismatch mid-index). The pending OpenRouter `:free` models (lfm2.5 1024d, vl-1b-v2 2048d) make dim collision a live scenario.

**Design:**
- Lane fingerprint = `"{provider}:{model}:{dim}"`; lane slug = sanitized `"{model}-{dim}"`.
- Collection name = `f"{base}-{slug}"` where `base` = F6's default_collection. **Lane naming lives in the caller (ragmcp), never in the VectorStore impls** — impls stay lane-agnostic.
- New module `tools/ragmcp/indexer/lanes.py`: `resolve_lane(base, provider, model, dim)`, `list_lanes(base, vector_store)` (collections filtered by base prefix), `legacy_lane(base)` (the pre-lane collection name, still searchable).
- Metadata sidecar becomes per-lane: `.indexing_metadata/indexing_metadata.<slug>.json`; the legacy filename maps to the legacy lane (F7).
- `index_code(embedding_model=...)` → resolves lane, `ensure_collection` per lane (dims consistent within a lane by construction).
- Search: query all existing lanes (+ legacy), per-lane `k*3` capped like odysseus, fuse dense ranks with the existing `_reciprocal_rank_fusion`; each hit labeled with its lane. Sparse-only mode is model-independent → single shared collection, lanes bypassed.
- **Pre-flight dim check (immediate cheap win, ship first):** before indexing into an existing collection, `get_collection` → compare `dense_dim` → abort with an actionable message instead of F4's mid-run failure.
- Dead-endpoint latch in the embedding HTTP path: N consecutive connection failures ⇒ latch 60s, fail fast with a clear message (odysseus pattern; protects Azure/OpenRouter probing loops).

**Edge-case catalogue:**
1. Fresh install → one lane, no legacy collection → search unaffected.
2. Existing deployment → legacy collection kept searchable; new indexes go to slug lanes; no auto-migration (document `clear_index` + reindex path instead).
3. Index request naming an existing lane with a different dim → pre-flight abort with message.
4. Cross-lane search with mixed models → RRF fusion across lanes; `k` split per lane.
5. OpenRouter `:free` models → new `provider` values in ragmcp `embedding` config; lanes make them safe from day one.
6. Turso/PG vector impls: lane = name suffix; validate naming per impl at implementation time (**OQ2** — probe on the grounds store before merging).
7. `clear_index` becomes lane-scoped with an explicit all-lanes option (typed confirm).

**Failure playbook:** pre-flight abort ⇒ user picks another lane or clears; mid-run endpoint death ⇒ latch + incremental indexer resumes on retry (already skips indexed files).

**Tests (~8):** lane resolver unit tests; pre-flight dim-mismatch (mock store); cross-lane RRF fusion; per-lane metadata path; legacy-lane search inclusion.

---

## P3 — Hybrid memory recall (memorymcp · keyword lane in queryMemory)

**Problem (F8):** `queryMemory` is vector-only although `SqlStore.search_text` (pg_trgm/FTS5) already exists unused. Exact-term memory queries ("which port did we pin") underperform on pure vector.

**Design:**
- Final score = `(1 - kw_weight) · current_blend(F11) + kw_weight · kw_norm`, `kw_weight` = `MEMORY_KW_WEIGHT` (default 0.35, orthogonal to the existing α/β/γ redistribution via `recency_weight`).
- Keyword lane: `sql_store.search_text(query, limit=k*3)` → normalize to [0,1] by rank; when SQL unavailable, fall back to `word_jaccard_similarity` (text_utils, zero new deps) over the vector candidates' texts.
- Candidates = union(vector hits, kw hits); dedupe by id taking the max normalized score; type-aware additive boost on `kw_norm` when query tokens overlap the memory-type label (odysseus identity/contact/preference table adapted to code_pattern/decision/lesson), capped at +0.10.
- Gate: keep current effective threshold for vector-only candidates (**OQ3** — confirm current behavior before changing what "no result" means).

**Multi-mode interaction (honest limitation):** under P1 design (a) the SQL table has no owner column, so `search_text` cannot owner-filter. Kw hits are post-filtered by fetching owner from Qdrant payload (k·3 is small) and refilling. If this proves hot, the P1-rejected owner column becomes the unlock — record the trigger condition.

**Edge-case catalogue:** SQL down mid-query → vector-only + warning line in output; pg_trgm unavailable → HIGH-14 degrade path already returns quietly, skip lane; empty/stopword-only query → skip kw lane; no SQL at all → Jaccard fallback; hit in both lanes → max; owner post-filter (above).

**Failure playbook:** `search_text` raising → caught per-lane, degrade to vector-only, never fail the whole query.

**Tests (~8):** blend math unit tests (weights env matrix); **exact param-list assertion on the FTS query with a filter case** (sql-param-order blindspot, HIGH-14 precedent); Jaccard fallback; owner post-filter; degrade-on-SQL-down.

---

## P4 — Maintenance-pass fingerprints + over-deletion guards (memorymcp)

**Problem (F9+F10):** `mergeDuplicates` is O(n²) with artifact rehydration on every invocation; `decayOrExpire`'s empty scroll is indistinguishable from a healthy empty store — the exact bug class that once destroyed odysseus's memory store (degraded read → persisted empty view).

**Design (odysseus patterns, re-derived):**
- Tidy-state sidecar `~/.memorymcp/tidy_state.json`: `{"fp": sha256(sorted(id + text_hash + last-modified)), "ts"}` written after each successful **non-dry** pass.
- Skip-if-unchanged: `mergeDuplicates` computes a cheap fingerprint via `iter_all` **without** vectors first; identical to sidecar ⇒ short-circuit report unless `force=true`.
- Over-deletion guard: a non-dry pass that would delete >50% of the current count aborts with the full report (`MEMORY_TIDY_MAX_DELETE_FRACTION`, default 0.5).
- Fail-closed scroll: if a pass observes 0 points while the sidecar says the store was non-empty, require `force=true` (suspected degraded read).

**Edge cases:** concurrent passes (operator-invoked; document, don't lock); corrupt sidecar ⇒ fail **open** (run the pass); Qdrant-only installs (sidecar is local, works); artifact rehydration cost now paid only when a pass actually runs.

**Tests (~6):** skip, force, >50% guard, empty-scroll guard, corrupt sidecar, fingerprint stability across ordering.

---

## P5 — Quick wins batch

1. **utf-8-sig dotenv (F12):** `load_dotenv(..., encoding="utf-8-sig")` at all three sites — harmless for BOM-less files.
2. **Policy anti-pollution line:** `tools/memorymcp/auto_use_policy.md` + cheatsheet — "keep each memory ≤2 short sentences; split compound learnings instead of one dump" (odysseus's ≤15-words/max-2-facts adapted; this repo allows longer texts by design via the artifact store).
3. **SSRF guard for embedding endpoints** — deferred until the OpenRouter keys land (validate scheme+host, reject link-local/metadata IPs; applies to the ragmcp embedding URL and `SMARTGRAPH_BASE_URL`).
4. **Threat-model doc:** `plans/threat-model.md` seeded from `plans/security-audit-2026-10-01.md` — honest framing, roles→capability matrix, explicit Known Gaps; pair every applied audit fix with a pinned regression test (odysseus `test_security_regressions.py` pattern).
5. **Drive-by:** P1's duplicate-stamping cleanup (F13).

---

## Open questions (resolve at implementation start, not from assumptions)

- **OQ1:** does `upsertMemory` expose an explicit `owner` param for admin-created shared memories? (F2 comment suggests yes.)
- **OQ2:** lane naming against Turso/PG vector impls — probe on the grounds store before P2 merge.
- **OQ3:** current effective `queryMemory` result threshold — read the code path, don't guess.
- **OQ4:** who (if anyone) calls `SqlStore.search_text` today — grep callers before wiring it into `queryMemory`.

## Deliberately out of scope

Odysseus's Chroma stack, JSON-file memory store, stdio builtin-server duplication, hand-rolled BM25 (this repo's `CodeSparseVectorGenerator` + FTS5 already cover sparse), their embedding HTTP client (only the latch idea transfers), MCP OAuth client plumbing (static API keys are correct for a localhost threat model), reranker/cross-encoder (neither project has one; RRF + keyword lane first).
