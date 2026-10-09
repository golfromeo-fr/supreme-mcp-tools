# Audit: `supreme-mcp-tools` against the vector-DB-death lessons

**Date:** 2026-10-08 · **Repo:** `~/supreme-mcp-tools` on `debian-vscode` (branch `feature/pod-ui-gaps`, HEAD `b644df5`)
**Scope:** `tools/ragmcp` (4667 LOC), `tools/memorymcp` (3155 LOC), `tools/shared`
**Method:** source read of the 8 handoff questions. No code changed.

---

## Verdict first

The handoff's premise — "BM25+dense+RRF is table stakes, the openings are deployment / cost-reporting / compiled-folder" — is **only true of ragmcp, and that tool already has the keyword arm**. The two tools differ sharply:

| | `ragmcp` (code search) | `memorymcp` (memory) |
|---|---|---|
| Retrieval | dense + **native BM25 sparse** + hybrid RRF | **dense only** |
| Keyword arm | yes, first-class, auto-selected | **none** |
| Cost per query | not reported | not reported |
| Provenance | n/a (code has paths/commits) | stored, **not returned** |

The real finding: **`memorymcp` is the vector-first design the video argues against, and `ragmcp` is the counter-example.** The competitive openings from §4 of the handoff are narrower than assumed — the compiled-folder failure and cost-reporting gaps are real, but "add BM25" is *already done* on the code side.

---

## The 8 questions, answered

### 1. Tool surface — one `search()` or a loop of narrow tools?

**Both, deliberately.** `ragmcp` exposes one `search(query, mode=...)` where `mode ∈ {auto, dense, sparse, hybrid}` (`ragmcp/ragmcp_fastmcp.py:868`). `mode="auto"` inspects the collection and picks: hybrid if both indexes exist, sparse if only sparse, dense if only dense.

`search_code` / `search_code_sparse` still exist as **deprecated shims** that log a warning and forward to `search(mode=...)` (`:817`, `:837`).

Assessment: the video's "one `search()` = one-shot RAG" trap is **avoided**. `auto` is capability detection, not a fixed pipeline. But it *is* still one-shot — there is no agent loop, no query planning, no "none of these are good, search again." The loop is the model's job, and the model cannot see that `sparse` exists unless the docstring is read. The docstrings do say it (`"Excellent for finding exact identifiers... Works offline without API costs"`), so an agent that reads tool descriptions will reach for sparse. That is the design working as intended.

### 2. Retrieval stack — is there a keyword arm, or vector-first?

**`ragmcp`: real BM25, hand-rolled, code-tuned. This is the strongest part of the codebase.**

`ragmcp/indexer/sparse_vector_gen.py` implements BM25 (K1=1.5, B=0.75) over code with:
- camelCase / snake_case / identifier splitting
- **domain boosting**: table names ×3.0, function names ×2.5, type names ×2.0, SQL keywords ×1.8, constants ×1.5, generic identifiers ×1.2
- PL/SQL + Pro*C aware patterns (`PROCEDURE`, `FROM <CAPS>`, `void/int/char/struct` signatures)

The file's own header claims ~95% identifier accuracy vs ~70% for embeddings — **unverified, no benchmark in-repo**. Directionally consistent with the video's EnterpriseRAG-Bench result, but treat the numbers as marketing until measured.

Hybrid uses **RRF** (`_reciprocal_rank_fusion`, k=60, `:1078`), with native Qdrant `FusionQuery`/`Prefetch` when `qdrant-client ≥1.7` and a Python RRF fallback below that (`shared/impls/qdrant_vector.py:26-32`).

**`memorymcp`: dense only.** `queryMemory` calls `vector_store.query_dense(...)` unconditionally (`memory_tools.py:395`). No sparse mode, no `mode` parameter, no hybrid, no RRF. Re-ranking is recency + usage weighted (`ScoringWeights` alpha/beta/gamma, `recency_weight` default 0.5) — **not** lexical. The docstring calls it "semantic search."

This is the gap. Given the video's finding that embedding models don't know your internal vocabulary, and given this user's domain (Retail ERP — SKU codes, `STOMVT`-style table names, ticket IDs), dense-only memory recall is the wrong default for exactly the identifier-heavy queries that matter. The sparse machinery already exists in the same repo and is 385 lines.

### 3. Storage — pgvector? dedicated? scale?

**Qdrant**, not pgvector, behind a neutral adapter layer.

`tools/shared/store_models.py:21` defines `VectorBackend = Literal["qdrant", "turso", "postgres", "none"]`. Implementations in `tools/shared/impls/`: `qdrant_vector.py`, `postgres_sql.py`, libSQL/Turso. Tool code never imports `qdrant_client` directly — it goes through the abstraction (`shared/store_models.py:5`).

`ragmcp` docstrings reference "the Turso FTS5 sidecar" for sparse (`_no_sparse_index_error`, `:856`), and there is a `migrate_store repair-sparse` command — so the sparse index has been retrofitted onto collections that were originally dense-only.

Relevance to the pgvector lesson: **not applicable as stated.** This isn't an OSS-distributed tool with a Docker image; it's a private multi-tool launcher. The pgvector ≥0.8.7 CVE pin is worth noting only if any image is ever published.

### 4. Does search re-paste compiled context? *(highest-value question)*

**No — and this is the strongest answer in the audit.**

- `queryMemory` returns `hit.text[:150]` per hit, hard-truncated in the formatted output (`memory_tools.py:~427`). Default `k=10` → ~1,500 chars maximum, regardless of what is stored.
- No tool returns "the compiled folder." `getMemorySystemPrompt` reads a static markdown file (`memory_system_prompt.md`) — that's a fixed instruction file, not compiled user knowledge.
- `textToGraph` / `textToSmartGraph` produce graph structures, not a re-pasted context blob.

The Aug-2026 study's failure mode (compiled artifact re-pasted per query, ~21× token cost) **is not present**.

**But there is a related inefficiency.** The artifact sidecar works the other way: memories above `ARTIFACT_INLINE_THRESHOLD_BYTES` are offloaded to an `ArtifactStore` blob, and `queryMemory` **rehydrates the full text for every hit** via `_rehydrate_payload_text` (`memory_tools.py:125`) — then truncates to 150 chars for display. Full blob fetched, 95% discarded. That's per-query sidecar I/O for text nobody reads in the response. `text_preview` (200 chars, always present) would serve the display path with zero fetches; full rehydration should be reserved for `getMemory`.

This is worth fixing but it is a *bandwidth* problem, not the 21× *token* problem the paper describes. Different failure, same neighborhood.

### 5. Does memory carry provenance?

**Yes on write, mostly no on read.** This is the asymmetry.

Written: `upsertMemory` payload carries `source`, `path`, `commit`, `file_range`, `agent_id`, `created_at`, `provenance{}`, `sensitivity`, `tags` (`memory_core.py:memory_item_to_payload`). `attachProvenance` (`:810`) appends to a **history list** — `source`, `model_version`, `confidence`, `timestamp`, `notes` — so successive attributions accumulate rather than overwrite.

Read back: `payload_to_memory_hit` (`memory_core.py:253`) **does** map `provenance=payload.get("provenance")` into the `MemoryHit`. But `queryMemory`'s output formatter prints only type, score, 150-char text, tags, usage count, and last-accessed. **Provenance is dropped at render time.** It's available to code, invisible to the calling agent.

The handoff's lesson (b) — "optimize memory for attribution, not similarity," where the 40%-vs-19% citation-quality gap is the real metric — is **half-implemented**. The data is there. The agent can't see it. Two lines of formatting fix it.

Related: `auditTrail` (`:1151`), `redactSensitive` (`:878`), `decayOrExpire` (`:717`), `mergeDuplicates` (`:1216`) all exist as tools.

### 6. Is the index disposable?

**Yes, and unusually well.**

- `reindexMemory(new_model)` (`memory_tools.py:1067`) re-embeds everything with `scroll_all` + batches, handles the artifact-backed case (re-embeds from full text when `text` is absent), and **refuses on dimension mismatch** with a clear message rather than corrupting the collection.
- `ragmcp` has `clear_index` (`:1817`), `start_indexing`, `stop_indexing`, `check_indexing_progress`, plus an incremental indexer (`indexer/incremental_indexer.py`) — so indexing is restartable background work, not a blocking rebuild.
- Backends are swappable via the `VectorBackend` literal, so an index can be rebuilt into a different store.

Directly satisfies lesson (e). The Turbopuffer V3 analogy is apt: content address is decoupled from the index. One caveat — memories are the *source of truth* in Qdrant; there's no upstream source to re-derive them from. A `clear_index` on `memory-store` destroys the memories, unlike a code index which rebuilds from disk. Rebuilt-without-loss holds for ragmcp, not for memorymcp.

### 7. Does it report its own token cost per query?

**No — and this is the cleanest unclaimed opening.**

`ragmcp` metrics (`ragmcp_fastmcp.py:300-350`) track: call counts by tool name, `total_search_time_ms`, `min`/`max` search time, `search_errors`, `embedding_calls`. Pushed to `fef_manager.metrics.record_request(endpoint="tools/call", tool_name=..., success=..., duration_ms=...)`.

`getMemoryMetrics` (`:909`) reports: total memories, breakdown by type, top-5 by agent, total retrieval count, average usage. It even builds `sql_stats = sql_store.get_metrics()` then **never renders it** — `sql_stats` is computed and dropped.

**Zero token accounting in either tool.** The only `tiktoken` usage in the whole repo is `estimate_tokens()` in the *indexer*, for chunking decisions (`incremental_indexer.py:320`) — build-time, not query-time.

This is exactly lesson (f) and exactly §4/opening #2 of the handoff. Still unclaimed, still free. And it's cheap here specifically: retrieval output is already hard-truncated to 150 chars × k, so "tokens returned per query" is computable with the tokenizer already in `requirements.txt`.

### 8. Where does a search call actually route?

**No network hop. Fully local, in-process.**

- Embeddings: `get_local_embedding_model()` → `SentenceTransformer(os.getenv('LOCAL_EMBEDDING_MODEL', 'BAAI/bge-m3'))`, a module-level singleton (`memory_core.py:144`). `generate_embedding` just calls `model.encode`. Runs in `asyncio.to_thread` (`memory_tools.py:~360`).
- `ragmcp` matches: `local_embeddings.py` header — *"No GPU required, works offline, zero API cost."*
- Indexer `--mode` help text (`:742`): `sparse (BM25, $0), dense (embeddings, API cost), hybrid (both)` — so a dense path *can* use an API, but the local path exists and is default.
- Transport: streamable-HTTP via the launcher on port 8004/8005, so a client hits HTTP, but the server is co-located on the same host. No external service.

Routing consequence: **`sparse` mode is genuinely $0 and genuinely offline**, which makes "make keyword nearly free" (lesson c) structurally true rather than aspirational. Dense has a per-call `model.encode` cost — wall-clock on the local model, and optionally API money if configured that way. That cost is invisible to the caller today.

---

## Where this contradicts the handoff

1. **"Nobody publishes cost per query"** — still true for these tools, but note they're private tools, not OSS competitors. The opening is real only if published.
2. **"BM25+dense+RRF is table stakes … copying the video's advice wins nothing"** — ragmcp already did it, months ago, and did it well (code-tuned boosting). The gap is *memorymcp*, not the RAG tool.
3. **Opening #3, "the compiled-folder failure is unfixed"** — **unfixed in the industry, but not present here.** `ragmcp` and `memorymcp` both retrieve narrowly. The ArtifactStore rehydration inefficiency is adjacent, not the same bug.
4. **Lesson (d), "Postgres+pgvector is an adoption strategy"** — not applicable. This is a self-hosted launcher with Qdrant, multi-user, function masks, and owner scoping. The pgvector CVE note is inert.

## Not checked

- Whether any benchmark exists in `tests/` for the sparse arm (would confirm/refute the 95% claim).
- `shared/impls/postgres_sql.py` internals — is postgres a live backend or a stub?
- Turso FTS5 sidecar behavior.
- `fef_integration.py` — the metrics plane, ~13KB, unread.
- Deployment/pod topology (`deploy/`, `pod_ui/`) — 9 dirty files on this branch, untouched by this audit.
- All claims in the video's own verification log remain as verified in the cloud session; this audit says nothing about the video's accuracy.

## Suggested order of work

1. **Add `mode` to `queryMemory`, reuse `sparse_vector_gen`.** ~385 lines already exist in the same repo; memory identifiers are exactly what dense recall is worst at.
2. **Print provenance in `queryMemory` output.** Two lines, unlocks lesson (b).
3. **Per-query token accounting.** Returned-text tokens + which retrieval tier fired. Free, and nobody in OSS MCP-land does it.
4. **Drop full-text rehydration from the query path**; serve `text_preview`, rehydrate only in `getMemory`.
5. Minor: render the `sql_stats` you already computed in `getMemoryMetrics`.

## Cross-cutting notes

- **Contradiction to flag:** the repo's own log says indexing modes are `sparse (BM25, $0), dense (embeddings, API cost)`, while the local default model makes dense also $0. Not a bug — just be aware the cost story depends on `LOCAL_EMBEDDING_MODEL` config.
- **Working tree is dirty** (`ARCHITECTURE.md`, `README.md` modified; 7 untracked incl. `plans/embedding-*.md`). This audit reads HEAD + working tree as-is and changed nothing.

---

## Addendum — re-verification (2026-10-09 04:45)

The two findings most likely to drift were re-checked against the live tree after a bridge outage. Both **hold**.

**Q2 — `memorymcp` is dense-only: still true.**
- `memorymcp/memory_tools.py` mtime is **2026-09-28 02:57** — untouched since well before this audit was written.
- `grep -rn "sparse|bm25|lexical" memorymcp/` returns exactly **one** hit, and it is not a search path:
  `memory_core.py:116` — `COLLECTION_NAME, dense_dim=embedding_dim, sparse=False`
  That is the collection being *created* with sparse explicitly disabled. There is no keyword arm.

**Q5 — provenance not rendered by `queryMemory`: still true, with a refinement.**
The render block prints type, score, `text[:150]`, tags, usage count, last-accessed — no provenance.

Refinement worth recording: provenance **is** printed by `getMemory`, at `memory_tools.py:505-506`:
```python
if payload.get('provenance'):
    output += f"\nProvenance: {json.dumps(payload['provenance'], indent=2)}"
```
So the asymmetry is narrower than "provenance is invisible": it is fully visible on **single-memory fetch** and absent only on **multi-memory search** — which is the path an agent actually uses when recalling. The fix is therefore localized to the `queryMemory` formatter, not a data-model change.

**State at re-verification:** HEAD `b644df5`, branch `feature/pod-ui-gaps`, audit file present but uncommitted.