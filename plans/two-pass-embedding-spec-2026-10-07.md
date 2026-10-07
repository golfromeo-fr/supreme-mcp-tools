# Two-pass embedding for ragmcp — spec (2026-10-07)

**Status:** DRAFT for implementation (Lindsay handoff — written to spec-grade per
`memory/spec-grade-plans-for-smaller-model-handoff`: probe-verified facts, prescribed
edge cases, verbatim knobs, failure playbook; no gospel code).
**Decided variant:** B — local-only (2026-10-07, user: OpenRouter free-tier availability
is not dependable; quality lane must never depend on egress).
**Scope:** ragmcp only. memorymcp is EXCLUDED (memories are written one at a time — no
bulk-index problem; `reindexMemory` covers model upgrades there).
**Builds on:** `feature/odysseus-lessons` P2 embedding lanes
(`tools/ragmcp/indexer/lanes.py` — per-fingerprint collections, `__collection_metadata__`
provenance point, pre-flight `check_lane_compatibility`).

---

## 0. Measured foundations (2026-10-06/07 benches — do not re-derive)

SciFact (English prose, 300q/1200 docs) and in-domain code bench (1,186 real `.py` files
from ~/fastmcp, ~/fastapi, ~/nicegui, ~/graphify; 600 AST docstring queries, gold =
defining file, 512-token truncation):

| Model | SciFact nDCG@10 | Code nDCG@10 | Code MRR@10 | ms/doc (code) | dims |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.7730 | 0.5467 | 0.3358 | 82 | 384 |
| bge-base-en-v1.5 | 0.8549 | 0.5517 | 0.3862 | 356 | 768 |
| BAAI/bge-m3 | 0.7644 | 0.5483 | 0.3936 | 1211 | 1024 |
| nemotron vl-1b-v2 (remote, REFERENCE ONLY — not adoptable, see Variant B) | 0.8700 | — | — | 20 | 2048 |

Established facts the design relies on:
1. Dimension count is a storage knob, not a quality gate (same-model PCA 1024→512 costs
   ~1 pt).
2. On **code**, dense model choice is nearly irrelevant on nDCG (spread 0.005); MRR keeps a
   small ordering (MiniLM 0.336 < base 0.386 ≈ m3 0.394). The bigger code levers are
   chunking granularity, the sparse lane, and reranking — pass 2 is NOT the code lever.
3. `paraphrase-multilingual-MiniLM-L12-v2` costs −15.6 pt English vs MiniLM-L6 ⇒ pass 1 is
   English-only MiniLM-L6; multilingual coverage belongs to pass 2 (m3 is 100+ languages).
4. Corpus is English/code-dominant (user + expected users). **Language policy:** deployments
   with non-English-primary content must swap pass-1 for a language-matched model
   (e.g. `bge-small-zh-v1.5`); surface this as a model-choice note in ragmcp docs (§7).

## 1. Architecture

```
pass 1 (index_code, default):   MiniLM-L6 dense (384d) + sparse (BM25) — SAME run
                                → lane collection  <base>-local-minilm-l6-384
                                → hybrid RRF search live immediately
pass 2 (backfill, background):  bge-m3 dense (1024d), CPU-throttled (nice + 2 threads)
                                → lane collection  <base>-local-bge-m3-1024
search (steady state):          3-way RRF(fast-dense, sparse, quality-dense) — RANK FUSION
                                ONLY; never raw-merge scores across models
                                quality lane OPTIONAL: missing → 2-way RRF (current behavior)
```

Both lanes coexist (this is the delta from P2, which assumes ONE active lane).
Sparse stays in the plain base collection — term-hashing, model-independent, free
(P2 already keeps sparse there: lanes.py docstring).

### Why pass 1 = MiniLM (user's call, confirmed by bench)
- Weakest-fast pass maximizes the two-pass architecture's value: +10 pt pass-2 headroom on
  prose (SciFact) vs +1.5 if pass 1 were bge-base; freshness window (time until a chunk is
  searchable) ≈ 1.6 h/100k chunks at 82 ms/doc vs ≈ 8 h at 356 ms/doc.
- 384d halves vector storage vs m3 for the bulk lane.

### Why pass 2 = bge-m3 (only local candidate)
- Only installed model with multilingual + 8k context + prod track record. On code it buys
  MRR polish (+0.058), multilingual chunks, and long-context (untested by the bench —
  production CHUNK_SIZE=300 chunks are shorter than the bench's 512-truncated files, so
  per-chunk encode is FASTER than the 1,211 ms/doc whole-file number; expect ~2× slowdown
  from the 2-thread throttle → budget ≈ 2.4 s/chunk worst case, 100k chunks ≈ 3 days
  niced. This is FINE: background lane, search never waits).

## 2. Changes (file-by-file landing zones)

### 2.1 `tools/ragmcp/config.json` — new `two_pass` block
```json
"embedding": { ...existing... },
"two_pass": {
  "enabled": true,
  "fast":  { "provider": "local", "model": "sentence-transformers/all-MiniLM-L6-v2" },
  "quality": { "provider": "local", "model": "BAAI/bge-m3" }
}
```
Env overrides (factory precedence rules apply — config.json > env > default):
`RAG_TWO_PASS=0/1` (kill-switch), `RAG_QUALITY_LANE_QUERY=0/1` (per-query opt-out of the
m3 encode, §4), `RAG_BACKFILL_THREADS=2`, `RAG_BACKFILL_NICE=10`.

### 2.2 `tools/ragmcp/indexer/lanes.py` — role helpers (pure stdlib, unit-testable)
- `LANE_ROLES = {"fast", "quality"}` — extend the metadata point payload with
  `"lane_role": "fast"|"quality"` (backwards-compatible: missing = legacy).
- `resolve_lane(base, provider, model, dim, role=None)` — unchanged naming math; role is
  payload-only. Two fingerprints that differ ONLY by role share a slug only if
  provider/model/dim match (they don't here: 384 vs 1024 → distinct slugs anyway).

### 2.3 `tools/ragmcp/indexer/incremental_indexer.py` — pass-1 switch + backfill mode
- **Pass 1 (default path):** embed dense with the FAST model (MiniLM) instead of the
  `embedding.provider` resolution; sparse phase unchanged. Write `lane_role: "fast"` in the
  lane metadata point. The `embedding_model` preset argument keeps working (it overrides
  the fast lane for users who explicitly want m3-only — then no quality lane is built and
  search degrades to 2-way; document this).
- **New `--backfill-quality` CLI mode** (also exposed via a `backfill_quality` tool):
  1. Read the fast lane's file-state metadata (`metadata_config` path — one state file PER
     LANE: `<state>-<lane_slug>.json`; never share state across lanes).
  2. For each file in fast-lane state: if the quality lane's state lacks it or mtime/hash
     differs → re-chunk (same chunker, same chunk ids) → encode with m3 → upsert into the
     quality lane → mark done in quality-lane state ONLY after the file's upsert returns.
  3. Removed files: delete stale chunk ids from the quality lane (mirror of the existing
     stale-chunk removal — per lane).
- **Throttling (backfill only):** `torch.set_num_threads(int(env RAG_BACKFILL_THREADS, 2))`
  + `os.nice(int(env RAG_BACKFILL_NICE, 10))` (wrap in try/except — nice may raise without
  permissions) before model load. Runs via the EXISTING PID-tracked subprocess path
  (`start_indexing`/`check_indexing_progress`/`stop_indexing`) — no new runner.
- The existing dim pre-flight (`check_lane_compatibility`) runs for BOTH lanes: backfill
  aborts cleanly if the quality lane exists with a different fingerprint.

### 2.4 `tools/ragmcp/ragmcp_fastmcp.py` — 3-way search
- `search(mode="hybrid")`: after the current 2-way RRF (dense active lane + sparse), if
  `two_pass.enabled` and the quality lane exists and `RAG_QUALITY_LANE_QUERY != 0`:
  encode the query with m3 (LANE CONSISTENCY: each lane's query vector MUST come from that
  lane's model — cross-model query/doc vectors are invalid) → `_do_dense_search` on the
  quality lane → include as a third ranked list in `_reciprocal_rank_fusion` (it already
  takes `ranked_lists: list[list[dict]]` — N-way is a length change, verify per-list dict
  keys carry lane id for provenance: add `"lane": "fast"|"sparse"|"quality"` to each hit).
- Query-embedding memoization: tiny LRU (functools.lru_cache maxsize=128) keyed by
  (model-tag, sha256(query)) — repeated/refined queries skip the 1–2 s m3 encode.
- `list_collections`: already lane-aware via P2; add `"role"` from each lane's metadata
  point to its listing.
- Cost disclosure in the tool description: hybrid+quality adds one m3 encode per query
  (~1.2–1.9 s CPU on this VM class) unless LRU-hit.

### 2.5 `tests/` — unit + live
- Unit (no FastMCP import): role payload round-trip; per-lane state files don't collide;
  3-way RRF ordering + provenance key; LRU key includes model tag.
- Live (grounds, extends tests/test_pod_smoke.py — probe 15): index a small tree (fast
  pass) → start backfill → poll progress → search hybrid → assert some hits carry
  `lane: "quality"` → touch a file → re-backfill → assert no ghost chunk for the OLD text
  remains in the quality lane (search old text snippet returns nothing from quality lane).

## 3. Edge-case catalogue (prescribed — handle ALL)

1. **Quality lane missing** (never backfilled): search = current 2-way behavior, INFO log
   once per process (not per query).
2. **Ghost chunks:** file changed between pass 1 and backfill → backfill MUST chunk the
   CURRENT file content and delete the quality lane's ids for that file BEFORE upserting
   new ones (ids are content-derived per chunk index — same file, changed content, same id
   slots get overwritten; id scheme must match pass-1 chunking exactly — assert chunk ids
   equal between lanes for identical file state).
3. **Backfill crash mid-file:** quality-lane state marks done per FILE only after upsert;
   restart re-does the partial file. Qdrant/turso upsert is idempotent by point id, so
   re-doing is safe. Partial WRITE within a file (crash between chunk upserts of one file)
   → next run overwrites same ids — consistent.
4. **Model RAM:** search path holds MiniLM (~90 MB) + m3 (~2.3 GB) resident = ~2.4 GB in
   the node process (ground/pod RAM budget: fine at 16 GB host; flag in docs for small
   hosts; RAG_QUALITY_LANE_QUERY=0 releases nothing — the model loads lazily on first
   quality query; with the flag off from boot it never loads).
5. **Backfill + index_code concurrently on the same base:** allowed (different lanes), but
   chunk-id drift (file changed between the two runs) is resolved by rule 2 on the NEXT
   backfill. No lock needed in v1; document.
6. **User overrides `embedding_model` preset on index_code** (m3-only collection): search
   must NOT query a quality lane for a base whose fast lane fingerprint ≠ MiniLM — compare
   metadata points; role mismatch → 2-way.
7. **Backfill on a base that was only sparse-indexed:** no dense fast lane → skip with a
   clear message ("backfill needs a dense fast lane; index with mode=dense/hybrid first").
8. **Disk growth:** quality lane doubles vector storage for covered files (1024d vs 384d
   ≈ 2.7× per vector). Report per-lane sizes in `collection_stats`.
9. ** utf-8/BOM:** env knobs read via the repo's dotenv (P5 already utf-8-sig) — nothing
   extra.

## 4. Query-time budget (honest numbers, this VM)

| Path | Cost |
|---|---|
| fast-dense query (MiniLM) | ~30–50 ms |
| sparse query | ms (term hashing) |
| quality-dense query (m3, cold) | ~1.2–1.9 s CPU |
| quality-dense query (LRU hit) | ~0 |
| 3-way RRF merge | ms |
**Mitigation:** LRU (§2.4) + `RAG_QUALITY_LANE_QUERY=0` for latency-critical callers;
document that m3-encode cost is the price of the local-only Variant B.

## 5. Failure playbook

- Backfill subprocess dies (OOM/kill): PID-tracked runner reports non-zero; state file
  shows last complete file → restart resumes there. Alert = check_indexing_progress output.
- m3 load failure (HF cache missing/corrupt): backfill aborts BEFORE deleting anything;
  fast lane + sparse keep serving.
- Quality lane corrupt/incompatible fingerprint (pre-flight abort): backfill refuses;
  remediation = delete the quality lane collection only (fast lane untouched) and re-run.
- Search with quality lane present but m3 encode raises: log WARNING, return 2-way results
  (degrade, never fail the query) — mirror of the existing NullVectorStore philosophy.

## 6. Rollout sequence (each step shippable)

- **Phase A:** pass-1 switch to MiniLM + `lane_role` payload + per-lane state files.
  New indexes go fast; existing m3 lanes keep working (role=legacy). Unit tests.
- **Phase B:** backfill mode + throttling + progress reporting. Grounds probe 15.
- **Phase C:** 3-way RRF + LRU + lane provenance in hits + knobs. Unit + grounds re-probe.
- **Phase D:** docs (README embedding section): the language-policy note (§0.4), RAM
  budget, query-cost table, `RAG_TWO_PASS=0` escape hatch.

## 7. Docs note to ship in Phase D (user-mandated)

"Default embedding models are tuned for English-primary content. For other primary
languages, set the fast lane to a language-matched model (e.g. `bge-small-zh-v1.5` for
Chinese) or disable two-pass (`RAG_TWO_PASS=0`) and use bge-m3 directly — the quality lane
model (bge-m3) already covers 100+ languages."

## 8. Open items (not blockers)

- Hand-written 30-query in-domain set would firm up the pass-2 MRR argument (synthetic
  docstring queries may understate production differences — noted in the bench).
- Reranker probe (CPU-adjudication P1) composes with this design: rerank AFTER RRF over
  the fused list — out of scope here.
- If a dependable free-remote embedding ever materializes, Variant A remains available as
  a backfill PROVIDER swap (the lane abstraction is provider-agnostic by design).
