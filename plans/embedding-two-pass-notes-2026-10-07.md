# Odysseus lessons + two-pass embedding — session notes (2026-10-07)

**Purpose:** persist this session's findings as input for a future plan ("Lindsay" handoff).
**Status:** findings and measured data only — nothing implemented in the repo.
**Related:** `embedding-models-2026-10-04.md` (rounds 1–3, speed/RAM focus — this adds the quality axis), `plans/security-audit-2026-10-01.md` (P0/P1 pending).

> **IMPLEMENTATION STATUS (2026-10-07, post-session):** lessons §2 #1–#5 and the utf-8-sig
> quick hit were implemented on branch `feature/odysseus-lessons` (worktree
> `~/supreme-mcp-tools-odysseus`, cut from main c5f7237, commits b293677..eaa3ec6, suite
> 1047) — owner-mixed text_hash + bulk_upsert preservation + merge owner guard (#1),
> ragmcp embedding lanes with dim pre-flight + active-lane search (#2), queryMemory
> kw_weight + trgm/FTS5 lane + type boost (#3), shared/tidy_state.py fingerprint skip +
> shrink guard (#4). Remaining there: live sweeps after launcher restart from the worktree,
> grounds probe before P2 merge. This doc remains the design record for §4 (two-pass),
> which is NOT part of that branch.
>
> **GROUNDS GATE RESULT (2026-10-07 evening): PASS — 14 passed / 1 skipped** (skip =
> cross-node probe, needs the 2-node g2 ground). First run failed 2 probes; all root
> causes were REAL pre-merge defects, fixed on the branch: (1) stale
> `turso_vector.get_collection` swallowed the missing-table error → memorymcp ensure-at-init
> skipped on fresh turso files AND the lane pre-flight misread fresh lanes as sparse-only
> (pulled forward from pod-ui-gaps 3452dfd/193fca1 + tiktoken dep, commit 497cd98);
> (2) the indexer rewrite dropped the tools/ `sys.path` bootstrap → the spawned subprocess
> died silently at `from shared...`; (3) org-prefix normalization (`BAAI/bge-m3` vs bare
> `bge-m3`) → 768d derived for a 1024d model, lane/dim mismatch. Ground mechanics: image
> built FROM THE WORKTREE via a temporary build-context override in
> deploy/compose-g1-pg.run.yml (reverted after); probes executed from the pod-ui-gaps
> checkout (that branch carries probes 12–14 and the grounds tooling). Remaining merge
> gate: live sweeps (launcher restart from the worktree — user-timed).

---

## 1. What was compared

`~/odysseus` (self-hosted AI workspace, Python, AGPL-3.0-or-later) vs our ragmcp/memorymcp.
Three-way exploration: odysseus memory+RAG internals, odysseus MCP/platform layer, our current state.

**License rule (hard constraint):** our repo is **GPL-2.0**; odysseus is **AGPL-3.0**, which is
incompatible with GPL-2.0 (AGPL-3 pairs only with GPL-3+). Reuse **designs, algorithms,
constants, test ideas** — never copy code verbatim.

## 2. Lesson inventory (odysseus → our stack)

| # | Lesson | Reuse from odysseus | Plugs into (ours) | Size |
|---|--------|--------------------|-------------------|------|
| 1 | **Owner-scoped dedup** | `sha256(owner \0 text)` ids (`src/rag_vector.py:48`); owner check before dedup verdict (`memory_extractor.py:434`) | E3.5 already stamps `payload["owner"]` on Qdrant writes/reads (`memory_tools.py:244-273,370-374,562-569,642-647`). **Verified gap = SQL leg only:** `sql_store.upsert_memory()` (memory_tools.py:299) passes no owner, so `ON CONFLICT (text_hash, memory_type)` (postgres_sql.py:295, turso_sql.py:204) remaps user B's identical upsert onto user A's row and returns A's id to B | M |
| 2 | **Embedding lanes** | Per-model collections + fingerprint + re-embed preserved rows on change (`src/embedding_lanes.py:252`); dead-endpoint latch (`src/embeddings.py`) | `ensure_collection` fixes dim at creation (qdrant_vector.py:197-219, named `dense`/`sparse`); dimension derived per process (incremental_indexer.py:85-94); incremental indexer's stale-chunk removal = re-embed machinery | M |
| 3 | **Hybrid memory retrieval** | `0.55·vec + 0.40·kw + 0.05·recency` + gates + category boosts (identity 1.4 / contact 1.3 / preference 1.2) (`chat_processor.py:158`) — starting defaults, retune later | `relevance_scorer.py` (extend with keyword term + type boost on our `memory_type`); pg_trgm/FTS5 exist in both SQL impls, **verified unused by queryMemory** (zero hits); `word_jaccard_similarity` as no-SQL fallback | M |
| 4 | **Maintenance-pass fingerprint** | `sha256(sorted(id+text+category))` sidecar skips no-op LLM/scan passes (30–120s each) (`memory_extractor.py:34`); >50% over-deletion guard | `decayOrExpire` (has dry_run) and `mergeDuplicates` | S |
| 5 | **Fail-closed read-modify-write** | `MemoryStoreUnreadable` — a degraded read must never be persisted (`src/memory.py:14`; their store-destroying bug) | Same class as grounds finding: users_store failure paths silently fall back to JSON | S |
| 6 | **Policy anti-pollution line** | "max 2 facts/turn, ≤15 words" extractor constraint | one line in `tools/memorymcp/auto_use_policy.md` + cheatsheet | S |
| 7 | **Threat-model doc + pinned regression tests** | THREAT_MODEL.md structure (honest framing, roles→capability matrix, Known Gaps); `tests/test_security_regressions.py` (1,546 LOC) | content source = plans/security-audit-2026-10-01.md (1 P0 + 5 P1, fixes pending) | S/M |
| 8 | Quick hits | `utf-8-sig` dotenv (BOM); SSRF guard on saved embedding endpoints (loopback default, reject cloud-metadata); fail-closed read-only tool classifier from `readOnlyHint`/verb prefixes; schema caps for prompt-spliced tool schemas (12 params/40-char tokens/300-char hints) | launcher env loading; ragmcp embedding config + `SMARTGRAPH_BASE_URL` (timely with OpenRouter); complement to E3 masks | S each |

**Sequencing:** #2 before any embedding-model switch (incl. OpenRouter adoption); #1 before E3
multi-user goes live; #3/#4 independent; #5–#8 filler.

**Not reused (we're ahead):** their flat-JSON memory store, no-decay memory (we have
`decayOrExpire` + retention policies), sentence-only chunking (we have PL/SQL/Java/Pro*C
chunkers), no incremental re-index, RRF-less retrieval, stdio-builtin MCP duplication.

## 3. Embedding benchmark — measured 2026-10-06/07

**Method:** BEIR SciFact, 300 test queries vs 1,200-doc corpus (283 gold + seeded fillers,
`random.Random(42)`, fillers from non-gold), nDCG@10 + MRR@10 (graded qrels, gain 2^rel−1),
cosine on L2-normalized vectors. Each model used its documented query-prefix convention
(bge-base + arctic: `Represent this sentence for searching relevant passages: ` prefix;
bge-m3/MiniLM/nemotron: none). CPU: 6-core Ryzen 3 4300GE, torch 2.11+cpu,
sentence-transformers 5.3.0; remote via OpenAI-compatible `/v1/embeddings`, batch 32.

| Rank | Model | Dims | nDCG@10 | MRR@10 | ms/doc | Where |
|---|---|---|---|---|---|---|
| 1 | `llama-nemotron-embed-vl-1b-v2:free` | 2048 | 0.8700 | 0.8478 | 20 | remote $0 |
| 2 | `nvidia/nemotron-3-embed-1b:free` | 2048 | 0.8627 | 0.8426 | 37 | remote $0 |
| 3 | BAAI/bge-base-en-v1.5 | 768 | 0.8549 | 0.8298 | 532 | local |
| 4 | bge-base PCA→512 | 512 | 0.8449 | 0.8206 | — | derived |
| 5 | all-MiniLM-L6-v2 (odysseus stack) | 384 | 0.7730 | 0.7441 | 58 | local fastembed |
| 5b | paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.6168 | 0.5752 | 72 | local fastembed |
| 6 | **BAAI/bge-m3 (incumbent)** | 1024 | 0.7644 | 0.7398 | 1914 | local |
| 7 | snowflake-arctic-embed-l-v2.0 | 1024 | 0.7435 | 0.7227 | 1878 | local |

*(multilingual sibling row: `paraphrase-multilingual-MiniLM-L12-v2` — pending, see §7.)*

**Findings (evidence, not inference):**
1. **Dimension effect ≈ zero.** Same-model PCA truncation costs ~1 pt: bge-m3
   1024→768→512 = 0.7644→0.7531→0.7522; bge-base 768→512 = 0.8549→0.8449. Dims are a
   storage/reindex knob, not a quality gate. Confirms Round-3 conclusion in
   `embedding-models-2026-10-04.md` ("dims demoted to a storage knob") with measurement.
2. **Training/domain dominates.** At equal 768 dims: bge-base 0.8549 vs bge-m3-PCA 0.7531 —
   10 pts from training, not dims. The earlier "768 worse than 1024" observation was
   generation mismatch (2023 English-only 512-ctx vs 2024 multilingual 8k-ctx), not the
   extra 256 dims. Arctic at the *same* 1024 dims only reaches parity with m3.
3. **384d beat the 1024d incumbent** on this task (MiniLM 0.7730 > bge-m3 0.7644) at 33×
   speed. Recorded prediction failure: I predicted MiniLM would land "clearly below" m3.
4. **Free remote models top the table** — best quality AND fastest (20–37 ms/doc including
   round-trip). Upgrades the 10-04 verdict ("indexing offload only") to "also quality-first
   on this bench". No rate-limit friction at 32-doc batches; liquid-502 reliability precedent
   still stands.
5. Caveats: SciFact is English (flatters English-only models; says nothing about our
   code+multilingual domain); corpus subsampled (absolute values not BEIR-comparable —
   relative ordering on identical corpus is the signal); PCA on non-MRL models likely
   *understates* what MRL-trained small dims keep; nemotron ran with no query instruction
   (none documented for the :free endpoint).

## 4. Two-pass embedding design (converged with user)

User goal: **dense-semantic-first**, sparse rides along (it's term-hashing, ~free), quality
upgrade in background. Scope: **ragmcp only** — memorymcp writes one memory at a time
(no bulk-index problem; `reindexMemory` covers upgrades).

```
pass 1 (minutes–hours): MiniLM dense (384d, 58 ms/doc) + sparse (free, same run)
                        → hybrid search live almost immediately
pass 2 (background):    nemotron vl-1b-v2 remote → 35 min / 100k chunks, +10 pts bench
                        OR bge-m3 local niced/2-threads → ~1 week / 100k, domain-fit
search (steady state):  RRF(fast-dense, sparse, quality-dense) — rank fusion ONLY
```

Rationale locked in session:
- Weakest-fast + strong-slow maximizes the architecture's value (+10 pass-2 gain vs +1.5
  if pass 1 were bge-base).
- Pass-1 coverage time = max staleness window for un-upgraded chunks; MiniLM shrinks it
  10× (1.6 h/100k vs 15 h) and barely touches the CPU (user's "no overload" requirement).
- If egress is refused, pass 2 = local m3 throttled buys domain fit (user's in-domain
  tests: m3 > base for our content), not bench quality.

**Implementation requirements (for the plan):**
1. Separate collection per embedding lane — never mix dims/models in one collection
   (Qdrant fixes dim at creation). Fast lane: 384d dense + sparse named vectors; quality
   lane: 2048d (or 1024d) dense-only.
2. Per-chunk freshness per lane (payload `emb_tier` / per-lane state next to the indexer's
   mtime metadata): a changed file must invalidate chunks in BOTH lanes or the quality lane
   confidently matches stale text (ghost chunks).
3. Per-lane query consistency: each lane queried with its own model's query embedding
   (MiniLM local ms-cheap; nemotron ~115 ms round-trip; sparse query = local term hashing).
4. RRF extension from 2-way (current dense+sparse) to N-way; never raw-merge scores across
   models (similarity scales differ — visible in mean_top1_cos spread 0.37–0.72).
5. Runner: ragmcp's existing PID-tracked background indexer (`start_indexing` /
   `check_indexing_progress` / `stop_indexing`) gains pass-2 mode; local throttling via
   `torch.set_num_threads(2)` + `nice`.

**Rough effort:** lane naming + freshness state + N-lane RRF + indexer modes ≈ M.

**DECISION (2026-10-07, user): Variant B locked — LOCAL ONLY.** Rationale: OpenRouter cannot
guarantee free-model availability; the quality lane must never depend on it. Pass 2 =
**bge-m3 local, throttled** (nice + `torch.set_num_threads(2)`), ~4 days/100k chunks,
domain-fit is the goal (not bench quality). The nemotron rows in §3 stand as evidence of the
ceiling forgone; the 10-04 "remote indexing offload" idea remains available for non-critical
bulk jobs only.

**Language policy (user note):** user and expected users are English-first. Pass-1 candidates
(MiniLM-L6, bge-base) are English-only; pass-2 bge-m3 is already multilingual (100+ languages,
strong zh). Non-English-primary deployments should swap pass-1 for a language-matched model
(e.g. `bge-small-zh-v1.5`/`bge-base-zh-v1.5` for Chinese; the multilingual MiniLM sibling
exists but costs −15.6 pt English per §7). When two-pass lands, surface this as a model-choice
note in ragmcp docs.

## 5. Harness (rebuild spec)

Scripts live in `/tmp/emb_bench/` (volatile — re-create if rebooted):
- `bench.py` — local sentence-transformers models + PCA controls (numpy SVD, fit on corpus,
  renormalize), resume via `emb_<tag>.npz` sidecars.
- `bench2.py` / `bench4.py` — fastembed models in odysseus venv
  (`~/odysseus/.venv/bin/python`), cache_dir `~/odysseus/data/fastembed_cache`.
- `bench3.py` — OpenRouter `POST /api/v1/embeddings`, key from project `.env`
  `OPENROUTER_KEY` (never printed), 429/5xx retry with backoff, batch 32.
- Subset rule must stay identical across runs: seed 42, gold docs + fillers from non-gold,
  queries filtered to qrels. Adding a model = ~10 lines in any of the three scripts.

## 6. Open decisions (for the future plan)

1. **Egress for pass 2 — DECIDED 2026-10-07: Variant B, local only** (OpenRouter free-tier
   availability is not dependable; see §4 decision block).
2. **Pass-1 model:** `all-MiniLM-L6-v2` — pending in-domain confirmation (§6.3).
3. **In-domain bench (code-first):** SciFact settled the generic lessons, not our domain.
   Corpora = user's $HOME Python projects: `~/fastmcp` (435 py), `~/fastapi` (533),
   `~/nicegui` (524), `~/graphify` (205). The PL/SQL/Java/Pro*C chunkers are work-legacy;
   personal-project content is Python — no special chunkers needed for the bench. Method:
   CodeSearchNet-style — docstring first sentence as the natural-language query, gold =
   the defining file, nDCG@10/MRR@10 as in §3. Decides: MiniLM-vs-alternatives for pass 1
   on real code, and whether pass-2 bge-m3 actually improves over pass-1 on code (the
   Variant-B value proposition).
4. Lesson table §2: which items become the plan (recommended: #2 lanes + #1 owner-dedup first).
5. From the 10-04 doc, still open: keep-or-delete arctic-embed-l-v2.0 (2.2G, referenced by
   nothing).

### 6.3 RESULT (2026-10-07 run) — dense model choice is NOT the code lever

Corpus: 1,186 real `.py` files (fastmcp 305 gold / graphify 152 / nicegui 131 / fastapi 12 —
fastapi's docstring style resisted extraction; noted skew), 600 AST-mined queries,
content-only docs at 512-token truncation (production CHUNK_SIZE=300 is shorter).

| Model | nDCG@10 | MRR@10 | ms/doc |
|---|---|---|---|
| MiniLM-L6 (pass-1 candidate) | 0.5467 | 0.3358 | 82 |
| bge-base (anchor) | 0.5517 | 0.3862 | 356 |
| bge-m3 (pass-2 candidate) | 0.5483 | 0.3936 | 1211 |

**Findings:**
1. **nDCG@10 is flat** (spread 0.005 over 600 queries) — on whole-file Python retrieval with
   natural-language queries, all three local models sit at the same quality plateau. The
   SciFact gaps (8–10 pts) do not transfer to code.
2. **MRR@10 keeps a small ordering** (MiniLM 0.336 < base 0.386 ≈ m3 0.394): the bigger
   models put the right file at rank 1 somewhat more often — the only measured pass-2 gain
   on code.
3. **Implication for the two-pass plan:** Variant B's pass-2 (m3) on code buys ~nothing in
   nDCG and a modest MRR polish. Its remaining justifications: multilingual content, 8k
   context (this bench truncated at 512 — m3's long-context edge untested), future reranker
   carrier. The bigger quality levers on code appear to be **chunking granularity, the
   sparse lane, and reranking** — not the dense model.
4. Bench-shape caveat: synthetic docstring queries saturate lexical overlap and may
   understate differences that production-style queries ("where is X handled") would show.
   A 30-query hand-written set would be the higher-fidelity arbiter before major effort
   goes into pass-2 tuning.

(All-MiniLM-L6-v2 was downloaded via sentence-transformers for this run — ~90MB now in the
project HF cache; delete with the other test artifacts if unwanted.)

## 7. Addendum — multilingual sibling bench (2026-10-07)

`paraphrase-multilingual-MiniLM-L12-v2` (384d, the multilingual sibling of odysseus's
fallback, same fastembed stack, same corpus): **nDCG@10 = 0.6168**, MRR@10 = 0.5752,
72 ms/doc.

**Interpretation:** multilingual training costs ~15.6 pts of English quality vs the
English L6 (0.7730) — a much bigger gap than the English-only models' spread. Two
consequences for the pass-1 decision (§4, §6.2):

- If the corpus is English/code-dominant, pass 1 should be **all-MiniLM-L6-v2**; the
  multilingual sibling is not a free swap ("same dims, still fast" — but not the same
  quality).
- If genuine multilingual pass-1 coverage is required, neither MiniLM variant is
  attractive (0.617 English, and the sibling's multilingual strength is untested here);
  the honest options become bge-base (English-only but 0.855) at 532 ms/doc, or leaning
  harder on pass 2 (bge-m3/nemotron are multilingual and land in the background lane).

The sibling's multilingual *advantage* was not measured — SciFact can't show it. An
in-domain bench (§6.3) is the arbiter.
