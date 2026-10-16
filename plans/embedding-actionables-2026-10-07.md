# Embedding actionables — what's actually actionable for this repo (2026-10-07)

**Status:** PLAN (nothing executed). Tracker: `TODO.md`.
**Source:** two external LLM replies on "CPU-only embedding picks" (bge-small/OpenVINO-int8/nomic-MRL recommendations), cross-checked against measured reality in [`embedding-models-2026-10-04.md`](../embedding-models-2026-10-04.md) (repo root, 3 elimination rounds on this VM). The replies' relative claims are sound; their absolute throughput tiers assume 16–32-core AVX2 servers and are 10–100x off for this box. This doc extracts only what survives the cross-check.

## Constraints that gate everything (measured/documented)

1. **bge-m3's learned-sparse output powers ragmcp hybrid search.** Dense-only alternatives (bge-small/base/e5/nomic/arctic) lose it. This killed every candidate in the 2026-10-04 rounds and kills the replies' picks for the same reason.
2. **1024d production collections + model-mixing rule:** vectors from different models never mix; any model or dim change = fresh collections + full reindex. memorymcp is single-collection, so the "small model for queries, big model for indexing" trick would need a second collection + dual writes.
3. **VM profile:** 6c/6t Ryzen 3 4300GE (Zen 2), AVX2 but **no AVX-512 / no AVX-VNNI** — vendor int8 speedup claims (5–8x) shrink here; expect 2–4x (*inference*; the missing VNNI is documented).
4. **Pods:** HF cache bind-mount is WORK-pod only; TEST pod has no cache (embeddings broken there until the TODO 3.3 probe passes).
5. Loaders hardcode `trust_remote_code=False`; runtime is torch 2.11+cpu **fp32** (fp16 on CPU is a 5x slowdown — the one reply warning that is simply true and already avoided).

## P0 — trivial correctness/housekeeping (no benchmarks needed)

- [x] **Fix stale ragmcp local presets** — DONE 2026-10-17. Cache re-check found `small` (`BAAI/bge-small-en-v1.5`) was ALSO gone (the plan's "only small/base resolve locally" was half wrong: only `base` resolved). The dead `sbert-large`/`e5-large`/`small` presets are dropped from `tools/ragmcp/config.json` + `config.example.json` (`embedding.local` = `bge-m3` + `base`, both cached); `LOCAL_EMBEDDING_MODEL` env block and `tools/ragmcp/README.md` aligned. Bonus bug found in the same sweep: `indexer/local_embeddings.py` mapped alias `base` → `BAAI/bge-base-en` (no `-v1.5` — NOT the cached model, silent re-download on the indexer path); fixed to `BAAI/bge-base-en-v1.5`. Note: `ragmcp_fastmcp.py` never reads config.json — the code maps (`LOCAL_EMBEDDING_MODELS`, `EMBEDDING_MODEL_PRESETS`) were already clean; the config block is display metadata only.
- [x] **Close the arctic-embed-l-v2.0 open decision** — DECIDED 2026-10-17: **KEEP** (2.2G, dense-only labeled backup; zero code/config references; disk is not scarce at 28G free). Not added to the ragmcp presets — adopting it costs a full reindex and loses hybrid search.

## P1 — probes with no reindex cost (quality/storage wins on existing data)

- [ ] **Qdrant store-side scalar int8 quantization on existing collections.** Enable via collection config patch (`quantization_config.scalar: int8, quantile≈0.99, always_ram: true`) — Qdrant reindexes in the background, no re-embedding. Probe on a scratch copy of `memory-store` first: measure recall delta (reuse the `/tmp/emb_bench` SciFact harness pattern) + RAM/disk savings. The "4x storage, ~1.5–3.5% loss" figure is a reply claim — *unverified here; the probe decides*. Binary quantization (32x, rescoring required) is the aggressive follow-up only if int8 passes. Turso vector has no equivalent — this is a PG/Qdrant-only win.
- [ ] **Cross-encoder reranker latency probe.** `bge-reranker-v2-m3` (multilingual, pairs with bge-m3) on this VM: measure ms/pair at 512 tokens. Viability gate: top-10 rerank ≤ ~2 s. If it passes → optional `rerank: true` param on ragmcp search (no reindex, pure quality add). If too slow → try an ONNX-int8 reranker or skip. The replies' 20–40 ms/pair assumes bigger CPUs (*inference* that we'll be slower).

## P2 — runtime speed (trigger-gated; do not start without a trigger)

- [ ] **ONNX Runtime/OpenVINO int8 for the dense fast preset (bge-base) only.** Precedent exists: odysseus runs fastembed all-MiniLM via ONNX. This is an *indexing-throughput* and *fast-preset* play — **not for bge-m3**: its sparse head does not survive trivial ONNX export (custom work) and hybrid search is load-bearing (constraint 1).
  - **Trigger:** measured pain first — (a) time a real short-query embed on the bge-m3 live path (the 1.55 s/doc figure is ~1 KB chunks; short queries should be far cheaper — *inference, measure it*); (b) a concrete need for faster `base`-preset indexing.
- [x] **Archive the dims bench** — CLOSED 2026-10-17 with a caveat: the bench COMPLETED (results in the session memory `embedding-dims-bench-2026-10-07`: dims = storage knob, m3 PCA 1024→512 ~1.2pt, MiniLM-384 beat m3 on SciFact, CODE bench flat on real Python) and is written up in `plans/embedding-models-2026-10-04.md` (committed) — but the `/tmp/emb_bench` scripts themselves were LOST to a reboot before archiving. Only the summary numbers survive. If the PCA harness is ever needed again, it must be rebuilt from the writeup.

## Standing items (pre-existing, unchanged by the replies)

- **OpenRouter `nemotron-3-embed-1b:free` ragmcp indexing offload** (2048d, 115 ms/doc, ~100x batch): blocked only on the user's egress + free-tier-reliability decision. Fresh collection namespace required; keep bge-m3 for memorymcp and the sparse half. (2026-10-04 doc §Round 3.)
- **TEST-node embedding probe** (TODO 3.3) — first TEST boot.

## Rejected by measurement — do not re-propose

Qwen3-0.6B (13 s/doc here, deleted), LFM2.5-350M local (12.5 s/doc), gte family + jina-v3 (crash on transformers 5.5.4), mxbai-large (9 s/doc), gte-Qwen2-1.5B (4.5 s/doc, 6.4G), e5-large (bge-m3 strict subset), nomic-v2-moe (needs `trust_remote_code` + dense-only + 4.8G; revisit only on a transformers-native release), EmbeddingGemma (gated repo), fp16-on-CPU (never), blanket `trust_remote_code=True` (policy).

## Measurement-first rule

No adoption from external model tables without this-VM numbers. Precedent: the viral 105–580 chunks/s table measured at 0.1–0.8 chunks/s here (2026-10-04 doc). The two replies' tier numbers repeat the same hardware-blindness class of error.
