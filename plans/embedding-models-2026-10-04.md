# Local embedding models — status & benchmark (2026-10-04)

## What's installed right now

| Model | Size | Embedding types | Who uses it (and how) | Special |
|---|---|---|---|---|
| **BAAI/bge-m3** | 4.3G | **dense 1024d + sparse (learned lexical weights) + ColBERT multi-vector** — one model, three output kinds | **memorymcp**: embeds every stored/queried memory (default in `memory_core.py`, `LOCAL_EMBEDDING_MODEL` in host `.env`, `reindexMemory` default). **ragmcp**: server default, the `auto` indexing preset, dense+sparse **hybrid search**, and the fallback for collections built with an unknown model. Pinned the same way inside the pods. | The only installed model with a **sparse** output — the hybrid-search path dies without it. Long context (8,192 tokens), multilingual (100+ languages). |
| **BAAI/bge-base-en-v1.5** | 419M | **dense only, 768d** | **ragmcp only**, and only when you index with the `fast`/`base` `embedding_model` preset (also listed in `tools/ragmcp/config.json`). memorymcp never touches it. | The speed escape hatch: 3× faster than bge-m3 (0.5 vs 1.6 s/doc), smallest RAM (1.0G). English-focused, 512-token context. |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 2.2G | **dense only, 1024d** | **Nothing** — no config, no env, no code references it. Adopting it would mean repointing `LOCAL_EMBEDDING_MODEL` and reindexing every collection. | The only tested peer at bge-m3's level (same backbone class, parity speed, parity quality, Apache-2.0) — but no sparse output, so adopting it would **lose ragmcp's hybrid search**. 8,192-token context, multilingual. |

Cache total: **6.9G** (was 14G plus test downloads).

## What was removed (2026-10-04, disk free 18G → 28G)

- `gte-Qwen2-1.5B-instruct` (6.7G) — legacy experiment, unreachable through the repo's own loader
- `multilingual-e5-large` (2.2G) — zero code references; a strict subset of bge-m3
- `Qwen3-Embedding-0.6B` (1.2G) — test download, 8× too slow (see below)

Also deleted after testing (transient test downloads, same day): jina-v3 partial (1.1G), gte-multilingual-base (599M), nomic-v2-moe (1.8G), mxbai-large-v1 (641M), gte-large-en-v1.5 (1.7G), plus orphaned custom-code caches.

## Benchmark verdicts (on this VM: 6-core Ryzen 3 4300GE, CPU-only, 16 × ~1KB chunks)

| Model | Speed | Peak RAM | Verdict |
|---|---|---|---|
| bge-base-en-v1.5 | 0.5 s/doc | 1.0G | keep — fast preset |
| **bge-m3** | 1.6 s/doc | 2.3G | **keep — default; comfortable on this VM** |
| arctic-embed-l-v2.0 | 1.6 s/doc | 2.4G | parity speed/quality, but dense-only → no reason to switch |
| e5-large | 1.5 s/doc | 2.3G | removed — same cost as bge-m3, used by nothing |
| gte-Qwen2-1.5B | 4.5 s/doc | 6.4G | removed — slow + heavy, degraded under the repo's loader |
| Qwen3-Embedding-0.6B | 13.0 s/doc | 2.2G | removed — decoder-LLM embedder, wrong compute class for 6 CPU cores |

## Rejected candidates (2026-10-04 test round)

- **jina-embeddings-v3** — crashes on transformers 5.5.4 (its custom `XLMRobertaLoRA` code); also CC-BY-NC license and needs `trust_remote_code`
- **gte-multilingual-base** — crashes at encode (IndexError in its custom RoPE code)
- **google/embeddinggemma-300m** — gated repo (needs HF login + Google terms); 768d native anyway
- **Microsoft BitNet** — no embedding model exists (BitNet is a text-generation LLM family)
- **nvidia/llama-embed-nemotron-8b** — 8B ≈ 32G fp32 weights vs 22G total RAM: impossible

## Viral comparison table vs measured reality (2026-10-04)

A circulated table claims 105–580 chunks/s on CPU for five models. **Every one of those CPU numbers is physically impossible** (580 chunks/s × 305M active params × 2 FLOPs × ~250 tokens ≈ 88 TFLOPS sustained — top-GPU territory, ~500× this VM's CPU) and no published benchmark backs them. Measured on this VM with the standard 16-chunk sample:

| Claim | Reality on this VM |
|---|---|
| nomic-embed-text-v2-moe — 580 chunks/s | **1,253 ms/doc (0.8 chunks/s)** — works; 768d, 512 ctx, 4.8G RAM, needs `trust_remote_code` |
| jina-embeddings-v3 — 220 chunks/s, "best accuracy" | **cannot load** on transformers 5.5.4 |
| arctic-embed-l-v2.0 — 130 chunks/s | **1,561 ms/doc (0.64 chunks/s)** — works (blue row above) |
| gte-large-en-v1.5 — 110 chunks/s | **cannot load** (same custom-RoPE IndexError as gte-multilingual-base) |
| mxbai-embed-large-v1 — 105 chunks/s | **9,069 ms/doc (0.11 chunks/s)** — works but the *slowest* working model measured (likely eager-attention vs SDPA; 1.5× slower than bge-m3 even on 10-token inputs) |

The "retrieval@10 87–92%" column likewise matches no standard metric (BEIR-class nDCG@10 for these models sits in the 50s–low-60s). Takeaway unchanged: **bge-m3 remains the fastest *usable* multilingual dense+sparse model here** — nomic-v2-moe edges it on raw dense speed (~20%) but is trust_remote_code-blocked, 768d, and eats 4.8G RAM.

### The interesting survivor: nomic-v2-moe (kept on file, not installed)

`nomic-ai/nomic-embed-text-v2-moe` is the only local model tested that beats bge-m3 at anything: **1,253 vs 1,554 ms/doc dense (~20% faster)**, multilingual (100+), MoE with ~305M active params, and — unlike jina-v3 and the gte family — its custom code actually runs on transformers 5.5.4.

Why it stays out today:
- **`trust_remote_code=True` required** — the repo's loaders hardcode it off (`local_embeddings.py`, `memory_core.py`); enabling it means executing model-repo code on every load
- **768d vs your 1024d collections** — adoption means fresh collections + full reindex
- **4.8G peak RSS** — the MoE keeps all experts resident; heavier than bge-m3's 2.3G
- **dense-only** — no sparse output, so ragmcp's hybrid search is lost on any nomic-backed collection

What would change the calculus: a `transformers`-native (no custom code) release of nomic-v2-moe, or a deliberate decision to allow trust_remote_code for ragmcp's local path. Until then it's a footnote, not a candidate — a ~20% dense gain doesn't buy back hybrid search, dimensions, and 2.5G of RAM.

## OpenRouter free embeddings — remote, tested live (2026-10-04)

Key: `OPENROUTER_KEY` in the project `.env`; endpoint `POST https://openrouter.ai/api/v1/embeddings` (OpenAI-compatible). Same 16-chunk sample as the local benchmark; latencies are full round-trip from this VM.

| Model | Dims (measured) | Single-doc RTT | 16-chunk batch | Context | Status |
|---|---|---|---|---|---|
| `nvidia/nemotron-3-embed-1b:free` | **2048** | **115 ms** | **0.21 s** | 32,768 | works, $0 |
| `nvidia/llama-nemotron-embed-vl-1b-v2:free` | **2048** | **126 ms** | **0.24 s** | ~10,240 | works, $0 — also embeds page images/charts |
| `liquid/lfm-2.5-embedding-350m:free` | — | — | — | 512 | route down at test time (HTTP 502 ×3) |

Against the local baseline (bge-m3: 1,554 ms/doc, 24.9 s per 16-doc batch, 1024d, on-VM): the remote 1B models are **~13× faster per call and ~100× on batch throughput**, with zero VM CPU cost.

Trade-offs that stop a blind adoption:
- **2048d ≠ 1024d** — every existing collection must be reindexed; mixed dimensions across tools break shared stores
- **dense-only** — ragmcp's sparse/hybrid path exists only with local bge-m3
- **free tier** — rate limits, no SLA; chunk text leaves the machine; network hop lands in every query path
- **integration shape** — ragmcp's remote provider is OpenAI-compatible with a configurable `api_url`, so OpenRouter fits it directly; memorymcp is local-only today

Verdict: attractive as a ragmcp *indexing* offload (100× batch), not a drop-in default. If adopted: fresh collection namespace, keep bge-m3 for memorymcp and the sparse half, and treat free-tier reliability as best-effort.

## Round 3 (2026-10-04): refined criteria — semantic quality per VM cost, dims as a storage knob

User's working criteria, refined mid-round: **quality first, CPU/RAM/time on the VM; dimension count demoted to a storage knob** (correct call — newer 768d models routinely outscore older 1024d ones; dims only hard-matter below ~256 and for the reindex-when-changing fact).

New tests this round:

- **LFM2.5-Embedding-350M local** — **NEVER on this VM**: measured **12,478 ms/doc** (8× bge-m3) even loading natively (no trust_remote_code; sentence-transformers fell back to generic mean pooling since the repo ships no ST config — a tuned path might close some gap, not 8×). 1024d ✓, CPU/time ✗. Download deleted.
- **OpenRouter slug probes**: `nvidia/llama-embed-nemotron-8b` and `qwen/qwen3-embedding-4b` do **not exist** there — the remote menu stays at the two 1B models.
- **liquid/lfm-2.5-embedding-350m:free recovered** (was 502): re-measured — **1024d, 552 ms/doc, 1.57 s/16-batch**. Same dims as existing collections (no dimension migration, though vectors from different models still can't mix — reindex required), 11-language coverage only.

### Standing ranking (verdict first)

| Verdict | Option | Why |
|---|---|---|
| **USE** (incumbent) | bge-m3 local | 1024d, 1.55 s/doc, dense+sparse hybrid, proven in prod |
| **USE — if egress accepted** | OpenRouter `nemotron-3-embed-1b:free` | 2048d, **115 ms/doc**, 0.21 s/16-batch, 32k ctx, $0 — the only verified big win (13× single-call, ~100× batch). Costs: reindex into a fresh namespace, dense-only, chunk text leaves the machine, free-tier reliability (liquid's 502 outage is the proof) |
| NOT NOW | nomic-v2-moe local | good quality-per-dim (768d), 1.25 s/doc — but needs a repo code change (`trust_remote_code`) + reindex + 4.8G RAM. Trigger: transformers-native release |
| NOT NOW | EmbeddingGemma | quality-strong at 768d, but gated repo blocks automated testing. Trigger: ungated mirror |
| NEVER (this VM) | LFM2.5 local, Qwen3-0.6B, gte family, jina-v3, mxbai, gte-Qwen2-1.5B | measured crashes or 4.5–12.9 s/doc |
| Backup | arctic-embed-l-v2.0 local | parity with bge-m3, dense-only, referenced by nothing |

## Bottom line

**Local verdict is closed:** bge-m3 stays as the default, bge-base as the fast preset. Across three test rounds, no local alternative beats it on the quality-per-VM-cost combo; nomic-v2-moe is the only local model faster at dense (~20%) and it is trust_remote_code-blocked.

**The one real upgrade path is remote:** OpenRouter's free `nemotron-3-embed-1b` (2048d, 115 ms/doc, ~100× batch, $0) — adopt as a ragmcp indexing offload into a fresh collection namespace if the egress + free-tier-reliability tradeoff is accepted; liquid (1024d, 552 ms/doc) is the dimension-compatible fallback route.

**Open decisions:** (1) keep arctic-embed-l-v2.0 (2.2G, dense-only backup, referenced by nothing) or delete it; (2) adopt the OpenRouter offload for ragmcp indexing or stay all-local.

Companion chart: `embedding-model-comparison-2026-10-03.png` (same folder). Benchmark conditions: sentence-transformers 5.3.0, torch 2.11+cpu (6 threads), batch 32, production load path; sources and per-candidate failure details live in `embedding-models-inventory-2026-10-04.xlsx` (archived locally in `~/supreme-mcp-tools-backups/`, deliberately not in the repo).
