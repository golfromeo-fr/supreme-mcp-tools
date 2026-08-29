# MCP Live Tool Sweep Report — 2026-08-27

Live validation of the supreme-mcp-tools servers via native `mcp__<server>__<function>`
calls from this session — the same path a real harness client uses
(harness MCP client → auth → session → server → tool). Not scripted HTTP; no
companion scripts were substituted.

**Summary: 26 passed / 4 failed / 1 skipped-by-design across 4 reachable servers.**
No `Session not found` (-32600) anywhere; all four sessions healthy end-to-end.

---

## Server coverage

| Server | Port | Wired into session | Sweep result |
|---|---|---|---|
| simplemcp | 8002 | yes | 4/4 PASS |
| webmcp | 8001 | yes | 3 PASS, 1 FAIL (known), 1 SKIP (not tested — user directive) |
| ragmcp | 8004 | yes | 4 PASS, 3 FAIL (**new defect**) |
| memorymcp | 8005 | yes | 15/15 read-only probes PASS |
| convertermcp | 8003 | **no native tools wired** | NOT TESTABLE |
| oraclemcp | 8000 | **no native tools wired** | NOT TESTABLE |

convertermcp and oraclemcp are configured in their respective `config.json`
files and exposed by the launcher, but this harness session has no
`mcp__convertermcp__*` / `mcp__oraclemcp__*` tools available. They need to be
wired into the harness config before a live sweep can validate them.

---

## Detailed results

### simplemcp — deterministic assertions, all exact

| Function | Args | Expected | Actual | Verdict |
|---|---|---|---|---|
| `double` | `value: 21` | 42 | 42.0 | PASS |
| `square` | `value: 7` | 49 | 49.0 | PASS |
| `greet` | `name: "smoke"`, `greeting: "Howdy"` (optional param exercised) | `"Howdy, smoke!"` | `"Howdy, smoke!"` | PASS |
| `get_secret` | `{}` | fixture value set | set, length 12 | PASS |

### webmcp

| Function | Args | Result | Verdict |
|---|---|---|---|
| `fetch_url` | `https://example.com`, images off | status 200, but body ≈ 318 bytes of raw gzip mojibake; content does NOT contain "Example Domain", U+FFFD garbage present | **FAIL — KNOWN DEFECT** |
| `post_url` | `https://httpbin.org/post`, JSON body `{"probe": "live-sweep", "n": 1}` | payload echoed verbatim (`json.n = 1`) | PASS |
| `brave_search_api` | query "Model Context Protocol specification", count 3 | 3 structured web results | PASS |
| `google_search_api` | query "FastMCP python framework", num 3 | organic results + related searches | PASS |
| `brave_search_web` | — | not tested — user directive (2026-08-27): the function is not meant to be used but cannot be hidden server-side; removed from all testing | NOT TESTED |

### ragmcp

Environment: Qdrant has 4 collections (`fastmcp-code` 424 chunks/1024d,
`memory-store` 155 chunks, `fastapi-code` 3196 chunks, `sparse-fix-probe3`
1 chunk/8d+sparse).

| Function | Args | Result | Verdict |
|---|---|---|---|
| `list_collections` | `{}` | 4 collections with chunk/vector stats | PASS |
| `search` mode=auto (dense hit) | `fastmcp-code`, "client connection management" | 2 correct chunks (transports/base.py, providers/proxy.py), scores 0.598/0.587 | PASS |
| `search_code` (dense legacy) | `fastmcp-code`, "streamable http transport" | exact hit: `fastmcp/client/transports/http.py`, score 0.655 | PASS |
| `search` mode=sparse | `fastmcp-code`, "Mcp-Session-Id" → then "ClientTransport connect_session" | zero hits both times | **FAIL** |
| `search_code_sparse` (legacy) | `fastmcp-code`, "initialize handshake" → then "httpx client" | zero hits both times | **FAIL** |
| `get_copilot_context` (sparse consumer) | `fastmcp-code`, "ClientTransport connect_session httpx" | `"No relevant context found"` | **FAIL** |
| `check_indexing_progress` | `{}` | clean "No indexing log found" state response | PASS |

### memorymcp — read-only + dry-run probes only

Write/mutating tools (`upsertMemory`, `deleteMemory`, `decayOrExpire(dry_run=false)`,
`mergeDuplicates(dry_run=false)`, `clear_index`, migration/reindex) were not
exercised beyond dry-run previews, per sweep policy.

| Function | Args | Result | Verdict |
|---|---|---|---|
| `queryMemory` | "mcp launcher backend abstraction smoke test", k=3, recency_weight 0.7 | 3 ranked hits incl. score 0.852 recency-weighted top hit | PASS |
| `getMetaDecisions` | "storage backend", k=2 | 2 level:meta/priority:A decisions returned | PASS |
| `decayOrExpire` | dry_run=true, ttl_days=1 | preview only: would delete 114 | PASS |
| `mergeDuplicates` | dry_run=true, threshold 0.99 | 0 dupes / 0 deletions | PASS |
| `textToGraph` | tiny 2-step doc, output=text | structured graph text returned | PASS |
| `textToSmartGraph` | one-node doc | LLM compression pass worked (cluster + rules + code emitted) | PASS |
| `getMemoryCheatsheet` | {} | policy cheatsheet body | PASS |
| `getMemorySystemPrompt` | {} | full guidelines body | PASS |
| `listMemoryTypes` | {} | 8 types listed | PASS |
| `getMemoryMetrics` | {} | 155 memories, breakdown by type/agent, 149 retrievals | PASS |
| `memoryTypeChart` | {} | ASCII bar chart consistent with metrics | PASS |
| `exportGraphAsMarkdown` | {} | 155-memory export (output truncated client-side, server success) | PASS |
| `getMemory` | known id `02e99440…` | full record with updated usage count/last-accessed | PASS |
| `auditTrail` | same id, limit 5 | created/accessed/usage metadata correct | PASS |
| `getMemoryGraph` | id `0d76dcbb…`, depth 1 | mermaid graph with 2 bidirectional edges resolved | PASS |

Not exercised: `onAgentAction`, `redactSensitive`, `createMemoryEdge`,
`attachProvenance`, `verifyBackendParity`, `reindexMemory`, `migrateMemoryBackend`.

---

## Findings

### F1 — NEW DEFECT: ragmcp sparse retrieval path is dead on populated collections

All three sparse consumers return zero hits on collection `fastmcp-code`
(424 chunks, verified populated via `list_collections`):

- `search(mode=sparse)` — "ClientTransport connect_session": no results
- `search_code_sparse` — "httpx client": no results
- `get_copilot_context` — "ClientTransport connect_session httpx": no context

Meanwhile dense mode on the **same collection** returns those exact chunks —
the terms searched for are provably present in the corpus. Conclusion: sparse
(BM25) vectors are missing or never built for `fastmcp-code`; queries fall
through cleanly instead of erroring, which makes the failure silent.

Affected surface: `search` (sparse mode and any auto-mode sparse fallback),
`search_code_sparse`, `get_copilot_context`. Likely fix: rebuild/sync the
sparse index for existing collections (verify what `index_code` writes vs.
what the sparse query path reads). Note collection `sparse-fix-probe3`
(1 chunk, sparse-dim 8d) suggests a prior probe of this area.

### F2 — CONFIRMED STILL OPEN: webmcp Content-Encoding not honored

`fetch_url("https://example.com")` reproduces the documented regression probe
failure exactly (~318 raw gzipped bytes, no "Example Domain" text):
origin bodies are stored verbatim without honoring `Content-Encoding`. This is
the bug recorded on 2026-08-23 in the live-test skill — unchanged, so re-adding
`brave_search_web` to sweeps remains blocked. Fix belongs in
`tools/webmcp/webmcp_fastmcp.py` decode handling.

### F3 — Session health

Zero occurrences of `-32600 Session not found`. All 30+ calls reused sessions
without churn. `/admin/flush-sessions` was deliberately NOT called.

---

## Artifacts

- Sweep findings captured as memorymcp entry `79742c30-9f39-4d3e-9571-709bef765abb`
  (queryable next sweep for regression diffing).
- Procedure source of truth: `.agents/skills/mcp-live-tool-test/SKILL.md`.

---

## Resolution (2026-08-27, same day — ZCode session, verified live)

- **F1 FIXED — three layers** (`28deb77`): (1) the migrated collections had
  no FTS5 sidecar and empty `text_content` — new
  `python -m tools.shared.migrate_store repair-sparse` backfills text from
  payloads and rebuilds the index (applied: fastmcp-code 424/424,
  fastapi-code 3196/3196); (2) sparse/hybrid queries on sidecar-less
  collections now fail loudly with the repair command instead of silent
  "No results found"; (3) `escape_fts5_query` OR-joins terms instead of
  whole-string phrases — multi-word queries previously required the words
  to be ADJACENT. Verified live: all four sparse consumers hit on
  `fastmcp-code`; auto mode upgrades to hybrid.
- **F2 FIXED** (`eec887d`): root cause was the manual
  `Accept-Encoding: …, br` header — servers answered brotli and, without
  the brotli package installed, httpx returned raw compressed bytes as
  text. Manual header removed at all three call sites; httpx now advertises
  only decodable codecs. `fetch_url("https://example.com")` returns clean
  "Example Domain" markup. Full sweep now exits 0 with zero defects.
- **brave_search_web**: removed from all testing by user directive
  (not meant to be used; cannot be hidden server-side) — see `383bbd5`.
- Suite: 528 passed after fixes. Coverage-table ragmcp row corrected
  (3 FAIL, not 4); memorymcp port 8005 filled in.
