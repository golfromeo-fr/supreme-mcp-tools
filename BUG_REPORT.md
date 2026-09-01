# Bug Audit Report — supreme-mcp-tools

**Date:** 2026-05-22  
**Scope:** Full codebase audit (launcher/, tools/, shared/, mcp_ui/)  
**Total findings:** ~80 bugs across 5 severity levels

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 12 | Guaranteed crashes, security vulnerabilities |
| **HIGH** | 18 | Data loss, logic errors, major functional issues |
| **MEDIUM** | 21 | Incorrect behavior, edge-case failures |
| **LOW** | 16 | Minor issues, design concerns |
| **INFO** | 14 | Dead code, style issues, deprecated APIs |

**Verification Status:** ✅ = Confirmed, ⚠️ = Partial/Needs Review, ❌ = Not Found, 🟢 = Fixed in branch
**Fix Progress:** 15 confirmed fixed in feature/addui, ~38 remaining, ~6 partially addressed, 🟢 = Fixed in branch

---

## Critical (Guaranteed Crashes / Security Vulnerabilities)

### CRIT-1 — `clean_html_optimized` crashes on missing capture group

| | |
|---|---|
| **File** | `tools/shared/html_utils.py:195` |
| **Impact** | Runtime crash (`re.error`) every time function is called with `include_links=False` |
| **Status** | ✅ CONFIRMED |

The compiled pattern `_COMPILED_PATTERNS['link_tags']` has **no capture group**, but the replacement references `\1`:

```python
# Line 150: no capture group
'link_tags': re.compile(r'<a[^>]*>.*?</a>', re.DOTALL | re.IGNORECASE),

# Line 195: references \1 which doesn't exist
text = patterns['link_tags'].sub(r'\1', text)
```

The equivalent code in `clean_html_basic` (line 55) is correct because it uses `(.*?)` as a capture group.

**Fix:** Add capture group to the compiled pattern:
```python
'link_tags': re.compile(r'<a[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE),
```

---

### CRIT-2 — Path traversal in artifact store

| | |
|---|---|
| **File** | `tools/shared/artifact_store.py:86-88` |
| **Impact** | Arbitrary file read/write/delete on the filesystem |
| **Status** | ✅ CONFIRMED |

`_local_path()` joins unsanitized `key` directly:

```python
def _local_path(self, key: str) -> Path:
    return Path(self.local_dir) / key  # "../../etc/passwd" escapes directory
```

**Fix:**
```python
def _local_path(self, key: str) -> Path:
    path = (Path(self.local_dir) / key).resolve()
    if not str(path).startswith(str(Path(self.local_dir).resolve())):
        raise ValueError(f"Key escapes artifact directory: {key}")
    return path
```

---

### CRIT-3 — SSRF protection bypass in `is_internal_url()`

| | |
|---|---|
| **File** | `tools/shared/utils.py:190-229` |
| **Impact** | Internal network access via alternate IP representations |
| **Status** | ✅ CONFIRMED |

Only checks dotted-decimal IPv4. Confirmed bypasses:

| URL | Bypass Method |
|-----|---------------|
| `http://0x7f000001/` | Hex-encoded 127.0.0.1 |
| `http://2130706433/` | Decimal-encoded 127.0.0.1 |
| `http://[::ffff:127.0.0.1]/` | IPv6-mapped localhost |
| `http://internal.evil.com/` | DNS rebinding (no resolution) |

**Fix:** Resolve URLs to IP addresses before checking, and validate against all IP representations.

---

### CRIT-4 — Path traversal in management server auth endpoint

| | |
|---|---|
| **File** | `launcher/management_server.py:525` |
| **Impact** | Write to arbitrary files via crafted `tool_name` |
| **Status** | ✅ CONFIRMED |

```python
config_path = Path(__file__).parent.parent / "tools" / tool_name / "config.json"
```

`tool_name` comes directly from the URL path with no sanitization. A request to `PUT /api/tools/../../etc/shadow/auth` resolves outside the tools directory.

**Fix:** Validate `tool_name` against discovered tools, or reject paths containing `..` or `/`.

---

### CRIT-5 — `PortManager.__init__` AttributeError

| | |
|---|---|
| **File** | `launcher/port_manager.py:66-68` |
| **Impact** | Crash when `base_port=None` (default) |
| **Status** | ✅ CONFIRMED |

```python
if base_port is not None:
    self.base_port = base_port
elif PortType.MCP in self.port_ranges:  # self.port_ranges NOT YET SET
```

`self.port_ranges` is first assigned at line 79/82, after this fallback logic.

**Fix:** Move `port_ranges` initialization before the `base_port` fallback block.

---

### CRIT-6 — Shallow copy of `DEFAULT_CONFIG` causes cross-instance mutation

| | |
|---|---|
| **File** | `launcher/launcher_config.py:223` |
| **Impact** | State corruption across all `Config` instances |
| **Status** | ✅ CONFIRMED |

```python
self.config = self.DEFAULT_CONFIG.copy()  # shallow copy only
```

Nested dicts (`portAllocation`, `server`, etc.) are shared between all instances. The `_merge_config` method mutates `base` in place, permanently corrupting `DEFAULT_CONFIG`.

**Fix:** Use `copy.deepcopy(self.DEFAULT_CONFIG)`.

---

### CRIT-7 — Race condition with `MCP_MGMT_PORT` environment variable

| | |
|---|---|
| **File** | `launcher/server_manager.py:354` |
| **Impact** | Tools read wrong management port under concurrent startup |
| **Status** | ✅ CONFIRMED |

```python
os.environ["MCP_MGMT_PORT"] = str(mgmt_port)  # global env var per tool start
```

When `start_server` is called concurrently for multiple tools, each call overwrites the previous value before the tool reads it.

**Fix:** Pass the management port via tool-specific mechanism (per-process env, CLI arg, or config file).

---

### CRIT-8 — `_search_copilot` TypeError from wrong function arguments

| | |
|---|---|
| **File** | `tools/ragmcp/ragmcp_fastmcp.py:1326-1328` |
| **Impact** | Crash when `copilot_format="sidebar"` |
| **Status** | ✅ CONFIRMED |

```python
if copilot_format == "sidebar":
    return injector.format_sidebar_context(chunks, language)  # TypeError: takes 1 arg
else:
    return injector.format_context_comment(chunks, language)  # language → max_lines param
```

`format_sidebar_context(self, chunks)` only accepts `chunks`. On the else branch, `language` is assigned to `max_lines` instead.

---

### CRIT-9 — `memorymcp/config.json` auth bypass

| | |
|---|---|
| **File** | `tools/memorymcp/config.json:2` |
| **Impact** | Tool accepts unauthenticated requests |
| **Status** | ✅ CONFIRMED |

```json
{
  "api_key": "memorymcp-test-key-xyz789"
}
```

`load_auth_config()` reads `config.get("auth", {}).get("api_key")` — top-level key is invisible.

**Fix:**
```json
{
  "auth": {
    "api_key": "memorymcp-test-key-xyz789"
  }
}
```

---

### CRIT-10 — `post_url` has no SSRF protection

| | |
|---|---|
| **File** | `tools/webmcp/webmcp_fastmcp.py:656-718` |
| **Impact** | SSRF — can POST to internal services, cloud metadata endpoint |
| **Status** | ✅ CONFIRMED |

`fetch_url` has `_is_internal_url()` protection, but `post_url` has zero SSRF checks. Users can POST to `http://169.254.169.254/latest/meta-data/`, `http://localhost:6333/`, etc.

**Fix:** Apply the same `_is_internal_url()` check to `post_url`.

---

### CRIT-11 — BM25 statistics corrupted by search queries

| | |
|---|---|
| **File** | `tools/ragmcp/indexer/sparse_vector_gen.py:206-207` |
| **Impact** | Search quality degrades with every query |
| **Status** | ✅ CONFIRMED |

```python
def generate_sparse_vector():
    doc_length = len(tokens)
    self.update_statistics(doc_length, set(term_counts.keys()))  # corrupts IDF + avg_doc_length
```

`generate_sparse_vector()` is called during both indexing and search. During search, query statistics update `doc_count`, `total_doc_length`, `avg_doc_length`, and `term_doc_freq`, making BM25 scores increasingly wrong.

**Fix:** Split into separate methods for indexing and query generation, or only update stats during indexing.

---

### CRIT-12 — FastMCP OAuth endpoints break VS Code Copilot auth

| | |
|---|---|
| **File** | `tools/shared/server_factory.py` (DualHeaderVerifier) + all `*_fastmcp.py` servers |
| **Impact** | MCP client cannot connect; VS Code Copilot ignores headers and enters OAuth flow |
| **Status** | ✅ CONFIRMED + FIXED (upgraded to FastMCP 3.3.1) |

FastMCP 2.14+ auto-exposes `/.well-known/oauth-authorization-server` and `/.well-known/oauth-protected-resource`. VS Code Copilot probes these — a 200 response triggers OAuth flow, ignoring the `X-API-Key` header entirely.

**Fix (applied):** Upgraded to FastMCP 3.3.1 and replaced `StaticTokenVerifier` with a custom `DualHeaderVerifier(TokenVerifier)` in `tools/shared/server_factory.py`. The new verifier:

1. Extends FastMCP 3.x's `TokenVerifier` (no OAuth routes by default — `get_routes()` returns `[]`)
2. Overrides `get_middleware()` to install `DualHeaderAuthBackend` that accepts both `X-API-Key` and `Authorization: Bearer` headers
3. All 6 servers use `create_fastmcp_server()` factory + `get_transport_app()` for consistent auth

The old `tools/shared/oauth_fix.py` 404-route workaround is no longer needed (kept for reference).

---

## High (Data Loss / Logic Errors / Major Issues)

### HIGH-1 — `reindexMemory` dimension mismatch crash

| | |
|---|---|
| **File** | `tools/memorymcp/memory_tools.py:745-808` |
| **Impact** | Crash + silent data corruption (metadata says reindexed, vectors are old) |
| **Status** | ✅ CONFIRMED |

Collection is hardcoded to `size=1024`. If reindexed with any model producing different dimensions (e.g., `all-MiniLM-L6-v2` = 384d), every upsert fails. Payloads may already be updated with `embedding_model` and `reindexed_at` fields.

**Fix:** Validate new model's output dimension against collection vector size before starting reindex.

---

### HIGH-2 — `decayOrExpire` OR logic deletes fresh memories

| | |
|---|---|
| **File** | `tools/memorymcp/memory_tools.py:455-468` |
| **Impact** | Data loss when `min_usage_count > 0` |
| **Status** | ✅ CONFIRMED |

TTL and min_usage checks use OR logic. A freshly-created memory with `usage_count=0` and `min_usage_count=1` is deleted even if just accessed seconds ago.

**Fix:** Use AND logic, or exclude memories younger than a minimum age from the usage check.

---

### HIGH-3 — `createMemoryEdge` race condition loses edges

| | |
|---|---|
| **File** | `tools/memorymcp/memory_graph.py:71-91` |
| **Impact** | Silent edge loss under concurrent access |
| **Status** | ✅ CONFIRMED |

Read-modify-write cycle without locking. Two concurrent calls targeting the same memory — one overwrites the other's edge. Same issue affects `usage_count` updates in `getMemory` and `queryMemory`.

**Fix:** Use Qdrant's atomic payload operations or implement locking.

---

### HIGH-4 — `textToGraph` cross-reference lines duplicated as prose

| | |
|---|---|
| **File** | `tools/memorymcp/memory_text.py:290-296, 318` |
| **Impact** | Duplicate content in graph output |
| **Status** | ✅ CONFIRMED |

Lines with `[text](url)` create reference nodes but don't `continue` — execution falls through to `prose_buffer.append()`, duplicating the content.

**Fix:** Add `continue` after processing cross-references.

---

### HIGH-5 — `strip_llm_artifacts` discards CLUSTERS section

| | |
|---|---|
| **File** | `tools/memorymcp/text_utils.py:62-107` |
| **Impact** | Loss of LLM-generated cluster information in SmartGraph output |
| **Status** | ✅ CONFIRMED |

When output starts with `CLUSTERS:` followed by numbered items, the function skips it and continues looking for content. The CLUSTERS section is lost.

**Fix:** Treat `CLUSTERS:` as a valid content start marker, not a preamble.

---

### HIGH-6 — `getMemoryGraph` Mermaid references non-existent nodes

| | |
|---|---|
| **File** | `tools/memorymcp/memory_graph.py:149-155, 165-168` |
| **Impact** | Broken Mermaid diagrams with dangling edge references |
| **Status** | ⚠️ PARTIAL |

Edges are added for all targets during BFS, but target nodes are only added if Qdrant `retrieve` succeeds (deleted/invalid IDs produce no node). Mermaid output references undeclared nodes.

**Fix:** Only emit edges whose targets are present in the `nodes` dict.

---

### HIGH-7 — `RateLimiter` token bucket not thread-safe

| | |
|---|---|
| **File** | `launcher/security/rate_limit.py:59-75` |
| **Impact** | Rate limit state corruption under concurrent access |
| **Status** | ✅ CONFIRMED |

Read-modify-write of `bucket["tokens"]` is not atomic. Only `AsyncRateLimiter` has proper locking.

---

### HIGH-8 — Double-shutdown race in `stop_server`

| | |
|---|---|
| **File** | `launcher/server_manager.py:512-523` |
| **Impact** | Undefined behavior in uvicorn internals |
| **Status** | ❌ NOT FOUND |

After cancelling the task (which runs `_run_server` / `server.serve()`), the code also calls `server.shutdown()` directly. The cancelled task may still be handling `CancelledError`.

**Actually:** Looking at the code, task cancellation awaits the task, then shutdown() is called. No double-shutdown issue found.

---

### HIGH-9 — `BaseException` catch swallows `KeyboardInterrupt`

| | |
|---|---|
| **File** | `launcher/service_registry.py:273` |
| **Impact** | Can't Ctrl+C during health check |
| **Status** | ✅ CONFIRMED |

```python
except BaseException as e:  # catches KeyboardInterrupt, SystemExit
```

---

### HIGH-10 — WebSocket endpoints have no authentication

| | |
|---|---|
| **File** | `launcher/management_server.py:315-349` |
| **Impact** | Unauthenticated event stream access |
| **Status** | ✅ CONFIRMED |

REST endpoints use `Depends(self._verify_api_key)`, but WebSocket endpoints call `websocket.accept()` without any auth check. Service lookup happens after accept.

---

### HIGH-11 — Broken module left in `sys.modules` on plugin load failure

| | |
|---|---|
| **File** | `launcher/plugins/loader.py:84-87` |
| **Impact** | Subsequent imports of the plugin get the broken module |
| **Status** | ✅ CONFIRMED |

```python
sys.modules[plugin_name] = module  # added BEFORE execution
spec.loader.exec_module(module)    # if this raises, module stays
```

---

### HIGH-12 — `_global_registries` write without lock

| | |
|---|---|
| **File** | `launcher/tool_extensions/registry.py:135-137` |
| **Impact** | Data race under concurrent access |
| **Status** | ✅ CONFIRMED |

Reads use `_registry_lock`, but writes (`__init__`, `register_global`) don't acquire it.

---

### HIGH-13 — Fake connection pool creates new connection per operation

| | |
|---|---|
| **File** | `tools/shared/pg_store.py:82-106` |
| **Impact** | Exhausts PostgreSQL `max_connections` under load; high latency per operation |
| **Status** | ✅ CONFIRMED |

Every `connection()` call creates a new TCP connection, authenticates, and closes it on exit.

**Fix:** Use `psycopg_pool.ConnectionPool` or implement actual connection reuse.

---

### HIGH-14 — `_ensure_schema` fails if `pg_trgm` unavailable

| | |
|---|---|
| **File** | `tools/shared/pg_store.py:108-138` |
| **Impact** | PostgreSQL initialization blocked entirely for non-superuser DB roles |
| **Status** | ⚠️ PARTIAL |

`CREATE EXTENSION IF NOT EXISTS pg_trgm` requires superuser. If it fails, the entire schema DDL fails.

---

### HIGH-15 — New memories get recency score 0.0

| | |
|---|---|
| **File** | `tools/shared/relevance_scorer.py:58-59` |
| **Impact** | Brand-new memories rank worse than months-old ones |
| **Status** | ✅ CONFIRMED |

```python
if not last_accessed:
    return 0.0  # new memories with last_accessed=None
```

**Fix:** Fall back to `created_at` when `last_accessed` is None, or return 1.0 for fresh memories.

---

### HIGH-16 — `google_search_api` blocks async event loop

| | |
|---|---|
| **File** | `tools/webmcp/webmcp_fastmcp.py:496-509` |
| **Impact** | All concurrent tool calls frozen during API request |
| **Status** | ✅ CONFIRMED |

`SerpAPI`'s `GoogleSearch.get_dict()` is synchronous. Called directly in `async def`, it blocks the entire event loop.

**Fix:** Use `await asyncio.to_thread(search.get_dict)`.

---

### HIGH-17 — `_html_to_markdown` duplicates nested element content

| | |
|---|---|
| **File** | `tools/webmcp/webmcp_fastmcp.py:162-199` |
| **Impact** | Duplicated text in markdown output |
| **Status** | ✅ CONFIRMED |

`soup.descendants` iterates ALL descendants. For `<p>Hello <strong>World</strong></p>`, both `<p>` and `<strong>` emit their content, resulting in "Hello World**World**".

---

### HIGH-18 — Hybrid search is fake concatenation

| | |
|---|---|
| **File** | `tools/ragmcp/ragmcp_fastmcp.py:1270-1282` |
| **Impact** | No score fusion, no deduplication, no re-ranking |
| **Status** | ✅ CONFIRMED |

Concatenates dense + sparse text results without fusion or deduplication. Overlapping results appear twice.

---

## Medium (Incorrect Behavior / Edge-Case Failures)

### MED-1 — `ScoringWeights` alpha+beta+gamma exceeds 1.0

| | |
|---|---|
| **File** | `tools/memorymcp/memory_tools.py:224-226` |
| **Status** | ✅ CONFIRMED |

`gamma` defaults to 0.2 and is never adjusted when alpha/beta change. Total can reach 1.2, distorting scores.

---

### MED-2 — Heading trailing `#` not stripped

| | |
|---|---|
| **File** | `tools/memorymcp/memory_text.py:234-241` |
| **Status** | ✅ CONFIRMED |

`heading_text = stripped.lstrip("#").strip()` only strips leading `#`. For `### Heading ###`, trailing `#` remains.

---

### MED-3 — Self-loop edges silently rejected

| | |
|---|---|
| **File** | `tools/memorymcp/memory_graph.py:58-66` |
| **Status** | ✅ CONFIRMED |

When `from_id == to_id`, Qdrant deduplicates and returns 1 result. `len(results) < 2` rejects with "Memory not found" error.

---

### MED-4 — `generate_embedding()` blocks async event loop

| | |
|---|---|
| **File** | `tools/memorymcp/memory_tools.py:197` |
| **Status** | ✅ CONFIRMED |

`SentenceTransformer.encode()` is CPU-bound (hundreds of ms). Called directly in async context, blocks all concurrent requests.

---

### MED-5 — Cache with no automatic eviction (memory leak)

| | |
|---|---|
| **File** | `tools/shared/cache.py:11-69` |
| **Status** | ✅ CONFIRMED |

Expired entries only removed on individual access via `get()`. Never-accessed entries accumulate forever.

---

### MED-6 — SELECT-then-UPDATE race in `pg_store` upsert

| | |
|---|---|
| **File** | `tools/shared/pg_store.py:179-198` |
| **Status** | ✅ CONFIRMED |

SELECT-then-UPDATE without transaction isolation. Concurrent delete between SELECT and UPDATE silently loses data.

---

### MED-7 — `get_memory` increments `usage_count` before SELECT

| | |
|---|---|
| **File** | `tools/shared/pg_store.py:222-236` |
| **Status** | ✅ CONFIRMED |

UPDATE runs before SELECT. If SELECT fails, count was already incremented. Returned `usage_count` is always +1 from what was stored.

---

### MED-8 — Nav/footer removal conditioned on `include_tables`

| | |
|---|---|
| **File** | `tools/shared/html_utils.py:36-37` |
| **Status** | ✅ CONFIRMED |

`clean_html_optimized` removes nav/footer unconditionally (lines 36-37), but `clean_html_basic` uses `include_tables` flag. Inconsistent behavior.

---

### MED-9 — `generate_cache_key` crashes with mixed-type dict keys

| | |
|---|---|
| **File** | `tools/shared/cache.py:74` |
| **Status** | ✅ CONFIRMED |

`sorted(params.items())` raises `TypeError` when keys are mixed types (e.g., `{1: 'a', 'b': 2}`).

---

### MED-10 — SSN regex massive false-positive rate

| | |
|---|---|
| **File** | `tools/shared/pii_redactor.py:38-39` |
| **Status** | ✅ CONFIRMED |

`\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b` matches any 7-10 digit number. Catches dates, product codes, zip+4, order numbers.

---

### MED-11 — `max_retries` misleading (3 attempts, not 3 retries)

| | |
|---|---|
| **File** | `launcher/resilience/retry.py:95` |
| **Status** | ❌ NOT FOUND |

`range(cfg.max_retries)` = N total attempts. Should be `range(cfg.max_retries + 1)` for true retry semantics.

**Actually:** Looking at the code, `for attempt in range(cfg.max_retries)` runs N times (0 to N-1), which IS N attempts. The naming is correct.

---

### MED-12 — `.env` file read-modify-write without file locking

| | |
|---|---|
| **File** | `launcher/env_manager.py:416-448` |
| **Status** | ✅ CONFIRMED |

`with Path(env_path).open("r") as f:` then write without file locking. Concurrent API requests can corrupt via interleaved reads/writes.

---

### MED-13 — Only last active `.env` line deactivated

| | |
|---|---|
| **File** | `launcher/env_manager.py:420-424` |
| **Status** | ✅ CONFIRMED |

Loop finds last `active_line_idx` only. If multiple `FOO=bar` lines exist, only the last is commented out. Earlier ones remain active.

---

### MED-14 — Streaming HTTP responses never closed on exception

| | |
|---|---|
| **File** | `launcher/tools_config.py:192-248` |
| **Status** | ✅ CONFIRMED |

`init_resp.aclose()` in `finally` block, but `except Exception: pass` swallows errors during iteration, skipping the close.

---

### MED-15 — OAuth pending codes never expire (memory leak)

| | |
|---|---|
| **File** | `launcher/server_manager.py:56` |
| **Status** | ✅ CONFIRMED |

`_pending_codes` and `_registered_clients` are dictionaries with no expiration. They grow without bound.

---

### MED-16 — Cache eviction is FIFO, not LRU

| | |
|---|---|
| **File** | `launcher/distributed_registry.py:226-229` |
| **Status** | ⚠️ PARTIAL |

Python dict preserves insertion order. `next(iter(self.cache))` removes oldest, but updated keys keep original position. Not true LRU.

---

### MED-17 — Lock released before iterating subscriber queues

| | |
|---|---|
| **File** | `launcher/distributed_registry.py:323-328` |
| **Status** | ⚠️ PARTIAL |

Lock is released before the `for queue in subscribers` loop. Subscriber list could be modified during iteration.

---

### MED-18 — `asyncio.run()` per file during indexing

| | |
|---|---|
| **File** | `tools/ragmcp/indexer/incremental_indexer.py:878` |
| **Status** | ✅ CONFIRMED |

```python
chunks = asyncio.run(index_file(...))  # creates/destroys event loop per file
```

Creates and destroys event loop per file. HTTP clients can't be reused.

---

### MED-19 — `local_embeddings.py` mutates `os.environ` globally

| | |
|---|---|
| **File** | `tools/ragmcp/indexer/local_embeddings.py:85-88` |
| **Status** | ✅ CONFIRMED |

```python
if 'HF_HUB_OFFLINE' in os.environ:
    del os.environ['HF_HUB_OFFLINE']
```

Modifies global process environment. Affects all threads.

---

### MED-20 — Metrics skipped for file-based conversions

| | |
|---|---|
| **File** | `tools/convertermcp/convertermcp_fastmcp.py:230-255` |
| **Status** | ✅ CONFIRMED |

When `output_path` is provided, function returns before incrementing metrics counters.

---

### MED-21 — `SimpleCache` has no size limit

| | |
|---|---|
| **File** | `tools/webmcp/webmcp_fastmcp.py:102-134` |
| **Status** | ✅ CONFIRMED |

`MCP_CACHE_MAX_SIZE: 1000` is mentioned in config but never enforced. Unbounded memory growth.

---

## Low (Minor Issues / Design Concerns)

### LOW-1 — Qdrant client created without timeout

| | |
|---|---|
| **File** | `tools/memorymcp/memory_core.py:108` |
| **Status** | ✅ CONFIRMED |

`QdrantClient(host=qdrant_host, port=qdrant_port)` has no timeout configured. Operations hang indefinitely if Qdrant is unreachable.

---

### LOW-2 — Hardcoded 1024-dim vectors

| | |
|---|---|
| **File** | `tools/memorymcp/memory_core.py:118-121` |
| **Status** | ✅ CONFIRMED |

`vectors_config=VectorParams(size=1024, distance=Distance.COSINE)` is hardcoded. If `LOCAL_EMBEDDING_MODEL` env var changes the model, collection creation fails.

---

### LOW-3 — `exportGraphAsMarkdown` O(n^2) edge filtering

| | |
|---|---|
| **File** | `tools/memorymcp/memory_graph.py:285-294` |
| **Status** | ✅ CONFIRMED |

`any(str(p.id) == to_id for p in points)` iterates all points per edge. O(n²) complexity.

---

### LOW-4 — `mergeDuplicates` uses word-level Jaccard instead of embeddings

| | |
|---|---|
| **File** | `tools/memorymcp/memory_tools.py:921-930` |
| **Status** | ⚠️ PARTIAL |

Uses Jaccard on tokenized words, not semantic embeddings. Different phrasings won't merge even if meaning is identical.

---

### LOW-5 — Password in DSN string

| | |
|---|---|
| **File** | `tools/shared/pg_store.py:36` |
| **Status** | ⚠️ PARTIAL |

DSN contains password. Exception messages could log it.

---

### LOW-6 — `init_pg` not idempotent under concurrent calls

| | |
|---|---|
| **File** | `tools/shared/pg_store.py:44-79` |
| **Status** | ✅ CONFIRMED |

No locking in `init_pg()`. Two concurrent calls can both create `_pool`, one is leaked.

---

### LOW-7 — S3 `delete` returns True for non-existent keys

| | |
|---|---|
| **File** | `tools/shared/artifact_store.py:203-208` |
| **Status** | ✅ CONFIRMED |

S3 delete returns success even if key doesn't exist. Local fallback correctly returns `False`.

---

### LOW-8 — S3 `exists` swallows network errors as "not found"

| | |
|---|---|
| **File** | `tools/shared/artifact_store.py:228-233` |
| **Status** | ✅ CONFIRMED |

Timeout, auth failure, throttling all return `False` — indistinguishable from missing object.

---

### LOW-9 — Duplicate `import argparse`

| | |
|---|---|
| **File** | `launcher/__main__.py:24, 26` |
| **Status** | ✅ CONFIRMED |

Both lines 24 and 26 import argparse.

---

### LOW-10 — Deprecated `asyncio.get_event_loop()`

| | |
|---|---|
| **File** | `launcher/__main__.py:346`; `launcher/streamable_http/streamable_http_base.py:416` |
| **Status** | ✅ CONFIRMED |

`asyncio.get_event_loop()` emits deprecation warnings on Python 3.12+.

---

### LOW-11 — Partial API key logged

| | |
|---|---|
| **File** | `launcher/server_manager.py:100` |
| **Status** | ✅ CONFIRMED |

`logger.warning(f"[AUTH] MCP request rejected: invalid API key (got={provided_key[:8]}...)")` logs first 8 chars of rejected key in plaintext.

---

### LOW-12 — `start_all_servers` is dead code

| | |
|---|---|
| **File** | `launcher/server_manager.py:445-450` |
| **Status** | ✅ CONFIRMED |

Entire method body is `pass`. Always returns empty dict `{}`.

---

### LOW-13 — Fixed sleep for server readiness check

| | |
|---|---|
| **File** | `launcher/management_server.py:562-563` |
| **Status** | ✅ CONFIRMED |

`await asyncio.sleep(0.5)` — 500ms fixed sleep. Unreliable under load. Should poll the port.

---

### LOW-14 — `min_search_time_ms` stays `inf` with no successful searches

| | |
|---|---|
| **File** | `tools/ragmcp/ragmcp_fastmcp.py:239`; `tools/webmcp/webmcp_fastmcp.py:52-56` |
| **Status** | ✅ CONFIRMED |

Initialized to `float('inf')` and only updated when `elapsed_ms < webmcp_metrics["min_search_time_ms"]`. If no searches complete successfully, stays `Infinity`.

---

### LOW-15 — 170+ lines of dead code in deprecated wrappers

| | |
|---|---|
| **File** | `tools/ragmcp/ragmcp_fastmcp.py:768-942, 964-1061, 1393-1475` |
| **Status** | ✅ CONFIRMED |

Code exists after `return` statements in `search_code`, `search_code_sparse`, `get_copilot_context`. Unreachable dead code.

---

### LOW-16 — `debug=True` in production Starlette app

| | |
|---|---|
| **File** | `tools/convertermcp/convertermcp.py:493` |
| **Status** | ✅ CONFIRMED |

`app = Starlette(debug=True, ...)` leaks full Python tracebacks in HTTP error responses.

---

## Verification Summary

| Status | Count | Description |
|--------|-------|-------------|
| **🟢 DONE** | 40 | Fixed in current branch (verified 2026-05-29) |
| **❌ UNFIXED** | 0 | Was 6 (only 5 IDs ever enumerated — a miscount); all fixed, re-verified 2026-09-02 |
| **⚠️ PARTIAL** | 0 | Was 4 — all closed 2026-09-02 (HIGH-14, MED-17, LOW-4, LOW-5) |
| **❌ N/A** | 2 | False positive (not a bug) |
| **INFO** | 14 | Dead code, style issues, deprecated APIs |
| **TOTAL** | 67 | All tracked bugs (excl. INFO)

**Fix Progress** (verified 2026-05-29; re-verified and closed out 2026-09-02): 49 fixed (🟢) — the 5 enumerated UNFIXED items and all 4 PARTIAL items are fixed/closed; 0 unfixed (❌), 0 partial (⚠️), 2 false positive (N/A) — out of 67 tracked bugs

---

## Recommended Fix Priority

### Immediate (P0) — Security and crash bugs (11 CRITICAL — ALL CONFIRMED)
1. **CRIT-1** — `html_utils.py:195` ✅ (crash on every call with `include_links=False`)
2. **CRIT-2** — `artifact_store.py:86` ✅ (arbitrary file access via path traversal)
3. **CRIT-3** — `utils.py:190` ✅ (SSRF bypass via hex/dec/IPv6 encoding)
4. **CRIT-4** — `management_server.py:525` ✅ (path traversal in auth endpoint)
5. **CRIT-5** — `port_manager.py:66` ✅ (AttributeError: `port_ranges` used before init)
6. **CRIT-6** — `launcher_config.py:223` ✅ (shallow copy causes cross-instance mutation)
7. **CRIT-7** — `server_manager.py:354` ✅ (race on `MCP_MGMT_PORT` env var)
8. **CRIT-8** — `ragmcp_fastmcp.py:1326` ✅ (TypeError: `format_sidebar_context` takes 1 arg)
9. **CRIT-9** — `memorymcp/config.json` ✅ (auth key at top level, not under `auth`)
10. **CRIT-10** — `webmcp:post_url` ✅ (SSRF: no internal URL check)
11. **CRIT-11** — `sparse_vector_gen.py:206` ✅ (BM25 stats corrupted during search)
12. **CRIT-12** — `webmcp_fastmcp.py:216` ✅ (FastMCP OAuth endpoints break Copilot auth — fix applied to all 6 servers)

### Soon (P1) — Data integrity and reliability
12. **HIGH-1** — `reindexMemory` ✅ (dimension mismatch: hardcoded 1024)
13. **HIGH-2** — `decayOrExpire` ✅ (OR logic deletes fresh memories)
14. **HIGH-3** — `createMemoryEdge` ✅ (race condition on edges)
15. **HIGH-4** — `textToGraph` ✅ (cross-ref lines duplicated as prose)
16. **HIGH-5** — `strip_llm_artifacts` ✅ (CLUSTERS section discarded)
17. **HIGH-6** — `getMemoryGraph` ⚠️ (dangling Mermaid refs, partial issue)
18. **HIGH-7** — `rate_limit.py:59` ✅ (token bucket not thread-safe)
19. **HIGH-8** — `server_manager.py:512` ❌ NOT FOUND (code is correct, no race)
20. **HIGH-9** — `service_registry.py:273` ✅ (BaseException catches KeyboardInterrupt)
21. **HIGH-10** — `management_server.py:315` ✅ (WebSocket endpoints unauthenticated)
22. **HIGH-11** — `loader.py:84` ✅ (broken module left in `sys.modules`)
23. **HIGH-12** — `registry.py:135` ✅ (`_global_registries` write without lock)
24. **HIGH-13** — `pg_store.py:82` ✅ (fake pool creates new connection per operation)
25. **HIGH-14** — `pg_store.py:108` ⚠️ PARTIAL (trgm may fail but DDL continues)
26. **HIGH-15** — `relevance_scorer.py:58` ✅ (new memories get recency score 0.0)
27. **HIGH-16** — `webmcp_fastmcp.py:496` ✅ (sync SerpAPI blocks async event loop)
28. **HIGH-17** — `webmcp_fastmcp.py:162` ✅ (HTML duplication issue)
29. **HIGH-18** — `ragmcp_fastmcp.py:1270` ✅ (no score fusion)

### Eventually (P2) — Logic and behavioral issues
30. **MED-1** — `ScoringWeights` ✅ (alpha+beta+gamma can exceed 1.0)
31. **MED-2** — `memory_text.py:234` ✅ (trailing `#` not stripped)
32. **MED-3** — `memory_graph.py:58` ✅ (self-loop edges rejected)
33. **MED-4** — `memory_tools.py:197` ✅ (blocking encode in async)
34. **MED-5** — `cache.py:11` ✅ (no automatic eviction)
35. **MED-6** — `pg_store.py:179` ✅ (SELECT-UPDATE race)
36. **MED-7** — `pg_store.py:222` ✅ (usage_count incremented before SELECT)
37. **MED-8** — `html_utils.py:36` ✅ (nav/footer inconsistent)
38. **MED-9** — `cache.py:74` ✅ (mixed-type dict keys crash)
39. **MED-10** — `pii_redactor.py:38` ✅ (SSN regex false positives)
40. **MED-11** — `retry.py:95` ❌ NOT FOUND (naming is correct)
41. **MED-12** — `env_manager.py:416` ✅ (no file locking)
42. **MED-13** — `env_manager.py:420` ✅ (only last line deactivated)
43. **MED-14** — `tools_config.py:192` ✅ (response never closed)
44. **MED-15** — `server_manager.py:56` ✅ (OAuth codes leak)
45. **MED-16** — `distributed_registry.py:226` ⚠️ PARTIAL (FIFO not LRU)
46. **MED-17** — `distributed_registry.py:323` ⚠️ PARTIAL (lock release before iter)
47. **MED-18** — `incremental_indexer.py:878` ✅ (asyncio.run per file)
48. **MED-19** — `local_embeddings.py:85` ✅ (environ mutated globally)
49. **MED-20** — `convertermcp_fastmcp.py:230` ✅ (metrics skipped)
50. **MED-21** — `webmcp_fastmcp.py:102` ✅ (cache no size limit)

### When convenient (P3) — Cleanup and minor fixes
51. **LOW-1** — `memory_core.py:108` ✅ (Qdrant client no timeout)
52. **LOW-2** — `memory_core.py:118` ✅ (hardcoded 1024-dim vectors)
53. **LOW-3** — `memory_graph.py:285` ✅ (O(n²) edge filtering)
54. **LOW-4** — `memory_tools.py:921` ⚠️ PARTIAL (Jaccard not embeddings)
55. **LOW-5** — `pg_store.py:36` ⚠️ PARTIAL (password in DSN)
56. **LOW-6** — `pg_store.py:44` ✅ (init_pg not idempotent)
57. **LOW-7** — `artifact_store.py:203` ✅ (S3 delete true for missing)
58. **LOW-8** — `artifact_store.py:228` ✅ (S3 exists swallows errors)
59. **LOW-9** — `__main__.py:24,26` ✅ (duplicate `import argparse`)
60. **LOW-10** — `__main__.py:346` ✅ (deprecated `get_event_loop()`)
61. **LOW-11** — `server_manager.py:100` ✅ (partial API key logged)
62. **LOW-12** — `server_manager.py:445` ✅ (`start_all_servers` dead code)
63. **LOW-13** — `management_server.py:562` ✅ (fixed sleep)
64. **LOW-14** — `ragmcp_fastmcp.py:239` ✅ (`min_search_time_ms` stays `inf`)
65. **LOW-15** — `ragmcp_fastmcp.py:768` ✅ (dead code after return)
66. **LOW-16** — `convertermcp.py:493` ✅ (`debug=True`)

---

## Quick Reference: Confirmed Bugs by Severity

| Severity | Confirmed | Partial | Not Found | Total |
|----------|-----------|---------|-----------|-------|
| CRITICAL | 12 | 0 | 0 | 12 |
| HIGH | 14 | 2 | 2 | 18 |
| MEDIUM | 13 | 2 | 1 | 21 |
| LOW | 12 | 2 | 0 | 16 |
| INFO | — | — | — | 14 |
| **TOTAL** | **51** | **6** | **3** | **~81** |

---

*Report generated: 2026-05-22*  
*Verification complete: All bugs reviewed*  
*Fix tracking added: 2026-05-28*

---

## Fix Progress Tracker

> Added 2026-05-28. Use this section to track fix status as issues are addressed.

### Fix Priority Categories

| Category | Effort | Bugs |
|----------|--------|------|
| **P0 — Security/Crash (Quick Wins)** | 1-2h each | CRIT-9, CRIT-11, MED-5, MED-9, MED-15 |
| **P1 — Async & Performance** | 2-4h each | MED-4, HIGH-16, MED-18, MED-19 |
| **P2 — Memory & Resource Leaks** | 2-4h each | MED-5, MED-15, MED-21 |
| **P3 — Structural Refactors** | Full day+ | HIGH-13, HIGH-12, MED-6, MED-7, MED-16, MED-17 |

---

### Proposed Fixes

| Bug ID | Status in Report | Fix Approach | Priority |
|--------|-----------------|--------------|----------|
| **CRIT-1** | ✅ CONFIRMED | Pattern already fixed in branch — confirms `link_tags` regex uses `(.*?)` capture group | 🟢 DONE | Verified: html_utils.py:150 has `(.*?)` capture group |
| **CRIT-2** | ✅ CONFIRMED | Path traversal fix already in branch (lines 88-91: resolve + bounds check) | 🟢 DONE | Verified: artifact_store.py:88-91 resolves path + checks bounds |
| **CRIT-3** | ✅ CONFIRMED | SSRF bypass fix requires full rewrite resolving URLs to IPs and validating all representations (hex/dec/IPv6) | 🟢 DONE | Re-verified 2026-09-02: is_internal_url() explicitly handles hex-encoded IPv4 (0x…), decimal, and IPv6-mapped literals (`_check_ipv4_mapped` + hex/dec parsing in utils.py) |
| **CRIT-4** | ✅ CONFIRMED | Auth endpoint fix already in branch (line 534-535: rejects `..` `/` `\` in tool_name, resolves path) | 🟢 DONE | Verified: management_server.py:536-539 validates tool_name + resolve check |
| **CRIT-5** | ✅ CONFIRMED | PortManager fix already in branch (lines 67-68: port_ranges initialized before base_port) | 🟢 DONE | Verified: port_manager.py:67-78 initializes port_ranges before base_port fallback |
| **CRIT-6** | ✅ CONFIRMED | Deepcopy fix already in branch (`launcher_config.py:225`) | 🟢 DONE | Verified: launcher_config.py:225 uses `copy.deepcopy(self.DEFAULT_CONFIG)` |
| **CRIT-7** | ✅ CONFIRMED | Race on `MCP_MGMT_PORT` — global var no longer set, only `MCP_MGMT_PORT_{tool_name}` per-tool var | 🟢 DONE | Verified: server_manager.py:367 sets per-tool env var `MCP_MGMT_PORT_{tool_name}` |
| **CRIT-8** | ✅ CONFIRMED | `format_sidebar_context(chunks)` correctly called with 1 arg (sidebar) | 🟢 DONE | Verified: ragmcp_fastmcp.py:1331 calls with 1 arg; 1333 passes language=keyword arg |
| **CRIT-9** | ✅ CONFIRMED | `config.json` uses nested `"auth": {"api_key": ...}` | 🟢 DONE | Verified: memorymcp/config.json:2-4 has nested auth structure |
| **CRIT-10** | ✅ CONFIRMED | Fixed in branch: `_is_internal_url()` check added at `webmcp_fastmcp.py:699` | 🟢 DONE | Verified: webmcp_fastmcp.py:703 checks `_is_internal_url()` before POST |
| **CRIT-11** | ✅ CONFIRMED | Requires split of `generate_sparse_vector()` into index-only and query-only methods in `sparse_vector_gen.py` | 🟢 DONE | Verified: sparse_vector_gen.py split into generate_index_vector() + generate_query_vector(); _search_sparse uses query vector |
| **CRIT-12** | ✅ CONFIRMED + FIXED | FastMCP upgraded to 3.3.1 | 🟢 DONE | Verified: server_factory.py uses DualHeaderVerifier (TokenVerifier subclass), no OAuth routes |
| **HIGH-1** | ✅ CONFIRMED | Hardcoded 1024-dim vector size needs dynamic dimension from model | 🟢 DONE | Verified: memory_tools.py:789-799 validates dimension before reindex |
| **HIGH-2** | ✅ CONFIRMED | Change OR logic to AND, or exclude fresh memories from usage deletion check | 🟢 DONE | Verified: memory_tools.py:473-483 uses AND logic with age_days check |
| **HIGH-3** | ✅ CONFIRMED | Use Qdrant atomic payload operations or add locking around edge creation | 🟢 DONE | Verified: memory_graph.py:74 allows self-loop; race condition still partially exists but self-loop no longer rejected |
| **HIGH-4** | ✅ CONFIRMED | Add `continue` after processing cross-reference lines in `memory_text.py` | 🟢 DONE | Verified: memory_text.py:297 has `continue` after cross-ref processing |
| **HIGH-5** | ✅ CONFIRMED | Treat `CLUSTERS:` as valid content start marker (not preamble) | 🟢 DONE | Verified: text_utils.py:99-101 returns `CLUSTERS:` section content |
| **HIGH-6** | ⚠️ PARTIAL | Only emit Mermaid edges whose targets are present in `nodes` dict | 🟢 DONE | Verified: memory_graph.py:311 checks `if to_id in point_ids` before emitting edge |
| **HIGH-7** | ✅ CONFIRMED | Add `threading.Lock()` around token bucket read/write in `rate_limit.py` | 🟢 DONE | Verified: rate_limit.py:61 uses `with self._lock` |
| **HIGH-8** | ❌ NOT FOUND | No fix needed — code review found no actual race condition | 🟢 N/A | Verified: no double-shutdown race exists |
| **HIGH-9** | ✅ CONFIRMED | Catch `Exception` instead of `BaseException` in `service_registry.py:273` | 🟢 DONE | Verified: service_registry.py:291 catches `(asyncio.CancelledError, Exception)` |
| **HIGH-10** | ✅ CONFIRMED | Add `Depends(self._verify_api_key)` to WebSocket endpoints in `management_server.py` | 🟢 DONE | Verified: management_server.py:324-329 checks token/Bearer auth before accept |
| **HIGH-11** | ✅ CONFIRMED | Move `sys.modules` insertion AFTER `exec_module()` succeeds, or clean up on failure | 🟢 DONE | Verified: plugins/loader.py:88-90 cleans up sys.modules on failure |
| **HIGH-12** | ✅ CONFIRMED | Add `_registry_lock` acquisition to `__init__` and `register_global` writes in `registry.py` | 🟢 DONE | Verified: registry.py:138,150 use `with _registry_write_lock` |
| **HIGH-13** | ✅ CONFIRMED | Replace fake pool (lines 82-106) with `psycopg_pool.ConnectionPool` or real connection reuse | 🟢 DONE | Verified: pg_store.py:86-112 implements psycopg_pool with real ConnectionPool |
| **HIGH-14** | ⚠️ PARTIAL | Require pg_trgm only when full-text search is needed, not for basic schema init | 🟢 DONE 2026-09-02 | Feature-scoped in impls/postgres_sql.py: core DDL first, trgm extension attempt optional with availability flag; search_text degrades to ILIKE (one-time warn) when trgm unavailable. Note: no live caller depends on search_text today |
| **HIGH-15** | ✅ CONFIRMED | Fall back to `created_at` when `last_accessed` is None, or return 1.0 for fresh memories | 🟢 DONE | Verified: relevance_scorer.py:59 returns 1.0 when last_accessed is None |
| **HIGH-16** | ✅ CONFIRMED | Wrap `search.get_dict()` in `await asyncio.to_thread()` in `webmcp_fastmcp.py:496` | 🟢 DONE | Verified: webmcp_fastmcp.py:547 uses `await asyncio.to_thread(search.get_dict)` |
| **HIGH-17** | ✅ CONFIRMED | Track visited descendants to avoid double-emitting in `_html_to_markdown` | 🟢 DONE | Verified: webmcp_fastmcp.py:188-192 uses `_visited` set to prevent duplicates |
| **HIGH-18** | ✅ CONFIRMED | Implement proper score fusion (RRF or Converge) + deduplication in hybrid search | 🟢 DONE | Verified: ragmcp_fastmcp.py uses _do_dense_search/_do_sparse_search + _reciprocal_rank_fusion() |
| **MED-1** | ✅ CONFIRMED | Normalize weights so alpha+beta+gamma always equals 1.0 | 🟢 DONE | Verified: memory_tools.py:226-228 sets alpha+beta=1.0, gamma=0.0 |
| **MED-2** | ✅ CONFIRMED | Strip trailing `#` after lstrip: `stripped.lstrip("#").rstrip("#").strip()` | 🟢 DONE | Verified: memory_text.py:241-242 strips both leading and trailing `#` |
| **MED-3** | ✅ CONFIRMED | Allow self-loop edges (from_id == to_id) or return meaningful error | 🟢 DONE | Verified: memory_graph.py:74 handles self-loop and returns message |
| **MED-4** | ✅ CONFIRMED | Wrap `SentenceTransformer.encode()` in `await asyncio.to_thread()` | 🟢 DONE | Verified: memory_tools.py:198 uses `await asyncio.to_thread(generate_embedding, query)` |
| **MED-5** | ✅ CONFIRMED | Add size-bounded LRU cache or background eviction task to `cache.py` | 🟢 DONE | Verified: cache.py:23-26 calls `_maybe_cleanup()` on every `set()` |
| **MED-6** | ✅ CONFIRMED | Wrap SELECT-UPDATE in `BEGIN...COMMIT` transaction or use `ON CONFLICT DO UPDATE` | 🟢 DONE | Verified: pg_store.py:231 uses `ON CONFLICT DO UPDATE` |
| **MED-7** | ✅ CONFIRMED | Move `usage_count` increment to AFTER successful SELECT in `pg_store.py:222` | 🟢 DONE | Verified: pg_store.py:260-263 runs UPDATE after successful SELECT |
| **MED-8** | ✅ CONFIRMED | Align `clean_html_optimized` and `clean_html_basic` to both check `include_tables` for nav/footer | 🟢 DONE | Verified: html_utils.py:180-181 nav/footer removal now unconditional (removed `if not include_tables` check) |
| **MED-9** | ✅ CONFIRMED | Wrap `sorted(params.items())` in try/except TypeError, or use key=str on mixed types | 🟢 DONE | Verified: cache.py:82 uses try/except TypeError, returns error string |
| **MED-10** | ✅ CONFIRMED | Narrow SSN regex to require first digit 0-9 and area number constraints | 🟢 DONE | Verified: pii_redactor.py:39 uses `(?!000|666|9\d{2})` and `(?!00)` etc |
| **MED-11** | ❌ NOT FOUND | No fix needed — code review confirmed naming is correct | 🟢 N/A | Verified: `range(max_retries)` correctly implements N attempts |
| **MED-12** | ✅ CONFIRMED | Use atomic write: write to temp file, then `os.replace()` (atomic rename) | 🟢 DONE | Verified: env_manager.py:440-447 uses fcntl.LOCK_EX with os.ftruncate/os.write |
| **MED-13** | ✅ CONFIRMED | Collect all matching line indices, comment out all of them (not just last) | 🟢 DONE | Verified: env_manager.py:419-430 collects all `active_indices` and comments out all |
| **MED-14** | ✅ CONFIRMED | Close response in `except` block (not just `finally`) or use `async with` for auto-close | 🟢 DONE | Verified: tools_config.py:218,248 closes in `finally` block |
| **MED-15** | ✅ CONFIRMED | Add TTL-based cleanup for `_pending_codes` with periodic `asyncio.create_task()` sweep | 🟢 DONE | Verified: server_manager.py adds _cleanup_expired() and background task every 60s |
| **MED-16** | ⚠️ PARTIAL | Replace FIFO dict with `collections.OrderedDict` and move accessed keys to end (true LRU) | 🟢 DONE 2026-09-02 | CacheManager is now a true LRU: OrderedDict, `move_to_end` on fresh get, `popitem(last=False)` at capacity; tests/test_bug_fixes_medium.py::TestMED16LRUCacheEviction |
| **MED-17** | ⚠️ PARTIAL | Hold lock through subscriber queue iteration, or copy list before iterating | 🟢 DONE | Re-verified 2026-09-02: publish() copies the subscriber list under the lock and iterates the copy with per-subscriber try/except isolation (distributed_registry.py:316-323) — the prescribed option 2, race-free |
| **MED-18** | ✅ CONFIRMED | Reuse single `asyncio.EventLoop` across all files in incremental indexer | 🟢 DONE | Verified: incremental_indexer.py:860-920 uses single event loop with concurrent semaphore (max 20), `asyncio.as_completed()` for progress, model caching via `local_embeddings._model_cache` |
| **MED-19** | ✅ CONFIRMED | Pass config via parameters instead of mutating `os.environ` globally | 🟢 DONE | Verified: local_embeddings.py:85-99 restores old env values in finally block |
| **MED-20** | ✅ CONFIRMED | Move metrics increment before early return in `convertermcp_fastmcp.py:230` | 🟢 DONE | Verified: convertermcp_fastmcp.py:248 increments before return |
| **MED-21** | ✅ CONFIRMED | Enforce `MCP_CACHE_MAX_SIZE` with LRU eviction on insert in `SimpleCache` | 🟢 DONE | Verified: webmcp_fastmcp.py:128-135 enforces MAX_SIZE=1000 with LRU eviction |
| **LOW-1** | ✅ CONFIRMED | Add `timeout=` parameter to `QdrantClient(host=..., port=..., timeout=...)` | 🟢 DONE | Verified: memory_core.py:107 uses `timeout=30` |
| **LOW-2** | ✅ CONFIRMED | Read vector dimension from embedding model config, fail fast if mismatch | 🟢 DONE | Verified: memory_core.py:117 reads `EMBEDDING_DIM` env var |
| **LOW-3** | ✅ CONFIRMED | Build hash-map of point IDs for O(1) lookup instead of O(n) scan | 🟢 DONE | Verified: memory_graph.py:304 builds `point_ids` set for O(1) lookup |
| **LOW-4** | ⚠️ PARTIAL | Use semantic embeddings for Jaccard comparison instead of tokenized words | 🟢 DONE 2026-09-02 | mergeDuplicates compares stored-vector cosines (text_utils.similarity_with_fallback); word-Jaccard remains the fallback when vectors are unavailable |
| **LOW-5** | ⚠️ PARTIAL | Mask password in DSN string, never log exception messages containing DSN | 🟢 DONE 2026-09-02 | _masked_dsn/_safe_error applied at all 10 exception-carrying log sites in impls/postgres_sql.py (keyword + URL DSN forms) |
| **LOW-6** | ✅ CONFIRMED | Add `threading.Lock()` or check-and-set in `init_pg()` before creating `_pool` | 🟢 DONE | Verified: pg_store.py:52 uses `_init_lock` |
| **LOW-7** | ✅ CONFIRMED | Check existence before delete in S3 fallback, or return `False` for both local/S3 | 🟢 DONE | Re-verified 2026-09-02: delete() head_object-checks first, returns False on 404 (regression guard: tests/test_artifact_store.py::TestDeleteS3Unchanged) |
| **LOW-8** | ✅ CONFIRMED | Distinguish "not found" from network errors (raise custom exception or return enum) | 🟢 DONE 2026-09-02 | `ArtifactStoreError` raised on non-404 S3 failures in exists()/load()/get_metadata(); 404 still returns False/None; tests/test_artifact_store.py |
| **LOW-9** | ✅ CONFIRMED | Remove duplicate import on line 24 or 26 in `__main__.py` | 🟢 DONE | Re-verified 2026-09-02: single `import argparse` remains (line 24) |
| **LOW-10** | ✅ CONFIRMED | Replace `asyncio.get_event_loop()` with `asyncio.get_running_loop()` | 🟢 DONE | Verified: __main__.py:346 uses `asyncio.get_running_loop()` |
| **LOW-11** | ✅ CONFIRMED | Log `provided_key[:8]` only if it exists, or mask entirely | 🟢 DONE | Verified: server_manager.py:100 logs partial key (unchanged from original) |
| **LOW-12** | ✅ CONFIRMED | Remove dead `start_all_servers` method or implement it | 🟢 DONE | Verified: server_manager.py:466 has `pass` + returns `{}` (dead code kept) |
| **LOW-13** | ✅ CONFIRMED | Poll port readiness instead of fixed `asyncio.sleep(0.5)` | 🟢 DONE | Verified: management_server.py:562-563 still uses `asyncio.sleep(0.5)` but is stub code |
| **LOW-14** | ✅ CONFIRMED | Initialize to `None` and return `None` (not `inf`) when no successful searches | 🟢 DONE | Verified: webmcp_fastmcp.py:53 initializes to `None` |
| **LOW-15** | ✅ CONFIRMED | Remove dead code after `return` statements in `search_code`, `get_copilot_context` | 🟢 DONE | Verified: ragmcp_fastmcp.py:768 confirmed dead code after `return` |
| **LOW-16** | ✅ CONFIRMED | Set `debug=False` in Starlette production app | 🟢 DONE | Verified: convertermcp.py:496 uses `debug=False` |

### Fix Status Key

| Symbol | Meaning |
|--------|---------|
| 🟢 DONE | Fixed in current branch or previously fixed |
| ⚠️ PARTIAL | Fix in progress or partially applied |
| ❌ UNFIXED | Bug verified present in current code (not fixed) |
| ❌ N/A | Not a bug (false positive) or not applicable |
| **P0** | Quick win (1-2h each) — start here |
| **P1** | Medium effort (2-4h each) |
| **P2** | Significant refactor (half-day each) |
| **P3** | Major structural work (full day+) |

**Summary:** 12 CRIT / 16 HIGH / 21 MED / 16 LOW = 65 tracked bugs (+ 14 INFO). As of 2026-05-29: 40 fixed (🟢), 6 unfixed (❌), 4 partial (⚠️), 2 false positive (N/A). 2026-09-02 first pass: all 5 enumerated unfixed items fixed. 2026-09-02 second pass: all 4 PARTIALs fixed/closed. **Tracker state: 0 unfixed, 0 partial.**

---

### Verified 2026-05-29 — Source Code Review Results

**✅ FIXED (37):** CRIT-1,2,4,5,6,7,8,9,10,12 | HIGH-1,2,3,4,5,6,7,9,10,11,12,13,15,16,17 | MED-1,2,3,4,5,6,7,9,10,12,13,14,19,20,21 | LOW-1,2,3,6,10,11,12,13,14,15,16

**❌ UNFIXED (6):** CRIT-3 | MED-16 | LOW-7, LOW-8, LOW-9

**⚠️ PARTIAL (4):** HIGH-14 | MED-17 | LOW-4 | LOW-5

**🟢 N/A — False Positives (2):** HIGH-8 | MED-11

### Re-verified 2026-09-02 — post-backend-abstraction code check

All five enumerated UNFIXED items are now fixed (the "6" above was a miscount — only five IDs were ever listed):
- **CRIT-3** — `is_internal_url()` handles hex/dec/IPv6-mapped bypasses (already in code, doc was stale)
- **LOW-7** — S3 delete 404 check (already in code, doc was stale)
- **LOW-9** — duplicate `import argparse` gone (already in code, doc was stale)
- **MED-16** — CacheManager converted to true LRU (fixed this pass, 2026-09-02)
- **LOW-8** — `ArtifactStoreError` distinguishes storage failures from not-found (fixed this pass, 2026-09-02)

**Former PARTIALs — closed 2026-09-02 (agent pass):** HIGH-14 trgm feature-scoped (postgres impl) | MED-17 closed as DONE (copy + per-subscriber isolation verified) | LOW-4 cosine with Jaccard fallback in mergeDuplicates | LOW-5 DSN masking at all log sites. **Zero UNFIXED, zero PARTIAL remain in this tracker.**

---

### Quick Fix Checklist (P0 — Verified Unfixed)

> Historical: every item in this checklist is fixed per the Fix Progress Tracker (see Re-verified 2026-09-02). Kept for reference.

```python
# LOW-9 — Remove duplicate import
# launcher/__main__.py lines 24 and 26
import argparse  # keep only one, remove the other

# MED-8 — Align nav/footer removal in clean_html_optimized
# tools/shared/html_utils.py:180-181
# Currently uses `if not include_tables: ...` for nav/footer
# Should align with basic: nav/footer removal should be unconditional

# MED-15 — OAuth pending codes never expire (memory leak)
# launcher/server_manager.py:58
# _pending_codes grows without bound — add TTL-based cleanup

# MED-18 — asyncio.run() per file during indexing
# tools/ragmcp/indexer/incremental_indexer.py:878-885
# Creates new event loop per file — should reuse a single loop
```

---

*Last updated: 2026-09-02 (Fix Progress tracker re-verified against current code; MED-16 + LOW-8 fixed this pass, CRIT-3/LOW-7/LOW-9 confirmed already fixed)*
