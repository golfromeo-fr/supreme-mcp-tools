> **SUPERSEDED (2026-11-04):** design snapshot from 2026-06-04 — predates the
> fastmcp 4 migration (pinned `fastmcp==4.0.0`), the centralized transport/auth
> selection in `tools/shared/server_factory.py:get_transport_app`, and the
> multi-transport default. Kept for design rationale; current behavior =
> root `AGENTS.md`.

# Meta Design Choices — supreme-mcp-tools

**Date:** 2026-06-04
**Scope:** 14 deliberately-chosen architectural decisions, each with a rationale and a trade-off.
**Why this doc:** the previous `DISCOVERIES.md` catalogued ~67 small patterns. This doc captures only the BIG calls — the ones that would change direction if you rebuilt the system from scratch.

Each decision is stored as a memory of type `architectural_decision` with tags `meta, architecture` so they surface on `queryMemory` for "how should I do X" questions.

---

## The 14 Meta Decisions

### 1. **One process, N tools + management API + registry**
- **Decision:** `python -m launcher` runs every tool server, the management API, the service registry, and Prometheus metrics inside a single Python interpreter.
- **Rationale:** ~50% memory reduction vs N processes (no per-interpreter startup, shared imports). Simpler ops (one PID to monitor, one log stream).
- **Trade-off:** A bug in one tool can affect the others. Mitigated by per-tool try/except in `ServerManager` and port cleanup on partial failure.
- **Memory:** `b174eae7`

### 2. **streamable-http is the default, SSE is legacy**
- **Decision:** `MCP_TRANSPORT` env var set by launcher before tool imports; `get_transport_app()` abstracts both `mcp.http_app()` (≥3.3) and `mcp.streamable_http_app()` (legacy).
- **Rationale:** streamable-http is simpler, native streaming, less protocol overhead. SSE kept for back-compat.
- **Trade-off:** The dual-path code is the cost of supporting both. Auto-detect is fragile across FastMCP versions.
- **Memory:** `d6cbaddd`

### 3. **Auth is a defense-in-depth combo, not a single mechanism**
- **Decision:** Accept BOTH `X-API-Key` AND `Authorization: Bearer` in `DualHeaderAuthBackend`; return 404 for `/.well-known/oauth-*` to make VS Code Copilot skip OAuth.
- **Rationale:** Either alone is broken. Copilot probes `.well-known/oauth-authorization-server` and enters OAuth flow that ignores configured headers. Without 404-suppression, no header auth will work.
- **Trade-off:** Two coupled fixes that must be applied in every `_fastmcp.py`. Easy to forget one.
- **Memory:** `b3e7551f`

### 4. **Qdrant + PostgreSQL dual-store for memorymcp**
- **Decision:** Qdrant for vectors + payload; PG for metadata + `text_hash` dedup + `pg_trgm` fuzzy search.
- **Rationale:** Qdrant has no cross-document transactions → PG gives true ACID `ON CONFLICT (text_hash, memory_type) DO UPDATE`. PG also gives easy `pg_dump` backups.
- **Trade-off:** Two stores to keep consistent. **Crucially OPTIONAL** — if `POSTGRES_HOST` unset, Qdrant-only mode kicks in automatically. Never blocks the system.
- **Memory:** `1f31a040`

### 5. **Hybrid retrieval (dense + sparse + RRF) is first-class, not an afterthought**
- **Decision:** Three modes (sparse-only / dense-only / hybrid) auto-detected from collection config. RRF with k=60 merges ranked lists. Domain boost factors (table=3.0, function=2.5, sql_keyword=1.8) push code-specific signals.
- **Rationale:** Dense embeddings miss exact identifiers (STOMVT, getMovementType). BM25 misses semantic similarity. Neither wins alone. `sparse-only` mode = $0 cost.
- **Trade-off:** More moving parts (sparse_vector_gen, dimension validation, mode auto-detect). But code search is the killer use case.
- **Memory:** `e14d6a92`

### 6. **LLM behavior policies as repo-local .md, not in-code strings**
- **Decision:** `auto_use_policy.md` + `auto_use_cheatsheet.md` live in `tools/memorymcp/`. Served by `getMemoryAutousePolicy()` / `getMemoryCheatsheet()` MCP tools. Inline fallbacks in `memory_autouse.py` ensure flow never breaks.
- **Rationale:** Diffable in PRs, editable by non-developers, version-controlled with code, human-readable as docs.
- **Trade-off:** Drift between file and fallback. Mitigated by tests asserting both contain key terms.
- **Memory:** `e14d6a92` (linked)

### 7. **FEF V3 distributed registry replaces the V1/V2 singleton**
- **Decision:** Each tool process runs its own `ExtensionRegistry`. Launcher exposes a `ManagementServer` aggregating via HTTP. Cross-process calls go through `DistributedExtensionRegistry` (circuit breaker, cache, request coalescer, event aggregator).
- **Rationale:** Singletons don't work when N tool processes run in parallel. Distributed = HA-friendly, multi-process.
- **Trade-off:** HTTP+WebSocket everywhere. Eventual consistency on config changes. Significantly more code.
- **Memory:** `cc4bfd53`

### 8. **Observability is a first-class concern, not a bolt-on**
- **Decision:** Per-tool `ToolMetrics` dataclass, Prometheus exporter (`fef_metrics`), `AuditLogger` (JSONL + rotation), `EventStore` (SQLite + replay), 3-state health checks, `MCP_HEALTH_CHECK_LOGS` to silence polling noise.
- **Rationale:** Every failure must be inspectable after the fact, every state reconstructable, every load measurable. Operations survive production only if observability is the default.
- **Trade-off:** ~15% of codebase is observability. Worth it.
- **Memory:** `72704979`

### 9. **ports.json is the single source of truth for all infrastructure ports**
- **Decision:** Five ranges (mcp/mgmt/system/metrics/ui), reserved system ports, per-type manual assignments — all in ONE file. All reads go through `PortManager`. Config facade + focused frozen dataclasses (`LoggingConfig`, `PortConfig`, `ServerConfig`). Env var overrides via dot-path.
- **Rationale:** Change infrastructure without touching code. Subsystems import only the focused config they need (no god-object).
- **Trade-off:** One more file to keep in sync. But it's the price of "no code changes for infra changes".
- **Memory:** `6c195417`

### 10. **Tool registration via import-time side effects (no plugin system)**
- **Decision:** Each `_fastmcp.py` ends with `import memory_tools; import memory_graph; ...` which triggers `@mcp.tool()` decorators as side effects. Set `MCP_TRANSPORT` before the import. Discovery uses non-recursive glob, prefers `_fastmcp.py` over `_sse.py`/bare `.py`. Auto-detects container-vs-tool dirs.
- **Rationale:** Adding a tool = "drop a folder with `_fastmcp.py` in `tools/`". No config edit, no plugin registration, no manifest.
- **Trade-off:** Implicit side effects are harder to debug than explicit registration. But the zero-friction adding is worth it.
- **Memory:** `8203424a`

### 11. **Optional dependencies are a philosophy, not a convenience**
- **Decision:** EVERY external service (PG, Prometheus, sentence-transformers, OpenAI, BeautifulSoup, Redis, Azure OpenAI) wrapped in `try/except` at import time. Each module exports `FOO_AVAILABLE = False` when missing. Functions check `FOO_AVAILABLE` and return degraded responses.
- **Rationale:** System runs with ANY subset of dependencies. Dev/prod parity without env juggling. CI doesn't need full infra.
- **Trade-off:** More conditional code paths. But the alternative is "crash on import" which is much worse.
- **Memory:** `3ae3861a`

### 12. **State-machine parser over regex soup (textToGraph)**
- **Decision:** Maintain a stack of `(node_id, heading_level)` to track document hierarchy as lines are walked. On heading: pop until parent, push new node. On numbered list: create step, chain via `then` edge. Multi-format output (adjacency/text/dot/mermaid/json).
- **Rationale:** LLMs reason over structured graphs much better than flat prose. Graph extraction is a quality multiplier for any long doc going into context.
- **Trade-off:** Hand-written state machine is more code than a regex. But correctness is way better.
- **Memory:** `7051b1b3`

### 13. **Three orthogonal relevance signals, not one**
- **Decision:** `relevance = α * semantic_sim + β * recency_decay + γ * usage_boost`. Recency uses exponential half-life (default 14d), usage uses min-max normalization (max=100). All weights env-configurable.
- **Rationale:** No single signal wins. Semantic-only is blind to "I used this 50 times yesterday". Usage-only is blind to similarity. Recency-only is blind to relevance. Combination is what makes the system feel intelligent.
- **Critical pitfall:** Use `is not None` not truthy — `α=0` or `β=0` is valid (single-signal mode).
- **Memory:** `2f96ab7a`

### 14. **Fail closed on security, fail open on availability**
- **Decision:** SSRF check returns `True` (is_internal) on ANY exception. Health check catches all exceptions, marks service unhealthy but launcher stays up. Path-traversal protection rejects `..` and absolute paths. The asymmetry is deliberate.
- **Rationale:** If the bad outcome is data exfiltration / RCE → fail closed. If the bad outcome is user-facing downtime → fail open. Pick the one that hurts less.
- **Trade-off:** Sometimes blocks legitimate traffic (false positive on security). Worth it.
- **Memory:** `2172daa6`

---

## Cross-Decision Graph (10 edges)

```
unified_process ──── distributed_registry (trade-off)
streamable_http ──── auth_combo (depends_on)
dual_store ────────── optional_deps (depends_on)
hybrid_search ────── state_machine_parser (refines)
hybrid_search ────── auto_use_policy (refines)
auto_use_policy ──── state_machine_parser (depends_on)
observability ────── distributed_registry (refines)
ports.json ───────── unified_process (related_to)
ports.json ───────── import_side_effect_discovery (depends_on)
fail_mode_asymmetry ── dual_store (refines)
```

---

## What I did NOT memorize (deliberately)

- The specific code patterns (scoped regexes, BM25 K1/B constants, exact error code lists) — these live in `DISCOVERIES.md` as `code_pattern` memories. The meta decisions are orthogonal: if you change a constant, no decision changes; if you change a meta decision, dozens of patterns must move.
- The tooling/UI layer details (mcp_ui) — those are implementation, not architecture.
- The detailed FEF V3 docs (18 markdown files in plans/) — the decision to use FEF V3 is meta; the individual features are implementation.

---

## How to use this list

When designing a new feature, ask: "Which of these 14 does this touch?" If you can't answer, the feature probably needs to make its own meta decision — and that should be added here.
