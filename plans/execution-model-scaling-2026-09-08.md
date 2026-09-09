# Execution model & scaling analysis (2026-09-08)

> STATUS: verified facts (fastmcp 4.0.0 installed source + this machine's
> runtime) and the scaling analysis for the CURRENT single-instance
> architecture. Companion: `plans/m4-multihost-2026-09-08.md` (the N-node
> future). This doc answers: "how do tool calls actually execute, and will
> the architecture scale without clustering?"

## Execution model (VERIFIED)

**No fork, no multiprocessing.** One Python process per tool server (5 via
the launcher + central 8200 + metrics 8300 + UI 8400). A `tools/call`
flows:

1. **uvicorn event loop** (one per tool server) accepts the request —
   async, cheap, thousands of concurrent connections.
2. **ASGI middleware chain**: ApiKeyFallback → auth (hmac + map lookup) →
   identity gate (dict lookup) → request-log. All cheap and off-loop-safe.
3. **fastmcp dispatches the tool with `run_in_thread=True` (default,
   verified in `fastmcp/tools/function_tool.py`)** — the tool body runs in
   an **anyio worker thread**, limiter = **40 concurrent tokens** (call 41
   queues).
4. **databasemcp's async tools push the blocking DB work one level deeper**
   via `asyncio.to_thread` → the asyncio default executor: on this 6-core
   box **10 threads** (`min(32, cpu_count+4)`). That is the true ceiling
   for concurrent DB-driver operations per tool process.
5. **databasemcp per-entry locks** serialize access to the SAME connection;
   different connections run in parallel. DB pool caps: PG max 5, Oracle
   max 10, libsql single connection (C-binding serializes).

A resource-intensive call therefore occupies: 1 anyio thread (≤40) + 1
executor thread (≤10, DB ops) + 1 pool session (≤5/≤10) + per-entry lock —
held until it finishes. It does NOT block other users' requests as long as
the work is I/O-bound (DB waits).

## Per-tool load profile

| Tool | Nature | Scaling behavior / ceiling |
|---|---|---|
| simplemcp | stateless math | trivial; scales forever |
| webmcp | I/O-bound HTTP to upstream APIs | threads sleep while waiting; ceiling = upstream API quotas (per deployment key) + cache |
| databasemcp | DB-bound, pool+locks | ceiling = min(10 executor threads, pool size, per-entry lock); heavy calls hold a pool session |
| memorymcp | local Turso + vector search | **WATCH: `query_dense` runs ON the event loop** (only the embedding is off-loaded) — with the brute-force scan (HNSW unavailable) a growing corpus blocks the WHOLE server's loop per query. 158 records today = fine; refactor item: wrap store calls in `to_thread` |
| ragmcp | local embedding model (bge-m3) | CPU-heavy sync work — occupies a worker thread for seconds; indexing is the heavy path; remote embedding provider (config) offloads it |
| convertermcp | not running | — |

## Scaling knobs (in order — no clustering needed)

1. **DB pool sizes** via env (`ORACLE_MAX_CONNECTIONS`, PG pool max) —
   raise to the DB server's real capacity.
2. **Executor sizes**: anyio limiter (40) and the asyncio default executor
   (10 here) are code-level defaults; raising them is a two-line change in
   the tool entry files (proposed env knob: `MCP_THREAD_POOL_SIZE`).
3. **More named connections** in databasemcp (runtime `connect_database`)
   — relieves per-entry lock and E4 one-tx-per-entry contention for
   multi-user workloads.
4. **ragmcp embeddings**: switch provider local → remote (config) to move
   the heaviest CPU work off the node.
5. **Per-tool vertical split** (before full clustering): run a SECOND
   launcher instance on another host/ports serving ONLY the heavy tool
   (e.g. databasemcp), point the heavy tool's clients at it. Tools are
   independent servers; this needs duplicated users/config (E3 .env seeds
   converge) but zero new architecture.

## What does NOT scale on one instance (honest ceiling)

- **CPU parallelism = 6 cores**: local embeddings and any CPU-heavy sync
  work saturate the box regardless of async. (Mitigations: remote
  embeddings, per-tool split.)
- **Shared fate**: one process serves all users of a tool — a C-extension
  segfault or OOM kills that tool for everyone. The launcher watchdog
  restarts it, but there is no per-user/per-request process isolation.
- **Result memory**: big result sets build in shared process memory
  (mitigated: `max_rows` caps, artifact offload for >8KB payloads).
- **DB server capacity itself** — the final ceiling for databasemcp-style
  load, node count irrelevant.

## Anticipating load — concrete preparations

1. **Monitor the queue, not just latency**: rising `tools/call` durations
   in `mcp.access` (per-user attribution included) and the metrics server
   (8300, Prometheus format) are the early signals that the 10-thread
   executor or a pool is saturating.
2. **Keep the off-loop discipline**: every blocking call in a tool must go
   through `to_thread` (databasemcp does; memorymcp partially — see watch
   item).
3. **Sizing playbook when load comes**: raise `ORACLE_MAX_CONNECTIONS` /
   PG pool → raise executor caps → split the heavy tool onto a second
   launcher → then (and only then) M4 clustering.
4. **Load probe** (when needed): a small script firing N concurrent
   `tools/call`s with per-request timing against a scratch tool would give
   the real saturation curve. Not built yet — offered as a next step.

## Watch items (refactors, small)

- memorymcp: off-load `query_dense`/`scroll`/store calls via `to_thread`
  (loop-blocking today by design shortcut).
- Propose `MCP_THREAD_POOL_SIZE` env knob (anyio limiter + executor
  sizing) instead of code defaults.
- Consider per-user fairness if one user's heavy calls starve others
  (today: queue-based, no reservation).
