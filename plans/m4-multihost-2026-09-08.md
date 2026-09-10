# M4 — Multi-host design (2026-09-08)

> STATUS: H1 + H2(masks/inventory) BUILT on `feature/m4-multihost`
> (2026-09-10). See "Build status" at the bottom. Builds on E3
> (identity/multi-user) and the 12-surface inventory
> (`plans/e3-m1-identity-spike-2026-09-07.md`).
> Goal: run **N launcher nodes** serving the same tools/data behind a load
> balancer, with a single consistent identity/config plane.

## What already multi-hosts (verified in earlier phases)

- **`/mcp-stateless`** — no session affinity; any node answers any request.
- **Network backends** — Turso/PG/Qdrant data planes are already remote.
- **Identity** — bearer-token model (E3): every request carries its key;
  verification is stateless (map lookup). No server-side session = no
  shared auth state needed for the token check itself.
- **Per-user visibility** — driven entirely by the token map entries.

## The blocker: node-local state (inventory)

Everything below lives on **local disk per node** today and would drift or
fork across hosts:

| State | Path/mechanism | Multi-host problem |
|---|---|---|
| User accounts/keys | `users.json` (E3 store) | users created on node A invisible on B; rotations diverge |
| Function masks + tools inventory | `tools_config.json` + boot **sync from the node's own (masked) tools/list** | per-node drift; cleanup history: this sync already self-destructed masks once (E3 fix 07bdb1a) |
| Env vars / auth keys per tool | central env/auth API → tool `config.json` | same drift class |
| Artifact payloads (>8KB memory) | local filesystem (ArtifactStore) | a memory upserted via node A is unreadable via B |
| Logs | local `logs/` + per-tool logs | per-node history; fine for ops, no cross-node view |
| Metrics | per-node 8300 (Prometheus text) | scrapers can aggregate; fine |
| `.env` | per node | must be identical (bootstrap secrets + seeds) |
| MCP sessions (stateful `/mcp`) | node-local session manager | sticky LB or stateless clients required (already documented) |

## Design options for the shared identity/config plane

**Option A — file sync between nodes (rsync/NFS/shared volume).**
Cheapest, but write races across nodes defeat the atomic-write design
(flock is per-filesystem), and NFS locking is notoriously unreliable.
Rejected.

**Option B — network-backed stores behind the EXISTING interfaces (recommended).**
E3/M2 deliberately made `users_store.py` the only door to user data; the
same interface-swap pattern extends to the other state:
- `users_store` gains a Turso (or PG) backend: same function set
  (`tokens_map_for_tool`, CRUD, authenticate), tables instead of JSON.
  A user created via any node's Users tab appears on all nodes within one
  request (the mtime cache becomes a short TTL or poll).
- `tools_config` (masks + inventory) and the env/auth key/value store move
  to the same network backend (keyed by `server`/`name`), replacing the
  JSON files. Runtime pushes (E1) then apply locally AND persist shared —
  masks become cluster-wide automatically.
- `tools` inventory per server: stored centrally from the PRIMARY node's
  (unmasked) discovery, never from per-node masked lists (E3 self-destruct
  lesson generalized: inventory must come from a source that cannot be
  narrowed by enforcement).
- `.env` per node shrinks to bootstrap secrets only: `MCP_AUTH_MODE`,
  backend DSNs (`TURSO_DATABASE_URL`/PG), `MCP_MANAGEMENT_API_KEY`,
  `MCP_UI_USERNAME/PASSWORD`, tool keys (or they move central too).
This is the point where the earlier **"Turso DB for auth"** idea returns —
it was premature at single-host scale, but N hosts sharing auth/config
state is exactly its use case. NOTE: the current "Turso" is a LOCAL file
(`file:/home/gr/turso_data/memorymcp.db`) — a real network Turso (turso://
cloud or a TCP-served libSQL) is required; a local file cannot be shared
across machines.

**Option C — auth/config micro-service (one authority node).**
A dedicated service owns all state; tools call it over HTTP per request.
Most "correct" (single writer, no drift by construction), heaviest: one
more service to run/HA. This is Option B's evolution if Turso is replaced
by a purpose-built API — keep as the fallback if B's table design strains.

## Recommended architecture (Option B, staged)

```
                    ┌───────── LB (TLS, health-checked) ─────────┐
                    ▼                       ▼                    ▼
              node 1 (launchmcp)      node 2 (launchmcp)   … node N
              code + .env only        same .env             same .env
                    │ shared network state │
        Turso/PG: users, tokens, tools_config (masks+inventory),
                  env/auth kv, (later: artifacts)
        Data plane unchanged: Turso/PG/Qdrant tool backends
```

- **Identity**: unchanged from E3 — `UserStoreVerifier` + gate, fed by the
  network-backed `tokens_map_for_tool`. Admin bootstrap seeds once
  (idempotent seeds converge; last-writer-wins per user record).
- **Management**: every node runs its central API; all are equivalent
  because they share the network stores. Pick one hostname for humans
  (or round-robin). No federation protocol needed for v1.
- **Session handling**: stateless clients float freely; stateful `/mcp`
  clients get sticky sessions at the LB (source-IP or cookie). Long-term,
  prefer `/mcp-stateless` everywhere.
- **Masks/env/auth writes**: any node → stored centrally → visible to all
  nodes on next read (TTL ≤ a few seconds, or per-write broadcast later).

## Migration path (from today's single node)

1. **H1** — `users_store` Turso backend (interface already clean; E3 seeds
   converge). One table: `users(username, password_hash, role, mcp_key,
   servers, masked_functions, enabled, created_at, key_rotated_at,
   updated_at)`. Records carry `updated_at` (stamped on every mutation
   since 2026-09-08) — it is H1's last-writer-wins comparison key. CAVEAT:
   JSON deletes are lossy for sync — H1 needs either tombstones or a
   full-table reconcile for removals.
2. **H2** — `tools_config` + env/auth key-value to network tables
   (largest slice: touches masks manager, env_manager, sync; the
   self-destruct lesson requires the inventory source to be unmasked
   discovery from ONE primary).
3. **H3** — ArtifactStore to shared storage (needed only if memorymcp
   artifacts must be readable cross-node; today single-node fine).
4. **H4** — second node runbook: identical `.env` (bootstrap secrets),
   LB config with health checks (`/health` exists per tool + central),
   sticky-session note for `/mcp`, log aggregation choice.

## Open questions (decide before H2)

- Network backend dialect: Turso (libSQL TCP) vs PG — both proven in-repo;
  PG brings schemas/transactions, Turso keeps the "one network service"
  ops profile. Default recommendation: same backend family as the
  deployment's data plane.
- Management-plane topology: shared-state equivalent-centrals (v1,
  recommended) vs single primary + workers (v2 if write contention shows).
- Log aggregation: ship `mcp.access` (+ user= attribution) to a shared
  store for cross-node per-user history (the natural completion of E3
  attribution).

## Single-instance scalability (user decision 2026-09-08: ONE instance now, M4 later)

**Decision recorded:** Option B (network-backed stores) is the chosen M4
direction; until then the deployment runs ONE startlauncher instance, and
multi-user must scale within it. Also recorded: the future cluster may be
**Podman instances** — data circulation must be clean (containers must hold
NO unique local state; everything mutable either in network backends or in
declared volumes).

### Does one instance limit multi-user? Mostly NO.

- **Identity scales free**: a user's key is one dict entry; verification is
  a map lookup + one filter per request. User count costs ~nothing.
- **The real per-user costs are data-plane, and a second instance does NOT
  fix them** (same backends behind both nodes): databasemcp shared
  connections (users sharing a preset serialize on its pool — E4 txs: one
  per connection), memorymcp's shared memory data (declared non-goal),
  ragmcp's shared indexes and upstream API quotas.
- **True single-instance ceilings** (team-scale, not reached at dozens of
  users): uvicorn concurrent connections per tool server; the GIL-bound
  local embedding model (ragmcp bge-m3) which is the biggest CPU consumer;
  ONE transaction per databasemcp connection entry (cross-user tx
  contention if users share connections — mitigation already exists:
  connect more NAMED connections, no code needed).

### Details to preserve NOW so the M4 swap stays trivial

1. **`users_store` function set is the migration seam** — never let tool
   code reach users.json directly. A future backend (Turso/PG) re-implements
   the same functions; callers don't change.
2. **Record hygiene for a future table**: stable field names, no
   process-local data, plaintext `mcp_key` (a DB column maps 1:1). When
   migrating, ADD `updated_at` per record then (not before — no speculative
   fields).
3. **Key format is DB-friendly** (`token_urlsafe(32)`), unique by
   construction, and globally unique checks already run at create/rotate.
4. **Latency note for the future DB backend**: per-request map reads become
   DB reads — plan a short TTL cache (2–5s) or keep a JSON mirror; do NOT
   query the DB per tool-call, only per verify.
5. **Log attribution (`user=` in mcp.access) is storage-agnostic** — it
   keeps working no matter where users live.
6. **Podman notes (for H4)**: rootless podman binds ports fine (publish
   8000-8200 range); keep logs and any transition-state files in DECLARED
   volumes; run backends and nodes on one podman network so nodes reach
   Turso/PG by name; the "identical .env" rule becomes "same image, one
   mounted .env".

### What would justify moving to a DB backend BEFORE M4

- Multiple simultaneous writers become routine (UI + scripts + nodes).
- Audit/compliance needs DB-level backup/restore of accounts.
- User count grows past what a JSON file review comfortably handles
  (hundreds).
Until then: JSON + atomic writes + hot-reload is the right amount of
machinery (single-instance multi-user is NOT limited by the JSON store).

## Honest limits

- Write contention: last-writer-wins per key; no distributed transactions.
- A node with stale network access serves stale identity/config until its
  TTL/poll catches up (bounded, logged).
- This remains a DESIGN — none of H1–H4 is built; E3's mono/multi flag and
  the users_store interface are the only multi-host-ready pieces that exist.


---

## Build status (feature/m4-multihost, 2026-09-10)

### H1 — DONE, with a design deviation (improvement)

Shipped on main as `MCP_USERS_BACKEND=db` (859537b): the whole users
document lives in ONE row (`mcp_users_store`) of the shared SqlStore
instead of the per-user table envisioned here. Deviation rationale:
- the module API is document-shaped (load_users/save_users), so the
  whole-doc row is the natural mapping and needs no ORM layer;
- the "JSON deletes are lossy / tombstones" caveat disappears by
  construction — every save is a full-document reconcile;
- reads are fresh per call (no cache), so rotation/disable propagate on
  the next request on ANY node;
- seeds converge across nodes (seed_from_env only ADDS missing users and
  syncs declarative env-pinned keys — verified idempotent).

### H2 — masks + inventory DONE (masks half)

`tools/shared/state_docs.py`: named JSON documents in the shared SqlStore
(table `mcp_state_docs`), selected by `MCP_STATE_BACKEND=db` (json default).
Routed through it: `launcher/tools_config.py` (load/save),
`tools/shared/function_masks.py` (per-boot mask reads),
`mcp_ui/components/tool_settings.py` (delegated to launcher.tools_config —
duplicate implementation removed).

**Self-destruct class KILLED as a side effect:** the live recovery during
this build proved `update_config_with_discovered_tools` replaced inventory
from masked `tools/list` (get_secret vanished from simplemcp's inventory on
a real re-discovery — E3 lesson re-confirmed in the wild). It now UNIONS
discovered tools with existing inventory, so enforcement can never narrow
the cluster inventory.

Verified: two independent processes sharing one Turso file — node A writes
masks+inventory, node B reads them via `load_tools_config` and
`function_masks.masked_tools`; json default untouched; suite 881, e35 45/0.

### H2b — env/auth key-value: DEFERRED

Per-tool `config.json` env/auth values stay node-local this slice: they are
the tools' import-time bootstrap (a node must be able to boot before any
central read). Multi-host v1 contract: nodes carry IDENTICAL tool
config.json files (deployment concern, same as .env). Revisit when a real
second node demands central mutation of env/auth.

### Honest limits (current)

- Runtime mask push (E1) applies to the node that received the API call;
  other nodes pick the mask up at next boot (or H2v2 fan-out).
- Writes are last-writer-wins per document.
- Only ONE node may run the inventory sync until discovery reads through a
  mask-immune surface (union merge already bounds the damage).
