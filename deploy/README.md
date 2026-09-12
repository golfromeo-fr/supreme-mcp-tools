# deploy — podman environments (startcluster)

Two ISOLATED podman environments, managed by the (local) `startcluster`
script:

| Environment | Purpose | Ports | Data |
|---|---|---|---|
| `work`   | pod-based DAILY DRIVER (alternative to host startlauncher) | canonical 8000-8005, central 8200, UI 8400 (profile) | own pgvector state plane + YOUR real data planes (host .env, host turso dir, host users.json imported once) |
| `test`   | M4 multi-host test bench (two light nodes) | 18200 / 19200 (+18xxx/19xxx tools) | own volumes (pgdata / sqldata) |

Isolation rules:
- separate compose projects (`mcp-work` / `mcp-multihost`) and volumes;
- the work cluster and the host `startlauncher` are mutually exclusive
  (same canonical ports) — the script refuses to start one while the other
  holds the ports;
- automated tests target the canonical ports: stop the work cluster before
  running the suite, or the suite tests the work environment (it is
  self-cleaning).

nodes, optional management UI. This is the H4 practical step: it gives you a
real second node so the shared-state code paths (`MCP_USERS_BACKEND=db`,
`MCP_STATE_BACKEND=db`) run against a real network backend instead of a
local file.

## Files

| File | Purpose |
|---|---|
| `Containerfile` | One launcher process per container. Reads `/app/.env` (mounted). |
| `compose-common.yml` | `node1` + `node2` (+ optional `ui` profile) — always used. |
| `compose-pg.yml` | Topology: embedded postgres state plane. |
| `compose-turso.yml` | Topology: embedded Turso/libSQL (sqld) state plane. |
| `node.env.template` | The mounted `.env` — copy to `node.env` and fill in (secrets + the state-plane block matching your topology). |

## Topologies (first arg of `startlauncher-podman`)

| Topology | State plane | db container | Use when |
|---|---|---|---|
| `pg` (default) | embedded postgres | yes (`compose-pg.yml`) | self-contained demo/cluster |
| `turso` | embedded libsql-server (sqld) | yes (`compose-turso.yml`) | same engine family as the tools' Turso stores |
| `external-pg` | your Postgres (managed/HA) | no | DB operated elsewhere |
| `external-turso` | Turso cloud / TCP sqld | no | cloud libSQL |

Each embedded backend owns its volume (`pgdata` / `sqldata`) — switching
topologies switches datasets; nothing is lost.

## Quickstart

```bash
cd deploy
cp node.env.template node.env       # fill in secrets + keep ONE state-plane block
cd ..
./startlauncher-podman pg           # or: turso | external-pg | external-turso

# node centrals:
#   node 1 -> http://localhost:18200/health
#   node 2 -> http://localhost:19200/health
```

(Manual equivalent, without the script:
`podman-compose -f compose-common.yml -f compose-pg.yml up -d` — the script
adds code pull, stale-image removal, and health waiting.)

Tool endpoints on the host (stateless URLs float across nodes):
`localhost:18002` / `18003` (node 1) and `19002` / `19003` (node 2) for
simplemcp / convertermcp. The prototype launches `simplemcp convertermcp
webmcp` per node — light tools with no external data-plane dependencies.
`memorymcp`/`ragmcp`/`databasemcp` need their backends reachable (add
qdrant/turso services to the compose file, or point the mounted `.env` at
shared instances) — see "Extending" below.

## Prove the shared plane (the H1/H2 demo)

```bash
# 1) create a user via NODE 1's central API (admin break-glass key)
curl -X POST http://localhost:18200/api/users \
  -H "Authorization: Bearer $MCP_MANAGEMENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"username":"clusteruser","password":"cluster-pass-9","role":"user","servers":["simplemcp"]}'

# 2) list users via NODE 2 — the user is already there
curl -H "Authorization: Bearer $MCP_MANAGEMENT_API_KEY" \
  http://localhost:19200/api/users
```

Same demo at the MCP surface: fetch `clusteruser`'s key from node 1, call
`simplemcp` on **node 2** (`localhost:19002/mcp-stateless`) with it — any
node answers any request, because both verify against the same shared store.

## How state flows

| State | Where | Mechanism |
|---|---|---|
| Users / keys / grants | `mcp_users_store` (postgres) | `MCP_USERS_BACKEND=db`; reads fresh per verify |
| Function masks + inventory | `mcp_state_docs` (postgres) | `MCP_STATE_BACKEND=db`; per-boot reads + central writes |
| env/auth snapshot | `env_auth_snapshot` (postgres) | mirror + additive restore for bootstrap |
| `.env` | mounted `node.env` | identical on every node (deployment concern) |
| tool `config.json` | node-local | import-time bootstrap (H2b deferral) |
| MCP sessions | node-local | stateful `/mcp` needs sticky LB; `/mcp-stateless` floats |
| logs / metrics | node-local | aggregate via your stack (Prometheus scrapes 8300) |

## Extending

- **Data planes for the heavy tools**: add `qdrant` (image
  `qdrant/qdrant`) and a network Turso (libsql-server) or point the mounted
  `.env` at your existing instances; then set
  `TOOLS_LIST: "simplemcp convertermcp webmcp memorymcp ragmcp databasemcp"`.
- **LB**: any reverse proxy with health checks on the node centrals
  (`/health`); sticky sessions only if a client insists on stateful `/mcp`.
- **Management UI**: `--profile ui` adds a UI service talking to node 1.

## Current limits

- Runtime mask pushes (E1) apply to the node that received the API call;
  other nodes pick masks up at next boot.
- Inventory sync: run on one node only (union merge bounds the damage).
- Writes are last-writer-wins per document.
