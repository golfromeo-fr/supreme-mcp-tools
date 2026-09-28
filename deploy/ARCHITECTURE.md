# Pod architecture — the podman environments explained

> Plain-language version: [PODS-EXPLAINED.md](PODS-EXPLAINED.md).

This document describes the pods you see in Podman Desktop, container by
container: what each is for, what it is connected to, and why. It covers the
two pods `startcluster` manages plus the classic host stack, so the whole
picture is in one place.

---

## The three runtimes (only one "work" runtime can run at a time)

| | What it is | How it starts | Ports |
|---|---|---|---|
| **Host stack** | The classic way: `startlauncher` runs all 6 tools as host processes, `startui` the management UI | `./startlauncher`, `startui` | canonical: 8000–8005 tools, 8200 central, 8400 UI |
| **`pod_mcp-work`** | The pod-based daily driver: ONE launcher container (all 6 tools) + its own state DB, host networking | `./startcluster work` | same canonical ports (that's the point) |
| **`pod_mcp-multihost`** | The multi-host TEST bench: TWO launcher clones + shared DB + MinIO + load balancer | `./startcluster test pg` | 18200/19200 centrals, 18080 LB, 18xxx/19xxx tools |

```
┌────────────────────────────┐
│        HOST STACK          │  startlauncher + startui
│   8000–8005 · 8200 · 8400  │  identity: ~/.config/supreme-mcp-tools JSON
└─────────────┬──────────────┘
              │  ⚠ same canonical ports — the script refuses
              │    to run both at once
┌─────────────┴──────────────┐
│       pod_mcp-work         │  ONE launcher container + its own db
│   8000–8005 · 8200 (8400)  │  identity: its own postgres
└────────────────────────────┘

┌────────────────────────────┐
│     pod_mcp-multihost      │  2 nodes + db + minio + lb
│   18xxx/19xxx · 18080      │  identity: its own postgres
└────────────────────────────┘
   no line up to the two above — shares nothing with them,
   which is exactly what makes it the safe testing ground
```

Key rules:
- `pod_mcp-work` and the host stack are **mutually exclusive** (same
  canonical ports). The script refuses to start one while the other holds
  the ports.
- `pod_mcp-multihost` is fully isolated from both (own ports, own volumes,
  own database) — it is the safe testing ground.
- The three environments share **no state** with each other. Each has its
  own identity store, its own masks/inventory, its own volumes.

---

## pod_mcp-work — the daily driver

```
        HOST
   users.json ────────┐  (imported ONCE at first boot: same accounts,
   tools_config ──────┤   same keys, same masks afterwards)
   turso dir ─────────┤  (mounted at the SAME path: real memories)
   .env ──────────────┘  (mounted as /app/.env: real data-plane settings)
        │  read-only mounts
        ▼
┌── pod_mcp-work ───────────────────────────────────────┐
│  work ─ ONE launcher process, ALL 6 tools             │
│         canonical ports 8000–8005, central 8200       │
│         (host networking → same ports as the host     │
│          stack, hence the mutual exclusion)           │
│  db ─── pgvector Postgres: THIS env's identity plane  │
│         (mcp_users_store, mcp_state_docs)             │
└───────────────────────────────────────────────────────┘
```

- `work` is exactly "running startlauncher", but in a container: one
  launcher process, all six tools, canonical ports.
- `db` exists so the identity/config plane (users, masks, inventory) is
  DB-backed per the M4 design. Your host `users.json` is imported once at
  first boot; afterwards the DB is authoritative for this environment.
- Start/stop: `./startcluster work` / `startcluster work stop`.
  UI optional: `startcluster work ui` (port 8400 — conflicts with a host
  `startui`).
- Wipe: `startcluster work clean` (typed confirmation; removes its data
  volume).

---

## pod_mcp-multihost — the multi-host test bench

First the functional view — who does what. (The same bench as wiring and
ports follows just below.)

```
LAYER 1 · ENTRY — the admin door (MCP clients skip it)
──────────────────────────────────────────────────────
  admins ──►   ┌─────────────┐
               │    nginx    │  loopback :18080 — the ONE door
               └──────┬──────┘  for MANAGEMENT traffic: UI · user
                      │         admin · mask pushes · health checks.
                      │         Spreads requests · pins a session
                      │         (ip_hash). The MCP tool ports are
                      │         NOT behind it
                      ▼
LAYER 2 · SERVING — the MCP workhorses
──────────────────────────────────────
        ┌─────────────┬─────────────┐           identical launcher
        │             │             │           clones — add a node,
        │             │             │           gain capacity,
  ┌───────────┐ ┌───────────┐ ┌───────────┐      no code change
  │  node 1   │ │  node 2   │ │  node X   │      MCP clients dial node
  │ launcher  │ │ launcher  │ │ launcher  │      tool ports DIRECTLY —
  │ + tools   │ │ + tools   │ │ + tools   │      a port per tool per node
  │ + central │ │ + central │ │ + central │      (18002/18003 ·
  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘      19002/19003 · …), like
        │             │             │            the host 8000-8005 model
        │             │             │            — nginx is NOT in this path
        └─────────────┼─────────────┘
                      │   every node reaches every store
                      ▼   directly — no go-between
LAYER 3 · BACK OFFICE — what the nodes lean on
──────────────────────────────────────────────
        ┌─────────────┬─────────────┐
        │             │             │
  ┌───────────┐ ┌───────────┐ ┌───────────┐
  │ postgres  │ │ minio S3  │ │ extend:   │
  │ STATE     │ │ DATA      │ │ qdrant ·  │
  │ users·keys│ │ large     │ │ turso ·   │
  │ masks·inv.│ │ artifacts │ │ postgres  │
  └───────────┘ └───────────┘ └───────────┘
```

Reading it:

- **Layer 1** serves MANAGEMENT traffic only (UI, user admin, mask
  fan-out, health): one address, no logic beyond spreading requests and
  pinning sessions. An MCP client never passes through it.
- **Layer 2** is where all MCP work happens — and where MCP clients enter:
  like the host stack (8000–8005), every tool keeps **a port per server
  per node** (container 8002/8003, published as 18002/18003 on node 1 and
  19002/19003 on node 2). There is no LB in the MCP path: a stateful
  `/mcp` session lives on the node whose port you dialed; a
  `/mcp-stateless` client can hop nodes between requests — floating is
  the client's choice, not nginx's. Ports are shared infrastructure, not
  per-user: every user configures the same `18002` for node-1 simplemcp
  (remote clients: `<host-ip>:18002`); who you are rides on your API key,
  verified against the shared identity plane — never on the port. The
  nodes are interchangeable clones — that interchangeability is exactly
  what M4 sets out to prove.
- **Layer 3** is everything a node would otherwise keep to itself, in two
  flavors: **STATE** (identity + config — the multi-host trick: a user
  created on node 1 is valid on node 2 immediately) and **DATA**
  (artifacts — the statelessness trick: nodes hold nothing, so any node
  can serve any request). The state box is the swappable topology
  (`pg` / `turso` / `external-pg` / `external-turso`); the extend box is
  where qdrant / turso / postgres land when the heavy tools
  (memory / rag / database) join the bench.

```
   ADMIN CLIENTS (UI · user admin · mask pushes)
     ├─ stateful central-API session ─ ip_hash pins it to ONE node
     └─ stateless calls ───────────── may land on either node
        │
        │  :18080 (LB — round-robin over the two node centrals)
        ▼
┌── pod_mcp-multihost ───────────────────────────────────────────┐
│                                                                │
│   lb (nginx) ──forwards──> node1          node2                │
│                             │            │                    │
│        each node = ONE launcher clone (3 light tools,          │
│        central API on 8200 — separate network namespaces,      │
│        hence per-node published ports 18002/3 and 19002/3)     │
│                             │            │                    │
│              read/write ─────┘            └──── read/write     │
│                             ▼            ▼                     │
│   db (postgres) ── shared identity plane: users, masks,        │
│                     inventory, node registry                   │
│   minio (S3) ──── shared artifact store: large-text memories   │
│                     upserted via node1 are readable via node2  │
└────────────────────────────────────────────────────────────────┘
```

| Container | Job | Talks to |
|---|---|---|
| `db` (postgres) | The **shared identity/config plane**: users + keys, masks, tools inventory, node registry, env/auth overrides | node1, node2 |
| `node1` | Launcher clone #1 (3 light tools + central API) | db (identity), minio (artifacts) |
| `node2` | Launcher clone #2 — **identical to node1** | db, minio |
| `lb` (nginx) | Round-robin load balancer over node1/node2 centrals, published on host :18080 (`X-Upstream` header shows which node answered). MANAGEMENT traffic only — the MCP tool ports (18002/18003 · 19002/19003) are node-direct doors, no LB in that path | node1, node2 |
| `minio` | Shared S3 artifact store — makes the nodes stateless (large-text memories live here, not on a node) | node1, node2 |

The feature being tested: **a user (or a mask change) created via node1 is
immediately valid on node2**, because both verify every request against the
same shared db. That is multi-host behavior, reproducible on one machine.

---

## Pod internals — inspecting the deployment

`podman-compose -p <project>` groups each project's containers into an
**infraless pod** (`podman pod ps` shows an empty INFRA ID for both
`pod_mcp-work` and `pod_mcp-multihost`). Three practical consequences when
inspecting a running cluster:

- **Pod membership is metadata, not a process.** `ps axf` shows every
  workload under its own `conmon` — nothing is named after the pod, and
  there is no infra container to look for. `podman ps --pod` / `podman pod
  ps` are the authoritative views.
- **The pod is not a shared network namespace.** It is a lifecycle/metadata
  grouping only: every container keeps its own netns and own IP on the
  project bridge (`mcp-multihost_default` / `mcp-work_default`); the
  service-name DNS the containers use comes from the bridge, not the pod.
  The check, if ever in doubt — five containers must print five distinct
  `net:[…]` inodes:

  ```bash
  for c in node1 node2 db minio lb; do
    n=mcp-multihost_${c}_1
    printf '%-8s %s\n' "$c" \
      "$(readlink "/proc/$(podman inspect -f '{{.State.Pid}}' $n)/ns/net")"
  done
  ```
- **`Exited (137)` after a stop is usually benign.** 137 = 128 + 9
  (SIGKILL), which means either the kernel OOM-killer or a
  stop/compose-teardown that outlived its grace period. `OOMKilled=false`
  settles it — the stop was deliberate, not memory pressure:

  ```bash
  podman inspect mcp-work_work_1 \
    --format 'ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}}'
  # observed on the 2026-09-15 work-env stop: ExitCode=137 OOMKilled=false
  ```

**Why infraless — what the infra container even is.** A classic pod makes
all its members share namespaces (network, hostname, IPC). A namespace
dies with the process that owns it and Podman has no daemon, so the pod
runs a tiny placeholder container — the **infra container** (the "pause"
image: a program that does nothing but sleep forever) — to hold the
shared network namespace open. Every member joins *its* netns: one
shared IP, one port space, one localhost. Kubernetes pods contain the
same thing (a hidden pause container); Podman just shows it to you as
the INFRA ID column.

These pods are infraless because compose semantics need **per-service
IPs**: nginx reaches `node1:8200` and `node2:8200` by container DNS
(`10.89.0.67` / `10.89.0.68`), and the two nodes bind the SAME internal
port 8200 — impossible inside one shared namespace, so a classic pod
would break the bench. The pod therefore groups lifecycle only; the
network comes from the project bridge. Flip side: nothing hosts a pause
container (zero pause images pulled on this machine) and no phantom
entry clutters `podman ps`.

---

## The shared state plane — where identity actually lives

Every environment keeps its identity/config state in ONE place:

| Environment | Identity/config state lives in |
|---|---|
| Host stack | `~/.config/supreme-mcp-tools/` JSON files (`users.json`, `tools_config.json`) |
| `pod_mcp-work` | its own Postgres (`work_pgdata` volume): `mcp_users_store`, `mcp_state_docs` |
| `pod_mcp-multihost` | its own Postgres (`pgdata` volume): same tables + `cluster_nodes`, `env_overrides`, `auth_overrides` |

This is why the environments are isolated: a user created in the work pod
does not exist on the host stack or in the test cluster, and vice versa.
It is also why the multi-host feature works *inside* a pod: both nodes of
`pod_mcp-multihost` read the same `db`.

State docs per environment:
- `mcp_users_store` — all users/keys/masks/grants (one JSON document)
- `tools_config` — function masks + tools inventory
- `cluster_nodes` — registered nodes (mask fan-out targets)
- `env_overrides` / `auth_overrides` — centrally-mutated env vars / tool
  auth keys, adopted by every node at boot
- `env_auth_snapshot` — point-in-time mirror for bootstrapping new nodes

---

## Ports at a glance

| Port | Environment | What |
|---|---|---|
| 8000–8005 | host stack / `pod_mcp-work` | the six MCP tools (canonical) |
| 8200 | host stack / `pod_mcp-work` | central management API |
| 8400 | host `startui` / `pod_mcp-work` ui profile | management UI |
| 18200 / 19200 | `pod_mcp-multihost` | node1 / node2 central APIs |
| 18002, 18003 | `pod_mcp-multihost` | node1 tools (simplemcp, convertermcp) |
| 19002, 19003 | `pod_mcp-multihost` | node2 tools |
| 18080 | `pod_mcp-multihost` | LB over both centrals |
| 127.0.0.1:19000 | `pod_mcp-multihost` | MinIO S3 API (artifacts) |
| 127.0.0.1:5433 | `pod_mcp-work` | work Postgres (host-side access for backup/debug) |

---

## Lifecycle cheat sheet

```bash
./startcluster work            # start/rebuild the daily-driver pod
./startcluster work stop       # stop it (volumes kept)
./startcluster work backup     # pg_dump its state plane
./startcluster work clean      # remove it (typed confirmation; wipes volume)

./startcluster test pg         # start/rebuild the two-node test bench
./startcluster test stop       # stop it (volumes kept)
./startcluster test clean      # remove it (volumes kept)

./startcluster                 # no args: prints this architecture help
```

Code changes reach the containers on the next `up` — the script removes
stale per-service images first, so images are always rebuilt from the
current tree.

## What is persistent vs disposable

| Data | Where | Survives container/pod removal? |
|---|---|---|
| work: users/masks/inventory | volume `work_pgdata` | yes |
| work: large-text artifacts | volume `work_artifacts` | yes |
| test: users/masks/inventory + registry | volume `pgdata` | yes |
| test: turso-topology state | volume `sqldata` | yes |
| test: artifacts (minio topology) | volume `miniodata` | yes |
| containers themselves | disposable | rebuilt from image + code |
| host stack identity | `~/.config/supreme-mcp-tools/*.json` | always there |

The only destructive action is deleting a **volume** (Podman Desktop →
Volumes, or `podman volume rm`).
