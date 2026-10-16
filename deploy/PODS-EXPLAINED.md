# The pods, explained simply

This is the plain-language guide. The technical reference lives in
[ARCHITECTURE.md](ARCHITECTURE.md). You don't need both — start here.

---

## The one-sentence version

**You have three "places" where your MCP tools can run. They can never run
at the same time, and they don't share anything.**

---

## The three places

### 1. The host stack — "the old reliable"

What you've always used. `startlauncher` starts all six tools directly on
your machine (ports 8000–8005), plus the central API (8200). `startui` adds
the management website (8400).

- **Identity:** a file on your machine (`users.json`)
- **Start it:** `./startlauncher`
- **Good for:** your normal daily work

### 2. `pod_mcp-work` — "the same thing, but in a box"

One container that does exactly what `startlauncher` does — all six tools,
same ports (8000–8005, central 8200) — plus a small database next to it
that holds the users and masks for this box. The database engine is YOUR
choice, made at boot: `./startcluster work pg` (Postgres-with-pgvector,
the default) or `./startcluster work turso` (libSQL/sqld). Switching is
non-destructive: the other engine's database goes dormant (kept, never
wiped) and comes back intact when you switch back. Memories created while
on turso live in the turso store and wait there while you are on pg.

- **Identity:** its own private database. Your host users were imported
  into it once, so your accounts and keys are the same.
- **Your memories (pg identity):** it reads the same real data as the host
  (your turso folder is mounted inside it).
- **Start it:** `./startcluster work` (boots with the last-used engine)
- **Good for:** working exactly like before, but containerized — with a
  choice of identity database the host stack doesn't offer.

⚠️ **This and the host stack cannot run at the same time** — same ports.
Starting one while the other is up will refuse with a clear message.

### 3. `pod_mcp-multihost` — "the laboratory"

Two small launcher copies + a database + a load balancer + a file store,
all in one group. Its whole purpose: prove that two launchers can share
one set of users. A user created on node 1 works immediately on node 2.
Nothing here touches your real data — break it freely.

- **Start it:** `./startcluster test pg`
- **Its web door:** `localhost:18080` (a balancer that alternates between
  the two nodes) — for management only: the admin website, users, masks
- **MCP tool doors:** straight to the workers, no doorman — node 1
  `18002`/`18003`, node 2 `19002`/`19003` (one port per tool; every user
  configures the same ports — your API key says who you are)
- **Good for:** testing the multi-host features without any risk

The lab has three jobs, stacked: **the door** (nginx lets *management* in)
→ **the workers** (the nodes do the serving) → **the memory** (the shared
notebook + filing cabinet remember everything). How a management request
travels through them (MCP tool requests skip the doorman and enter at the
workers directly):

```
localhost:18080
      │
      ▼
┌───────────┐
│   nginx   │  the doorman — sends each request to copy #1 or #2
└─────┬─────┘
      ├──────────────┐
      ▼              ▼
┌──────────┐   ┌──────────┐
│  node 1  │   │  node 2  │   two identical launcher copies
└────┬─────┘   └────┬─────┘
     └───────┬──────┘
             ▼
       ┌──────────┐
       │    db    │   the shared user notebook — this is WHY a
       └──────────┘   user from node 1 works immediately on node 2

node 1 and node 2 also both read/write MinIO (the filing cabinet)
```

---

## Config bundles — "photocopies of your real settings"

By default the lab starts empty (no users) and the box copies your settings
once, on first boot. A **config bundle** fixes both: `./startcluster harvest`
photocopies your real settings (`users.json`, masks, tool configs, `.env`)
into a folder, and `--config-bundle <folder>` moves that photocopy into the
box or the lab when they start — same users, same keys, same masks as your
daily `startlauncher`. The photocopy is the new original: **edit the bundle
folder, start the environment again, your change is live** (no rebuild).
One safety rule: in the lab, settings that point at files on your machine
(your real memory store) are left out on purpose, so lab experiments can
never touch real data.

---

## What you see in Podman Desktop

The 7 stopped containers from your screenshot, color-coded:

**Group `pod_mcp-multihost` — the laboratory (6 containers):**

| Container | In plain words |
|---|---|
| `..._db_1` (postgres) | The lab's user notebook: accounts, keys, masks |
| `..._node1_1` | Launcher copy #1 |
| `..._node2_1` | Launcher copy #2 |
| `..._lb_1` (nginx) | The doorman — spreads requests over copy #1 and #2 |
| `..._minio_1` (MinIO) | The lab's filing cabinet: big memory blobs |

**Group `pod_mcp-work` — the boxed launcher bench (2 containers):**

| Container | In plain words |
|---|---|
| `..._work_1` (mcp-node) | `startlauncher` in a box — all six tools |
| `..._db_1` (postgres) | This box's own user notebook |

They are grouped into "pods" so Podman Desktop (and the script) can treat
each environment as ONE thing: press play on the pod, everything in it
starts; press stop, everything stops.

---

## What is connected to what

```
THE LAB (pod_mcp-multihost)
   node1 ──┬── reads/writes ──> db          (users, masks, inventory)
   node2 ──┘
   lb  ──sends requests to──> node1, node2
   node1, node2 ──read/write──> minio      (big memory files)

THE BOX (pod_mcp-work)
   work ──reads/writes──> its own db    (users, masks)
   work ──reads──> your real memory data (mounted from the host)
   work ──imported once──> your host users file

THE LAB and THE BOX: no connection at all, on purpose.
```

---

## Everyday commands

| I want to... | Command |
|---|---|
| Work as usual (host) | `./startlauncher` |
| Work in the box instead | `./startcluster work` |
| Stop the box | `./startcluster work stop` |
| Check what's running | `./startcluster work status` or `./startcluster status` |
| Play with multi-host | `./startcluster test pg` |
| Photocopy your settings | `./startcluster harvest` |
| Start the box from a photocopy | `./startcluster work --config-bundle <bundle-folder>` |
| Start the lab with your real users | `./startcluster test pg --config-bundle <bundle-folder>` |
| Save the box's users/masks to a file | `./startcluster work backup` |
| Remove the lab (data kept) | `./startcluster test clean` |

The golden rule: **after changing the project's code, start the
environment again with its start-script** (it rebuilds). The ▶ button in
Podman Desktop only re-runs the old image.

---

## Glossary

- **pod** — a group of containers managed as one unit (one play/stop
  button for the whole group)
- **infraless pod** — the kind of pod used here: a pure bookkeeping group
  (no hidden helper container, no shared network — each container has its
  own address). Pod membership shows in `podman ps --pod`, not in `ps axf`
- **container** — one running box (e.g. one launcher, or one postgres)
- **volume** — permanent storage that survives restarts; the only thing
  that holds real data
- **identity plane** — wherever users, keys and masks are stored
- **state plane** — identity plane + masks + tools inventory
- **LB (load balancer)** — a doorman that spreads requests over several
  identical servers
- **MinIO** — a file store that speaks the S3 protocol (Amazon-S3-style)
