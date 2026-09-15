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
that holds the users and masks for this box.

- **Identity:** its own private database. Your host users were imported
  into it once, so your accounts and keys are the same.
- **Your memories:** it reads the same real data as the host (your turso
  folder is mounted inside it).
- **Start it:** `./startcluster work`
- **Good for:** working exactly like before, but containerized.

⚠️ **This and the host stack cannot run at the same time** — same ports.
Starting one while the other is up will refuse with a clear message.

### 3. `pod_mcp-multihost` — "the laboratory"

Two small launcher copies + a database + a load balancer + a file store,
all in one group. Its whole purpose: prove that two launchers can share
one set of users. A user created on node 1 works immediately on node 2.
Nothing here touches your real data — break it freely.

- **Start it:** `./startcluster test pg`
- **Its web door:** `localhost:18080` (a balancer that alternates between
  the two nodes)
- **Good for:** testing the multi-host features without any risk

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

**Group `pod_mcp-work` — the boxed daily driver (2 containers):**

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
| Save the box's users/masks to a file | `./startcluster work backup` |
| Remove the lab (data kept) | `./startcluster test clean` |

The golden rule: **after changing the project's code, start the
environment again with its start-script** (it rebuilds). The ▶ button in
Podman Desktop only re-runs the old image.

---

## Glossary

- **pod** — a group of containers managed as one unit (one play/stop
  button for the whole group)
- **container** — one running box (e.g. one launcher, or one postgres)
- **volume** — permanent storage that survives restarts; the only thing
  that holds real data
- **identity plane** — wherever users, keys and masks are stored
- **state plane** — identity plane + masks + tools inventory
- **LB (load balancer)** — a doorman that spreads requests over several
  identical servers
- **MinIO** — a file store that speaks the S3 protocol (Amazon-S3-style)
