# Architecture — supreme-mcp-tools

*Last verified: 2026-10-17 against branch `feature/pod-ui-gaps` (tip `bf75e24`, includes the tools/\*/config.json untracking).*

This document describes the architecture that **actually ships today**. The previous version of this
file (dated 2026-02-20) was a design proposal for a layout that was never built — it named
`launcher/discovery.py`, `launcher/config.py`, `launcher/exceptions.py` and `tools/webmcp/webmcp.py`,
none of which exist. It has been replaced, not amended.

**Canonical guidance is [`AGENTS.md`](AGENTS.md)** — commands, conventions, pitfalls. This file is the
top-down mental model. Where the two disagree, `AGENTS.md` and `--help` win.

## How to re-verify

Every claim below cites a file. To re-check after a refactor:

```bash
python -m launcher --help              # real flag set of the primary CLI
python launchmcp.py --help             # legacy CLI (different flags)
python launchmcp.py --list-tools       # what discovery actually finds
cat config/ports.json                  # port corridors + assignments
grep -rn "@mcp.tool()" tools/<name>/   # the real per-tool function lists
```

Discovery is the ground truth for tool names and ports — a tool that is not in `--list-tools` is not
part of the default set. A linter/typechecker is not configured, so these reads are the only
available checks.

---

## 1. The process model: one process, many FastMCP instances

The unifying premise is memory. Running each MCP tool as its own OS process costs roughly 2× the
memory of the same tools hosted together (README quotes ~50–65% savings across 3/5/10 tools), because
each interpreter carries its own copy of the FastMCP/Starlette/uvicorn stack.

So: **one Python process hosts all tools.** Each tool is an independent `FastMCP` instance mounted
on its own port, each fronted by its own uvicorn server. The launcher's `server_manager` starts them
concurrently (`run_servers_concurrently`) inside the single process — you see one
`Started server process [pid]` in the log with N uvicorn instances behind it.

The isolation boundary is therefore the **port**, not the process. Tools cannot reach each other's
in-memory state; they communicate only over HTTP, exactly as an external MCP client would. The
memory win is real but the coupling it implies is genuine, and it shows up in session semantics
(§5) and shutdown ordering.

Consequence worth internalizing: because the process is shared, a crash or hang in one tool's event
loop is a launcher-wide event, and a launcher restart invalidates every client's session at once.

## 2. Tool discovery (`launcher/tool_discovery.py`)

Discovery is the contract that lets tools be added without touching the launcher. The rules, all
enforced in this one module:

- **Non-recursive glob** over each configured tool directory — only *top-level* `.py` files are
  considered. A tool's helper submodules are never scanned.
- **`<name>_fastmcp.py` is canonical.** The `_fastmcp` suffix is stripped to derive the tool name
  (`ragmcp_fastmcp.py` → `ragmcp`).
- **Auto-excluded**: `*_streamable.py`, `migrate_*`, `copilot_context_injector`, `test_*`,
  `__init__.py`.
- If both `<name>_fastmcp.py` and a bare `<name>.py` exist, `_fastmcp` wins.
- Container-style directories with no `*_fastmcp.py` are detected and descended one level, so
  `tools/foo/<sub>/foo_fastmcp.py` still resolves.

A tool's entry point must export an `app` produced by the shared server factory:

```python
from tools.shared.server_factory import create_fastmcp_server, get_transport_app

mcp = create_fastmcp_server("my_tool")   # @mcp.tool() functions register here
app = get_transport_app(mcp)            # multi-transport app
```

`tools/<name>/AGENTS.md` holds the full tool-directory convention. The contribution loop is
`launchmcp.py --list-tools` (discovery) then `python -m launcher --tools <name>`.

## 3. Transport multiplexing (`tools/shared/server_factory.py`)

`get_transport_app(mcp)` is the **single selection point** for transport. By default every tool
serves all three client dialects simultaneously from the one FastMCP instance, so the client
picks by URL and nothing needs configuring:

| Endpoint | Mode | Serves |
|---------|------|--------|
| `/mcp` | streamable HTTP, stateful | sessions; `/admin/flush-sessions`; idle TTL |
| `/mcp-stateless` | streamable HTTP, stateless | per-request, no session ids — load-balanced or handshake-less clients |
| `/sse` + `/messages` | legacy SSE | outdated harnesses (self-heal via EventSource reconnect) |

This replaced a thrash: SSE was removed 2026-08-14, re-added 2026-08-30, then generalized into the
multi-transport default the same day (spike-proven against fastmcp 4.0.0b3). The goal is that a
client upgrade never requires a server change.

Single-transport pinning remains an escape hatch, with precedence
**explicit argument > `config.json` `transport` key > `MCP_TRANSPORT` env > default `multi`**:

```bash
python -m launcher --transport sse
# or per-tool: tools/<name>/config.json  {"transport": "sse"}
```

The launcher only sets `MCP_TRANSPORT` when explicitly asked, so the default reaches the tools
unmodified. (Legacy dict-shaped `transport` blocks in some older configs are ignored.)

Auth is uniform across all three endpoints: dual-header `X-API-Key` **or** `Authorization: Bearer`,
with every request logged on the `mcp.access` logger.

## 4. Port corridors (`config/ports.json`)

Ports are partitioned into ranges by *role*, so a new tool never collides with a system service:

| Range | Role |
|-------|------|
| 8000–8099 | MCP tool endpoints |
| 8100–8199 | Per-tool management servers |
| 8200–8299 | System — central management API on 8200 |
| 8300–8399 | Metrics (8300 reserved; see caveat) |
| 8400–8499 | UI — management UI 8400, pod UI 8401 |

Fixed assignments: `databasemcp` 8000 (mgmt pinned 8110), `webmcp` 8001, `simplemcp` 8002,
`convertermcp` 8003, `ragmcp` 8004, `memorymcp` 8005.

`port_manager.py` detects conflicts by **socket binding, not by reading config** — a free config slot
is not evidence the port is free. The asymmetry between port kinds is deliberate:

- A pinned **MCP** port whose holder is stale **fails loudly**, because clients hardcode these ports
  and silently moving the tool would break them invisibly.
- A pinned **management** port whose stale holder outlives the 20s busy-retry window **degrades to
  corridor auto-allocation** with a loud WARNING. Consumers follow `service_registry.py`, so a tool
  never loses its run over a management port. This was the fix for the recurring
  "first `./startlauncher` starts without databasemcp, second run is fine" failure.

`databasemcp`'s mgmt port is pinned (8110) because 8100 collides with the auto-allocation floor.

## 5. Sessions

Sessions live **in-process**, in FastMCP's `StreamableHTTPSessionManager._server_instances`. There
is no session store: a launcher restart drops every client session, and stale `Mcp-Session-Id` values
must be re-initialized. That is the direct cost of the single-process design (§1), and it is why
`flush-sessions` exists as a first-class operation rather than a debugging afterthought.

Controls, wired in `get_transport_app`:

| Control | Effect | Default |
|---------|--------|---------|
| `POST /admin/flush-sessions` | Terminate all in-memory sessions; stale ids then get HTTP 404 and clients re-initialize. Auth = the tool's API key. | enabled |
| `MCP_SESSION_IDLE_TIMEOUT` | Idle-session TTL in seconds; `0`/unset disables. | 1800 |
| `MCP_DISABLE_FLUSH_ENDPOINT` | Suppress the flush route (TTL still applies). | unset |
| `MCP_DISABLE_REQUEST_LOGS` | Suppress the per-request `mcp.access` line. | unset |

The `mcp.access` line is the **only** uniform client-connect forensic trail, because the uvicorn
access log is off and session reuse is otherwise silent:

```
[simplemcp] POST /mcp from 127.0.0.1 v=2026-07-28 session=NEW -> 200 in 3ms tools/call echo
```

`v=` is the `MCP-Protocol-Version`; `2026-07-28` identifies the modern session-less single-exchange
path, which logs nothing on its own. Tool failures ride inside HTTP 200s, so without the
`rpc_err=<code>` / `tool_error` markers a failing tool looks healthy. Never grep for `initialize` —
that word only appears in backend-store startup lines.

## 6. The shared library layer (`tools/shared/`)

Cross-tool code lives here so tools stay self-contained but not redundant. The important members:

| Module | Role |
|--------|------|
| `server_factory.py` | `create_fastmcp_server`, `get_transport_app` — transport + session wiring, the single transport selection point |
| `sql_store.py` / `vector_store.py` | The two storage ABCs + `get_sql_store()` / `get_vector_store()` factories |
| `store_models.py` | Backend-neutral types (`PointStruct`, `Filter`, …) |
| `impls/` | Concrete backends: `postgres_sql`, `turso_sql`, `qdrant_vector`, `turso_vector`, `postgres_vector`, plus `libsql_reconnect.py` |
| `function_masks.py` | `apply_function_masks(mcp, TOOL_NAME)` — server-enforced masks |
| `identity.py` | Per-request identity resolution for multi-user mode |
| `pii_redactor.py` | `redact_sensitive_text`, used on the memory write path |
| `relevance_scorer.py` | Shared memory-result scoring |
| `hashing.py` | `text_hash` — pure function |
| `atomic_io.py` | One atomic write helper (flock on a `.lock` sidecar + tmp + `os.replace`); flock on the replaced file itself is inode-unsafe |
| `artifact_store.py` | Offload of >8KB memory texts with transparent rehydration |
| `migrate_store.py` | Backend migration CLI |
| `cache.py`, `html_utils.py`, `memory_models.py`, `cluster.py`, `migrate_mcp.py`, `utils.py` | Assorted shared utilities |

Two contracts are easy to get wrong and are enforced across every tool:

- **`ScoredPoint.score` is ALWAYS similarity — higher is better.** All impls normalize, and results
  sort descending.
- **`iter_all` MUST stream.** Backends yield one record at a time; full materialization is a bug, not
  a style choice.

## 7. Storage backend abstraction

Storage is two independent axes, so any SQL backend pairs with any vector backend:
`SqlStore` (`get_sql_store()`) × `VectorStore` (`get_vector_store()`).

| | SQL | Vector |
|---|---|---|
| Postgres | `PostgresSqlStore` | `PostgresVectorStore` (pgvector) |
| Turso/libSQL | `TursoSqlStore` (FTS5) | `TursoVectorStore` (libSQL vector + FTS5 sidecar) |
| Qdrant | — | `QdrantVectorStore` |

Seven supported combinations: PG+Qdrant (default), Turso+Turso (zero containers), PG+Turso,
Turso+Qdrant, Qdrant-only, PG+PG(pgvector), Turso+PG(pgvector). Selection precedence is
**`config.json` `storage` block > env vars > default** — Turso+Turso is the deployment that needs no
containers at all.

Migration is a first-class CLI, JSONL with `_meta`/`_sql`/`_vec` keys:

```bash
python -m tools.shared.migrate_store export --out backup.jsonl
python -m tools.shared.migrate_store import --in backup.jsonl --backend turso+turso
python -m tools.shared.migrate_store verify --left postgres+qdrant --right turso+turso
```

`--backup-before` and `--progress-every` are available; import filters `__collection_metadata__`.
`tools/memorymcp/BACKENDS.md` is the long-form reference.

## 8. The two UIs

Both are NiceGUI apps, both **separate processes** from the launcher.

**`mcp_ui` (port 8400)** — the management dashboard: tool status, Function Masks tab, memory
explorer, DB presets, and an admin-only Users tab. Components in `mcp_ui/components/`
(`tool_card.py`, `tool_list.py`, `tool_settings.py`, `users_tab.py`, `memory_tab.py`,
`presets_panel.py`, `rag_collections_panel.py`, `mutators_box.py`, `env_var_editor.py`, …). The
memory explorer talks to memorymcp **through its MCP surface only** (`mcp_ui/memory_client.py`,
fastmcp Client + tool-config key), so no backend credentials live in the UI and every action it
performs is something an MCP client could also do.

**`pod_ui` (port 8401)** — operations for the container cluster: manage the `work`/`test`
environments, config bundles, and backups. Pages in `pod_ui/pages_v3/` (`home.py`, `operations.py`,
`bundles.py`, `backups.py`, `logs.py`, `chrome.py`).

Single-instance rule: only one `mcp_ui` process at a time. Agents running scratch instances for
testing must stop them — verify with `pgrep -af "mcp_[u]i"` and confirm 8400/8401 return 000.

## 9. Auth and masking

**Auth mode** is `MCP_AUTH_MODE=mono|multi`, default `mono` — one system key per tool, which is
exactly pre-multi-user behavior. `multi` switches identity to the user store
(`~/.config/supreme-mcp-tools/users.json`, overridable via `MCP_USERS_STORE`; or
`MCP_USERS_BACKEND=db` for the shared SQL backend). Each tool's verifier map is
`{system key → admin username} ∪ {per-user mcp_keys}`, mtime-cached so store edits hot-apply with
no restart. `IdentityGateMiddleware` filters `tools/list` and rejects `tools/call` with "Unknown
tool" semantics; `mcp.access` lines gain `user=<client_id>`.

The central API on 8200 requires `MCP_MANAGEMENT_API_KEY`; 81xx action routes and
`/admin/flush-sessions` require the admin role. `mcp_ui` logs in against the store, with the
`MCP_UI_USERNAME`/`MCP_UI_PASSWORD` env pair bootstrapping the store admin once.

**Function masks** are server-enforced, not UI cosmetics. `disabled_tools` in
`~/.config/supreme-mcp-tools/tools_config.json` is applied by every `<name>_fastmcp.py` calling
`apply_function_masks(mcp, TOOL_NAME)` after registration. A masked tool is disabled on the FastMCP
instance itself, so clients cannot see it in `tools/list` and direct calls fail as "Unknown tool".
Since E1, mask changes reach the **running** server: the central endpoints persist file-first, then
push to the tool's instance via its registry (`mcp_instance`, wired by `server_manager`), with
`runtime_applied`/`runtime_note` in the response so the UI can report honestly. A missing or corrupt
mask file means "no masks" — never a startup failure.

Per-user data-plane scoping (E3.5): memorymcp stamps an `owner` on upsert and scopes queries,
ragmcp gates collections, databasemcp denies ungranted presets per call.

## 10. Deployment (`deploy/`)

A Podman/compose stack with two named environments, `work` and `test`, plus a config-bundle workflow.
`Containerfile`, `compose-common.yml`, `compose-{work,test}.bundle.yml`, `compose-work.run.yml`,
`compose-lb.yml`, `compose-pg.yml`, `compose-turso.yml`, `compose-artifacts.yml`, `nginx-lb.conf`,
`node.env`, `clusters.d/`, and the helpers `healthcheck.py`, `gen_custom_compose.py`,
`bundle_node_env.py`, `env_strip.py`, `harvest-config.sh`, `publish-image.sh`.

```bash
./startcluster work|test        # bring an environment up
./startcluster harvest [NAME]   # snapshot config → ./my-bundles/<NAME>
./startcluster work -b <path>   # seed from a bundle
```

Narrative docs live in `deploy/README.md`, `deploy/PODS-EXPLAINED.md`,
`deploy/ARCHITECTURE.md`, and `deploy/automation.md`.

## 11. Repository map

```
launcher/          orchestration: discovery, ports, server lifecycle, central API, FEF V3
launchmcp.py       legacy entry point (different, larger flag set)
tools/<name>/<name>_fastmcp.py   per-tool entry point; must export `app`
tools/shared/      cross-tool libraries + impls/ backends
mcp_ui/            management UI (8400)
pod_ui/            pod/cluster UI (8401)
config/            ports.json, launcher_config.json, monitoring_config.json
deploy/            compose stack, LB, cluster scripts
tests/             pytest suite; tests/fef_v3/ holds FEF V3 fixtures
plans/             design docs (historical — read for intent, not current state)
```

`config.json` files under `tools/<name>/` are **machine-written** by the management server API —
preserve their JSON structure when editing. The `auth.api_key` nesting is mandatory
(`load_auth_config()` reads `config.get("auth", {}).get("api_key")`); a bare top-level `api_key` is
invisible to the auth system. They are also **gitignored runtime state** (untracked 2026-10-17 —
they carry the real key): the committed template is `tools/<name>/config.example.json`, and the
launcher bootstraps `config.json` from it on first boot (`tool_discovery._bootstrap_config`).

## Known gaps

- **Metrics endpoints are not implemented.** `config/ports.json` reserves 8300 and
  `config/monitoring_config.json` ships with `"enabled": true`, and
  `launcher/monitoring/fef_metrics.py` provides a `FEFMetrics` collector — but there is no metrics
  **server**, and no `/metrics`, `/metrics/health`, or `/metrics/stats` routes exist anywhere in the
  tree. The README's monitoring section is stale on this point. Treat 8300 as reserved, not live.
- **Multi-user is implemented but unproven at scale.** The README's own status line says the system
  is "tested as single user server (not yet tested as multi-user)".
- `tools/oraclemcp` no longer exists — it was renamed to `tools/databasemcp` (2026-09-06). Any doc
  still naming `oraclemcp` as an available tool is wrong.
