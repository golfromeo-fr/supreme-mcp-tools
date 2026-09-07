# E3 — Multi-user overhaul: detailed implementation plan (2026-09-07)

> STATUS: spec-grade plan for coding (user-directed). Scope = the WHOLE project
> (user: "multi user implies every aspect of the project like mcp_ui secret keys
> to auth etc."). Companion docs: `plans/e3-m1-identity-spike-2026-09-07.md`
> (spike answers + the 12-surface auth inventory — READ FIRST),
> `plans/evolutions-plan-2026-09-06.md` §E3.
> Implementable by flash or GLM-5.3; suite gate per phase; one commit per phase.
> Tracker: TODO.md (E3 entry).

## Goal

One user store is the single source of identity for the entire deployment:
per-user MCP keys (all tool transports), per-user tool visibility, roles
(admin/user), mcp_ui per-user login, role-gated admin surfaces (81xx actions,
flush, central writes) — with the existing per-tool system keys kept working
(your current clients never break), and everything hot-reloading without
restarts (E1 lesson).

## Non-goals (v1)

- Per-user DATA isolation in memorymcp (tag-by-owner) — separate, harder.
- Per-user DB credentials / preset bypass runs as the server (inventory #10).
- OAuth/LDAP/SSO — static user store only (DualHeaderVerifier stays local).
- Multi-host replication of the user store (M4 designs it).

## VERIFIED FACTS (probes, 2026-09-07 — fastmcp 4.0.0, real stack)

- **F1** `get_http_request()` (from `fastmcp.server.dependencies`) inside
  `on_call_tool`/`on_list_tools` returns the caller's Starlette request;
  raw `Authorization: Bearer <token>` visible per call, on `/mcp`,
  `/mcp-stateless`, single-child, and in-memory paths.
- **F2** The VERIFIED AccessToken is NOT exposed (`request.state` empty, no
  `rc.access_token`); identity = token → `client_id` lookup in the same map
  auth verified (exact, post-auth).
- **F3** Per-user visibility demonstrated end-to-end: `on_list_tools` filter +
  `on_call_tool` `ToolError("Unknown tool")` gate (alice ok / bob empty+rejected).
- **F4** Middleware fires on every transport path (the one "not firing" report
  was a probe-reporting bug).
- **F5** Hook surface: `on_request/on_message/on_list_tools/on_call_tool/
  on_list_resources/on_list_prompts/on_initialize/on_discover…`; register via
  constructor `middleware=[...]` or `add_middleware()` before `http_app()`.
- **F6** `DualHeaderVerifier(tokens: dict[str, dict])` already multi-token;
  today each tool passes ONE key with `client_id` = tool name.
- **F7** `get_http_request()` ALSO works inside tool functions (stateless
  probe: tool returned the caller's Bearer header) → E4 `tx_id` binding is
  direct, no middleware relay needed.
- **F8 (inventory)** Central 8200 is UNAUTHENTICATED (`ManagementServer`
  built without `api_key` → `_verify_api_key` returns True; no-header probes
  200). Metrics 8300 open. mcp_ui = shared env login + `MCP_UI_SECRET`
  sessions; 81xx X-API-Key = the tool's MCP key (`load_auth_config`).

## General architecture

### Single source of truth

`~/.config/supreme-mcp-tools/users.json` (path override: `MCP_USERS_STORE`).
Everything derives from it at REQUEST time (mtime-cached read, atomic file —
`tools/shared/atomic_io`), so a store edit reaches every running process
within one request, no restarts:

- **Tool servers** build their token map from it: each tool's
  `DualHeaderVerifier` map = `{<tool system key> → client_id "<toolname>",
  role admin}` ∪ `{<user mcp_key> → client_id "<username>", role <role>}` for
  every ENABLED user whose `servers` includes that tool. The tool system
  key keeps working forever (backward compat for existing harness bindings).
- **Visibility** = `IdentityGateMiddleware` filtering `tools/list` /
  gating `tools/call` by the caller's `servers` + `masked_functions`.
- **mcp_ui** logs in against it; the UI then makes tool calls AS the acting
  user (their `mcp_key`) → per-user visibility for free, enforced server-side.
- **Roles**: `admin` (manage users/masks/env/auth, run mgmt actions, flush) vs
  `user` (use allowed tools, browse UI). Role travels IN the token's map entry
  — no separate lookup after auth.

### Request flow (one picture)

```
client (user key) ──▶ ApiKeyFallbackMiddleware (X-API-Key → Bearer)
                  ──▶ fastmcp BearerAuth → UserStoreVerifier.verify_token()
                        (mtime-cached users.json; system keys ∪ user keys)
                  ──▶ IdentityGateMiddleware (resolve client_id+role from the
                        same map; on_list_tools filter; on_call_tool gate;
                        "Unknown tool" semantics — no existence leak)
                  ──▶ tool function (get_http_request() → same identity;
                        E4: begin_transaction stamps tx owner)
```

### Store schema (exact)

```json
{
  "version": 1,
  "users": {
    "golfromeo": {
      "username": "golfromeo",
      "password_hash": "pbkdf2_sha256$390000$<salt_b64>$<hash_b64>",
      "role": "admin",
      "mcp_key": "<secrets.token_urlsafe(32)>",
      "servers": ["simplemcp", "databasemcp"],
      "masked_functions": {"databasemcp": ["execute_sql", "deleteMemory"]},
      "enabled": true,
      "created_at": "2026-09-07T12:00:00+00:00",
      "key_rotated_at": "2026-09-07T12:00:00+00:00"
    }
  }
}
```

- Password hashing: stdlib only — `hashlib.pbkdf2_hmac("sha256", password
  .encode(), salt, 390_000)`, salt = `secrets.token_bytes(16)`; serialized
  Django-style `pbkdf2_sha256$iters$saltb64$hashb64`; verify with
  `hmac.compare_digest` on the derived bytes. NO new dependencies.
- `servers` (round 3): the MCP servers this user may reach AT ALL —
  absent server = no access (your "no key for that server" case: the user's
  key simply fails auth there, because tokens_map_for_tool omits them).
- `masked_functions` (round 3): PER-USER function masks, deny-list per
  server — same UX/semantics as E1's global masks, scoped to one user.
  Effective visibility = server ∈ user.servers
                       AND function ∉ global disabled_tools[server] (E1)
                       AND function ∉ user.masked_functions[server].
  Hides compose by UNION; a user with every function masked sees an empty
  server (bob case, F3). TRADE-OFF (veto-able): within an allowed server
  this is default-OPEN — a NEWLY added function becomes visible to that
  server's users until masked; grant-servers-then-mask-specifics is the
  mental model, and the Users tab is literally the Functions matrix per user.
- `mcp_key` shown ONCE at creation/rotation (UI), stored plaintext — it is a
  bearer credential like today's config.json keys, same threat model.

## Specific architecture (exact signatures)

### M1 — `tools/shared/identity.py`

```python
def resolve_identity(request) -> tuple[str, str] | None:
    """(client_id, role) from the request's Bearer/X-API-Key token, via the
    ACTIVE tokens map (injected below). None when no HTTP scope or token
    unknown (in-memory tests, pre-auth paths)."""

class IdentityGateMiddleware(Middleware):
    def __init__(self, tokens_map_fn: Callable[[], dict[str, dict]],
                 require_identity: bool | None = None):  # None → env MCP_REQUIRE_IDENTITY
        ...
    # on_list_tools: filter by allowed; on_call_tool: ToolError("Unknown tool")
    # when not allowed; on_discover: same filter; absent identity →
    # fail-OPEN (logged) unless require_identity.
```

`tokens_map_fn` is a CALLABLE so M2's mtime-cached store read supplies live
data with zero wiring changes.

### M2 — `tools/shared/users_store.py`

```python
USERS_PATH = Path(os.environ.get("MCP_USERS_STORE",
                  Path.home() / ".config/supreme-mcp-tools/users.json"))

def hash_password(password: str) -> str          # pbkdf2_sha256$390000$s$b$
def verify_password(password: str, stored: str) -> bool
def load_users() -> dict                          # tolerant: corrupt/missing → {} + loud log
def save_users(store: dict) -> None               # atomic_write_json (flock sidecar)
def list_users() -> list[dict]                    # never returns mcp_key/password_hash
def create_user(username, password, role="user", servers=None, masked_functions=None) -> dict  # returns mcp_key ONCE
def delete_user(username) -> None
def set_enabled(username, enabled: bool) -> None
def rotate_key(username) -> dict                  # new mcp_key, returns it ONCE (globally unique)
def revoke_system_key(tool_name) -> None          # OPTIONAL (round 2): retire one tool's system key
def set_servers(username, servers: list) -> None
def set_masked_functions(username, masks: dict) -> None   # per-user function masks
def authenticate(username, password) -> dict | None   # constant-time verify
def tokens_map_for_tool(tool_name: str, system_key: str) -> dict[str, dict]:
    # {system_key: {"client_id": <ADMIN_USERNAME>, "role": "admin", "allowed": "*"}} ∪
    # (round 2: system keys are ATTRIBUTED TO THE ADMIN — the current mono
    #  user — so logs attribute them from day one; toolname no longer used)
    # {u.mcp_key: {"client_id": u.username, "role": u.role,
    #              "masked": u.masked_functions.get(tool_name, [])}}
    # mtime-cached internally (stat per call; reload on change)
```

Username rules: `^[a-z0-9_-]{2,32}$` (lowercased on create); collisions
rejected. `create_user` refuses usernames equal to an existing tool name
(`client_id` namespace clash with system keys).

### M2 — factory wiring (`tools/shared/server_factory.py`)

`create_fastmcp_server(name, api_key=None, ...)` — NO new param: the factory
reads `MCP_AUTH_MODE` (round 2; `multi` AND a readable store activate the
user layer; `mono` = exactly today). When active, the verifier becomes a thin
`UserStoreVerifier(DualHeaderVerifier)` whose `_tokens` property reads
`tokens_map_for_tool(name, system_key)`; `IdentityGateMiddleware` is added
with the same callable. Store absent → EXACTLY today's behavior (single key,
no middleware) — bootstrap-safe, and tools never fail on a bad store
(E1-masks lesson: tolerant).

### M2 — central API (`launcher/management_server.py` + `launchmcp.py`)

- `ManagementServer(..., api_key=os.environ.get("MCP_MANAGEMENT_API_KEY"))`
  in launchmcp.py (P0).
- M2 adds `/api/users` CRUD (create/list/delete, rotate-key,
  set-allowed-tools, set-enabled) — admin-key-protected once P0 is on;
  responses NEVER echo `mcp_key` except the create/rotate response body.

### M2 — mcp_ui

- Login (`auth` flow in management_ui.py): try `authenticate()` from the
  store; session gets `{"username", "role", "mcp_key"}` in
  `app.storage.user` (signed by the system-level `MCP_UI_SECRET`). Bootstrap:
  store absent OR `MCP_UI_LEGACY_LOGIN=1` → today's env-cred login, and a
  SUCCESSFUL legacy admin login WRITE-CREATES the `admin` user with a
  generated key (shown once in a dismissable dialog) — no lockout, no CLI step.
- `api_client`/`memory_client`: send the ACTING user's `mcp_key` on tool
  calls; central calls use the admin system key when the actor is admin,
  else the user key (central role check = M3).
- New **Users tab** (admin only): user list (role, enabled, tool count),
  create user (username/password/role → key shown once), server checkboxes + per-server
  FUNCTION-MASK matrix reusing the Functions-tab mask-grid pattern (round 3:
  the two dimensions the user asked for — servers reached, functions masked), rotate key, enable/disable,
  delete with confirm. Tab hidden for role=user.

### M3 — role enforcement + tx binding

- 81xx (`launcher/tool_extensions/http_server.py`): `_verify_api_key`
  becomes `_verify_identity` returning `(client_id, role)`; ACTION-type
  routes (execute, and `/admin/function-masks`) require `role == "admin"`.
  System keys carry role admin (F6 namespace).
- `/admin/flush-sessions` (server_factory): same role check via the map.
- Central writes (disabled-tools, users, env, auth PUT/POST/DELETE): require
  admin role when the caller presented a user key; the P0 system key is admin.
- E4: `begin_transaction` stamps `entry.tx_owner = client_id` via
  `get_http_request()` (F7); `finish_tx`/`reap` log the owner; commit/rollback
  by a DIFFERENT identity than the owner → allowed but logged (v1: shared
  workbench; strict ownership = knob for later).
- `MCP_REQUIRE_IDENTITY=1` flips the gate fail-closed (deny identity-less
  calls over HTTP; in-memory still allowed for tests).

## File tree (new/changed)

```
tools/shared/identity.py                 NEW  M1
tools/shared/users_store.py              NEW  M2
tools/shared/server_factory.py           M2 verifier+middleware wiring
launcher/management_server.py            P0 key; M2 /api/users; M3 role deps
launchmcp.py                             P0 pass MCP_MANAGEMENT_API_KEY
launcher/tool_extensions/http_server.py  M3 identity+role on actions
mcp_ui/management_ui.py                  M2 login-from-store, role tabs
mcp_ui/components/users_tab.py           NEW  M2
mcp_ui/api_client.py, memory_client.py   M2 acting-as-user keys
tools/databasemcp/db_tools.py            M3 tx_owner stamp
tests/test_identity_middleware.py        NEW  M1 (re-encodes F1-F4)
tests/test_users_store.py                NEW  M2
tests/test_e3_integration.py             NEW  M2/M3 (two users over real app)
```

## Instance & resource model (round 3 — "1 instance for all, or fork per user?")

**One shared instance per MCP server serves ALL users. No forking.** The
spike proved identity is per-REQUEST metadata (header → map lookup → filter),
not per-process state: alice and bob were served correctly by the SAME
instance in F3, and the stateless endpoint exists precisely so a request can
be answered by any process. Multi-user costs a dict lookup per request —
process count, ports, memory baseline are UNCHANGED as users are added.
Per-user forks would only ever buy HARD security isolation (separate
processes per tenant), which a bearer-key shared instance already covers for
a team workbench (the standard multi-tenant SaaS-API pattern); M4 multi-host
is where N processes reappear (N nodes, not N×users).

Per-tool resource reality as users grow:

| Tool | Per-user state? | Shared-instance verdict |
|---|---|---|
| simplemcp | none | trivial (stateless math) |
| webmcp | none that matters | "universal" as you said; search/fetch CACHE and history are shared (cache sharing is a feature); upstream API quotas are per-deployment, not per-user |
| databasemcp | connections are NAMED+SHARED by design | registry does NOT multiply per user (two users using preset 02 share one pool — overhead actually stays flat); E4 txs: one per connection regardless of user; per-user DB CREDENTIALS remain a v1 non-goal — a future `per-user connection namespace` knob exists in the plan's non-goals lineage if ever needed |
| memorymcp | the memories THEMSELVES are shared | data isolation stays the declared non-goal; cheapest future path is the EXISTING `agent_id` field (stamp/filter by client_id in middleware) — data-model work, not forking; per-user collections would multiply vector stores (heavy, rejected) |
| ragmcp | indexes are shared artifacts | read access gated per user normally; destructive actions (reindex/start_indexing are mgmt ACTIONS) → admin role under M3, so only the admin mutates indexes |

**The honest overhead curve:** auth/visibility scales to dozens of users at
~zero marginal cost. The costs that DO grow per user are data-plane choices
we deliberately deferred (whose memories, whose DB creds) — when they come,
they are schema/filter work inside one process, never process forks. The
only structural fork is M4 multi-host (node count ∝ load, not users).

## Environment (new)

| Var | Default | Purpose |
|---|---|---|
| `MCP_AUTH_MODE` | `mono` | `mono` = exactly today (single key per tool, no identity layer — zero risk). `multi` = store-driven (user round 2) |
| `MCP_MANAGEMENT_API_KEY` | unset (=open, loud startup WARNING until set) | P0 central 8200 bearer |
| `MCP_USERS_STORE` | `~/.config/supreme-mcp-tools/users.json` | store path |
| `MCP_UI_LEGACY_LOGIN` | unset | force env-cred login (break-glass) |
| `MCP_REQUIRE_IDENTITY` | unset | fail-closed identity gate (M3) |

No new ports. Metrics 8300: P0 documents bind/firewall guidance only.

## Design decisions — round 2 (user input, 2026-09-07)

1. **Mode flag `MCP_AUTH_MODE=mono|multi`** (absorbs the earlier
   `MCP_USERS_STORE_DISABLE` kill-switch — mono IS the kill-switch). `mono`
   must be behavior-identical to today (no middleware, no store reads, env
   login). `multi` activates the store; **the current mono user becomes the
   admin**: each tool's EXISTING system key is attributed to the ADMIN'S
   USERNAME (client_id = admin username, role admin — not "system"), so
   attribution in logs is correct from the first request and the admin's
   clients keep working unchanged. Admin bootstrap password = the current
   `MCP_UI_USERNAME`/`MCP_UI_PASSWORD` (write-through on first login).
2. **One key per user (v1).** The key asserts WHO you are; WHAT you can reach
   is `servers` + `masked_functions` (the store's job). One rotation, one revoke, one Users
   row; cutting a user off from one server is a checkbox, which is the same
   protection per-server keys buy. Per-server keys (`mcp_keys: {server: key}`
   map, schema v2) remain a compatible LATER extension if blast-radius
   isolation is ever needed.
3. **Store stays a JSON file in v1 — no Turso/DB.** Deliberate: the file is
   tiny, all processes share the filesystem, atomic-write + mtime hot-reload
   are proven (D2/E1), and a DB would put connection management into every
   tool process (the side-effect-import trap) plus migration/backup surface.
   `users_store.py`'s function set IS the interface — M4 (multi-host) is
   where a central auth service earns its keep and gets designed.
4. **`revoke_system_key(tool)` — OPTIONAL admin action, off by default.**
   Retires ONE tool's system key from its map (only user keys remain valid
   there). Existing keys keep working until explicitly revoked.
5. **Attribution in logs (promoted into M1).** `mcp.access` lines and the
   mutation logs gain the resolved `client_id` (username) wherever identity
   is available — the daily payoff of multi-user ("who ran that
   execute_sql?"). Identity is resolvable exactly where those lines emit.
6. **UI login = username + password (pbkdf2).** Key-as-login (paste the
   mcp_key, zero passwords) stays possible later — verification is a lookup
   either way; default follows the explicit mcp_ui-secret-keys requirement.
7. **Global key uniqueness** enforced at create/rotate: a user key must not
   collide with any system key or another user's key (regenerate on the
   astronomically unlikely collision; test the guard).
8. Out of scope, noted: per-user rate limits/quotas; per-user memory data.

## Edge-case catalogue (prescribe a test or an explicit note each)

1. Store missing/corrupt at tool startup → single-key behavior, loud log
   (never a startup failure — masks lesson).
2. Store edited while serving → next request sees it (mtime cache; verify
   with a live enable/disable round trip like E1's mask toggle).
3. Disabled user with a live session → next request 401 (verifier skips
   disabled), UI tab render re-checks role from the SESSION store.
4. Key rotation race → old key invalid the moment the map swaps (atomic
   read); client sees 401 and must re-bind (documented).
5. Username = tool-name collision → rejected at create (client_id namespace).
6. `servers` ∅ / `masked_functions` shapes; all-functions-masked user sees an
   empty server (allowed to auth, nothing listed); NEW function on an allowed
   server is default-OPEN to that server's users until masked (documented
   trade-off of deny-list, round 3).
7. Masks ∩ identity = intersection (E1 mask hides globally; identity scopes
   per user) — test both orders.
8. Identity-less call (in-memory `Client(mcp)`) → fail-open + log;
   with `MCP_REQUIRE_IDENTITY=1` over HTTP → deny; in-memory still allowed.
9. `/sse` GET stream carries no messages → hooks never fire there (note only).
10. Prompts/resources/discover parity: v1 filters tools + discover;
    prompts/resources = TODO note in code.
11. `on_call_tool` rejection text identical to unknown-tool ("Unknown tool")
    — no existence leak (F3 pattern).
12. UI user (role=user) hitting admin endpoints → 403 from central/81xx once
    M3 on; before M3, tabs are hidden (defense in depth order documented).
13. Legacy env login bootstrap → writes admin user exactly once; concurrent
    first-logins race → atomic write, one winner, both sessions valid.
14. All admins disabled/deleted → break-glass `MCP_UI_LEGACY_LOGIN=1` +
   delete/fix users.json directly (documented in README).
15. pbkdf2 verify timing → constant-time compare; wrong-username short-
    circuits to a dummy hash compare (no user-enumeration timing).
16. User deleted with an open E4 transaction → tx continues under stale
    client_id; reaper unaffected; logs note the unknown owner.
17. Central key set but UI not restarted → api_client gets 401s: UI reads the
    env at startup; README notes restart-after-set.
18. Concurrent store writes (UI + central API) → atomic_io flock (D2) —
    add a concurrency test.

## Failure playbook

- Store write fails (disk full/permissions) → API returns 500 with the
  atomic_io error; in-memory map unchanged (last-good file still read).
- `get_http_request()` raises in middleware (non-HTTP context) → caught →
  identity None → fail-open branch (M3 knob flips).
- Verifier map swap mid-request → per-request snapshot: read the map ONCE
  per verify/resolve (bind locally), so a request is decided by one version.
- UI session survives user deletion → next tool call 401s (server-side truth);
  UI surfaces re-login.
- Central locked out (key lost) → edit `.env` locally (documented).
- users.json hand-edited badly → load_users tolerant → {} → single-key
  behavior + loud WARNING (same shape as case 1).

## Phases (suite-gated; one commit each; branch `feature/e3-multiuser`)

- **P0 (~1h)** central key: launchmcp passes env → ManagementServer; unset →
  keep open but log a loud startup warning; .env gets a generated key (user
  informed); README security note. Tests: set/unset behavior of
  `_verify_api_key`.
- **M1 (~half day)** `identity.py` + factory hook + client_id attribution in `mcp.access` lines (round 2 #5) + `tests/test_identity_
  middleware.py` (re-encode F1-F4: two identities over the real multi app,
  list filter, call gate, 401, fail-open).
- **M2 (~1 day)** `users_store.py` (+tests: hashing, CRUD, tolerant load,
  tokens_map_for_tool shapes, mtime cache), factory wiring + kill-switch,
  central `/api/users`, mcp_ui login + Users tab + acting-as-user clients;
  integration test: create alice(user, one tool) + bob(admin) → alice's key
  sees only her tools on the live app, bob manages.
- **M3 (~half-1 day)** role gates (81xx actions, flush, central writes),
  `MCP_REQUIRE_IDENTITY`, E4 tx_owner (F7), docs (AGENTS.md auth section,
  README multi-user, CHANGELOG), live browser pass on the Users tab.
- **M4** multi-host design doc (stateless + LB + per-node identity notes) —
  separate.

## Open decisions (defaulted, veto-able)

- Tool system keys remain valid indefinitely (backward compat); a future
  `revoke_system_key` per tool can retire them.
- v1 tx ownership is LOG-ONLY (commit by non-owner allowed + logged).
- Password policy: min length 8, no complexity rules (local tool).
- `mcp_key` stored plaintext in users.json (same threat model as today's
  config.json keys; hashing tokens would break constant-time map lookup).
- mcp_ui admin identity for central calls: the acting admin's USER key (not
  the system key) once M3 role checks exist — one credential per human.
