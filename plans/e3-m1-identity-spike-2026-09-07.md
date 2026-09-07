# E3/M1 — Multi-user spike: caller identity in fastmcp middleware (2026-09-07)

> STATUS: **spike questions ANSWERED live** (probes below, run 2026-09-07 against
> fastmcp 4.0.0 on the real multi-transport stack from `tools/shared/server_factory.py`).
> This document records the verified answer, the architectural decision it unlocks,
> and the scoped M1 deliverable (productionizing the probe into shared code).
> Tracker: TODO.md evolutions section (E3). Companion: `plans/evolutions-plan-2026-09-06.md` §E3.

## The spike question

Can fastmcp 4 middleware see **who** is calling — enough to filter `tools/list`
and gate `tools/call` per identity, on the live serving path, without
per-identity server instances?

## VERIFIED FACTS (live probes, 2026-09-07)

**F1 — the documented identity path exists and works.**
`from fastmcp.server.dependencies import get_http_request` inside
`on_call_tool` / `on_list_tools` returns the caller's Starlette `Request`:
the raw `Authorization: Bearer <token>` header is visible **per call, per
caller** (probe: two clients, `Bearer ali…` vs `Bearer bob…` captured
correctly). Works on the stateful `/mcp`, stateless `/mcp-stateless`, single-
transport child, and in-memory paths alike.

**F2 — the VERIFIED AccessToken is NOT exposed.** `request.state` is empty;
`FastMCPRequestContext` has no `access_token` attribute. fastmcp's auth stack
consumes `DualHeaderVerifier.verify_token()` and does not stamp the result
anywhere middleware can read. **Consequence:** identity must be resolved from
the raw token — which is safe here: a request only reaches middleware AFTER
auth passed, so looking the token up in the same `tokens` map
(`token → {"client_id", "scopes"}`) that the verifier checked is exact, not
a re-implementation of auth.

**F3 — per-user visibility WORKS end-to-end (the M3 mechanism, demonstrated).**
Probe middleware: `on_list_tools` filtered by the caller's resolved
`client_id` against an `ALLOWED = {"alice": {"ping"}, "bob": set()}` map;
`on_call_tool` rejected non-allowed tools with `ToolError("Unknown tool")`.
Live result over `/mcp`: **alice sees `ping` and calls it (`pong`); bob sees
an empty list and his call is rejected.** Wrong tokens still 401 at the auth
layer before middleware ever runs.

**F4 — middleware fires on every transport path.** Verified: in-memory
`Client(mcp)`, single HTTP child (`stateless_http=True`), the multi app via
`/mcp` and via `/mcp-stateless`. (An earlier "middleware didn't fire" reading
was a probe-script reporting bug — an empty placeholder list crashed the
print loop before the real captures; the hooks had fired all along. Lesson:
test the reporter, not just the subject.)

**F5 — hook surface (fastmcp 4.0.0).** `Middleware` provides `on_request`,
`on_message`, `on_list_tools`, `on_call_tool`, `on_list_resources`,
`on_list_prompts`, `on_initialize`, `on_discover`, … — all receive
`MiddlewareContext` (`message`, `fastmcp_context`, `method`, `source`).
Registration: constructor `FastMCP(..., middleware=[...])` or
`mcp.add_middleware(...)` before `http_app()` — both verified.

**F6 — auth wiring today.** `create_fastmcp_server(name, api_key=...)`
builds `FastMCP(name, auth=DualHeaderVerifier(tokens={key: {"client_id": name,
"scopes": ["mcp"]}}))` — ONE token per tool, `client_id` = the tool name.
`DualHeaderVerifier.verify_token()` already accepts a MULTI-token dict
(`tests/test_era_negotiation.py:56-58` proves the shape) — multi-key auth is
native; we simply pass one key today.

## DECISION (unlocked by F1–F6)

**Outcome A: per-user visibility is a middleware filter — no per-identity
server instances, no proxies.** The E3 architecture proceeds as planned:
M2 (user store + per-user keys feeding a MULTI-token `DualHeaderVerifier`)
→ M3 (one shared identity/authorization middleware on every tool, driven by
per-user allow-lists) → M4 (multi-host design doc). The stateless endpoint
already makes horizontal scaling possible; identity composes on top.

## M1 deliverable (productionize the probe) — ~half day

1. **`tools/shared/identity.py`** — two small pieces:
   - `resolve_client_id(request) -> str | None`: parse `Authorization:
     Bearer <token>` (plus a tolerant `X-API-Key` fallback for the case where
     the normalizer is disabled), look the token up in the active tokens map.
     Outside an HTTP scope (`get_http_request()` raises) → `None`.
   - `class IdentityGateMiddleware(Middleware)`: reads the allow-map from an
     injected callable (so M2's user store can supply it live later);
     `on_list_tools` filters, `on_call_tool` raises `ToolError("Unknown
     tool")` for non-allowed tools (masks' no-leak UX), `on_discover`
     filtered the same way. Absent identity (in-memory/dev path) → **fail
     OPEN, logged** (matches today's single-user behavior; tightening to
     fail-closed is an M3 config knob: `MCP_REQUIRE_IDENTITY=1`).
2. **Factory hook** — `create_fastmcp_server(..., allow_map=...)` optional
   param wires the middleware when provided (tools unchanged until M3).
3. **Tests** (`tests/test_identity_middleware.py`) — the probe scenarios as
   regression: two identities over the real multi app (list filter + call
   gate + wrong-key 401), in-memory no-identity fail-open, stale-token
   resolution → deny, middleware-order note (identity outermost).

### Edge-case catalogue (each needs a test or an explicit note)

1. Caller with a valid token that was rotated out of the map mid-session →
   resolver returns `None` → fail-open (v1) — logged loudly; revisit at M3.
2. `X-API-Key` client (normalizer rewrites to Bearer pre-auth; resolver sees
   Bearer — F1). Tolerant fallback parse keeps correctness if the normalizer
   is ever bypassed.
3. Stateful `/mcp` sessions: every JSON-RPC POST carries headers (verified);
   legacy `/sse` GET stream carries no messages (hooks never fire there) —
   note only.
4. `on_discover`/`on_list_prompts`/`on_list_resources` parity — same filter
   applied; v1 covers tools + discover, prompts/resources noted as TODO.
5. Identity middleware vs E1 masks: both filter `tools/list`; composition is
   intersection (mask hides globally, identity scopes per user) — ordering
   irrelevant for the result, but identity should be registered FIRST
   (outermost) so its logging sees masked-out calls too.
6. Tool-error surface: rejected calls must be indistinguishable from unknown
   tools (`ToolError("Unknown tool")`) — no existence leak.
7. `get_http_request()` outside HTTP (in-memory `Client(mcp)`) raises →
   caught → `None` identity → fail-open branch.
8. Performance: one dict lookup per request — negligible; no new I/O.

### Failure playbook

- `get_http_request()` returns a request WITHOUT an Authorization header in
  middleware (should be impossible post-auth): log ERROR, treat as `None`,
  fail-open — auth already vetted the caller, so this is a wiring anomaly to
  surface, not a user-facing failure.
- Middleware raising inside a hook: fastmcp propagates → client sees an MCP
  error; wrap resolver body in try/except returning `None` + log.
- Tokens map mutated while serving (M2 rotation): resolver reads the map per
  request; rotation is atomic-swap of the dict reference (same pattern as E1
  runtime masks).

## Bridge to M2–M4 (pointers, not scope)

- **M2**: user store (`~/.config/supreme-mcp-tools/users.json`, hashed keys,
  per-user allowed-tools map) + central `POST /api/users` + mcp_ui Users tab;
  the store REPLACES the single-token dict in `create_fastmcp_server`
  (multi-token `DualHeaderVerifier`, F6). tx ownership (E4) should bind
  `tx_id` to the caller's client_id when this lands.
- **M3**: flip tools to `allow_map` from the store; `MCP_REQUIRE_IDENTITY`
  knob; runtime push like E1.
- **M4**: multi-host design doc (stateless endpoint + LB; identity travels
  in the token, no server-side session needed).

## Probe artifacts

The three probe cells are embedded in this document's git history
(session 2026-09-07): middleware-firing matrix (F4), identity capture (F1/F2),
per-user gate demo (F3). The regression tests in the M1 deliverable
re-encode all three.
