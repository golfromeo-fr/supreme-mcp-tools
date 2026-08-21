# FastMCP 4 Phase 0 Spike Verdict

**Date:** 2026-08-21
**fastmcp version tested:** fastmcp==4.0.0b3 (still latest pre-release on 2026-08-21; stable 4.0.0 NOT out — trigger #1 unmet)
**mcp version installed transitively:** mcp==2.0.0 (+ mcp-types==2.0.0 standalone, re-exported as `mcp.types`)
**Environment:** throwaway venv `/tmp/fmcp4` (starlette 1.6.0, pydantic 2.14.0b1, uvicorn 0.52.4; **no fastapi**, **httpx2 2.12.0** instead of httpx)

## Questions (mirrors plan Q1–Q9)

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | `TokenVerifier.get_middleware()` still overridable? | **YES** — same shape `(self) -> list`; called at `fastmcp/server/http.py:458` and `:601` | `inspect.signature`; source grep |
| 2 | `mcp.server.auth.middleware.auth_context/bearer_auth` still import? | **YES** — both classes at identical paths under mcp 2.0.0 | live import in spike |
| 3 | `FastMCP(name, auth=<TokenVerifier>)` constructs? | **YES** — `s.auth is verifier` holds | live construct |
| 4 | Native idle-session-TTL kwarg on `http_app()`? | **YES — `session_idle_timeout: float \| None`** (also on `create_streamable_http_app`). Retire `_apply_session_idle_timeout_via_lifespan`. | `http_app` signature; live instance attr `session_idle_timeout` |
| 5 | Where is the session manager reachable? | **Same walk as 3.x**: `/mcp` route → `RequireAuthMiddleware` → `.app` → `StreamableHTTPASGIApp.session_manager` (instance class: `FastMCPStreamableHTTPSessionManager`). `session_manager` is `None` until lifespan runs (same as today) | live endpoint walk |
| 6 | Per-session `terminate()` API? | **YES — `_server_instances` dict exists at RUNTIME** on the manager instance (`{session_id: StreamableHTTPServerTransport}`; invisible at class level — instance attr). Entries have `async def terminate()` (`mcp/server/streamable_http.py:807`). Flush mechanism ports unchanged. Note: modern (sessionless-era) clients leave no lingering entry; legacy sessions do. | live manager `vars()` dump + SDK source |
| 7 | `http_app(transport=...)` accepted values? | `Literal['http', 'streamable-http', 'sse']` — current code's `"http"` remains valid | signature |
| 8 | Legacy-era client knob? | **YES — `Client(transport, mode="legacy")`** (full Client params include `mode`; no `protocol_version`). Also `verify=`, `auto_initialize`. | live: legacy client called a tool successfully |
| 9 | Does dual-header auth run for legacy clients too? | **YES, one era-agnostic app-level middleware stack** — both modern and `mode="legacy"` clients authenticated + called tools. **BUT new gate:** 4.0b3 `RequireAuthMiddleware` rejects requests with no `Authorization` header (RFC 6750 presence check) — **X-API-Key-only requests get 401 even when authenticated**. Live matrix: Bearer-only 200 · both 200 · X-API-Key-only **401** · bad token 401 | live header-variant matrix |

## Phase 2 path chosen

**Mechanical port (branch 1), simplified, plus one small addition:**

1. **Add** an outermost ~10-line ASGI middleware (`api_key_fallback`) that copies `X-API-Key` → `Authorization: Bearer <key>` when Authorization is absent, wrapping the app returned by `http_app()`. **Live-validated**: X-API-Key-only → 200 + full tool-call round-trip.
2. **Keep** `DualHeaderVerifier.verify_token()` as-is (TokenVerifier subclass).
3. **DELETE** `DualHeaderAuthBackend` + `_AuthenticatedUser` — with the normalizer, FastMCP's default `BearerAuthBackend` + our `verify_token` override delivers dual-header auth. Less custom code than 3.x.
4. Keep `test_auth_c1_oauth_removal.py` green to re-verify the no-OAuth/no-WWW-Authenticate guarantees under 4.0 (TokenVerifier route advertising needs a re-check there).

## Notes for Phase 3 REWORK

- Idle-timeout kwarg name: **`session_idle_timeout`** — set directly via `http_app(session_idle_timeout=...)`; delete the lifespan-wrap hack.
- Session-manager reach path: unchanged (`/mcp` route endpoint → `.app` chain → `StreamableHTTPASGIApp.session_manager`).
- Per-session termination: `_server_instances` values expose `async def terminate()` — flush loop ports as-is.

## Dependency findings (Phase 1 amendments)

- **fastapi is NOT in FastMCP 4's dependency set** — the planned `fastapi>=0.133` floor bump is only needed while the launcher's FastAPI-tool validation tier exists (independent decision, not a FastMCP 4 requirement).
- **httpx2 replaces httpx** for FastMCP's client — `tests/test_era_negotiation.py` gets httpx2 transitively via fastmcp; our tools' own httpx usage is unaffected.
- `fastmcp-slim` 4.0.0b3 installed alongside; pin both when pinning the beta.
- `mcp.types` re-export works (`mcp.types.JSONRPCError` imports); `LATEST_PROTOCOL_VERSION == "2026-07-28"`.
- Beta status 2026-08-21: b3 still latest; no b4/stable. Weekly re-check stands.

## Key snippets (the port's load-bearing code)

The X-API-Key → Authorization normalizer (outermost wrapper — live-validated end-to-end):

```python
def api_key_fallback(app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers") or []
            if not any(k.lower() == b"authorization" for k, _ in headers):
                key = next((v for k, v in headers if k.lower() == b"x-api-key"), None)
                if key:
                    scope["headers"] = list(headers) + [(b"authorization", b"Bearer " + key)]
        await app(scope, receive, send)
    return wrapped

app = api_key_fallback(mcp.http_app(transport="http", session_idle_timeout=...))
```

Flush-relevant runtime facts (live manager `vars()`): `_server_instances: dict[session_id →
StreamableHTTPServerTransport]` + `_session_owners: dict` + `session_idle_timeout` attr; entries
expose `async def terminate()` (mcp/server/streamable_http.py:807). Reach path:
`/mcp` route → `RequireAuthMiddleware` → `.app` → `StreamableHTTPASGIApp.session_manager`.

**Risk caveat (per plan risk register):** `_server_instances` is a *private* instance attribute —
it moved from class-source-visible (3.x) to runtime-only (4.0b3) already. If stable 4.0 moves it
again, the fallback is the idle-TTL (native kwarg) + process restart; re-run this check when stable
ships (the verdict's Q6 must be re-verified, not assumed).

## Spike artifacts

Throwaway venv `/tmp/fmcp4` (delete freely); scripts `spike.py`, `spike_final.py` (auth pattern),
`spike_live2.py` (two-era + manager internals) are the reproducible evidence — key snippets live in
this file; nothing merges to main from the spike.
