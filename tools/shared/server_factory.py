"""
FastMCP Server Factory — supreme-mcp-tools
==========================================
Centralized factory for all MCP tool servers.

Key design decisions (FastMCP 4 port, 2026-08-23):
- Uses DualHeaderVerifier (TokenVerifier subclass) — no OAuth endpoints
  advertised, no /.well-known/oauth-* routes, no WWW-Authenticate 401.
- Dual-header support (X-API-Key for VS Code Copilot, Authorization: Bearer
  for standard clients) comes from the outermost api_key_fallback ASGI
  middleware: it copies X-API-Key into Authorization: Bearer when the latter
  is absent, so FastMCP's default BearerAuthBackend + verify_token()
  authenticates both header styles (Phase 0 verdict Q9,
  plans/mcp-2026-07-28-phase0-verdict.md).
- Factory pattern: one call creates server, adds tools, and configures auth.
- No dependencies on launcher modules — runs standalone.

Usage in each *_fastmcp.py:

    from tools.shared.server_factory import create_fastmcp_server, get_transport_app, DEFAULT_HOST

    mcp = create_fastmcp_server("webmcp")
    app = get_transport_app(mcp)
    uvicorn.run(app, host=DEFAULT_HOST, port=MCP_PORT)
"""

from __future__ import annotations

import json
import logging
import os
import hmac
import secrets as _secrets
import time
import urllib.parse
import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

# ── Single source of truth for bind host ──────────────────────────────────
# On Debian/Linux, "::" binds IPv6-only despite bindv6only=0, breaking IPv4
# clients (VS Code Copilot). Use "0.0.0.0" which accepts both 127.0.0.1 and
# localhost (::1 → kernel maps to IPv4).
DEFAULT_HOST = "0.0.0.0"

if TYPE_CHECKING:
    from fastmcp import FastMCP

# ── FastMCP auth imports ──────────────────────────────────────────────────
from fastmcp.server.auth import TokenVerifier, AccessToken


class ApiKeyFallbackMiddleware:
    """Pure-ASGI middleware enabling dual-header auth (verdict Q9).

    Copies X-API-Key into Authorization: Bearer <key> when no Authorization
    header is present, so requests that only carry the Copilot-style header
    pass FastMCP's RequireAuthMiddleware RFC 6750 presence check and are
    validated by the same verify_token() as Bearer clients. When both
    headers are sent, the existing Authorization header wins.

    Registered via app.add_middleware() so it runs outside the router —
    ahead of FastMCP's per-route auth — on every route including
    /admin/flush-sessions.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers") or []
            if not any(k.lower() == b"authorization" for k, _ in headers):
                key = next((v for k, v in headers if k.lower() == b"x-api-key"), None)
                if key:
                    scope["headers"] = list(headers) + [
                        (b"authorization", b"Bearer " + key)
                    ]
        await self.app(scope, receive, send)


class DualHeaderVerifier(TokenVerifier):
    """
    FastMCP TokenVerifier that backs dual-header authentication.

    Replaces the old StaticTokenVerifier (removed in FastMCP 3.x).
    verify_token() checks a static token dict — no remote validation, no OAuth.
    The default BearerAuthBackend wiring calls verify_token(); the outermost
    api_key_fallback middleware normalizes X-API-Key into Bearer form first,
    so both header styles land here. No get_middleware() override — 4.x gates
    on Authorization-header presence, which the normalizer satisfies.
    """

    def __init__(self, tokens: dict[str, dict[str, Any]]) -> None:
        """Initialize with a static token map.

        Args:
            tokens: Mapping of token string → {"client_id": str, "scopes": list[str]}
        """
        super().__init__()
        self._tokens: dict[str, dict[str, Any]] = tokens

    async def verify_token(self, token: str) -> AccessToken | None:
        """Validate token against the static token dict."""
        found = None
        for key, val in self._tokens.items():
            if hmac.compare_digest(token, key):
                found = val
                break
        if found is None:
            return None
        info = found
        return AccessToken(
            token=token,
            client_id=info.get("client_id", "unknown"),
            scopes=info.get("scopes", []),
            claims=info,
        )


def _jsonrpc_failure_marker(body: bytes) -> str | None:
    """Best-effort failure marker from a JSON-RPC response body (JSON or SSE).

    Two failure shapes exist on the wire: a protocol-level ``error`` object
    (unknown method, bad params) and FastMCP tool failures, which come back as
    a successful result carrying ``"isError": true``. Both must be surfaced —
    both ride inside HTTP 200 responses.
    """
    if not body:
        return None
    payload = body
    idx = body.find(b"data:")
    if idx != -1:  # SSE-framed: the first data line carries the message
        chunk = body[idx + 5:].lstrip()
        end = chunk.find(b"\n")
        payload = chunk[: end if end != -1 else len(chunk)]
    try:
        decoded = json.loads(payload)
    except Exception:
        return None
    if not isinstance(decoded, dict):
        return None
    error = decoded.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return f"rpc_err={code if isinstance(code, (int, str)) else 'error'}"
    result = decoded.get("result")
    if isinstance(result, dict) and result.get("isError") is True:
        return "tool_error"
    return None


def _extract_log_session(
    headers: dict[str, str], path: str, query_string: bytes | str = b""
) -> str | None:
    """Session id for the access line: Mcp-Session-Id header, else SSE query param.

    The modern dialects carry the session in a header; legacy SSE clients
    cannot — the SSE transport hands the client an endpoint URL and every
    message POSTs to ``/messages/?session_id=<id>``. The header wins when
    present; the query param is the SSE-only fallback so /messages lines stop
    looking session-less. None → the log line prints ``NEW``.
    """
    session = headers.get("mcp-session-id")
    if session:
        return session
    if path.rstrip("/") == "/messages" and query_string:
        if isinstance(query_string, bytes):
            query_string = query_string.decode("latin-1")
        values = urllib.parse.parse_qs(query_string).get("session_id") or []
        return values[0] if values else None
    return None


class RequestLogMiddleware:
    """One INFO line per HTTP request hitting the tool app (logger ``mcp.access``).

    The single trace that works for every client dialect — legacy handshake-era
    clients AND the modern 2026-07-28 single-exchange dialect, whose serving
    path logs nothing at INFO:

        [simplemcp] POST /mcp from 127.0.0.1 v=2026-07-28 session=NEW -> 200 in 3ms tools/call echo

    Fields:
    - ``[name]``   — FastMCP server name; all tools share one launcher.log, so
      without it four servers' lines are indistinguishable
    - ``v=``       — MCP-Protocol-Version header (``-`` when absent); 2026-07-28
      names the modern session-less dialect
    - ``session=`` — Mcp-Session-Id prefix, ``NEW`` when the request carries
      none; legacy-SSE POSTs to ``/messages/?session_id=<id>`` show the
      query-param id (see _extract_log_session)
    - ``in Nms``   — handler duration
    - ``tools/call echo`` — JSON-RPC method + tool name, taken from the modern
      routing headers when present, else a bounded sniff of the request body
    - ``rpc_err=`` / ``tool_error`` — failure markers found in the response
      body: a JSON-RPC ``error`` object (code shown) or a result with
      ``"isError": true``. Both ride inside HTTP 200 responses, so without
      this a failing tool call looks healthy

    Sits outside the auth stack, so 401/404 rejections are logged too, making
    failed initial connections debuggable. Suppressed entirely by
    MCP_DISABLE_REQUEST_LOGS.
    """

    _MAX_SNIFF = 1_000_000       # request body bytes buffered for parsing
    _MAX_RESPONSE_SNIFF = 8192   # response body bytes kept for error extraction

    def __init__(self, app, name: str = "mcp"):
        self.app = app
        self.name = name
        self.logger = logging.getLogger("mcp.access")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in (scope.get("headers") or [])
        }
        proto = headers.get("mcp-protocol-version")
        client = scope.get("client")
        peer = f"{client[0]}:{client[1]}" if client else "-"
        method = scope.get("method", "-")
        path = scope.get("path", "-")
        # Header session when present (modern dialects); SSE POSTs carry theirs
        # only as a query param (/messages/?session_id=<id>) — fall back to it.
        session = _extract_log_session(
            headers, path, scope.get("query_string") or b""
        )

        # JSON-RPC method + tool name: the modern dialect's routing headers
        # identify it for free; legacy requests (streamable /mcp AND SSE-era
        # /messages) get a bounded body sniff.
        rpc_method = headers.get("mcp-method")
        rpc_name = headers.get("mcp-name")
        buffered = []
        if method == "POST" and path in ("/mcp", "/mcp-stateless", "/messages") and rpc_method is None:
            body = b""
            while True:
                message = await receive()
                buffered.append(message)
                if message["type"] != "http.request":
                    break
                body += message.get("body", b"")
                if len(body) > self._MAX_SNIFF or not message.get("more_body"):
                    break
            try:
                decoded = json.loads(body)
                rpc_method = decoded.get("method")
                name_value = (decoded.get("params") or {}).get("name")
                if rpc_name is None and isinstance(name_value, str):
                    rpc_name = name_value
            except Exception:
                pass

        async def receive_replay():
            # Replay the buffered body, then delegate to the live receive so
            # disconnect detection during long responses still works — faking
            # http.disconnect here makes servers close responses early.
            if buffered:
                return buffered.pop(0)
            return await receive()

        status = None
        response_sniff = bytearray()
        start = time.monotonic()

        async def send_wrapped(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif (message["type"] == "http.response.body"
                  and len(response_sniff) < self._MAX_RESPONSE_SNIFF):
                room = self._MAX_RESPONSE_SNIFF - len(response_sniff)
                response_sniff.extend(message.get("body", b"")[:room])
            await send(message)

        try:
            await self.app(scope, receive_replay if buffered else receive, send_wrapped)
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            rpc = f" {rpc_method}" if rpc_method else ""
            if rpc and rpc_name:
                rpc += f" {rpc_name}"
            error_marker = _jsonrpc_failure_marker(bytes(response_sniff))
            error_field = f" {error_marker}" if error_marker is not None else ""
            self.logger.info(
                "[%s] %s %s from %s v=%s session=%s -> %s in %dms%s%s",
                self.name, method, path, peer, proto or "-",
                session[:8] if session else "NEW", status or "-",
                elapsed_ms, rpc, error_field,
            )


# ── Factory function ──────────────────────────────────────────────────────

def create_fastmcp_server(
    name: str,
    api_key: str | None = None,
    tools_module: object | None = None,
) -> "FastMCP":
    """
    Create a FastMCP 3.x server with dual-header static-token auth.

    The returned server:
    - Validates X-API-Key AND Bearer tokens via DualHeaderVerifier
    - Does NOT advertise /.well-known/oauth-* endpoints
    - Does NOT return WWW-Authenticate on 401
    - Runs standalone (no launcher dependency)

    Args:
        name: Tool name, used as FastMCP server name
        api_key: Static API key. Resolved from <NAME>_API_KEY env var,
                 then MCP_API_KEY env var, then the provided fallback.
        tools_module: Module whose @mcp.tool() decorators should be registered.
                      The module is imported to trigger decorator side effects.

    Returns:
        Configured FastMCP server instance (call get_transport_app() or .http_app())
    """
    from fastmcp import FastMCP

    resolved_key = _resolve_api_key(name, api_key)

    verifier = DualHeaderVerifier(
        tokens={resolved_key: {"client_id": name, "scopes": ["mcp"]}},
    )

    mcp = FastMCP(name, auth=verifier)

    if tools_module is not None:
        import importlib
        importlib.import_module(tools_module.__name__)

    return mcp


def _tool_config_transport(name: str | None) -> str | None:
    """Per-tool transport override from tools/<name>/config.json ("transport" key).

    Lets one tool serve SSE for legacy harnesses while the rest of the launcher
    stays on streamable-http. Only accepts the two known transports, so a typo
    in config.json fails loudly here instead of silently doing nothing.
    """
    if not name:
        return None
    try:
        from pathlib import Path
        cfg = json.loads(
            (Path(__file__).resolve().parent.parent / name / "config.json").read_text()
        )
        value = cfg.get("transport")
        if isinstance(value, dict):
            # Legacy config shape ("transport": {type, endpoint, ...}) — not a
            # transport selection. Selection stays with env/default.
            return None
        value = (value or "").lower()
        if value == "http":
            value = "streamable-http"
        if value and value not in ("streamable-http", "streamable-stateless", "sse"):
            raise ValueError(
                f"tools/{name}/config.json: unsupported transport {value!r} "
                "(use 'streamable-http', 'streamable-stateless' or 'sse')"
            )
        return value or None
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise ValueError(f"tools/{name}/config.json is not valid JSON: {e}") from e


def get_transport_app(mcp, transport: str | None = None):
    """Get the ASGI app for the requested transport mode.

    **Default is the versatile multi-transport app** (2026-08-30): one FastMCP
    instance serving every client dialect at once —

    - ``/mcp``            streamable HTTP, stateful (sessions, flush, idle TTL)
    - ``/mcp-stateless``  streamable HTTP, per-request fresh (no session ids)
    - ``/sse`` + ``/messages``  legacy SSE

    Auth (dual-header via ApiKeyFallbackMiddleware + DualHeaderVerifier) and
    the ``mcp.access`` request log cover all endpoints. Client selection is by
    URL; nothing needs to be configured.

    Single-transport modes remain as escape hatches (``"streamable-http"``,
    ``"streamable-stateless"``, ``"sse"``). Precedence: explicit ``transport``
    argument > ``"transport"`` key in ``tools/<name>/config.json`` >
    ``MCP_TRANSPORT`` env var > default ``"multi"``.

    Args:
        mcp: FastMCP server instance
        transport: optional explicit override (see above)

    Returns:
        Starlette ASGI application
    """
    transport = (
        transport
        or _tool_config_transport(getattr(mcp, "name", None))
        or os.environ.get("MCP_TRANSPORT")
        or "multi"
    ).lower()
    if transport == "http":
        transport = "streamable-http"
    if transport not in ("multi", "streamable-http", "streamable-stateless", "sse"):
        raise ValueError(
            f"Unsupported MCP_TRANSPORT {transport!r}: only 'multi' (default), "
            "'streamable-http', 'streamable-stateless' or 'sse' are supported"
        )

    if not hasattr(mcp, "http_app"):
        raise RuntimeError(f"FastMCP server has no http_app method: {type(mcp)}")

    if transport == "multi":
        app = _build_multi_app(mcp)
    else:
        app = _build_http_app(mcp, transport)
        if transport == "streamable-http":
            _wire_session_management(app)
    # User middleware runs outside the router, so the normalizer executes
    # before FastMCP's per-route RequireAuthMiddleware on every route.
    app.add_middleware(ApiKeyFallbackMiddleware)
    # add_middleware stacks last-added = outermost, so the access line sees
    # every request including auth rejections and flush-endpoint calls.
    if not os.environ.get("MCP_DISABLE_REQUEST_LOGS"):
        app.add_middleware(RequestLogMiddleware, name=getattr(mcp, "name", "mcp"))
    return app


def _build_multi_app(mcp):
    """Versatile app: one tool, every client dialect (spike-proven 2026-08-30).

    Three child apps built from the same FastMCP instance have disjoint paths
    (/mcp, /mcp-stateless, /sse + /messages), so their routes flatten into one
    parent Starlette app. The parent lifespan runs the three child lifespans
    (each child's session manager starts inside its own lifespan). Child
    middleware is deduplicated onto the parent — all children share this
    factory's single auth provider, so one AuthenticationMiddleware stack
    serves every route. The flush route is wired onto the stateful child
    before flattening, and the idle TTL applies to it.
    """
    stateful = _build_http_app(mcp, "streamable-http")
    _wire_session_management(stateful)
    stateless = mcp.http_app(transport="http", stateless_http=True, path="/mcp-stateless")
    sse = mcp.http_app(transport="sse", path="/sse")

    routes = [*stateful.router.routes, *stateless.router.routes, *sse.router.routes]
    seen, middlewares = set(), []
    for child in (stateful, stateless, sse):
        for mw in child.user_middleware:
            if mw.cls not in seen:
                seen.add(mw.cls)
                middlewares.append(mw)

    from starlette.applications import Starlette

    @asynccontextmanager
    async def combined_lifespan(app):
        async with stateful.router.lifespan_context(stateful):
            async with stateless.router.lifespan_context(stateless):
                async with sse.router.lifespan_context(sse):
                    yield

    parent = Starlette(routes=routes, lifespan=combined_lifespan)
    for mw in middlewares:
        parent.add_middleware(mw.cls, **mw.kwargs)
    return parent


def _build_http_app(mcp, transport: str):
    """Build the ASGI app for the requested transport.

    streamable-http is stateful: MCP_SESSION_IDLE_TIMEOUT (default 1800s;
    <=0 disables) keeps stale sessions self-evicting instead of accumulating
    until process restart, applied natively via ``http_app(session_idle_timeout=...)``
    (FastMCP ≥4, Phase 0 verdict Q4). No fallback: the runtime is pinned to
    fastmcp 4.x, and if a future version drops the kwarg the TypeError at
    startup is the loud failure we want. SSE apps have no idle-TTL kwarg.
    streamable-stateless serves per-request fresh contexts (no session ids,
    no GET stream) — nothing to reap or flush, so no session wiring either.
    """
    if transport == "sse":
        return mcp.http_app(transport="sse")
    if transport == "streamable-stateless":
        return mcp.http_app(transport="http", stateless_http=True)

    raw_timeout = os.environ.get("MCP_SESSION_IDLE_TIMEOUT", "1800")
    try:
        idle_timeout = float(raw_timeout)
        if idle_timeout <= 0:
            idle_timeout = None
    except (TypeError, ValueError):
        idle_timeout = None

    if idle_timeout is not None:
        return mcp.http_app(transport="http", session_idle_timeout=idle_timeout)
    return mcp.http_app(transport="http")


def _wire_session_management(app) -> None:
    """Attach the flush-sessions admin route.

    The session idle TTL is applied at app construction (see _build_http_app).

    The flush route authenticates via the app-wide auth stack (default
    BearerAuthBackend + DualHeaderVerifier, with ApiKeyFallbackMiddleware
    normalizing X-API-Key) — the same tool API key that protects /mcp.
    """
    if os.environ.get("MCP_DISABLE_FLUSH_ENDPOINT"):
        return

    try:
        from starlette.routing import Route
        from starlette.responses import JSONResponse
    except ImportError:
        return

    async def flush_sessions(request):
        # Auth is enforced by app-wide AuthenticationMiddleware; require an
        # authenticated user (same tool API key that protects /mcp).
        user = getattr(request, "user", None)
        if user is None or not getattr(user, "is_authenticated", False):
            return JSONResponse(
                {"error": "unauthorized"}, status_code=401,
                headers={"www-authenticate": 'Bearer error="invalid_token"'},
            )
        # Find the /mcp route's endpoint — it holds the session_manager singleton.
        sm = _get_session_manager(app)
        if sm is None:
            return JSONResponse({"error": "session manager not initialized"}, status_code=503)
        count = 0
        # terminate() closes per-request streams and marks the transport _terminated;
        # the manager only auto-pops on idle-timeout/crash, so clear explicitly.
        for transport in list(getattr(sm, "_server_instances", {}).values()):
            try:
                await transport.terminate()
            except Exception:
                pass
            count += 1
        getattr(sm, "_server_instances", {}).clear()
        return JSONResponse({"flushed": count})

    routes = list(app.router.routes)
    routes.append(Route("/admin/flush-sessions", flush_sessions, methods=["POST"]))
    app.router.routes = routes


def _get_session_manager(app):
    """Return the StreamableHTTPSessionManager serving /mcp, or None.

    The /mcp route's endpoint is wrapped by auth middleware (RequireAuthMiddleware),
    so walk the ``.app`` chain inward to find the StreamableHTTPASGIApp singleton
    whose ``session_manager`` is populated by the lifespan. The manager only
    exists after startup (it is created inside the lifespan).
    """
    try:
        from fastmcp.server.http import StreamableHTTPASGIApp
    except ImportError:
        return None
    for r in getattr(app.router, "routes", []):
        if getattr(r, "path", None) != "/mcp":
            continue
        node = getattr(r, "endpoint", None)
        for _ in range(10):  # bound the walk in case of cycles
            if isinstance(node, StreamableHTTPASGIApp):
                return getattr(node, "session_manager", None)
            inner = getattr(node, "app", None)
            if inner is None or inner is node:
                break
            node = inner
    return None


def _resolve_api_key(name: str, fallback: str | None) -> str:
    """Resolve API key with priority: env var > config.json > fallback > dev default."""
    # 1. Check <NAME>_API_KEY env var
    env_key = f"{name.upper().replace('-', '_')}_API_KEY"
    if env_val := os.environ.get(env_key):
        return env_val
    # 2. Check MCP_API_KEY env var
    if env_val := os.environ.get("MCP_API_KEY"):
        return env_val
    # 3. Check config.json in tool directory
    try:
        from pathlib import Path
        import json
        config_path = Path(__file__).resolve().parent.parent / name / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                cfg = json.load(f)
            config_key = cfg.get("auth", {}).get("api_key") or cfg.get("api_key")
            if config_key:
                return config_key
    except Exception:
        pass
    # 4. Fallback
    if fallback:
        return fallback
    generated = f"{name}-dev-{_secrets.token_hex(16)}"
    warnings.warn(
        f"No API key configured for '{name}'. Generated random key: {generated[:12]}...",
        stacklevel=2,
    )
    return generated
