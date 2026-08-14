"""
FastMCP 3.x Server Factory — supreme-mcp-tools
==============================================
Centralized factory for all MCP tool servers.

Key design decisions for FastMCP 3.x (≥3.3):
- Uses DualHeaderVerifier (TokenVerifier subclass) — no OAuth endpoints
  advertised, no /.well-known/oauth-* routes, no WWW-Authenticate 401.
- Custom DualHeaderAuthBackend accepts BOTH X-API-Key (VS Code Copilot) and
  Authorization: Bearer (standard OAuth clients).
- Factory pattern: one call creates server, adds tools, and configures auth.
- No dependencies on launcher modules — runs standalone.

Usage in each *_fastmcp.py:

    from tools.shared.server_factory import create_fastmcp_server, get_transport_app, DEFAULT_HOST

    mcp = create_fastmcp_server("webmcp")
    app = get_transport_app(mcp)
    uvicorn.run(app, host=DEFAULT_HOST, port=MCP_PORT)
"""

from __future__ import annotations

import os
import hmac
import secrets as _secrets
import warnings
from typing import TYPE_CHECKING, Any

# ── Single source of truth for bind host ──────────────────────────────────
# On Debian/Linux, "::" binds IPv6-only despite bindv6only=0, breaking IPv4
# clients (VS Code Copilot). Use "0.0.0.0" which accepts both 127.0.0.1 and
# localhost (::1 → kernel maps to IPv4).
DEFAULT_HOST = "0.0.0.0"

if TYPE_CHECKING:
    from fastmcp import FastMCP

# ── FastMCP 3.x auth imports ──────────────────────────────────────────────
from starlette.authentication import AuthCredentials, AuthenticationBackend
from starlette.requests import HTTPConnection
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware

from fastmcp.server.auth import TokenVerifier, AccessToken
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser


class _AuthenticatedUser(AuthenticatedUser):
    """AuthenticatedUser subclass for DualHeaderAuthBackend.

    Extends the MCP SDK's AuthenticatedUser so that RequireAuthMiddleware's
    isinstance(user, AuthenticatedUser) check passes for X-API-Key auth too.
    """

    def __init__(self, auth_info) -> None:
        super().__init__(auth_info)


class DualHeaderAuthBackend(AuthenticationBackend):
    """
    Starlette AuthenticationBackend that accepts both X-API-Key and Bearer token.

    VS Code Copilot sends X-API-Key header.
    Other clients send Authorization: Bearer token.
    Both are validated via the TokenVerifier's verify_token().
    """

    def __init__(self, token_verifier: TokenVerifier) -> None:
        self.token_verifier = token_verifier

    async def authenticate(
        self, conn: HTTPConnection
    ) -> tuple[AuthCredentials, _AuthenticatedUser] | None:
        # Primary: X-API-Key header (VS Code Copilot)
        api_key = conn.headers.get("x-api-key")
        if api_key:
            auth_info = await self.token_verifier.verify_token(api_key)
            if auth_info:
                return (
                    AuthCredentials(auth_info.scopes),
                    _AuthenticatedUser(auth_info),
                )

        # Fallback: Authorization: Bearer <token>
        auth_header = conn.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:]
            auth_info = await self.token_verifier.verify_token(token)
            if auth_info:
                return (
                    AuthCredentials(auth_info.scopes),
                    _AuthenticatedUser(auth_info),
                )

        return None


class DualHeaderVerifier(TokenVerifier):
    """
    FastMCP 3.x TokenVerifier that accepts both X-API-Key and Bearer headers.

    Replaces the old StaticTokenVerifier (removed in FastMCP 3.x).
    verify_token() checks a static token dict — no remote validation, no OAuth.
    get_middleware() installs DualHeaderAuthBackend instead of the default
    Bearer-only backend, enabling dual-header authentication.
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

    def get_middleware(self) -> list[Middleware]:
        """Return middleware using DualHeaderAuthBackend (not default Bearer-only)."""
        return [
            Middleware(
                AuthenticationMiddleware, backend=DualHeaderAuthBackend(self)
            ),
            Middleware(AuthContextMiddleware),
        ]


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


def get_transport_app(mcp, transport: str | None = None):
    """Get the ASGI app for the streamable-http transport.

    SSE support was removed (2026-08, plans/mcp-2026-07-28-stateless-upgrade.md
    Phase -1): ``MCP_TRANSPORT=sse`` raises instead of silently switching.

    Also wires session management on the returned app:
    - ``POST /admin/flush-sessions`` — terminate all in-memory sessions so stale
      clients get HTTP 404 and re-initialize. Reuses the app-wide auth (same
      tool API key). Enabled unless ``MCP_DISABLE_FLUSH_ENDPOINT`` is set.
    - A session idle TTL (``MCP_SESSION_IDLE_TIMEOUT`` seconds, default 1800) so
      stale sessions self-evict instead of accumulating until process restart.
      FastMCP's ``http_app()`` doesn't expose this kwarg, so it is set on the
      session manager right after the lifespan creates it.

    Args:
        mcp: FastMCP server instance
        transport: optional override; only "streamable-http" (or "http") is
            accepted — anything else, notably "sse", raises.

    Returns:
        Starlette ASGI application
    """
    transport = (transport or os.environ.get("MCP_TRANSPORT", "streamable-http")).lower()
    if transport == "sse":
        raise ValueError(
            "SSE transport was removed (plans/mcp-2026-07-28-stateless-upgrade.md, "
            "Phase -1). Unset MCP_TRANSPORT or set it to 'streamable-http'."
        )
    if transport not in ("streamable-http", "http"):
        raise ValueError(
            f"Unsupported MCP_TRANSPORT {transport!r}: only 'streamable-http' is supported"
        )

    if not hasattr(mcp, "http_app"):
        raise RuntimeError(f"FastMCP server has no http_app method: {type(mcp)}")

    app = mcp.http_app(transport="http")
    _wire_session_management(app)
    return app


def _wire_session_management(app) -> None:
    """Attach the flush-sessions admin route and the session idle TTL.

    The session manager is created inside FastMCP's lifespan, so the idle TTL
    is applied by wrapping that lifespan to run after the manager exists.

    The flush route authenticates via the app-wide AuthenticationMiddleware
    (DualHeaderAuthBackend) — the same tool API key that protects /mcp.
    """
    if os.environ.get("MCP_DISABLE_FLUSH_ENDPOINT"):
        _apply_session_idle_timeout_via_lifespan(app)
        return

    try:
        from starlette.routing import Route
        from starlette.responses import JSONResponse
    except ImportError:
        _apply_session_idle_timeout_via_lifespan(app)
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

    _apply_session_idle_timeout_via_lifespan(app)


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


def _apply_session_idle_timeout_via_lifespan(app) -> None:
    """Set session_idle_timeout on the session manager once the lifespan creates it.

    FastMCP's http_app() does not expose session_idle_timeout, and the manager is
    instantiated inside the lifespan — so wrap the lifespan context factory to
    set the attribute after the original startup runs. The reaper reads this
    attribute live, so the value takes effect for new sessions immediately.
    """
    raw_timeout = os.environ.get("MCP_SESSION_IDLE_TIMEOUT", "1800")
    try:
        idle_timeout = float(raw_timeout)
        if idle_timeout <= 0:
            idle_timeout = None
    except (TypeError, ValueError):
        idle_timeout = None
    if idle_timeout is None:
        return

    router = getattr(app, "router", None)
    original_lifespan = getattr(router, "lifespan_context", None) if router else None
    if original_lifespan is None:
        return

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan_with_ttl(scoped_app):
        # Original lifespan creates + runs the session manager; once it has
        # yielded (startup complete), the manager exists and we can patch it.
        async with original_lifespan(scoped_app):
            sm = _get_session_manager(app)
            if sm is not None and getattr(sm, "session_idle_timeout", None) is None:
                try:
                    sm.session_idle_timeout = idle_timeout
                except Exception:
                    pass
            yield

    # Starlette's Router uses lifespan_context at startup.
    try:
        router.lifespan_context = lifespan_with_ttl
    except Exception:
        pass


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
