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
    """Get the ASGI app with the correct transport.

    Handles FastMCP version differences:
    - fastmcp ≥3.3: mcp.http_app(transport=...)
    - mcp.server.fastmcp (legacy): mcp.streamable_http_app() / mcp.sse_app()

    Args:
        mcp: FastMCP server instance
        transport: "streamable-http" (default), "sse"

    Returns:
        Starlette ASGI application
    """
    transport = (transport or os.environ.get("MCP_TRANSPORT", "streamable-http")).lower()
    transport = "sse" if transport == "sse" else "streamable-http"

    if hasattr(mcp, "http_app"):
        # fastmcp ≥3.3 — unified http_app with transport param
        return mcp.http_app(transport="sse" if transport == "sse" else "http")
    elif hasattr(mcp, "streamable_http_app"):
        # legacy mcp.server.fastmcp — separate methods
        if transport == "sse":
            return mcp.sse_app()
        return mcp.streamable_http_app()
    else:
        raise RuntimeError(
            f"FastMCP server has neither http_app nor streamable_http_app: {type(mcp)}"
        )


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
