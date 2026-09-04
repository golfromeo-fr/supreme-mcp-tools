"""
OAuth Suppression Fix for FastMCP 2.14+
=======================================
STATUS: SUPERSEDED — kept for reference only. Zero importers since the
FastMCP 4 auth redesign: no FastMCP instance exposes OAuth discovery routes
anymore (server_factory attaches the DualHeaderVerifier token-verifier stack
instead of OAuth endpoints), so there is nothing left to suppress. Do not
call apply_oauth_fix; if OAuth metadata ever reappears on the well-known
endpoints, fix it at the server_factory auth wiring, not here.

Historical context: FastMCP 2.14+ auto-exposed
/.well-known/oauth-authorization-server and
/.well-known/oauth-protected-resource. VS Code Copilot's MCP client probes
these endpoints — when it gets a 200 with OAuth metadata, it enters OAuth flow
and ignores configured headers (X-API-Key, Authorization, etc.) entirely.

This module provided a shared fix that suppressed those endpoints by
registering custom 404 routes on any FastMCP instance.

Former usage (in each *_fastmcp.py):
    from tools.shared.oauth_fix import apply_oauth_fix
    apply_oauth_fix(mcp)

Was applied to: webmcp, ragmcp, convertermcp, memorymcp, simplemcp, oraclemcp
"""

from starlette.requests import Request
from starlette.responses import Response
from fastmcp import FastMCP


def apply_oauth_fix(mcp: FastMCP) -> None:
    """
    Register OAuth discovery suppression routes on a FastMCP instance.

    Both /.well-known/oauth-authorization-server and
    /.well-known/oauth-protected-resource are mapped to return 404, so
    VS Code Copilot skips OAuth discovery and uses the configured headers.
    """
    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def suppress_oauth_as(request: Request) -> Response:
        return Response(status_code=404)

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def suppress_oauth_pr(request: Request) -> Response:
        return Response(status_code=404)