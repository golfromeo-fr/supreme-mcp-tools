"""E3/M1 — caller identity resolution + per-user visibility gate.

Identity model (spike-verified, plans/e3-m1-identity-spike-2026-09-07.md):
requests reach middleware only AFTER the auth layer accepted the token, so
resolving the raw token through the SAME map the verifier checked is exact.

Token-map entry shape (built by users_store.tokens_map_for_tool or the
single-key legacy map):
    {"client_id": str, "role": "admin" | "user", "masked": list[str]}

Gate semantics:
- role admin bypasses USER masks; E1 GLOBAL masks (fastmcp disable) always
  apply to everyone, admin included.
- identity absent (non-HTTP scope, e.g. in-memory Client) → fail-OPEN with a
  log line, unless require_identity and the request arrived over HTTP —
  then the call is rejected with "Unknown tool" semantics (no existence leak).
"""

import hmac
import logging
import os
from typing import Callable

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext

logger = logging.getLogger(__name__)


def resolve_identity(request, tokens_map_fn: Callable[[], dict]) -> tuple[str, dict] | None:
    """(client_id, entry) from the request's Bearer (fallback X-API-Key)
    token, looked up in ``tokens_map_fn()``. None when there is no HTTP
    scope, no usable header, or the token is unknown. Never raises."""
    try:
        auth = request.headers.get("authorization", "") or ""
        token = auth[7:] if auth.lower().startswith("bearer ") else None
        if token is None:
            token = request.headers.get("x-api-key")
        if not token:
            return None
        entry = tokens_map_fn().get(token)
        if entry is None:
            return None
        return entry.get("client_id", "unknown"), entry
    except Exception as e:  # resolver must never break a request
        logger.warning(f"identity resolution failed: {type(e).__name__}: {e}")
        return None


def current_identity(tokens_map_fn: Callable[[], dict]) -> tuple[str, dict] | None:
    """Identity of the in-flight request (for tool functions, E4 tx stamping).
    None outside an HTTP scope."""
    try:
        request = get_http_request()
    except Exception:
        return None
    return resolve_identity(request, tokens_map_fn)


class IdentityGateMiddleware(Middleware):
    """Per-user visibility on the LIVE server.

    on_list_tools / on_discover: drop masked tools from the result.
    on_call_tool: reject masked tools with "Unknown tool" (no existence leak).
    """

    def __init__(
        self,
        tokens_map_fn: Callable[[], dict],
        require_identity: bool | None = None,
    ) -> None:
        self._tokens_map_fn = tokens_map_fn
        if require_identity is None:
            require_identity = os.environ.get("MCP_REQUIRE_IDENTITY", "").strip() == "1"
        self._require_identity = require_identity

    # -- internals -------------------------------------------------------
    def _resolve_for_request(self) -> tuple[tuple[str, dict] | None, bool]:
        """((client_id, entry) | None, over_http)."""
        try:
            request = get_http_request()
        except Exception:
            return None, False  # non-HTTP scope (in-memory client)
        return resolve_identity(request, self._tokens_map_fn), True

    def _entry(self, context) -> dict | None:
        ident, over_http = self._resolve_for_request()
        if ident is None:
            if over_http and self._require_identity:
                raise ToolError("Unknown tool")
            logger.debug("identity absent — fail-open visibility")
            return None
        return ident[1]

    def _masked(self, entry: dict | None) -> set[str]:
        if entry is None or entry.get("role") == "admin":
            return set()
        return set(entry.get("masked") or [])

    # -- hooks -------------------------------------------------------------
    async def on_list_tools(self, context, call_next):
        entry = self._entry(context)
        tools = await call_next(context)
        masked = self._masked(entry)
        if not masked:
            return tools
        return [t for t in tools if t.name not in masked]

    async def on_discover(self, context, call_next):
        entry = self._entry(context)
        result = await call_next(context)
        masked = self._masked(entry)
        if not masked:
            return result
        if isinstance(result, dict):  # legacy dict shape
            return result
        try:
            return [t for t in result if t.name not in masked]
        except TypeError:
            return result

    async def on_call_tool(self, context, call_next):
        entry = self._entry(context)
        masked = self._masked(entry)
        name = getattr(context.message, "name", None)
        if masked and name in masked:
            raise ToolError("Unknown tool")
        return await call_next(context)
