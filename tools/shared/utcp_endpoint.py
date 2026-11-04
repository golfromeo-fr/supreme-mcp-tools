"""UTCP endpoint — a fourth client dialect on every tool server (2026-11-04).

UTCP (Universal Tool Calling Protocol, RFC 1.0.0) is a meta-protocol: a
client reads a JSON "manual" at GET /utcp and calls tools directly over
plain HTTP/JSON at POST /utcp/call/<tool> — no MCP JSON-RPC, no sessions,
same port, same auth stack.

Wire format verified against the reference client source
(universal-tool-calling-protocol/python-utcp, utcp-http 1.x, PyPI utcp
1.2.0), not the marketing docs:

- Manual: {"utcp_version", "manual_version", "tools": [...]}.
- Tool: name, description, inputs/outputs (JSON schema), tool_call_template.
- HttpCallTemplate.body_field defaults to "body": the reference client only
  sends a JSON body when the arg named body_field exists — without it, flat
  tool arguments degrade to URL query parameters. So each Tool's inputs is
  declared as {"body": <real schema>} and this route unwraps
  payload["body"]. Flat JSON bodies are also accepted (hand-rolled clients
  that just POST the arguments dict).

Auth: the app-wide auth stack (ApiKeyFallbackMiddleware + the tool's
verifier) covers these routes like every other dialect. The manual's call
templates reference a client-side variable (${UTCP_<NAME>_API_KEY}) instead
of carrying the real key; UTCP clients resolve variables from their own
config/env at call time.

Identity/masks (E3): these routes bypass fastmcp's MCP middleware chain
(the gate's get_http_request() ContextVar is unset on plain parent-app
routes), so the handlers close over `mcp` and re-implement the gate
semantics themselves: per-user masked tools are dropped from the manual and
rejected with the same "Unknown tool" (404, no existence leak) as
IdentityGateMiddleware. Global E1 masks need no code here — they are
fastmcp-native disables, so list_tools()/get_tool() already exclude them.

Knobs: MCP_DISABLE_UTCP_ENDPOINT=1 (env) or "utcp": false in
tools/<name>/config.json suppress the routes; default is ON — serving
another dialect is consistent with the multi-transport default (one app,
every client dialect, selection by URL).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Knob ──────────────────────────────────────────────────────────────────

def utcp_enabled(server_name: str | None, config_dir=None) -> bool:
    """UTCP routes serve by default; suppressed by env or per-tool config.

    A per-tool "utcp": false key in tools/<name>/config.json always
    suppresses; otherwise the MCP_DISABLE_UTCP_ENDPOINT env (=1/true/yes)
    suppresses; default enabled. Unlike the transport knob there is no
    config-side opt-IN: "utcp": true does not override the env kill-switch
    (env stays a global off switch). Invalid config values fail loudly (a typo should not silently
    flip the endpoint on or off). ``config_dir`` is a test seam overriding
    the tools/ directory the per-tool config is read from.
    """
    base = Path(config_dir) if config_dir else Path(__file__).resolve().parent.parent
    if server_name:
        try:
            cfg = json.loads(
                (base / server_name / "config.json").read_text(encoding="utf-8")
            )
            value = cfg.get("utcp")
            if value is not None and not isinstance(value, bool):
                raise ValueError(
                    f"tools/{server_name}/config.json: \"utcp\" must be a "
                    f"boolean, got {value!r}"
                )
            if value is False:
                return False
        except FileNotFoundError:
            pass
        except json.JSONDecodeError as e:
            raise ValueError(
                f"tools/{server_name}/config.json is not valid JSON: {e}"
            ) from e
    return os.environ.get("MCP_DISABLE_UTCP_ENDPOINT", "").strip() not in ("1", "true", "yes")


# ── Manual construction ───────────────────────────────────────────────────


def _var_name(server_name: str) -> str:
    """Client-side variable for the tool's API key, UTCP-style ${VAR}."""
    return "UTCP_" + server_name.upper().replace("-", "_") + "_API_KEY"


def _strip_vendor_keys(schema):
    """Drop fastmcp's internal markers (x-fastmcp-*) from a published schema.

    JSON Schema ignores unknown keywords, so they are harmless to clients —
    but the manual is a spec-facing artifact and shouldn't carry the
    implementation's vendor extensions.
    """
    if isinstance(schema, dict):
        return {k: _strip_vendor_keys(v) for k, v in schema.items()
                if not k.startswith("x-fastmcp-")}
    if isinstance(schema, list):
        return [_strip_vendor_keys(v) for v in schema]
    return schema


def _manual_tool(tool, base_url: str, server_name: str) -> dict:
    """One UTCP Tool object from a fastmcp Tool (schema-bearing attributes)."""
    real_schema = tool.parameters or {"type": "object", "properties": {}}
    inputs = {"type": "object", "properties": {"body": real_schema}}
    # Require the wrapper only when the real tool has required args — an
    # all-optional/zero-arg tool stays callable with an empty request.
    if real_schema.get("required"):
        inputs["required"] = ["body"]
    auth = {
        "auth_type": "api_key",
        "api_key": "${" + _var_name(server_name) + "}",
        "var_name": "X-API-Key",
        "location": "header",
    }
    url = f"{base_url}/utcp/call/{urllib.parse.quote(tool.name, safe='')}"
    template = {
        "call_template_type": "http",
        "url": url,
        "http_method": "POST",
        "content_type": "application/json",
        "body_field": "body",
        "auth": auth,
    }
    return {
        "name": tool.name,
        "description": tool.description or "",
        "inputs": inputs,
        "outputs": _strip_vendor_keys(tool.output_schema) or {"type": "object"},
        "tool_call_template": template,
    }


# ── Identity/masks (E3 gate semantics, route edition) ─────────────────────


def _resolve_masked(mcp, request) -> set[str]:
    """Per-user masked set for this request, mirroring the E3 gate.

    Mirrors IdentityGateMiddleware._masked: None entry or admin role means
    no filtering (mono fail-open; user-store entries are role-stamped, so
    the "user" default in resolve_role only guards malformed maps). Uses
    resolve_identity + resolve_role — the CR1-sanctioned single role
    default, never a local re-derivation.
    """
    from tools.shared.identity import resolve_identity, resolve_role

    verifier = getattr(mcp, "auth", None)
    if verifier is None or not hasattr(verifier, "tokens"):
        return set()
    identity = resolve_identity(request, verifier.tokens)
    if identity is None:
        if os.environ.get("MCP_REQUIRE_IDENTITY", "").strip() == "1":
            raise PermissionError("identity required")
        return set()
    _client_id, entry = identity
    if resolve_role(entry, bool(getattr(verifier, "is_user_store", False))) != "admin":
        return set(entry.get("masked") or [])
    return set()


async def _manual_response(mcp, request):
    """GET /utcp handler body: the UTCP manual, filtered for the caller."""
    from starlette.responses import JSONResponse

    user = getattr(request, "user", None)
    if user is None or not getattr(user, "is_authenticated", False):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        masked = _resolve_masked(mcp, request)
    except PermissionError:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    tools = await mcp.list_tools(run_middleware=False)
    # Global masks are fastmcp-native disables: already absent from this
    # list. Per-user masks (E3) are filtered here at route level, mirroring
    # IdentityGateMiddleware.on_list_tools.
    visible = [t for t in tools if t.name not in masked]
    server_name = getattr(mcp, "name", "mcp")
    base_url = str(request.base_url).rstrip("/")
    manual = {
        "utcp_version": "1.0",
        "manual_version": "1.0",
        "tools": [_manual_tool(t, base_url, server_name) for t in visible],
    }
    return JSONResponse(manual)


# ── Tool-call route ───────────────────────────────────────────────────────


def _split_body(payload) -> dict:
    """Tool arguments from the request body.

    Accepts {"body": {...}} — the shape the reference client sends, per
    HttpCallTemplate.body_field — or the flat args dict for hand-rolled
    clients that just POST the arguments.
    """
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    body = payload.get("body")
    if isinstance(body, dict):
        return body
    if "body" in payload:
        raise ValueError('"body" must be an object of tool arguments')
    return payload




def _result_body(result) -> dict:
    """JSON body for a successful call: structured content when the tool
    declares an output schema, else the text content under {"result": ...}."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    parts = [c.text for c in getattr(result, "content", [])
             if getattr(c, "text", None) is not None]
    return {"result": "\n".join(parts)}


async def _call_response(mcp, request):
    """POST /utcp/call/{tool_name} handler body: plain-HTTP tool execution.

    Identity/mask checks mirror IdentityGateMiddleware: a masked or unknown
    tool answers 404 "Unknown tool" (no existence leak); validation errors
    are 400; tool failures are 500 with {"error": ...} — the reference UTCP
    client surfaces non-2xx bodies in its raised error message.
    """
    from starlette.responses import JSONResponse

    user = getattr(request, "user", None)
    if user is None or not getattr(user, "is_authenticated", False):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    name = request.path_params["tool_name"]
    try:
        masked = _resolve_masked(mcp, request)
    except PermissionError:
        # Same 401 as the manual route (spec: no identity + require_identity
        # → 401); the no-existence-leak 404 below applies to tool resolution
        # only. Near-unreachable in practice: the outer auth middleware
        # rejects unknown tokens before this handler runs.
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if name in masked:
        return JSONResponse({"error": "Unknown tool"}, status_code=404)

    try:
        tool = await mcp.get_tool(name)
    except Exception:
        tool = None
    if tool is None:
        return JSONResponse({"error": "Unknown tool"}, status_code=404)

    try:
        payload = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"invalid JSON body: {e}"}, status_code=400)
    try:
        args = _split_body(payload)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    from fastmcp.exceptions import NotFoundError, ToolError, ValidationError

    try:
        result = await mcp.call_tool(name, args, run_middleware=False)
    except NotFoundError:
        return JSONResponse({"error": "Unknown tool"}, status_code=404)
    except (ValidationError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except ToolError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:  # noqa: BLE001 — the route reports, never raises
        # fastmcp wraps tool exceptions with their text ("Error calling tool
        # ...") — the same exposure the MCP dialects have; parity kept.
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    return JSONResponse(_result_body(result))


# ── Wiring ────────────────────────────────────────────────────────────────


def wire_utcp(app, mcp) -> int:
    """Append the UTCP routes to a built ASGI app (any transport mode).

    Same flatten-onto-the-parent pattern as the flush-sessions route in
    server_factory. Returns the number of routes added (0 when the mcp
    object lacks the fastmcp tool API — never a startup failure).
    """
    if not (hasattr(mcp, "list_tools") and hasattr(mcp, "get_tool")
            and hasattr(mcp, "call_tool")):
        logger.warning(
            "[%s] UTCP endpoints skipped: object lacks the fastmcp tool API",
            getattr(mcp, "name", "mcp"),
        )
        return 0

    from starlette.routing import Route

    async def manual(request):
        return await _manual_response(mcp, request)

    async def call_tool(request):
        return await _call_response(mcp, request)

    routes = list(app.router.routes)
    routes.append(Route("/utcp", manual, methods=["GET"]))
    routes.append(Route("/utcp/call/{tool_name}", call_tool, methods=["POST"]))
    app.router.routes = routes
    logger.info(
        "[%s] UTCP endpoints wired: GET /utcp + POST /utcp/call/{tool}",
        getattr(mcp, "name", "mcp"),
    )
    return 2
