"""Server-side Function Mask enforcement (D4, 2026-09-05).

Function masks — ``disabled_tools`` in
``~/.config/supreme-mcp-tools/tools_config.json``, managed by the management
API (``PUT /api/disabled-tools/...``) and the mcp_ui Functions tab — are
enforced at the tool-server boundary: masked tools are disabled on the
FastMCP instance before the transport app is built, so MCP clients cannot see
them in ``tools/list`` and direct calls fail with "Unknown tool" (fastmcp 4
native disable semantics).

Called from each ``<name>_fastmcp.py`` entry point after tool registration;
mask changes therefore take effect at the next server/launcher start.

Tolerant by design: a missing or corrupt config file means "no masks" —
never a server-startup failure.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Same file the launcher (launcher/tools_config.py) and mcp_ui
# (mcp_ui/components/tool_settings.py) write. Collapsing these three
# constants onto one is tracked as design-pass D2 follow-up.
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "supreme-mcp-tools" / "tools_config.json"


def masked_tools(server_name: str, config_path: Path | None = None) -> list[str]:
    """Return the masked tool names for one server; [] when unset or unreadable."""
    path = Path(
        config_path
        or os.environ.get("MCP_TOOLS_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        disabled = config.get("disabled_tools", {}).get(server_name, [])
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Function masks unavailable ({path}: {e}) — serving all tools")
        return []
    return [name for name in disabled if isinstance(name, str) and name]


def apply_function_masks(mcp, server_name: str, config_path: Path | None = None) -> list[str]:
    """Disable masked tools on a FastMCP instance; returns the names disabled.

    ``config_path`` overrides the mask source (tests, alternate setups);
    default is ``$MCP_TOOLS_CONFIG_PATH`` or the standard user config.
    """
    masks = masked_tools(server_name, config_path=config_path)
    if not masks:
        return []
    if not hasattr(mcp, "disable"):
        logger.warning(
            f"[{server_name}] FastMCP has no disable(); masks inactive: {masks}"
        )
        return []
    mcp.disable(names=set(masks))
    logger.info(f"[{server_name}] Function masks enforced: {', '.join(sorted(masks))}")
    return masks


def apply_mask_at_runtime(mcp, server_name: str, tool_name: str, masked: bool) -> dict:
    """E1: toggle ONE mask on a LIVE FastMCP instance (boot-time enforcement
    stays in ``apply_function_masks``; this is the runtime half).

    File persistence is the CALLER's job and must happen FIRST (file-first
    ordering: a failed runtime toggle leaves the file authoritative, so a
    restart converges to the user's intent — never the reverse).

    Returns {"applied": bool, "reason": str | None}.
    """
    if mcp is None:
        return {"applied": False,
                "reason": "server has no FastMCP instance wired (pre-E1 or not running)"}
    if not (hasattr(mcp, "disable") and hasattr(mcp, "enable")):
        return {"applied": False, "reason": "FastMCP lacks disable()/enable()"}
    try:
        if masked:
            mcp.disable(names={tool_name})
        else:
            mcp.enable(names={tool_name})
    except Exception as e:
        logger.warning(f"[{server_name}] runtime mask toggle failed for {tool_name}: {e}")
        return {"applied": False, "reason": str(e)}
    logger.info(
        f"[{server_name}] mask {'applied' if masked else 'lifted'} at runtime: {tool_name}"
    )
    return {"applied": True, "reason": None}
