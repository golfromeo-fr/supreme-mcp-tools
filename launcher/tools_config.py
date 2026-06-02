"""
Tools Configuration Module.

Manages configuration for MCP tools including disabled tools.
This allows disabling specific tools per MCP server without modifying code.
"""

import json
import asyncio
import logging
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config directory
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "supreme-mcp-tools"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "tools_config.json"


def _ensure_config_dir() -> None:
    """Ensure the config directory exists."""
    _DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_tools_config(config_path: Path | None = None) -> dict:
    """
    Load tools configuration from JSON file.

    Args:
        config_path: Optional path to config file

    Returns:
        Dictionary with disabled_tools configuration
    """
    path = config_path or _DEFAULT_CONFIG_FILE

    if not path.exists():
        # Return default config if file doesn't exist
        return {"disabled_tools": {}, "tools": {}, "version": 1}

    try:
        with Path(path).open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"disabled_tools": {}, "tools": {}, "version": 1}


def save_tools_config(config: dict, config_path: Path | None = None) -> None:
    """
    Save tools configuration to JSON file.

    Args:
        config: Configuration dictionary to save
        config_path: Optional path to config file
    """
    path = config_path or _DEFAULT_CONFIG_FILE
    _ensure_config_dir()

    with Path(path).open('w') as f:
        json.dump(config, f, indent=2)


def get_disabled_tools(server_name: str, config_path: Path | None = None) -> list[str]:
    """
    Get list of disabled tools for a specific MCP server.

    Args:
        server_name: Name of the MCP server (e.g., 'webmcp', 'ragmcp')
        config_path: Optional path to config file

    Returns:
        List of disabled tool names
    """
    config = load_tools_config(config_path)
    return config.get("disabled_tools", {}).get(server_name, [])


def set_disabled_tools(server_name: str, disabled_list: list[str], config_path: Path | None = None) -> None:
    """
    Set disabled tools for a specific MCP server.

    Args:
        server_name: Name of the MCP server (e.g., 'webmcp', 'ragmcp')
        disabled_list: List of tool names to disable
        config_path: Optional path to config file
    """
    config = load_tools_config(config_path)
    if "disabled_tools" not in config:
        config["disabled_tools"] = {}
    config["disabled_tools"][server_name] = disabled_list
    save_tools_config(config, config_path)


def enable_tool(tool_name: str, server_name: str, config_path: Path | None = None) -> None:
    """
    Enable a specific tool for an MCP server (remove from disabled list).

    Args:
        tool_name: Name of the tool to enable
        server_name: Name of the MCP server
        config_path: Optional path to config file
    """
    disabled = get_disabled_tools(server_name, config_path)
    if tool_name in disabled:
        disabled.remove(tool_name)
        set_disabled_tools(server_name, disabled, config_path)


def disable_tool(tool_name: str, server_name: str, config_path: Path | None = None) -> None:
    """
    Disable a specific tool for an MCP server (add to disabled list).

    Args:
        tool_name: Name of the tool to disable
        server_name: Name of the MCP server
        config_path: Optional path to config file
    """
    disabled = get_disabled_tools(server_name, config_path)
    if tool_name not in disabled:
        disabled.append(tool_name)
        set_disabled_tools(server_name, disabled, config_path)


def get_all_disabled_tools(config_path: Path | None = None) -> dict[str, list[str]]:
    """
    Get all disabled tools configuration.

    Args:
        config_path: Optional path to config file

    Returns:
        Dictionary mapping server names to lists of disabled tool names
    """
    config = load_tools_config(config_path)
    return config.get("disabled_tools", {})


def get_server_tools(server_name: str, config_path: Path | None = None) -> list[str]:
    """
    Get list of known tools for a specific MCP server from config.

    Args:
        server_name: Name of the MCP server
        config_path: Optional path to config file

    Returns:
        List of tool names
    """
    config = load_tools_config(config_path)
    return config.get("tools", {}).get(server_name, [])


def set_server_tools(server_name: str, tools_list: list[str], config_path: Path | None = None) -> None:
    """
    Set the list of tools for a specific MCP server in config.

    Args:
        server_name: Name of the MCP server
        tools_list: List of tool names
        config_path: Optional path to config file
    """
    config = load_tools_config(config_path)
    if "tools" not in config:
        config["tools"] = {}
    config["tools"][server_name] = tools_list
    save_tools_config(config, config_path)


async def discover_tools_from_server(server_url: str, timeout: float = 5.0, api_key: str = None) -> list[str]:
    """
    Discover tools from an MCP server by performing an initialize handshake
    then calling tools/list via the Streamable HTTP transport.

    Uses streaming to handle SSE responses and tracks the mcp-session-id
    header for session continuity between requests.

    Args:
        server_url: URL of the MCP server (e.g., 'http://localhost:8001/mcp')
        timeout: Request timeout in seconds
        api_key: Optional API key for authentication

    Returns:
        List of tool names advertised by the server
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if api_key:
            headers["X-API-Key"] = api_key
        request_timeout = httpx.Timeout(timeout, read=timeout)
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            init_resp = await client.send(
                client.build_request(
                    "POST",
                    server_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {"name": "launcher-discovery", "version": "1.0"},
                        },
                    },
                    headers=headers,
                ),
                stream=True,
            )
            try:
                if init_resp.status_code != 200:
                    return []
                session_id = init_resp.headers.get("mcp-session-id")
                async for line in init_resp.aiter_lines():
                    if line.startswith("data: "):
                        break
            finally:
                await init_resp.aclose()

            if not session_id:
                return []

            headers["mcp-session-id"] = session_id
            tools_resp = await client.send(
                client.build_request(
                    "POST",
                    server_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    },
                    headers=headers,
                ),
                stream=True,
            )
            try:
                if tools_resp.status_code != 200:
                    return []
                async for line in tools_resp.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        tools = data.get("result", {}).get("tools", [])
                        return [t.get("name") for t in tools if t.get("name")]
                return []
            finally:
                await tools_resp.aclose()
    except Exception as e:
        logger.error(f"Error discovering tools from {server_url}: {e}")
        pass
    return []


def discover_all_tools(
    server_urls: dict[str, str],
    timeout: float = 5.0,
    auth_keys: dict[str, str] = None
) -> dict[str, list[str]]:
    """
    Discover tools from all configured MCP servers.

    Args:
        server_urls: Dictionary mapping server names to their MCP URLs
        timeout: Request timeout in seconds
        auth_keys: Optional dictionary mapping server names to API keys

    Returns:
        Dictionary mapping server names to lists of tool names
    """
    _auth_keys = auth_keys or {}

    async def _discover():
        results = {}
        for server_name, url in server_urls.items():
            tools = await discover_tools_from_server(url, timeout, api_key=_auth_keys.get(server_name))
            results[server_name] = tools
        return results

    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context - run in a thread to get a fresh event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _discover())
                return future.result()
        else:
            return asyncio.run(_discover())
    except Exception as e:
        logger.error(f"Error discovering all tools: {e}")
        return {}


def update_config_with_discovered_tools(
    server_urls: dict[str, str],
    config_path: Path | None = None,
    timeout: float = 5.0,
    auth_keys: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """
    Discover tools from all servers and update the config file.

    Args:
        server_urls: Dictionary mapping server names to their MCP URLs
        config_path: Optional path to config file
        timeout: Request timeout in seconds
        auth_keys: Optional dictionary mapping server names to API keys

    Returns:
        Dictionary mapping server names to lists of discovered tool names
    """
    discovered = discover_all_tools(server_urls, timeout, auth_keys=auth_keys)

    config = load_tools_config(config_path)
    if "tools" not in config:
        config["tools"] = {}

    for server_name, tools in discovered.items():
        config["tools"][server_name] = tools

    save_tools_config(config, config_path)
    return discovered


def validate_and_cleanup_config(config_path: Path | None = None) -> dict[str, list[str]]:
    """
    Validate the config and clean up:
    - Remove disabled tools that no longer exist in the server's tool list
    - Remove tools entries for servers that are no longer configured

    Args:
        config_path: Optional path to config file

    Returns:
        Dictionary with validation results
    """
    config = load_tools_config(config_path)
    results = {"removed_invalid": [], "servers_updated": []}

    tools = config.get("tools", {})
    disabled = config.get("disabled_tools", {})

    # Clean up disabled tools that don't exist in tools list
    for server_name, disabled_list in list(disabled.items()):
        server_tools = tools.get(server_name, [])
        if server_tools:
            invalid = [t for t in disabled_list if t not in server_tools]
            if invalid:
                disabled[server_name] = [t for t in disabled_list if t in server_tools]
                results["removed_invalid"].extend(invalid)
                results["servers_updated"].append(server_name)

    config["disabled_tools"] = disabled
    save_tools_config(config, config_path)

    return results


def filter_tools_by_disabled(
    tools: list[dict],
    server_name: str,
    config_path: Path | None = None
) -> list[dict]:
    """
    Filter a list of tools, removing disabled ones.

    Args:
        tools: List of tool dictionaries with 'name' key
        server_name: Name of the MCP server
        config_path: Optional path to config file

    Returns:
        Filtered list of tools with disabled ones removed
    """
    disabled = set(get_disabled_tools(server_name, config_path))
    return [tool for tool in tools if tool.get("name") not in disabled]
