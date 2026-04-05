#!/usr/bin/env python3
"""
Simple test client to call simplemcp and see stats in web UI.

Usage:
    python -m launcher.test_simplemcp_client list    # List available tools
    python -m launcher.test_simplemcp_client call     # Call a tool (greet)
    python -m launcher.test_simplemcp_client loop N   # Loop N times to generate traffic
    python -m launcher.test_simplemcp_client secret   # Read SIMPLEMCP_SECRET value
    python -m launcher.test_simplemcp_client watch N  # Poll get_secret every N seconds to test hot-reload
"""

import argparse
import asyncio
import httpx
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from launcher.launcher_config import load_ports_config


def get_server_url(server_name: str) -> str:
    """Get MCP server URL from ports config."""
    config = load_ports_config()
    port = config.get("assignments", {}).get("mcp", {}).get(server_name)
    if not port:
        raise ValueError(f"Server '{server_name}' not found in ports.json")
    return f"http://localhost:{port}/mcp"


SIMPLEMCP_URL = get_server_url("simplemcp")

# API key for authentication
_api_key = None


def _load_tool_api_key(tool_name: str) -> str | None:
    """Auto-load API key from the tool's config.json."""
    config_path = Path(__file__).parent.parent / "tools" / tool_name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("auth", {}).get("api_key")
    return None


def _get_headers() -> dict:
    """Get common headers including API key if configured."""
    headers = {"Content-Type": "application/json"}
    if _api_key:
        headers["X-API-Key"] = _api_key
    return headers


async def list_tools():
    """Call tools/list on simplemcp."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }
        response = await client.post(SIMPLEMCP_URL, json=payload, headers=_get_headers())
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        return result


async def call_tool(tool_name: str = "greet", arguments: dict = None):
    """Call a tool on simplemcp."""
    if arguments is None:
        arguments = {"name": "World"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 2
        }
        response = await client.post(SIMPLEMCP_URL, json=payload, headers=_get_headers())
        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))
        return result


async def loop(count: int):
    """Call tools/list and call repeatedly to generate traffic."""
    print(f"Running {count} iterations...")
    for i in range(count):
        # List tools
        await list_tools()
        await asyncio.sleep(0.1)
        # Call a tool
        await call_tool()
        await asyncio.sleep(0.1)
        print(f"Iteration {i+1}/{count} complete")
    print("Done!")


async def get_secret():
    """Call get_secret tool to read the current SIMPLEMCP_SECRET value."""
    result = await call_tool("get_secret", {})
    if result and "result" in result:
        content = result["result"].get("content", [])
        for item in content:
            if item.get("type") == "text":
                print(f"\n{item['text']}")
    return result


async def watch(interval: int = 5):
    """Poll get_secret repeatedly to watch for hot-reload changes.

    Update SIMPLEMCP_SECRET via the WebUI while this runs to see changes appear.
    Press Ctrl+C to stop.
    """
    import time
    print(f"Watching SIMPLEMCP_SECRET every {interval}s (Ctrl+C to stop)")
    print("Tip: Update the value via the WebUI while this runs!\n")
    try:
        while True:
            ts = time.strftime("%H:%M:%S")
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "get_secret", "arguments": {}},
                    "id": 99,
                }
                response = await client.post(SIMPLEMCP_URL, json=payload, headers=_get_headers())
                data = response.json()

            if "result" in data:
                content = data["result"].get("content", [])
                text = content[0]["text"] if content else "(no response)"
                # Extract just the key info
                for line in text.split("\n"):
                    if "Length:" in line or "Is Set:" in line:
                        print(f"  [{ts}] {line.strip()}")
            else:
                print(f"  [{ts}] Error: {data}")

            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    global _api_key

    parser = argparse.ArgumentParser(description="Test client for simplemcp")
    parser.add_argument(
        "action",
        choices=["list", "call", "loop", "secret", "watch"],
        help="Action to perform",
    )
    parser.add_argument("count", nargs="?", type=int, default=5, help="Loop count or watch interval in seconds")
    parser.add_argument("--api-key", default=None, help="Override API key (auto-loaded from config.json by default)")

    args = parser.parse_args()

    _api_key = args.api_key or _load_tool_api_key("simplemcp")
    if _api_key:
        print(f"Using API key: {_api_key[:8]}...")

    if args.action == "list":
        asyncio.run(list_tools())
    elif args.action == "call":
        asyncio.run(call_tool())
    elif args.action == "loop":
        asyncio.run(loop(args.count))
    elif args.action == "secret":
        asyncio.run(get_secret())
    elif args.action == "watch":
        asyncio.run(watch(args.count))


if __name__ == "__main__":
    main()
