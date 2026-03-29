#!/usr/bin/env python3
"""
Simple test client to call simplemcp and see stats in web UI.

Usage:
    python -m launcher.test_simplemcp_client list    # List available tools
    python -m launcher.test_simplemcp_client call     # Call a tool (greet)
    python -m launcher.test_simplemcp_client loop N   # Loop N times to generate traffic
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


async def list_tools():
    """Call tools/list on simplemcp."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }
        response = await client.post(SIMPLEMCP_URL, json=payload)
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
        response = await client.post(SIMPLEMCP_URL, json=payload)
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


def main():
    parser = argparse.ArgumentParser(description="Test client for simplemcp")
    parser.add_argument("action", choices=["list", "call", "loop"], help="Action to perform")
    parser.add_argument("count", nargs="?", type=int, default=5, help="Loop count (for 'loop' action)")

    args = parser.parse_args()

    if args.action == "list":
        asyncio.run(list_tools())
    elif args.action == "call":
        asyncio.run(call_tool())
    elif args.action == "loop":
        asyncio.run(loop(args.count))


if __name__ == "__main__":
    main()
