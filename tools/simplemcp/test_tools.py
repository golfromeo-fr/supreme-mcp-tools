#!/usr/bin/env python3
"""Test script for simplemcp Streamable HTTP tools.

Hits the live simplemcp server (default port 8002) and exercises the
double / square / greet tools.  Authenticated via the X-API-Key header
that simplemcp's FastMCP server expects (see tools/simplemcp/config.json).

The streamable-HTTP transport returns Server-Sent Events, so we parse the
``data:`` lines out of the response body.  The session id returned by
``initialize`` must be sent back on subsequent requests.
"""

import json
import os
import requests

BASE_URL = os.environ.get("SIMPLEMCP_URL", "http://127.0.0.1:8002/mcp")
API_KEY = os.environ.get("SIMPLEMCP_API_KEY", "simplemcp-test-key-5678")

REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "X-API-Key": API_KEY,
}

_session_id: str | None = None


def _parse_sse_json(response: requests.Response) -> dict:
    """Extract the JSON-RPC payload from a streamable-HTTP SSE response.

    Returns the parsed dict of the first ``data:`` line that is not an
    empty heartbeat (``:``).  Notifications return 202 Accepted with an
    empty body — for those we return an empty dict so the caller can
    proceed without errors.  Stores the ``mcp-session-id`` response
    header for subsequent calls.
    """
    global _session_id
    sid = response.headers.get("mcp-session-id")
    if sid:
        _session_id = sid

    if response.status_code == 202 or not response.content:
        return {}

    for raw in response.text.splitlines():
        line = raw.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        return json.loads(payload)
    raise AssertionError(
        f"No JSON-RPC payload in SSE response (status={response.status_code}, "
        f"body={response.text[:200]!r})"
    )


def _send_request(request_data: dict) -> dict:
    """POST a JSON-RPC request and return the parsed response body."""
    headers = dict(REQUEST_HEADERS)
    if _session_id is not None:
        headers["mcp-session-id"] = _session_id
    response = requests.post(BASE_URL, json=request_data, headers=headers, timeout=10)
    if response.status_code in (401, 403):
        raise AssertionError(
            f"Auth rejected by {BASE_URL}: {response.status_code} {response.text[:200]!r}"
        )
    return _parse_sse_json(response)


def test_initialize():
    """Test initialize request."""
    print("Testing initialize...")
    request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "TestClient",
                "version": "1.0.0"
            }
        },
        "id": 0
    }
    response = _send_request(request)
    print(f"Initialize response: {json.dumps(response, indent=2)}")

    assert "result" in response, f"Initialize failed: no result ({response})"
    assert "capabilities" in response["result"], "Initialize failed: no capabilities"
    assert "tools" in response["result"]["capabilities"], "Initialize failed: tools not in capabilities"
    assert _session_id, "Initialize did not return an mcp-session-id header"
    print("✅ Initialize test passed\n")

    notify_request = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }
    _send_request(notify_request)
    print("✅ Initialized notification sent\n")


def test_tools_list():
    """Test tools/list request."""
    print("Testing tools/list...")
    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    response = _send_request(request)
    print(f"Tools list response: {json.dumps(response, indent=2)}")

    assert "result" in response, f"Tools list failed: no result ({response})"
    assert "tools" in response["result"], "Tools list failed: no tools"
    tools = response["result"]["tools"]

    tool_names = [tool["name"] for tool in tools]
    assert "double" in tool_names, "double tool not found"
    assert "square" in tool_names, "square tool not found"
    assert "greet" in tool_names, "greet tool not found"

    print(f"✅ Tools list test passed - found {len(tools)} tools: {tool_names}\n")


def test_double_tool():
    """Test double tool."""
    print("Testing double tool...")
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "double",
            "arguments": {"value": 5}
        },
        "id": 2
    }
    response = _send_request(request)
    print(f"Double response: {json.dumps(response, indent=2)}")

    assert "result" in response, f"Double tool failed: no result ({response})"
    assert "content" in response["result"], "Double tool failed: no content"
    content = response["result"]["content"][0]["text"]
    assert content in ("10", "10.0"), f"Expected '10' or '10.0', got '{content}'"
    print("✅ Double tool test passed\n")


def test_square_tool():
    """Test square tool."""
    print("Testing square tool...")
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "square",
            "arguments": {"value": 4}
        },
        "id": 3
    }
    response = _send_request(request)
    print(f"Square response: {json.dumps(response, indent=2)}")

    assert "result" in response, f"Square tool failed: no result ({response})"
    assert "content" in response["result"], "Square tool failed: no content"
    content = response["result"]["content"][0]["text"]
    assert content in ("16", "16.0"), f"Expected '16' or '16.0', got '{content}'"
    print("✅ Square tool test passed\n")


def test_greet_tool():
    """Test greet tool."""
    print("Testing greet tool...")
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "greet",
            "arguments": {"name": "World", "greeting": "Hello"}
        },
        "id": 4
    }
    response = _send_request(request)
    print(f"Greet response: {json.dumps(response, indent=2)}")

    assert "result" in response, f"Greet tool failed: no result ({response})"
    assert "content" in response["result"], "Greet tool failed: no content"
    content = response["result"]["content"][0]["text"]
    assert content == "Hello, World!", f"Expected 'Hello, World!', got '{content}'"
    print("✅ Greet tool test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing simplemcp Streamable HTTP Tools")
    print("=" * 60)
    print(f"Server: {BASE_URL}")
    print(f"Auth:   X-API-Key={API_KEY[:8]}…")
    print()

    try:
        test_initialize()
        test_tools_list()
        test_double_tool()
        test_square_tool()
        test_greet_tool()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"Test failed: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
