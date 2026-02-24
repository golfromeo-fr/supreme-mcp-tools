#!/usr/bin/env python3
"""Test script for simplemcp8 Streamable HTTP tools."""

import json
import requests
import time

BASE_URL = "http://localhost:8005/mcp"

def send_request(request_data):
    """Send a JSON-RPC request to the server."""
    response = requests.post(BASE_URL, json=request_data)
    return response.json()

def test_initialize():
    """Test initialize request."""
    print("Testing initialize...")
    request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "TestClient",
                "version": "1.0.0"
            }
        },
        "id": 0
    }
    response = send_request(request)
    print(f"Initialize response: {json.dumps(response, indent=2)}")
    
    # Check if response is valid
    assert "result" in response, "Initialize failed: no result"
    assert "capabilities" in response["result"], "Initialize failed: no capabilities"
    assert "tools" in response["result"]["capabilities"], "Initialize failed: tools not in capabilities"
    print("✅ Initialize test passed\n")
    
    # Send initialized notification
    notify_request = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }
    send_request(notify_request)
    print("✅ Initialized notification sent\n")

def test_tools_list():
    """Test tools/list request."""
    print("Testing tools/list...")
    request = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    response = send_request(request)
    print(f"Tools list response: {json.dumps(response, indent=2)}")
    
    # Check if tools are present
    assert "result" in response, "Tools list failed: no result"
    assert "tools" in response["result"], "Tools list failed: no tools"
    tools = response["result"]["tools"]
    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"
    
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
    response = send_request(request)
    print(f"Double response: {json.dumps(response, indent=2)}")
    
    assert "result" in response, "Double tool failed: no result"
    assert "content" in response["result"], "Double tool failed: no content"
    content = response["result"]["content"][0]["text"]
    assert content == "10", f"Expected '10', got '{content}'"
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
    response = send_request(request)
    print(f"Square response: {json.dumps(response, indent=2)}")
    
    assert "result" in response, "Square tool failed: no result"
    assert "content" in response["result"], "Square tool failed: no content"
    content = response["result"]["content"][0]["text"]
    assert content == "16", f"Expected '16', got '{content}'"
    print("✅ Square tool test passed\n")

def test_greet_tool():
    """Test greet tool."""
    print("Testing greet tool...")
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "World",
            "greeting": "Hello"
        },
        "id": 4
    }
    response = send_request(request)
    print(f"Greet response: {json.dumps(response, indent=2)}")
    
    assert "result" in response, "Greet tool failed: no result"
    assert "content" in response["result"], "Greet tool failed: no content"
    content = response["result"]["content"][0]["text"]
    assert content == "Hello, World!", f"Expected 'Hello, World!', got '{content}'"
    print("✅ Greet tool test passed\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing simplemcp8 Streamable HTTP Tools")
    print("=" * 60)
    print(f"Server: {BASE_URL}\n")
    
    try:
        test_initialize()
        test_tools_list()
        test_double_tool()
        test_square_tool()
        test_greet_tool()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
