#!/usr/bin/env python3
"""
Example Streamable HTTP Client for Testing

This is a simple example client that demonstrates how to use the
Streamable HTTP client utilities to communicate with a Streamable HTTP server.

Usage:
    python example_client.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from launcher.streamable_http import StreamableHttpClient, ClientConfig
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies:")
    print("  pip install httpx")
    sys.exit(1)


async def main():
    """Main client logic."""
    # Create client configuration
    config = ClientConfig(
        base_url="http://localhost:8000",
        endpoint="/mcp",
        request_timeout=30.0,
    )
    
    # Create client
    client = StreamableHttpClient(config)
    
    # Register event handlers
    def on_connected():
        print("✓ Connected to server")
    
    def on_disconnected():
        print("✓ Disconnected from server")
    
    def on_connection_failed(error):
        print(f"✗ Connection failed: {error}")
    
    client.on("connected", on_connected)
    client.on("disconnected", on_disconnected)
    client.on("connection_failed", on_connection_failed)
    
    try:
        # Connect to server
        print("Connecting to server...")
        await client.connect()
        print()
        
        # List available tools
        print("Listing available tools...")
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        print()
        
        # Test echo tool
        print("Testing 'echo' tool...")
        async for response in client.call_tool("echo", {"text": "Hello, Streamable HTTP!"}):
            if "result" in response:
                for content in response["result"].get("content", []):
                    print(f"  Response: {content.get('text')}")
        print()
        
        # Test add tool
        print("Testing 'add' tool...")
        async for response in client.call_tool("add", {"a": 5, "b": 3}):
            if "result" in response:
                for content in response["result"].get("content", []):
                    print(f"  Response: {content.get('text')}")
        print()
        
        # Test greet tool
        print("Testing 'greet' tool...")
        async for response in client.call_tool("greet", {"name": "World"}):
            if "result" in response:
                for content in response["result"].get("content", []):
                    print(f"  Response: {content.get('text')}")
        print()
        
        # Test unknown tool (should fail)
        print("Testing unknown tool (should fail)...")
        try:
            async for response in client.call_tool("unknown", {}):
                if "error" in response:
                    print(f"  Error (expected): {response['error']['message']}")
        except Exception as e:
            print(f"  Exception (expected): {e}")
        print()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect
        print("Disconnecting...")
        await client.disconnect()


def run_tests():
    """Run client tests."""
    print("=" * 60)
    print("Example Streamable HTTP Client")
    print("=" * 60)
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
