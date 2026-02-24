"""
Streamable HTTP Transport Package

This package provides the base infrastructure for Streamable HTTP transport
for MCP (Model Context Protocol) tools. It includes server and client utilities
for implementing JSON-RPC over HTTP with proper framing and connection management.

Based on the SSE to Streamable HTTP Migration Plan (Phase 6.1).

Usage:
    from launcher.streamable_http import (
        StreamableHttpConfig,
        StreamableHttpTransportBase,
        StreamableHttpClient,
    )
    
    # Server side
    config = StreamableHttpConfig()
    transport = StreamableHttpTransportBase("my_tool", config)
    
    # Client side
    client = StreamableHttpClient()
    await client.connect()
    tools = await client.list_tools()
"""

from .streamable_http_base import (
    StreamableHttpConfig,
    StreamableHttpFraming,
    StreamableHttpTransportBase,
    StreamableHttpError,
    StreamableHttpParseError,
    StreamableHttpInvalidRequestError,
    StreamableHttpMethodNotFoundError,
    StreamableHttpInvalidParamsError,
    StreamableHttpInternalError,
    MCP_SDK_AVAILABLE,
)

from .streamable_http_client import (
    ClientConfig,
    ConnectionState,
    StreamableHttpClient,
    StreamingJSONParser,
    ReconnectionManager,
    HTTPX_AVAILABLE,
)


__version__ = "1.0.0"
__all__ = [
    # Base module exports
    "StreamableHttpConfig",
    "StreamableHttpFraming",
    "StreamableHttpTransportBase",
    "StreamableHttpError",
    "StreamableHttpParseError",
    "StreamableHttpInvalidRequestError",
    "StreamableHttpMethodNotFoundError",
    "StreamableHttpInvalidParamsError",
    "StreamableHttpInternalError",
    "MCP_SDK_AVAILABLE",
    # Client module exports
    "ClientConfig",
    "ConnectionState",
    "StreamableHttpClient",
    "StreamingJSONParser",
    "ReconnectionManager",
    "HTTPX_AVAILABLE",
]


def __getattr__(name: str):
    """Lazy import for backward compatibility."""
    if name == "__all__":
        return __all__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
