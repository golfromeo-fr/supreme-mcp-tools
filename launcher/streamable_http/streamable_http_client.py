"""
Streamable HTTP Client Utilities

This module provides client utilities for testing and validating Streamable HTTP
transport implementations. It includes a streaming JSON parser, reconnection logic,
and error handling.

Based on the SSE to Streamable HTTP Migration Plan (Phase 6.1).
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Try to import httpx
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .streamable_http_base import (
    StreamableHttpConfig,
    StreamableHttpFraming,
    StreamableHttpError,
)


# Configure logging
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


@dataclass
class ClientConfig:
    """Configuration for Streamable HTTP client."""
    
    # Server connection
    base_url: str = "http://localhost:8000"
    endpoint: str = "/mcp"
    
    # Timeout settings
    request_timeout: float = 30.0
    connection_timeout: float = 10.0
    read_timeout: float = 60.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    
    # Reconnection settings
    enable_auto_reconnect: bool = True
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 10
    
    # Framing configuration
    framing_format: str = "newline-delimited"
    encoding: str = "utf-8"
    
    # Headers
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Content-Type": "application/json",
        "Accept": "application/json,text/event-stream",
    })
    
    # Session management
    session_id: Optional[str] = None
    enable_session_persistence: bool = True


class StreamableHttpClient:
    """
    Client for communicating with Streamable HTTP MCP servers.
    
    This client provides methods for sending requests, handling streaming
    responses, and managing reconnection logic.
    """
    
    def __init__(self, config: Optional[ClientConfig] = None):
        """
        Initialize the Streamable HTTP client.
        
        Args:
            config: Optional client configuration
        """
        self.config = config or ClientConfig()
        self._state = ConnectionState.DISCONNECTED
        self._http_client: Optional[httpx.AsyncClient] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info(f"StreamableHttpClient initialized for {self.config.base_url}")
    
    async def connect(self) -> None:
        """
        Connect to the Streamable HTTP server.
        
        Raises:
            StreamableHttpError: If connection fails
        """
        if not HTTPX_AVAILABLE:
            raise StreamableHttpError(
                -32603,
                "httpx library is required but not installed"
            )
        
        self._state = ConnectionState.CONNECTING
        
        try:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=self.config.read_timeout,
                    write=self.config.request_timeout,
                ),
                headers=self.config.headers,
            )
            
            # Test connection with initialize request
            await self._test_connection()
            
            self._state = ConnectionState.CONNECTED
            await self._emit_event("connected")
            logger.info(f"Connected to {self.config.base_url}")
            
        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            await self._emit_event("connection_failed", error=str(e))
            raise StreamableHttpError(
                -32603,
                f"Failed to connect: {str(e)}"
            )
    
    async def _test_connection(self) -> None:
        """Test the connection with an initialize request."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "streamable-http-client",
                    "version": "1.0.0",
                },
            },
            "id": 0,
        }
        
        response = await self._send_request(request)
        if "error" in response:
            raise StreamableHttpError(
                response["error"]["code"],
                response["error"]["message"],
                response["error"].get("data")
            )
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._state = ConnectionState.CLOSING
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        
        self._state = ConnectionState.DISCONNECTED
        await self._emit_event("disconnected")
        logger.info("Disconnected from server")
    
    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and get a single response.
        
        Args:
            method: JSON-RPC method name
            params: Optional parameters for the method
            request_id: Optional request ID (auto-generated if not provided)
            
        Returns:
            JSON-RPC response dictionary
            
        Raises:
            StreamableHttpError: If request fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise StreamableHttpError(
                -32603,
                "Not connected to server"
            )
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id if request_id is not None else id(request),
        }
        
        return await self._send_request(request)
    
    async def send_streaming_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a JSON-RPC request and stream multiple responses.
        
        Args:
            method: JSON-RPC method name
            params: Optional parameters for the method
            request_id: Optional request ID (auto-generated if not provided)
            
        Yields:
            JSON-RPC response dictionaries
            
        Raises:
            StreamableHttpError: If request fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise StreamableHttpError(
                -32603,
                "Not connected to server"
            )
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id if request_id is not None else id(request),
        }
        
        async for response in self._send_streaming_request(request):
            yield response
    
    async def _send_request(
        self,
        request: Dict[str, Any],
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Send a request and get a single response.
        
        Args:
            request: JSON-RPC request dictionary
            retry_count: Current retry attempt
            
        Returns:
            JSON-RPC response dictionary
        """
        if not self._http_client:
            raise StreamableHttpError(-32603, "HTTP client not initialized")
        
        headers = self.config.headers.copy()
        if self.config.session_id:
            headers["Mcp-Session-Id"] = self.config.session_id
        
        try:
            response = await self._http_client.post(
                self.config.endpoint,
                json=request,
                headers=headers,
                timeout=self.config.request_timeout,
            )
            
            response.raise_for_status()
            
            # Parse response (expecting single response)
            content = response.content
            if self.config.framing_format == "newline-delimited":
                # Take the first line
                lines = content.decode(self.config.encoding).strip().split("\n")
                return json.loads(lines[0])
            else:
                return StreamableHttpFraming.decode_message(content, self._convert_config())
            
        except httpx.HTTPStatusError as e:
            if retry_count < self.config.max_retries:
                delay = self.config.retry_delay * (self.config.backoff_multiplier ** retry_count)
                logger.warning(f"Request failed (attempt {retry_count + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                return await self._send_request(request, retry_count + 1)
            
            raise StreamableHttpError(
                -32603,
                f"HTTP error: {e.response.status_code}",
                str(e)
            )
        
        except httpx.RequestError as e:
            if retry_count < self.config.max_retries:
                delay = self.config.retry_delay * (self.config.backoff_multiplier ** retry_count)
                logger.warning(f"Request failed (attempt {retry_count + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                return await self._send_request(request, retry_count + 1)
            
            raise StreamableHttpError(
                -32603,
                f"Request error: {str(e)}"
            )
        
        except json.JSONDecodeError as e:
            raise StreamableHttpError(
                -32700,
                f"Failed to decode response: {str(e)}"
            )
    
    async def _send_streaming_request(
        self,
        request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a request and stream multiple responses.
        
        Args:
            request: JSON-RPC request dictionary
            
        Yields:
            JSON-RPC response dictionaries
        """
        if not self._http_client:
            raise StreamableHttpError(-32603, "HTTP client not initialized")
        
        headers = self.config.headers.copy()
        if self.config.session_id:
            headers["Mcp-Session-Id"] = self.config.session_id
        
        try:
            async with self._http_client.stream(
                "POST",
                self.config.endpoint,
                json=request,
                headers=headers,
                timeout=self.config.read_timeout,
            ) as response:
                response.raise_for_status()
                
                # Stream and parse responses
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode response line: {e}")
                            yield {
                                "jsonrpc": "2.0",
                                "error": {
                                    "code": -32700,
                                    "message": "Parse error",
                                    "data": str(e),
                                },
                                "id": request.get("id"),
                            }
        
        except httpx.HTTPStatusError as e:
            yield {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"HTTP error: {e.response.status_code}",
                    "data": str(e),
                },
                "id": request.get("id"),
            }
        
        except httpx.RequestError as e:
            yield {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Request error: {str(e)}",
                },
                "id": request.get("id"),
            }
    
    def _convert_config(self) -> StreamableHttpConfig:
        """Convert client config to server config."""
        return StreamableHttpConfig(
            endpoint=self.config.endpoint,
            framing_format=self.config.framing_format,
            encoding=self.config.encoding,
        )
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the server.
        
        Returns:
            List of tool definitions
        """
        response = await self.send_request("tools/list")
        
        if "error" in response:
            raise StreamableHttpError(
                response["error"]["code"],
                response["error"]["message"],
                response["error"].get("data")
            )
        
        return response.get("result", {}).get("tools", [])
    
    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Call a tool on the server.
        
        Args:
            name: Tool name
            arguments: Optional tool arguments
            
        Yields:
            Tool response messages
        """
        async for response in self.send_streaming_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments or {},
            }
        ):
            if "error" in response:
                raise StreamableHttpError(
                    response["error"]["code"],
                    response["error"]["message"],
                    response["error"].get("data")
                )
            yield response
    
    def on(self, event: str, handler: Callable) -> None:
        """
        Register an event handler.
        
        Args:
            event: Event name (e.g., "connected", "disconnected", "connection_failed")
            handler: Event handler function
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable) -> None:
        """
        Unregister an event handler.
        
        Args:
            event: Event name
            handler: Event handler function
        """
        if event in self._event_handlers:
            try:
                self._event_handlers[event].remove(handler)
            except ValueError:
                pass
    
    async def _emit_event(self, event: str, **kwargs) -> None:
        """Emit an event to registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(**kwargs)
                    else:
                        handler(**kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for '{event}': {e}")
    
    @property
    def state(self) -> ConnectionState:
        """Get the current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._state == ConnectionState.CONNECTED


class StreamingJSONParser:
    """
    Parser for streaming JSON-RPC messages.
    
    This parser handles incremental parsing of newline-delimited JSON
    messages from a byte stream.
    """
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the streaming JSON parser.
        
        Args:
            encoding: Text encoding to use
        """
        self.encoding = encoding
        self._buffer = ""
    
    def feed(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Feed data to the parser and return parsed messages.
        
        Args:
            data: Raw byte data to parse
            
        Returns:
            List of parsed JSON-RPC messages
        """
        messages = []
        
        # Decode and append to buffer
        self._buffer += data.decode(self.encoding)
        
        # Parse complete messages (newline-delimited)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            
            if line:
                try:
                    message = json.loads(line)
                    messages.append(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    messages.append({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                            "data": str(e),
                        },
                        "id": None,
                    })
        
        return messages
    
    def reset(self) -> None:
        """Reset the parser buffer."""
        self._buffer = ""
    
    def has_buffered_data(self) -> bool:
        """Check if there's buffered data waiting for more input."""
        return bool(self._buffer)


class ReconnectionManager:
    """
    Manages automatic reconnection for Streamable HTTP clients.
    """
    
    def __init__(
        self,
        client: StreamableHttpClient,
        max_attempts: int = 10,
        initial_delay: float = 2.0,
        backoff_multiplier: float = 2.0,
        max_delay: float = 60.0
    ):
        """
        Initialize the reconnection manager.
        
        Args:
            client: The client to manage reconnection for
            max_attempts: Maximum number of reconnection attempts
            initial_delay: Initial delay before first reconnection attempt
            backoff_multiplier: Multiplier for exponential backoff
            max_delay: Maximum delay between attempts
        """
        self.client = client
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self._attempt = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the reconnection manager."""
        if self._running:
            return
        
        self._running = True
        self._attempt = 0
        self._task = asyncio.create_task(self._reconnect_loop())
        logger.info("Reconnection manager started")
    
    async def stop(self) -> None:
        """Stop the reconnection manager."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Reconnection manager stopped")
    
    async def _reconnect_loop(self) -> None:
        """Main reconnection loop."""
        while self._running and self._attempt < self.max_attempts:
            if self.client.is_connected:
                await asyncio.sleep(1)
                continue
            
            self._attempt += 1
            delay = min(
                self.initial_delay * (self.backoff_multiplier ** (self._attempt - 1)),
                self.max_delay
            )
            
            logger.info(f"Reconnection attempt {self._attempt}/{self.max_attempts} in {delay}s")
            await asyncio.sleep(delay)
            
            try:
                await self.client.connect()
                self._attempt = 0  # Reset on successful connection
            except Exception as e:
                logger.warning(f"Reconnection attempt {self._attempt} failed: {e}")
        
        if self._attempt >= self.max_attempts:
            logger.error(f"Max reconnection attempts ({self.max_attempts}) reached")


# Export symbols
__all__ = [
    "ClientConfig",
    "ConnectionState",
    "StreamableHttpClient",
    "StreamingJSONParser",
    "ReconnectionManager",
    "HTTPX_AVAILABLE",
]
