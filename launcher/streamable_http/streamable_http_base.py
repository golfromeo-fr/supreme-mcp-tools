"""
Streamable HTTP Base Implementation

This module provides the base infrastructure for Streamable HTTP transport
for MCP (Model Context Protocol) tools. It implements JSON-RPC framing with
newline-delimited JSON, chunked transfer encoding, and connection management.

Based on the SSE to Streamable HTTP Migration Plan (Phase 6.1).
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Union
from dataclasses import dataclass, field

# Try to import from modelcontextprotocol SDK
try:
    from mcp.server.lowlevel import Server
    from mcp.types import (
        JSONRPCError,
        JSONRPCNotification,
        JSONRPCRequest,
        JSONRPCResponse,
    )
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False
    # Define stub types for when SDK is not available
    Server = None
    JSONRPCError = None
    JSONRPCNotification = None
    JSONRPCRequest = None
    JSONRPCResponse = None


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class StreamableHttpConfig:
    """Configuration for Streamable HTTP transport."""
    
    # Endpoint paths
    endpoint: str = "/mcp"
    messages_path: str = "/messages/"
    
    # Framing configuration
    framing_format: str = "newline-delimited"  # or "length-prefixed"
    encoding: str = "utf-8"
    
    # Timeout settings
    request_timeout: float = 30.0
    connection_timeout: float = 10.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Chunked transfer settings
    chunk_size: int = 8192
    enable_chunked_encoding: bool = True
    
    # Session management
    enable_session_management: bool = True
    session_timeout: float = 300.0  # 5 minutes
    
    # Headers
    required_headers: Dict[str, str] = field(default_factory=lambda: {
        "Content-Type": "application/json",
        "Accept": "application/json,text/event-stream",
    })
    
    # Error handling
    include_stack_traces: bool = False
    max_error_message_length: int = 1000


class StreamableHttpFraming:
    """Handles message framing for Streamable HTTP transport."""
    
    @staticmethod
    def encode_message(message: Dict[str, Any], config: StreamableHttpConfig) -> bytes:
        """
        Encode a message with appropriate framing.
        
        Args:
            message: The message dictionary to encode
            config: Configuration for framing format
            
        Returns:
            Encoded message as bytes
        """
        json_str = json.dumps(message, ensure_ascii=False)
        
        if config.framing_format == "newline-delimited":
            return (json_str + "\n").encode(config.encoding)
        elif config.framing_format == "length-prefixed":
            # 4-byte length prefix (big-endian)
            length = len(json_str.encode(config.encoding))
            length_bytes = length.to_bytes(4, byteorder='big')
            return length_bytes + json_str.encode(config.encoding)
        else:
            raise ValueError(f"Unsupported framing format: {config.framing_format}")
    
    @staticmethod
    def decode_message(data: bytes, config: StreamableHttpConfig) -> Dict[str, Any]:
        """
        Decode a message with appropriate framing.
        
        Args:
            data: The encoded message data
            config: Configuration for framing format
            
        Returns:
            Decoded message as dictionary
        """
        if config.framing_format == "newline-delimited":
            json_str = data.decode(config.encoding).strip()
            return json.loads(json_str)
        elif config.framing_format == "length-prefixed":
            # First 4 bytes are the length
            length = int.from_bytes(data[:4], byteorder='big')
            json_str = data[4:4+length].decode(config.encoding)
            return json.loads(json_str)
        else:
            raise ValueError(f"Unsupported framing format: {config.framing_format}")
    
    @staticmethod
    async def parse_stream(
        data_stream: AsyncGenerator[bytes, None],
        config: StreamableHttpConfig
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Parse a stream of framed messages.
        
        Args:
            data_stream: Async generator yielding byte chunks
            config: Configuration for framing format
            
        Yields:
            Decoded messages as dictionaries
        """
        buffer = b""
        
        if config.framing_format == "newline-delimited":
            async for chunk in data_stream:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        try:
                            yield StreamableHttpFraming.decode_message(line, config)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode message: {e}")
                            yield StreamableHttpFraming._create_error_response(
                                -32700, "Parse error", str(e)
                            )
        
        elif config.framing_format == "length-prefixed":
            async for chunk in data_stream:
                buffer += chunk
                while len(buffer) >= 4:
                    # Read length prefix
                    length = int.from_bytes(buffer[:4], byteorder='big')
                    if len(buffer) < 4 + length:
                        # Not enough data yet
                        break
                    
                    # Extract and decode message
                    message_data = buffer[4:4+length]
                    buffer = buffer[4+length:]
                    
                    try:
                        yield StreamableHttpFraming.decode_message(message_data, config)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message: {e}")
                        yield StreamableHttpFraming._create_error_response(
                            -32700, "Parse error", str(e)
                        )
    
    @staticmethod
    def _create_error_response(code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message,
            },
            "id": None,
        }
        if data is not None:
            error_response["error"]["data"] = data
        return error_response


class StreamableHttpTransportBase:
    """
    Base class for Streamable HTTP transport.
    
    This class provides the core functionality for handling Streamable HTTP
    requests and responses with proper framing, error handling, and session
    management.
    """
    
    def __init__(
        self,
        server_name: str,
        config: Optional[StreamableHttpConfig] = None
    ):
        """
        Initialize the Streamable HTTP transport.
        
        Args:
            server_name: Name of the MCP server
            config: Optional configuration (uses defaults if not provided)
        """
        self.server_name = server_name
        self.config = config or StreamableHttpConfig()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = asyncio.Lock()
        
        # Initialize MCP server if SDK is available
        self._mcp_server = None
        if MCP_SDK_AVAILABLE:
            self._mcp_server = Server(server_name)
        
        logger.info(f"StreamableHttpTransport initialized for '{server_name}'")
    
    async def handle_request(
        self,
        request_data: Dict[str, Any],
        headers: Dict[str, str],
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle an incoming Streamable HTTP request.
        
        Args:
            request_data: Parsed JSON-RPC request
            headers: HTTP headers from the request
            session_id: Optional session identifier
            
        Yields:
            JSON-RPC responses or notifications
        """
        # Validate request
        if not self._validate_request(request_data):
            yield StreamableHttpFraming._create_error_response(
                -32600, "Invalid Request"
            )
            return
        
        # Get or create session
        session = await self._get_or_create_session(session_id, headers)
        
        # Process request
        try:
            async for response in self._process_request(request_data, session):
                yield response
        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            yield StreamableHttpFraming._create_error_response(
                -32603, "Internal error", str(e) if self.config.include_stack_traces else None
            )
    
    async def _process_request(
        self,
        request_data: Dict[str, Any],
        session: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a JSON-RPC request.
        
        Args:
            request_data: The JSON-RPC request
            session: Session context
            
        Yields:
            JSON-RPC responses
        """
        method = request_data.get("method")
        params = request_data.get("params", {})
        request_id = request_data.get("id")
        
        if method == "initialize":
            # Handle initialize request
            response = await self._handle_initialize(params, session)
            if request_id is not None:
                response["id"] = request_id
            yield response
        
        elif method == "tools/list":
            # Handle tools list request
            response = await self._handle_tools_list(params, session)
            if request_id is not None:
                response["id"] = request_id
            yield response
        
        elif method == "tools/call":
            # Handle tool call request
            async for response in self._handle_tool_call(params, session, request_id):
                yield response
        
        elif method == "notifications/initialized":
            # Handle initialized notification
            yield await self._handle_initialized(params, session)
        
        else:
            # Unknown method
            if request_id is not None:
                yield StreamableHttpFraming._create_error_response(
                    -32601, "Method not found", f"Unknown method: {method}"
                )
    
    async def _handle_initialize(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle initialize request."""
        session["initialized"] = False
        session["capabilities"] = params.get("capabilities", {})
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "serverInfo": {
                    "name": self.server_name,
                    "version": "1.0.0",
                },
            },
        }
    
    async def _handle_initialized(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle initialized notification."""
        session["initialized"] = True
        return {
            "jsonrpc": "2.0",
            "method": "notifications/ready",
        }
    
    async def _handle_tools_list(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle tools/list request."""
        # Subclasses should override this to return actual tools
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": [],
            },
        }
    
    async def _handle_tool_call(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle tools/call request."""
        # Subclasses should override this to handle actual tool calls
        yield StreamableHttpFraming._create_error_response(
            -32601, "Method not found", "Tool calls not implemented"
        )
    
    def _validate_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate a JSON-RPC request."""
        if not isinstance(request_data, dict):
            return False
        
        if request_data.get("jsonrpc") != "2.0":
            return False
        
        if "method" not in request_data:
            return False
        
        return True
    
    async def _get_or_create_session(
        self,
        session_id: Optional[str],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Get or create a session."""
        if not self.config.enable_session_management:
            return {}
        
        async with self._session_lock:
            # Use provided session ID or generate one
            if session_id is None:
                session_id = headers.get("Mcp-Session-Id")
            
            if session_id is None:
                session_id = f"session_{id(self)}"
            
            # Get or create session
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "id": session_id,
                    "created_at": asyncio.get_event_loop().time(),
                    "last_activity": asyncio.get_event_loop().time(),
                    "initialized": False,
                    "capabilities": {},
                }
            
            # Update last activity
            self._sessions[session_id]["last_activity"] = asyncio.get_event_loop().time()
            
            return self._sessions[session_id]
    
    async def cleanup_sessions(self):
        """Clean up expired sessions."""
        if not self.config.enable_session_management:
            return
        
        async with self._session_lock:
            current_time = asyncio.get_event_loop().time()
            expired_sessions = [
                sid for sid, session in self._sessions.items()
                if current_time - session["last_activity"] > self.config.session_timeout
            ]
            
            for sid in expired_sessions:
                logger.info(f"Cleaning up expired session: {sid}")
                del self._sessions[sid]
    
    async def create_response_stream(
        self,
        responses: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Create a response stream with proper framing.
        
        Args:
            responses: Async generator of response dictionaries
            
        Yields:
            Framed response bytes
        """
        async for response in responses:
            yield StreamableHttpFraming.encode_message(response, self.config)
    
    def get_config(self) -> StreamableHttpConfig:
        """Get the current configuration."""
        return self.config
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)


class StreamableHttpError(Exception):
    """Base exception for Streamable HTTP errors."""
    
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


class StreamableHttpParseError(StreamableHttpError):
    """Error during message parsing."""
    
    def __init__(self, message: str, data: Any = None):
        super().__init__(-32700, "Parse error", data)


class StreamableHttpInvalidRequestError(StreamableHttpError):
    """Invalid request error."""
    
    def __init__(self, message: str, data: Any = None):
        super().__init__(-32600, "Invalid Request", data)


class StreamableHttpMethodNotFoundError(StreamableHttpError):
    """Method not found error."""
    
    def __init__(self, message: str, data: Any = None):
        super().__init__(-32601, "Method not found", data)


class StreamableHttpInvalidParamsError(StreamableHttpError):
    """Invalid params error."""
    
    def __init__(self, message: str, data: Any = None):
        super().__init__(-32602, "Invalid params", data)


class StreamableHttpInternalError(StreamableHttpError):
    """Internal error."""
    
    def __init__(self, message: str, data: Any = None):
        super().__init__(-32603, "Internal error", data)


# Export symbols
__all__ = [
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
]
