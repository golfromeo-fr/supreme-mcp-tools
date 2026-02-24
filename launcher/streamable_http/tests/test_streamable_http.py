"""
Test suite for Streamable HTTP transport implementation.

This module provides comprehensive tests for the Streamable HTTP base
implementation including framing, parsing, error handling, and
client-server communication.

Based on the SSE to Streamable HTTP Migration Plan (Phase 6.1).
"""

import asyncio
import json
import pytest

from launcher.streamable_http import (
    StreamableHttpConfig,
    StreamableHttpFraming,
    StreamableHttpTransportBase,
    StreamableHttpClient,
    ClientConfig,
    StreamingJSONParser,
    StreamableHttpError,
    StreamableHttpParseError,
    StreamableHttpInvalidRequestError,
    StreamableHttpMethodNotFoundError,
    StreamableHttpInvalidParamsError,
    StreamableHttpInternalError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def newline_config():
    """Create a newline-delimited configuration."""
    return StreamableHttpConfig(
        framing_format="newline-delimited",
        encoding="utf-8",
    )


@pytest.fixture
def length_prefix_config():
    """Create a length-prefixed configuration."""
    return StreamableHttpConfig(
        framing_format="length-prefixed",
        encoding="utf-8",
    )


@pytest.fixture
def sample_request():
    """Create a sample JSON-RPC request."""
    return {
        "jsonrpc": "2.0",
        "method": "test_method",
        "params": {"key": "value"},
        "id": 1,
    }


@pytest.fixture
def sample_response():
    """Create a sample JSON-RPC response."""
    return {
        "jsonrpc": "2.0",
        "result": {"data": "test_result"},
        "id": 1,
    }


@pytest.fixture
def sample_error():
    """Create a sample JSON-RPC error."""
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32600,
            "message": "Invalid Request",
        },
        "id": None,
    }


# ============================================================================
# StreamableHttpConfig Tests
# ============================================================================

class TestStreamableHttpConfig:
    """Tests for StreamableHttpConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamableHttpConfig()
        
        assert config.endpoint == "/mcp"
        assert config.messages_path == "/messages/"
        assert config.framing_format == "newline-delimited"
        assert config.encoding == "utf-8"
        assert config.request_timeout == 30.0
        assert config.connection_timeout == 10.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.chunk_size == 8192
        assert config.enable_chunked_encoding is True
        assert config.enable_session_management is True
        assert config.session_timeout == 300.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StreamableHttpConfig(
            endpoint="/custom",
            framing_format="length-prefixed",
            request_timeout=60.0,
        )
        
        assert config.endpoint == "/custom"
        assert config.framing_format == "length-prefixed"
        assert config.request_timeout == 60.0
    
    def test_required_headers(self):
        """Test default required headers."""
        config = StreamableHttpConfig()
        
        assert "Content-Type" in config.required_headers
        assert "Accept" in config.required_headers
        assert config.required_headers["Content-Type"] == "application/json"


# ============================================================================
# StreamableHttpFraming Tests
# ============================================================================

class TestStreamableHttpFraming:
    """Tests for StreamableHttpFraming."""
    
    def test_encode_newline_delimited(self, newline_config, sample_request):
        """Test encoding with newline-delimited format."""
        encoded = StreamableHttpFraming.encode_message(sample_request, newline_config)
        
        assert isinstance(encoded, bytes)
        assert encoded.endswith(b"\n")
        
        decoded_str = encoded.decode("utf-8").strip()
        decoded = json.loads(decoded_str)
        assert decoded == sample_request
    
    def test_encode_length_prefixed(self, length_prefix_config, sample_request):
        """Test encoding with length-prefixed format."""
        encoded = StreamableHttpFraming.encode_message(sample_request, length_prefix_config)
        
        assert isinstance(encoded, bytes)
        assert len(encoded) >= 4
        
        # Extract length prefix
        length = int.from_bytes(encoded[:4], byteorder='big')
        assert length == len(encoded) - 4
        
        # Decode message
        decoded = StreamableHttpFraming.decode_message(encoded, length_prefix_config)
        assert decoded == sample_request
    
    def test_decode_newline_delimited(self, newline_config, sample_request):
        """Test decoding with newline-delimited format."""
        encoded = StreamableHttpFraming.encode_message(sample_request, newline_config)
        decoded = StreamableHttpFraming.decode_message(encoded, newline_config)
        
        assert decoded == sample_request
    
    def test_decode_length_prefixed(self, length_prefix_config, sample_request):
        """Test decoding with length-prefixed format."""
        encoded = StreamableHttpFraming.encode_message(sample_request, length_prefix_config)
        decoded = StreamableHttpFraming.decode_message(encoded, length_prefix_config)
        
        assert decoded == sample_request
    
    def test_encode_decode_roundtrip(self, newline_config, sample_request):
        """Test encode-decode roundtrip."""
        encoded = StreamableHttpFraming.encode_message(sample_request, newline_config)
        decoded = StreamableHttpFraming.decode_message(encoded, newline_config)
        
        assert decoded == sample_request
    
    def test_invalid_framing_format(self, sample_request):
        """Test error handling for invalid framing format."""
        config = StreamableHttpConfig(framing_format="invalid")
        
        with pytest.raises(ValueError, match="Unsupported framing format"):
            StreamableHttpFraming.encode_message(sample_request, config)
    
    @pytest.mark.asyncio
    async def test_parse_stream_newline_delimited(self, newline_config):
        """Test parsing newline-delimited stream."""
        messages = [
            {"jsonrpc": "2.0", "method": "test1", "id": 1},
            {"jsonrpc": "2.0", "method": "test2", "id": 2},
            {"jsonrpc": "2.0", "method": "test3", "id": 3},
        ]
        
        # Create stream
        async def data_stream():
            for msg in messages:
                yield StreamableHttpFraming.encode_message(msg, newline_config)
        
        # Parse stream
        parsed = []
        async for msg in StreamableHttpFraming.parse_stream(data_stream(), newline_config):
            parsed.append(msg)
        
        assert len(parsed) == 3
        assert parsed == messages
    
    @pytest.mark.asyncio
    async def test_parse_stream_with_invalid_json(self, newline_config):
        """Test parsing stream with invalid JSON."""
        # Create stream with invalid JSON
        async def data_stream():
            yield b'{"jsonrpc":"2.0","method":"test","id":1}\n'
            yield b'invalid json\n'
            yield b'{"jsonrpc":"2.0","method":"test2","id":2}\n'
        
        # Parse stream
        parsed = []
        async for msg in StreamableHttpFraming.parse_stream(data_stream(), newline_config):
            parsed.append(msg)
        
        # Should have 3 messages (2 valid, 1 error)
        assert len(parsed) == 3
        assert parsed[0]["method"] == "test"
        assert "error" in parsed[1]  # Error response
        assert parsed[2]["method"] == "test2"


# ============================================================================
# StreamableHttpTransportBase Tests
# ============================================================================

class TestStreamableHttpTransportBase:
    """Tests for StreamableHttpTransportBase."""
    
    def test_initialization(self):
        """Test transport initialization."""
        transport = StreamableHttpTransportBase("test_tool")
        
        assert transport.server_name == "test_tool"
        assert isinstance(transport.config, StreamableHttpConfig)
        assert transport.get_session_count() == 0
    
    def test_initialization_with_config(self):
        """Test transport initialization with custom config."""
        config = StreamableHttpConfig(request_timeout=60.0)
        transport = StreamableHttpTransportBase("test_tool", config)
        
        assert transport.config.request_timeout == 60.0
    
    def test_validate_request_valid(self):
        """Test request validation with valid request."""
        transport = StreamableHttpTransportBase("test_tool")
        request = {
            "jsonrpc": "2.0",
            "method": "test",
            "id": 1,
        }
        
        assert transport._validate_request(request) is True
    
    def test_validate_request_invalid_jsonrpc(self):
        """Test request validation with invalid jsonrpc version."""
        transport = StreamableHttpTransportBase("test_tool")
        request = {
            "jsonrpc": "1.0",
            "method": "test",
            "id": 1,
        }
        
        assert transport._validate_request(request) is False
    
    def test_validate_request_missing_method(self):
        """Test request validation with missing method."""
        transport = StreamableHttpTransportBase("test_tool")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
        }
        
        assert transport._validate_request(request) is False
    
    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        """Test initialize request handling."""
        transport = StreamableHttpTransportBase("test_tool")
        session = {}
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
        }
        
        response = await transport._handle_initialize(params, session)
        
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert "capabilities" in response["result"]
        assert "serverInfo" in response["result"]
    
    @pytest.mark.asyncio
    async def test_handle_initialized(self):
        """Test initialized notification handling."""
        transport = StreamableHttpTransportBase("test_tool")
        session = {}
        params = {}
        
        response = await transport._handle_initialized(params, session)
        
        assert response["jsonrpc"] == "2.0"
        assert response["method"] == "notifications/ready"
        assert session["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_handle_tools_list(self):
        """Test tools/list request handling."""
        transport = StreamableHttpTransportBase("test_tool")
        session = {}
        params = {}
        
        response = await transport._handle_tools_list(params, session)
        
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert "tools" in response["result"]
    
    @pytest.mark.asyncio
    async def test_handle_request_invalid(self):
        """Test handling invalid request."""
        transport = StreamableHttpTransportBase("test_tool")
        request = {
            "jsonrpc": "1.0",
            "method": "test",
        }
        headers = {}
        
        responses = []
        async for response in transport.handle_request(request, headers):
            responses.append(response)
        
        assert len(responses) == 1
        assert "error" in responses[0]
        assert responses[0]["error"]["code"] == -32600
    
    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session creation and management."""
        config = StreamableHttpConfig(enable_session_management=True)
        transport = StreamableHttpTransportBase("test_tool", config)
        
        # Get or create session
        session = await transport._get_or_create_session(None, {})
        
        assert "id" in session
        assert "created_at" in session
        assert "last_activity" in session
        assert session["initialized"] is False
        
        # Check session count
        assert transport.get_session_count() == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_sessions(self):
        """Test session cleanup."""
        config = StreamableHttpConfig(
            enable_session_management=True,
            session_timeout=0.1,  # Very short timeout
        )
        transport = StreamableHttpTransportBase("test_tool", config)
        
        # Create session
        await transport._get_or_create_session(None, {})
        assert transport.get_session_count() == 1
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Cleanup
        await transport.cleanup_sessions()
        assert transport.get_session_count() == 0


# ============================================================================
# StreamingJSONParser Tests
# ============================================================================

class TestStreamingJSONParser:
    """Tests for StreamingJSONParser."""
    
    def test_initialization(self):
        """Test parser initialization."""
        parser = StreamingJSONParser()
        
        assert parser.encoding == "utf-8"
        assert parser._buffer == ""
    
    def test_feed_complete_message(self):
        """Test feeding a complete message."""
        parser = StreamingJSONParser()
        data = b'{"jsonrpc":"2.0","method":"test","id":1}\n'
        
        messages = parser.feed(data)
        
        assert len(messages) == 1
        assert messages[0]["method"] == "test"
    
    def test_feed_partial_message(self):
        """Test feeding a partial message."""
        parser = StreamingJSONParser()
        
        # Feed partial data
        messages1 = parser.feed(b'{"jsonrpc":"2.0","method"')
        assert len(messages1) == 0
        
        # Feed remaining data
        messages2 = parser.feed(b':"test","id":1}\n')
        assert len(messages2) == 1
        assert messages2[0]["method"] == "test"
    
    def test_feed_multiple_messages(self):
        """Test feeding multiple messages at once."""
        parser = StreamingJSONParser()
        data = (
            b'{"jsonrpc":"2.0","method":"test1","id":1}\n'
            b'{"jsonrpc":"2.0","method":"test2","id":2}\n'
            b'{"jsonrpc":"2.0","method":"test3","id":3}\n'
        )
        
        messages = parser.feed(data)
        
        assert len(messages) == 3
        assert messages[0]["method"] == "test1"
        assert messages[1]["method"] == "test2"
        assert messages[2]["method"] == "test3"
    
    def test_feed_invalid_json(self):
        """Test feeding invalid JSON."""
        parser = StreamingJSONParser()
        data = b'invalid json\n'
        
        messages = parser.feed(data)
        
        assert len(messages) == 1
        assert "error" in messages[0]
    
    def test_reset(self):
        """Test parser reset."""
        parser = StreamingJSONParser()
        parser.feed(b'{"jsonrpc":"2.0","method":"test"}\n')
        
        assert parser.has_buffered_data() is False
        
        # Feed partial data
        parser.feed(b'{"jsonrpc":"2.0","method"')
        assert parser.has_buffered_data() is True
        
        # Reset
        parser.reset()
        assert parser.has_buffered_data() is False
        assert parser._buffer == ""


# ============================================================================
# Error Classes Tests
# ============================================================================

class TestErrorClasses:
    """Tests for error exception classes."""
    
    def test_streamable_http_error(self):
        """Test base error class."""
        error = StreamableHttpError(-32603, "Test error", {"data": "test"})
        
        assert error.code == -32603
        assert error.message == "Test error"
        assert error.data == {"data": "test"}
        assert str(error) == "Test error"
    
    def test_parse_error(self):
        """Test parse error class."""
        error = StreamableHttpParseError("Invalid JSON")
        
        assert error.code == -32700
        assert error.message == "Parse error"
    
    def test_invalid_request_error(self):
        """Test invalid request error class."""
        error = StreamableHttpInvalidRequestError("Bad request")
        
        assert error.code == -32600
        assert error.message == "Invalid Request"
    
    def test_method_not_found_error(self):
        """Test method not found error class."""
        error = StreamableHttpMethodNotFoundError("Unknown method")
        
        assert error.code == -32601
        assert error.message == "Method not found"
    
    def test_invalid_params_error(self):
        """Test invalid params error class."""
        error = StreamableHttpInvalidParamsError("Bad params")
        
        assert error.code == -32602
        assert error.message == "Invalid params"
    
    def test_internal_error(self):
        """Test internal error class."""
        error = StreamableHttpInternalError("Server error")
        
        assert error.code == -32603
        assert error.message == "Internal error"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for Streamable HTTP transport."""
    
    @pytest.mark.asyncio
    async def test_simple_server_client_flow(self):
        """Test simple server-client communication flow."""
        # Create transport
        transport = StreamableHttpTransportBase("test_tool")
        
        # Simulate initialize request
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
            },
            "id": 1,
        }
        
        # Handle request
        responses = []
        async for response in transport.handle_request(request, {}):
            responses.append(response)
        
        assert len(responses) == 1
        assert "result" in responses[0]
        assert responses[0]["id"] == 1
        
        # Send initialized notification
        request = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        
        responses = []
        async for response in transport.handle_request(request, {}):
            responses.append(response)
        
        assert len(responses) == 1
        assert responses[0]["method"] == "notifications/ready"
        
        # List tools
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2,
        }
        
        responses = []
        async for response in transport.handle_request(request, {}):
            responses.append(response)
        
        assert len(responses) == 1
        assert "result" in responses[0]
        assert responses[0]["id"] == 2
    
    @pytest.mark.asyncio
    async def test_framing_integration(self):
        """Test framing integration with transport."""
        transport = StreamableHttpTransportBase("test_tool")
        config = transport.config
        
        # Create request
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 1,
        }
        
        # Encode request
        encoded = StreamableHttpFraming.encode_message(request, config)
        
        # Decode request
        decoded = StreamableHttpFraming.decode_message(encoded, config)
        
        assert decoded == request


# ============================================================================
# Test Utilities
# ============================================================================

def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
