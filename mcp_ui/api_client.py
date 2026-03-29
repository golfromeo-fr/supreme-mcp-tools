"""
Async HTTP client for Management API.

Non-blocking API client with comprehensive logging and error handling.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

import aiohttp

from .logging_config import get_logger, generate_trace_id, set_trace_id, get_trace_id
from .models import APIResponse, ToolInfo, ToolDetail, Extension, ExtensionType

logger = get_logger(__name__)


def _load_ports_config() -> dict:
    """
    Load the central ports configuration.

    Returns:
        Dictionary with ports configuration

    Raises:
        ConfigError: If ports.json is not found or invalid
    """
    # Try multiple locations for ports.json
    possible_paths = [
        # Standard location: config/ports.json
        Path(__file__).parent.parent / "config" / "ports.json",
        # Alternative: ports.json in root
        Path(__file__).parent.parent / "ports.json",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                logger.debug(f"Loaded ports config from {path}")
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}: {e}")
            except Exception as e:
                raise ValueError(f"Failed to load ports config from {path}: {e}")

    raise ValueError(
        "ports.json not found. Please create config/ports.json with "
        "'reserved.central_management' and 'reserved.management_ui' ports."
    )


def _get_default_base_url() -> str:
    """Get the default base URL from ports.json or environment."""
    # Check environment first
    api_url = os.environ.get("MCP_API_URL")
    if api_url:
        return api_url

    # Try ports.json
    ports_config = _load_ports_config()
    port = ports_config.get("reserved", {}).get("central_management")
    if port:
        return f"http://localhost:{port}"

    raise ValueError(
        "API base URL not configured. Set MCP_API_URL environment variable or "
        "create config/ports.json with reserved.central_management port."
    )


def _get_api_timeout() -> float:
    """Get the API request timeout from environment or config."""
    # Check environment first
    env_timeout = os.environ.get("MCP_API_TIMEOUT")
    if env_timeout:
        return float(env_timeout)

    # Try ports.json
    ports_config = _load_ports_config()
    timeout = ports_config.get("timeouts", {}).get("api_request")
    if timeout:
        return float(timeout)

    # Default
    return 30.0


class APIClient:
    """
    Async HTTP client for management API.

    Never blocks the UI - all calls are async with proper timeouts.
    Uses lazy session creation and comprehensive logging.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the Management API
            timeout: Request timeout in seconds
        """
        self.base_url = (_get_default_base_url() if base_url is None else base_url).rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout if timeout is not None else _get_api_timeout())
        self._session: Optional[aiohttp.ClientSession] = None
        self.api_key = os.environ.get("MCP_API_KEY")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session (lazy initialization)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            logger.debug(f"Created new aiohttp session for {self.base_url}")
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session (call on shutdown)."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed aiohttp session")

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> APIResponse:
        """
        Make an HTTP request with logging and error handling.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            path: API endpoint path
            **kwargs: Additional arguments for aiohttp request

        Returns:
            APIResponse with success status and data/error
        """
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        url = f"{self.base_url}{path}"
        logger.debug(f"--> {method} {url}")

        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            session = await self._get_session()
            async with session.request(method, url, headers=headers, **kwargs) as response:
                status = response.status
                logger.debug(f"<-- {method} {url} [{status}]")

                if status == 200:
                    data = await response.json()
                    return APIResponse.ok(data)
                elif status == 404:
                    error = f"Not found: {path}"
                    logger.warning(f"[{trace_id}] {error}")
                    return APIResponse.fail(error)
                elif status == 401:
                    error = "Unauthorized"
                    logger.warning(f"[{trace_id}] {error}")
                    return APIResponse.fail(error)
                else:
                    text = await response.text()
                    error = f"API error {status}: {text[:100]}"
                    logger.error(f"[{trace_id}] {error}")
                    return APIResponse.fail(error)

        except asyncio.TimeoutError:
            error = f"Request timeout: {path}"
            logger.error(f"[{trace_id}] {error}")
            return APIResponse.fail(error)
        except aiohttp.ClientError as e:
            error = f"Connection error: {e}"
            logger.error(f"[{trace_id}] {error}")
            return APIResponse.fail(error)
        except Exception as e:
            error = f"Unexpected error: {e}"
            logger.exception(f"[{trace_id}] {error}")
            return APIResponse.fail(error)

    async def get_health(self) -> APIResponse:
        """Check API health status."""
        return await self._request("GET", "/health")

    async def get_tools(self) -> APIResponse[list[ToolInfo]]:
        """
        Get all registered tools.

        Returns:
            APIResponse containing list of ToolInfo objects
        """
        response = await self._request("GET", "/api/tools")
        if not response.success:
            return response

        try:
            data = response.data
            # Handle different response formats
            if isinstance(data, dict):
                tools_data = data.get("tools", data.get("data", []))
            elif isinstance(data, list):
                tools_data = data
            else:
                tools_data = []

            tools = [ToolInfo(**item) for item in tools_data]
            logger.debug(f"Parsed {len(tools)} tools")
            return APIResponse.ok(tools)
        except Exception as e:
            error = f"Failed to parse tools: {e}"
            logger.error(f"[{get_trace_id()}] {error}")
            return APIResponse.fail(error)

    async def get_tool(self, name: str) -> APIResponse[ToolDetail]:
        """
        Get single tool details.

        Args:
            name: Tool name

        Returns:
            APIResponse containing ToolDetail object
        """
        response = await self._request("GET", f"/api/tools/{name}")
        if not response.success:
            return response

        try:
            data = response.data
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            tool = ToolDetail(**data)
            return APIResponse.ok(tool)
        except Exception as e:
            error = f"Failed to parse tool {name}: {e}"
            logger.error(f"[{get_trace_id()}] {error}")
            return APIResponse.fail(error)

    async def get_extensions(self, tool_name: str) -> APIResponse[list[Extension]]:
        """
        Get extensions for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            APIResponse containing list of Extension objects
        """
        response = await self._request("GET", f"/api/tools/{tool_name}/extensions")
        if not response.success:
            return response

        try:
            data = response.data
            if isinstance(data, dict):
                ext_data = data.get("extensions", data.get("data", []))
            elif isinstance(data, list):
                ext_data = data
            else:
                ext_data = []

            extensions = [Extension(**item) for item in ext_data]
            return APIResponse.ok(extensions)
        except Exception as e:
            error = f"Failed to parse extensions: {e}"
            logger.error(f"[{get_trace_id()}] {error}")
            return APIResponse.fail(error)

    async def query_extension(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[dict] = None,
    ) -> APIResponse:
        """Query a data source extension."""
        payload = params or {}
        return await self._request(
            "POST",
            f"/api/tools/{tool_name}/extensions/{extension_name}/query",
            json=payload,
        )

    async def mutate_extension(
        self,
        tool_name: str,
        extension_name: str,
        params: dict,
    ) -> APIResponse:
        """Submit a mutation to a mutator extension."""
        return await self._request(
            "POST",
            f"/api/tools/{tool_name}/extensions/{extension_name}/mutate",
            json={"params": params},
        )

    async def execute_extension(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[dict] = None,
    ) -> APIResponse:
        """Execute an action extension."""
        payload = params or {}
        return await self._request(
            "POST",
            f"/api/tools/{tool_name}/extensions/{extension_name}/execute",
            json=payload,
        )

    async def start_tool(self, name: str) -> APIResponse:
        """Start a tool."""
        return await self._request("POST", f"/api/tools/{name}/start")

    async def stop_tool(self, name: str) -> APIResponse:
        """Stop a tool."""
        return await self._request("POST", f"/api/tools/{name}/stop")

    async def restart_tool(self, name: str) -> APIResponse:
        """Restart a tool."""
        return await self._request("POST", f"/api/tools/{name}/restart")

    # === Disabled Tools Configuration ===

    async def get_disabled_tools(self) -> APIResponse:
        """Get all disabled tools configuration."""
        return await self._request("GET", "/api/disabled-tools")

    async def get_server_disabled_tools(self, server_name: str) -> APIResponse:
        """Get disabled tools for a specific server."""
        return await self._request("GET", f"/api/disabled-tools/{server_name}")

    async def set_server_disabled_tools(self, server_name: str, disabled_list: list) -> APIResponse:
        """Set disabled tools for a specific server."""
        return await self._request(
            "PUT",
            f"/api/disabled-tools/{server_name}",
            json=disabled_list,
        )

    async def disable_tool(self, server_name: str, tool_name: str) -> APIResponse:
        """Disable a specific tool for a server."""
        return await self._request(
            "POST",
            f"/api/disabled-tools/{server_name}/{tool_name}/disable",
        )

    async def enable_tool(self, server_name: str, tool_name: str) -> APIResponse:
        """Enable a specific tool for a server."""
        return await self._request(
            "POST",
            f"/api/disabled-tools/{server_name}/{tool_name}/enable",
        )


# For backwards compatibility
APIError = Exception


# Global client instance (lazy)
_client: Optional[APIClient] = None


def get_client() -> APIClient:
    """
    Get or create the global API client instance.

    Returns:
        APIClient instance
    """
    global _client
    if _client is None:
        _client = APIClient()
        logger.info(f"Created API client for {_client.base_url}")
    return _client


async def close_client() -> None:
    """Close the global client (call on shutdown)."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def reset_client() -> None:
    """Reset the global client (useful for testing)."""
    global _client
    _client = None
