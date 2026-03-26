"""
API Client for the Management UI.

Provides HTTP client for communicating with the Management API server.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional

import httpx

from .models import (
    ToolInfo,
    ToolDetail,
    Extension,
    ExtensionType,
    APIResponse,
)


class APIError(Exception):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ManagementAPIClient:
    """
    HTTP client for the Management API.
    
    Provides methods for interacting with the MCP Tools Management API,
    including fetching tool information, extensions, and executing
    extension operations.
    """
    
    def _get_default_base_url() -> str:
        """Get the default base URL from ports.json."""
        try:
            from launcher.launcher_config import load_ports_config
            ports_config = load_ports_config()
            port = ports_config.get("reserved", {}).get("central_management")
            if port:
                return f"http://localhost:{port}"
        except Exception:
            pass
        return None
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 1,
        retry_delay: float = 0.5,
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the Management API (default: from ports.json or MCP_API_URL env var).
            api_key: API key for authentication (default: MCP_API_KEY env var).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
        """
        if base_url is None:
            base_url = os.environ.get("MCP_API_URL")
        if base_url is None:
            base_url = self._get_default_base_url()
        if base_url is None:
            raise ValueError(
                "API base URL not configured. Set MCP_API_URL environment variable or "
                "create config/ports.json with reserved.central_management port."
            )
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("MCP_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Prepare headers
        self.headers: Dict[str, str] = {}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get an HTTPX async client."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
        )
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.).
            path: API endpoint path.
            **kwargs: Additional arguments for httpx request.
        
        Returns:
            Response JSON data.
        
        Raises:
            APIError: If request fails after retries.
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                async with self._get_client() as client:
                    response = await client.request(method, path, **kwargs)
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 401:
                        raise APIError("Unauthorized: Invalid API key", 401)
                    elif response.status_code == 404:
                        raise APIError(f"Not found: {path}", 404)
                    else:
                        raise APIError(
                            f"API error: {response.text}",
                            response.status_code
                        )
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            except httpx.ConnectError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            except APIError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        raise APIError(f"Request failed after {self.max_retries} retries: {last_error}")
    
    async def get_health(self) -> Dict[str, Any]:
        """
        Get API health status.
        
        Returns:
            Health check response with status information.
        """
        return await self._request_with_retry("GET", "/health")
    
    async def get_tools(self) -> List[ToolInfo]:
        """
        Get list of all registered tools.
        
        Returns:
            List of ToolInfo objects.
        """
        data = await self._request_with_retry("GET", "/api/tools")
        
        # Handle different response formats:
        # - {"tools": [...]} - standard format from Management API
        # - {"data": [...]} - alternative wrapped format
        # - [...] - direct list
        if isinstance(data, dict):
            if "tools" in data:
                tools_data = data["tools"]
            elif "data" in data:
                tools_data = data["data"]
            else:
                tools_data = []
        elif isinstance(data, list):
            tools_data = data
        else:
            tools_data = []
        
        return [ToolInfo(**tool) for tool in tools_data]
    
    async def get_tool(self, name: str) -> ToolDetail:
        """
        Get detailed information about a specific tool.
        
        Args:
            name: Tool name.
        
        Returns:
            ToolDetail with full information and extensions.
        """
        data = await self._request_with_retry("GET", f"/api/tools/{name}")
        
        # Handle both direct response and wrapped response
        if isinstance(data, dict) and "data" in data:
            tool_data = data["data"]
        else:
            tool_data = data
        
        return ToolDetail(**tool_data)
    
    async def get_extensions(self, tool_name: str) -> List[Extension]:
        """
        Get extensions for a specific tool.
        
        Args:
            tool_name: Name of the tool.
        
        Returns:
            List of Extension objects.
        """
        data = await self._request_with_retry("GET", f"/api/tools/{tool_name}/extensions")
        
        # Handle response formats:
        # - {"extensions": [...]} - standard format from Management API
        # - {"data": [...]} - alternative wrapped format
        # - [...] - direct list
        if isinstance(data, dict) and "extensions" in data:
            extensions_data = data["extensions"]
        elif isinstance(data, dict) and "data" in data:
            extensions_data = data["data"]
        elif isinstance(data, list):
            extensions_data = data
        else:
            extensions_data = []
        
        return [Extension(**ext) for ext in extensions_data]
    
    async def query_extension(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query a data source extension.
        
        Args:
            tool_name: Name of the tool.
            extension_name: Name of the extension.
            params: Optional query parameters.
        
        Returns:
            Query result data.
        """
        payload = params or {}
        data = await self._request_with_retry(
            "POST",
            f"/api/tools/{tool_name}/extensions/{extension_name}/query",
            json=payload,
        )
        return data
    
    async def mutate_extension(
        self,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Submit a mutation to a mutator extension.
        
        Args:
            tool_name: Name of the tool.
            extension_name: Name of the extension.
            params: Mutation parameters.
        
        Returns:
            Mutation result data.
        """
        data = await self._request_with_retry(
            "POST",
            f"/api/tools/{tool_name}/extensions/{extension_name}/mutate",
            json={"params": params},
        )
        return data
    
    async def execute_extension(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an action extension.
        
        Args:
            tool_name: Name of the tool.
            extension_name: Name of the extension.
            params: Optional execution parameters.
        
        Returns:
            Execution result data.
        """
        payload = params or {}
        data = await self._request_with_retry(
            "POST",
            f"/api/tools/{tool_name}/extensions/{extension_name}/execute",
            json=payload,
        )
        return data


# Global client instance (initialized lazily)
_client: Optional[ManagementAPIClient] = None


def get_client() -> ManagementAPIClient:
    """
    Get or create the global API client instance.
    
    Returns:
        ManagementAPIClient instance.
    """
    global _client
    if _client is None:
        _client = ManagementAPIClient()
    return _client


def reset_client() -> None:
    """Reset the global client instance (useful for testing)."""
    global _client
    _client = None
