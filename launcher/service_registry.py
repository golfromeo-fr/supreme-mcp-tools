"""
Service Registry

Manages tool discovery and health monitoring.
Provides automatic tool registration and health checks.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """
    Information about a registered service.
    
    Attributes:
        name: Tool name
        management_url: URL of the management HTTP server
        mcp_port: Port of the MCP server
        status: Current health status
        last_check: Timestamp of last health check
        capabilities: Optional capabilities information
    """
    name: str
    management_url: str
    mcp_port: int
    status: str = "unknown"  # healthy, degraded, unhealthy, unknown
    last_check: float = 0.0
    capabilities: dict[str, Any] | None = None
    registered_at: float = field(default_factory=time.time)


class ServiceRegistry:
    """
    Service registry for tool discovery and health monitoring.
    
    Maintains a registry of all running tools and their endpoints.
    Performs periodic health checks to monitor tool status.
    """
    
    def __init__(
        self,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 10.0
    ):
        """
        Initialize the service registry.
        
        Args:
            health_check_interval: Interval between health checks in seconds
            health_check_timeout: Timeout for health check requests
        """
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        
        self._services: dict[str, ServiceInfo] = {}
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task | None = None
        self._running = False
        self._session: aiohttp.ClientSession | None = None
    
    async def start(self) -> None:
        """Start the health check background task."""
        if self._running:
            return
        
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
        )
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service registry started")
    
    async def stop(self) -> None:
        """Stop the health check background task."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
        
        logger.info("Service registry stopped")
    
    async def register(
        self,
        name: str,
        management_url: str,
        mcp_port: int,
        capabilities: dict[str, Any] | None = None
    ) -> None:
        """
        Register a tool service.
        
        Args:
            name: Tool name
            management_url: URL of the management HTTP server
            mcp_port: Port of the MCP server
            capabilities: Optional capabilities information
        """
        async with self._lock:
            self._services[name] = ServiceInfo(
                name=name,
                management_url=management_url,
                mcp_port=mcp_port,
                capabilities=capabilities
            )
        
        # Perform initial health check
        await self._check_health(name)
        
        logger.info(
            f"Registered service '{name}' at {management_url} "
            f"(MCP port: {mcp_port})"
        )
    
    async def unregister(self, name: str) -> bool:
        """
        Unregister a tool service.
        
        Args:
            name: Tool name
            
        Returns:
            True if service was unregistered, False if not found
        """
        async with self._lock:
            if name in self._services:
                del self._services[name]
                logger.info(f"Unregistered service '{name}'")
                return True
        return False
    
    async def list_tools(self) -> list[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        async with self._lock:
            return list(self._services.keys())
    
    async def get_endpoint(self, name: str) -> ServiceInfo | None:
        """
        Get endpoint information for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            ServiceInfo if found, None otherwise
        """
        async with self._lock:
            return self._services.get(name)
    
    async def get_all_services(self) -> dict[str, ServiceInfo]:
        """
        Get all registered services.
        
        Returns:
            Dictionary mapping tool names to ServiceInfo
        """
        async with self._lock:
            return self._services.copy()
    
    async def update_capabilities(
        self,
        name: str,
        capabilities: dict[str, Any]
    ) -> bool:
        """
        Update capabilities for a registered service.
        
        Args:
            name: Tool name
            capabilities: New capabilities information
            
        Returns:
            True if updated, False if service not found
        """
        async with self._lock:
            if name in self._services:
                self._services[name].capabilities = capabilities
                return True
        return False
    
    async def _health_check_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self._check_all_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _check_all_health(self) -> None:
        """Check health of all registered services."""
        async with self._lock:
            names = list(self._services.keys())
        
        for name in names:
            await self._check_health(name)
    
    async def _check_health(self, name: str) -> None:
        """
        Check health of a specific service.
        
        Args:
            name: Tool name
        """
        async with self._lock:
            service = self._services.get(name)
        
        if service is None:
            return
        
        try:
            async with self._session.get(
                f"{service.management_url}/health"
            ) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                    except Exception as json_err:
                        # Log the raw response text for debugging
                        response_text = await response.text()
                        logger.error(
                            f"Health check JSON parse failed for '{name}': {json_err}. "
                            f"Response status: {response.status}, Text: {response_text[:500]}"
                        )
                        raise
                    
                    async with self._lock:
                        if name in self._services:
                            self._services[name].status = "healthy"
                            self._services[name].last_check = time.time()
                            
                            # Update capabilities if provided
                            if "extensions_count" in data:
                                if self._services[name].capabilities is None:
                                    self._services[name].capabilities = {}
                                self._services[name].capabilities["extensions_count"] = (
                                    data["extensions_count"]
                                )
                    
                    logger.debug(f"Health check passed for '{name}'")
                else:
                    async with self._lock:
                        if name in self._services:
                            self._services[name].status = "degraded"
                            self._services[name].last_check = time.time()
                    
                    logger.warning(
                        f"Health check returned {response.status} for '{name}'"
                    )
        except BaseException as e:
            # Catch BaseException to handle CancelledError and all other exceptions
            async with self._lock:
                if name in self._services:
                    self._services[name].status = "unhealthy"
                    self._services[name].last_check = time.time()
            
            # Check for timeout-related exceptions
            if isinstance(e, (asyncio.CancelledError, asyncio.TimeoutError)) or \
               (hasattr(e, '__cause__') and isinstance(e.__cause__, asyncio.CancelledError)):
                logger.warning(f"Health check timed out for '{name}'")
            else:
                import traceback
                logger.warning(
                    f"Health check failed for '{name}': {type(e).__name__}: {e}. "
                    f"Traceback: {traceback.format_exc()[:500]}"
                )
