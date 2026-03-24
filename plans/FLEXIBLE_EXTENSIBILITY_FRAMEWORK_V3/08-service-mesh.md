# Service Mesh and Discovery

## Service Discovery Integration

The service registry integrates with the launcher to automatically discover running tools:

```python
# launcher/server_manager.py (integration)

class ServerManager:
    """Manages tool server lifecycle."""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.port_manager = PortManager()
    
    async def start_tool(self, tool_name: str):
        """Start a tool and register it."""
        # Allocate ports
        mcp_port = self.port_manager.allocate()
        mgmt_port = self.port_manager.allocate()
        
        # Start tool process
        process = await self._start_process(
            tool_name,
            mcp_port=mcp_port,
            mgmt_port=mgmt_port
        )
        
        # Register with service registry
        self.service_registry.register(
            name=tool_name,
            management_url=f"http://127.0.0.1:{mgmt_port}",
            mcp_port=mcp_port
        )
        
        return process
```

## Health Monitoring

The service registry performs periodic health checks:

```python
# Health check response format
{
    "status": "healthy",  # "healthy", "degraded", "unhealthy"
    "tool": "webmcp",
    "timestamp": 1679064000,
    "checks": {
        "registry": "ok",
        "config": "ok",
        "dependencies": "ok"
    }
}
```

## Service Registry Class

```python
# launcher/service_registry.py

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    name: str
    management_url: str
    mcp_port: int
    status: str  # "healthy", "degraded", "unhealthy"
    last_health_check: float
    capabilities: Optional[Dict] = None


class ServiceRegistry:
    """
    Service registry for tool discovery and health monitoring.
    
    Integrates with the launcher to track running tools and their
    management endpoints.
    """
    
    def __init__(self, health_check_interval: float = 30.0):
        self.services: Dict[str, ServiceInfo] = {}
        self.health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the health check background task."""
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )
    
    async def stop(self):
        """Stop the health check background task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
    
    def register(
        self,
        name: str,
        management_url: str,
        mcp_port: int
    ) -> None:
        """
        Register a tool service.
        
        Args:
            name: Tool name
            management_url: Management API URL
            mcp_port: MCP server port
        """
        self.services[name] = ServiceInfo(
            name=name,
            management_url=management_url,
            mcp_port=mcp_port,
            status="unknown",
            last_health_check=0
        )
        logger.info(f"Registered service: {name} at {management_url}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool service."""
        if name in self.services:
            del self.services[name]
            logger.info(f"Unregistered service: {name}")
    
    async def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.services.keys())
    
    async def get_endpoint(self, name: str) -> Optional[ServiceInfo]:
        """
        Get endpoint information for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            ServiceInfo if found, None otherwise
        """
        return self.services.get(name)
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_all_health(self):
        """Check health of all registered services."""
        import aiohttp
        
        for name, service in self.services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{service.management_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            service.status = "healthy"
                        else:
                            service.status = "degraded"
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                service.status = "unhealthy"
            
            service.last_health_check = time.time()
```

## Health Check Configuration

```json
{
  "health_check": {
    "interval": 30,
    "timeout": 5,
    "unhealthy_threshold": 3,
    "healthy_threshold": 2
  }
}
```

## Service Discovery Flow

1. **Tool Startup**: Tool starts and allocates ports
2. **Registration**: Tool registers with service registry
3. **Health Checks**: Registry performs periodic health checks
4. **Discovery**: Management server queries registry for available tools
5. **Routing**: Management server routes requests to healthy tools
6. **Deregistration**: Tool deregisters on shutdown
