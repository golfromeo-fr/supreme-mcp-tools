# Core Components (Completion)

This section completes the Core Components that were cut off in the main document.

## DistributedExtensionRegistry (Complete)

The `DistributedExtensionRegistry` acts as a proxy that routes requests to appropriate tool processes based on service discovery information.

```python
# launcher/distributed_registry.py

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ToolEndpoint:
    """Represents a tool's management endpoint."""
    name: str
    url: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: float
    capabilities: Optional[Dict[str, Any]] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for resilient HTTP communication.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.states: Dict[str, CircuitBreakerState] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, float] = {}
    
    async def execute(
        self,
        key: str,
        func: callable,
        fallback: Optional[callable] = None
    ) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            key: Circuit breaker key (e.g., tool name)
            func: Async function to execute
            fallback: Optional fallback function when circuit is open
            
        Returns:
            Result from func or fallback
            
        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback
        """
        state = self._get_state(key)
        
        if state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery(key):
                self.states[key] = CircuitBreakerState.HALF_OPEN
            else:
                if fallback:
                    return await fallback()
                raise CircuitBreakerOpenError(f"Circuit open for {key}")
        
        try:
            result = await func()
            self._on_success(key)
            return result
        except Exception as e:
            self._on_failure(key)
            raise
    
    def _get_state(self, key: str) -> CircuitBreakerState:
        """Get current state for a key."""
        return self.states.get(key, CircuitBreakerState.CLOSED)
    
    def _should_attempt_recovery(self, key: str) -> bool:
        """Check if enough time has passed to attempt recovery."""
        last_failure = self.last_failure_times.get(key, 0)
        return (time.time() - last_failure) >= self.recovery_timeout
    
    def _on_success(self, key: str) -> None:
        """Handle successful execution."""
        self.states[key] = CircuitBreakerState.CLOSED
        self.failure_counts[key] = 0
    
    def _on_failure(self, key: str) -> None:
        """Handle failed execution."""
        self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        self.last_failure_times[key] = time.time()
        
        if self.failure_counts[key] >= self.failure_threshold:
            self.states[key] = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened for {key}")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CacheManager:
    """
    Async TTL-based cache manager for extension metadata and query results.
    
    Uses OrderedDict for LRU eviction and supports async operations.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry["expires_at"] > time.time():
                    return entry["value"]
                else:
                    del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 60) -> None:
        """Set cached value with TTL."""
        async with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl
            }
    
    async def invalidate(self, key: str) -> None:
        """Invalidate a cache entry."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
    
    async def invalidate_prefix(self, prefix: str) -> None:
        """Invalidate all entries with a prefix."""
        async with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self.cache[key]


class EventAggregator:
    """
    Aggregates events from tools and distributes to WebSocket subscribers.
    
    Supports pub/sub pattern for real-time event streaming.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, tool_name: str) -> asyncio.Queue:
        """Subscribe to events from a tool. Returns a queue to receive events."""
        queue = asyncio.Queue()
        async with self._lock:
            if tool_name not in self.subscribers:
                self.subscribers[tool_name] = []
            self.subscribers[tool_name].append(queue)
        return queue
    
    async def unsubscribe(self, tool_name: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from events."""
        async with self._lock:
            if tool_name in self.subscribers:
                self.subscribers[tool_name].remove(queue)
    
    async def publish(self, tool_name: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to all subscribers of a tool."""
        event = {
            "tool": tool_name,
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        async with self._lock:
            subscribers = self.subscribers.get(tool_name, [])
        
        for queue in subscribers:
            try:
                await queue.put(event)
            except Exception as e:
                logger.error(f"Error publishing event: {e}")


class HTTPClient:
    """
    Async HTTP client with connection pooling and timeout support.
    
    Wraps aiohttp for making requests to tool management servers.
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=self.timeout
                    )
        return self._session
    
    async def get(self, url: str) -> Any:
        """Execute HTTP GET request."""
        session = await self._get_session()
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(self, url: str, data: Dict[str, Any]) -> Any:
        """Execute HTTP POST request."""
        session = await self._get_session()
        async with session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def websocket(self, url: str):
        """Create WebSocket connection."""
        session = await self._get_session()
        return session.ws_connect(url)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class ConfigManager:
    """
    Manages configuration loading and persistence for a tool.
    
    Loads persisted configuration on startup and provides
    methods to update configuration at runtime.
    """
    
    def __init__(self, tool_name: str, config_dir: str = "~/.config/supreme-mcp-tools"):
        self.tool_name = tool_name
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / f"{tool_name}.json"
        self.config: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from disk."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config for {self.tool_name}")
            except Exception as e:
                logger.error(f"Error loading config for {self.tool_name}: {e}")
                self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    async def set(self, key: str, value: Any) -> None:
        """Set a configuration value and persist to disk."""
        async with self._lock:
            self.config[key] = value
            self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to disk."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config for {self.tool_name}: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()


class ConfigPersistence:
    """
    File-based configuration persistence.
    
    Stores configuration changes in JSON files that survive restarts.
    """
    
    def __init__(self, config_dir: str = "~/.config/supreme-mcp-tools"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_tool_config_path(self, tool_name: str) -> Path:
        """Get path to tool's config file."""
        return self.config_dir / f"{tool_name}.json"
    
    def load(self, tool_name: str) -> Dict[str, Any]:
        """
        Load persisted configuration for a tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Configuration dictionary
        """
        config_path = self._get_tool_config_path(tool_name)
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config for {tool_name}: {e}")
                return {}
        return {}
    
    async def save(
        self,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Save a configuration change.
        
        Args:
            tool_name: Tool name
            extension_name: Extension that was mutated
            params: New configuration values
        """
        config_path = self._get_tool_config_path(tool_name)
        
        # Load existing config
        config = self.load(tool_name)
        
        # Update with new values
        if "mutations" not in config:
            config["mutations"] = []
        
        config["mutations"].append({
            "extension": extension_name,
            "params": params,
            "timestamp": time.time()
        })
        
        # Save to file
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved config for {tool_name}.{extension_name}")


class DistributedExtensionRegistry:
    """
    Distributed extension registry that communicates with tool processes via HTTP.
    
    This registry acts as a proxy, routing requests to appropriate tool processes
    based on service discovery information.
    """
    
    def __init__(self, service_registry: 'ServiceRegistry'):
        self.service_registry = service_registry
        self.circuit_breaker = CircuitBreaker()
        self.cache = CacheManager()
        self.config_persistence = ConfigPersistence()
        self.event_aggregator = EventAggregator()
        self.http_client = HTTPClient()
    
    # ==================== DISCOVERY ====================
    
    async def list_tools(self) -> List[str]:
        """List all available tools."""
        return await self.service_registry.list_tools()
    
    async def list_extensions(
        self,
        tool_name: Optional[str] = None,
        ext_type: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        List extensions from tools with caching.
        
        Args:
            tool_name: Optional filter by tool name
            ext_type: Optional filter by extension type
            
        Returns:
            Dictionary mapping tool names to extension lists
        """
        cache_key = f"extensions:{tool_name}:{ext_type}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        tools = [tool_name] if tool_name else await self.list_tools()
        result = {}
        
        for tool in tools:
            endpoint = await self.service_registry.get_endpoint(tool)
            if not endpoint:
                continue
            
            try:
                extensions = await self._http_get(
                    f"{endpoint.url}/extensions",
                    circuit_breaker_key=tool
                )
                
                if ext_type:
                    extensions = [
                        e for e in extensions
                        if e["type"] == ext_type
                    ]
                
                if extensions:
                    result[tool] = extensions
                    
                    # Cache tool capabilities
                    await self.cache.set(
                        f"extensions:{tool}",
                        extensions,
                        ttl=60
                    )
            except CircuitBreakerOpenError:
                logger.warning(f"Circuit breaker open for {tool}")
                # Return cached data if available
                cached_tool = await self.cache.get(f"extensions:{tool}")
                if cached_tool:
                    result[tool] = cached_tool
        
        await self.cache.set(cache_key, result, ttl=60)
        return result
    
    # ==================== QUERY ====================
    
    async def query(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Query a data source extension in a tool process.
        
        Args:
            tool_name: Target tool name
            extension_name: Extension to query
            params: Query parameters
            
        Returns:
            Query result from the tool
        """
        endpoint = await self.service_registry.get_endpoint(tool_name)
        if not endpoint:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return await self._http_post(
            f"{endpoint.url}/extensions/{extension_name}/query",
            {"params": params or {}},
            circuit_breaker_key=tool_name
        )
    
    # ==================== MUTATE ====================
    
    async def mutate(
        self,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        Mutate configuration in a tool process.
        
        Args:
            tool_name: Target tool name
            extension_name: Extension to mutate
            params: Mutation parameters
            
        Returns:
            Mutation result from the tool
        """
        endpoint = await self.service_registry.get_endpoint(tool_name)
        if not endpoint:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        result = await self._http_post(
            f"{endpoint.url}/extensions/{extension_name}/mutate",
            {"params": params},
            circuit_breaker_key=tool_name
        )
        
        # Persist configuration change
        await self.config_persistence.save(
            tool_name,
            extension_name,
            params
        )
        
        # Invalidate cache
        await self.cache.invalidate(f"extensions:{tool_name}")
        
        # Notify subscribers
        await self.event_aggregator.publish(
            tool_name,
            "mutation",
            {
                "extension": extension_name,
                "params": params,
                "result": result
            }
        )
        
        return result
    
    # ==================== ACTIONS ====================
    
    async def execute(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute an action in a tool process.
        
        Args:
            tool_name: Target tool name
            extension_name: Action to execute
            params: Action parameters
            
        Returns:
            Action result from the tool
        """
        endpoint = await self.service_registry.get_endpoint(tool_name)
        if not endpoint:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        return await self._http_post(
            f"{endpoint.url}/extensions/{extension_name}/execute",
            {"params": params or {}},
            circuit_breaker_key=tool_name
        )
    
    # ==================== EVENTS ====================
    
    async def subscribe(
        self,
        tool_name: str,
        extension_name: str
    ) -> AsyncIterator[Any]:
        """
        Subscribe to events from a tool process.
        
        Args:
            tool_name: Target tool name
            extension_name: Event extension to subscribe to
            
        Yields:
            Event data from the tool
        """
        endpoint = await self.service_registry.get_endpoint(tool_name)
        if not endpoint:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        ws_url = endpoint.url.replace("http://", "ws://")
        ws_url = f"{ws_url}/extensions/{extension_name}/events"
        
        async for event in self._websocket_connect(ws_url):
            yield event
    
    # ==================== HTTP CLIENT ====================
    
    async def _http_get(
        self,
        url: str,
        circuit_breaker_key: str
    ) -> Any:
        """Execute HTTP GET with circuit breaker."""
        return await self.circuit_breaker.execute(
            circuit_breaker_key,
            lambda: self.http_client.get(url)
        )
    
    async def _http_post(
        self,
        url: str,
        data: Dict[str, Any],
        circuit_breaker_key: str
    ) -> Any:
        """Execute HTTP POST with circuit breaker."""
        return await self.circuit_breaker.execute(
            circuit_breaker_key,
            lambda: self.http_client.post(url, data)
        )
    
    async def _websocket_connect(
        self,
        url: str
    ) -> AsyncIterator[Any]:
        """Connect to WebSocket with reconnection logic."""
        while True:
            try:
                async with self.http_client.websocket(url) as ws:
                    async for message in ws:
                        yield message
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Backoff before reconnect
```

## ExtensionHTTPServer (Complete)

Each tool exposes an HTTP API for extension management.

```python
# launcher/tool_extensions/http_server.py

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import APIKeyHeader
from typing import Dict, Any, Set, Optional
import logging
import asyncio
import uvicorn

logger = logging.getLogger(__name__)

# API Key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class ExtensionHTTPServer:
    """
    HTTP server for extension management in tool processes.
    
    Exposes RESTful endpoints for querying, mutating, and executing
    extensions, plus WebSocket for real-time events.
    """
    
    def __init__(
        self,
        tool_name: str,
        registry: 'ExtensionRegistry',
        config_manager: 'ConfigManager',
        port: int,
        api_keys: Optional[Dict[str, Dict]] = None
    ):
        self.tool_name = tool_name
        self.registry = registry
        self.config_manager = config_manager
        self.port = port
        self.api_keys = api_keys or {}
        self.app = FastAPI(title=f"{tool_name} Management API")
        self.websocket_connections: Set[WebSocket] = set()
        self._server: Optional[uvicorn.Server] = None
        self._register_routes()
    
    def _verify_api_key(self, api_key: Optional[str] = Depends(API_KEY_HEADER)) -> dict:
        """Verify API key and return permissions."""
        if not self.api_keys:
            # No authentication configured
            return {"role": "admin"}
        
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        if api_key not in self.api_keys:
            raise HTTPException(status_code=403, detail="Invalid API key")
        
        return self.api_keys[api_key]
    
    def _register_routes(self):
        """Register HTTP routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "tool": self.tool_name}
        
        @self.app.get("/extensions")
        async def list_extensions():
            """List all extensions for this tool."""
            return self.registry.list_extensions()
        
        @self.app.get("/extensions/{name}")
        async def get_extension(name: str):
            """Get extension details."""
            ext = self.registry.get_extension(name)
            if ext:
                return {
                    "name": ext.name,
                    "type": ext.ext_type.value,
                    "schema": ext.schema,
                    "metadata": ext.metadata
                }
            raise HTTPException(status_code=404, detail="Not found")
        
        @self.app.post("/extensions/{name}/query")
        async def query_extension(name: str, request: Dict[str, Any]):
            """Query a data source extension."""
            try:
                result = self.registry.query(
                    name,
                    request.get("params")
                )
                return {"data": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/extensions/{name}/mutate")
        async def mutate_extension(name: str, request: Dict[str, Any]):
            """Mutate configuration via extension."""
            try:
                result = self.registry.mutate(
                    name,
                    request.get("params", {})
                )
                return {"result": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/extensions/{name}/execute")
        async def execute_extension(name: str, request: Dict[str, Any]):
            """Execute an action extension."""
            try:
                result = self.registry.execute(
                    name,
                    request.get("params")
                )
                return {"result": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.websocket("/extensions/{name}/events")
        async def event_websocket(websocket: WebSocket, name: str):
            """WebSocket for real-time events."""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            
            # Subscribe to events
            def event_handler(event_data: Dict[str, Any]):
                asyncio.create_task(
                    websocket.send_json(event_data)
                )
            
            self.registry.subscribe("event", event_handler)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.discard(websocket)
    
    async def start(self):
        """Start the HTTP server."""
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="info"
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()
    
    async def stop(self):
        """Stop the HTTP server gracefully."""
        if self._server:
            self._server.should_exit = True
            await self._server.shutdown()
            
        # Close all WebSocket connections
        for ws in self.websocket_connections.copy():
            try:
                await ws.close()
            except Exception:
                pass
        self.websocket_connections.clear()
```

## ServiceRegistry (Complete)

The service registry manages tool discovery and health monitoring.

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
