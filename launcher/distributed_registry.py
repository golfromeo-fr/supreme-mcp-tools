"""
Distributed Extension Registry

The distributed registry acts as a proxy that routes requests to appropriate tool processes
based on service discovery information. This is the central component of the FEF V3 architecture.
"""

import asyncio
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RequestCoalescer:
    """
    Coalesces duplicate concurrent requests.
    
    When multiple identical requests are made concurrently, only one
    actual request is executed and the result is shared with all callers.
    """
    
    def __init__(self):
        """Initialize the request coalescer."""
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def execute(self, key: str, func: callable) -> Any:
        """
        Execute request with coalescing.
        
        Args:
            key: Unique key for the request
            func: Async function to execute
            
        Returns:
            Result from the function
        """
        async with self._lock:
            if key in self._pending:
                future = self._pending[key]
                if not future.done():
                    # Wait for existing request
                    pass
                else:
                    # Future completed, start new one
                    future = asyncio.get_event_loop().create_future()
                    self._pending[key] = future
                    # Execute in background
                    asyncio.create_task(self._execute_func(key, func, future))
            else:
                # Start new request
                future = asyncio.get_event_loop().create_future()
                self._pending[key] = future
                # Execute in background
                asyncio.create_task(self._execute_func(key, func, future))
        
        return await future
    
    async def _execute_func(self, key: str, func: callable, future: asyncio.Future) -> None:
        """Execute function and set future result."""
        try:
            result = await func()
            if not future.done():
                future.set_result(result)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            async with self._lock:
                if key in self._pending and self._pending[key] is future:
                    del self._pending[key]


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
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
        """
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


class CacheManager:
    """
    Async TTL-based cache manager for extension metadata and query results.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached value if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry["expires_at"] > time.time():
                    return entry["value"]
                else:
                    del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 60) -> None:
        """
        Set cached value with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
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
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
    
    async def invalidate_prefix(self, prefix: str) -> None:
        """
        Invalidate all entries with a prefix.
        
        Args:
            prefix: Prefix to match
        """
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
        """Initialize the event aggregator."""
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, tool_name: str) -> asyncio.Queue:
        """
        Subscribe to events from a tool.
        
        Args:
            tool_name: Name of the tool to subscribe to
            
        Returns:
            Queue to receive events
        """
        queue = asyncio.Queue()
        async with self._lock:
            if tool_name not in self.subscribers:
                self.subscribers[tool_name] = []
            self.subscribers[tool_name].append(queue)
        return queue
    
    async def unsubscribe(self, tool_name: str, queue: asyncio.Queue) -> None:
        """
        Unsubscribe from events.
        
        Args:
            tool_name: Name of the tool
            queue: Queue to unsubscribe
        """
        async with self._lock:
            if tool_name in self.subscribers:
                if queue in self.subscribers[tool_name]:
                    self.subscribers[tool_name].remove(queue)
    
    async def publish(
        self,
        tool_name: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Publish an event to all subscribers of a tool.
        
        Args:
            tool_name: Name of the tool
            event_type: Type of event
            data: Event data
        """
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
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        Initialize the HTTP client.
        
        Args:
            timeout: Request timeout in seconds
        """
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
        """
        Execute HTTP GET request.
        
        Args:
            url: URL to request
            
        Returns:
            Response JSON data
        """
        session = await self._get_session()
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(self, url: str, data: Dict[str, Any]) -> Any:
        """
        Execute HTTP POST request.
        
        Args:
            url: URL to request
            data: Request body data
            
        Returns:
            Response JSON data
        """
        session = await self._get_session()
        async with session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def websocket(self, url: str):
        """
        Create WebSocket connection.
        
        Args:
            url: WebSocket URL
            
        Returns:
            WebSocket connection
        """
        session = await self._get_session()
        return session.ws_connect(url)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class ConfigPersistence:
    """
    File-based configuration persistence.
    
    Stores configuration changes in JSON files that survive restarts.
    """
    
    def __init__(self, config_dir: str = "~/.config/supreme-mcp-tools"):
        """
        Initialize the configuration persistence.
        
        Args:
            config_dir: Directory to store configuration files
        """
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
        """
        Initialize the distributed extension registry.
        
        Args:
            service_registry: Service registry for tool discovery
        """
        self.service_registry = service_registry
        self.circuit_breaker = CircuitBreaker()
        self.cache = CacheManager()
        self.config_persistence = ConfigPersistence()
        self.event_aggregator = EventAggregator()
        self.http_client = HTTPClient()
    
    # ==================== DISCOVERY ====================
    
    async def list_tools(self) -> List[str]:
        """
        List all available tools.
        
        Returns:
            List of tool names
        """
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
                    f"{endpoint.management_url}/extensions",
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
            f"{endpoint.management_url}/extensions/{extension_name}/query",
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
            f"{endpoint.management_url}/extensions/{extension_name}/mutate",
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
            f"{endpoint.management_url}/extensions/{extension_name}/execute",
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
        
        url = f"{endpoint.management_url}/extensions/{extension_name}/events"
        
        async with self.http_client._session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") != "ping":
                        yield data
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
    
    # ==================== INTERNAL HTTP HELPERS ====================
    
    async def _http_get(
        self,
        url: str,
        circuit_breaker_key: Optional[str] = None
    ) -> Any:
        """
        Execute HTTP GET with circuit breaker protection.
        
        Args:
            url: URL to request
            circuit_breaker_key: Optional circuit breaker key
            
        Returns:
            Response data
        """
        if circuit_breaker_key:
            return await self.circuit_breaker.execute(
                circuit_breaker_key,
                lambda: self.http_client.get(url)
            )
        return await self.http_client.get(url)
    
    async def _http_post(
        self,
        url: str,
        data: Dict[str, Any],
        circuit_breaker_key: Optional[str] = None
    ) -> Any:
        """
        Execute HTTP POST with circuit breaker protection.
        
        Args:
            url: URL to request
            data: Request body data
            circuit_breaker_key: Optional circuit breaker key
            
        Returns:
            Response data
        """
        if circuit_breaker_key:
            return await self.circuit_breaker.execute(
                circuit_breaker_key,
                lambda: self.http_client.post(url, data)
            )
        return await self.http_client.post(url, data)
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.close()
