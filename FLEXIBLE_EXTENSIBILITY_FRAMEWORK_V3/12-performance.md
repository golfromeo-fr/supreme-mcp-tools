# Performance Optimization

## Caching Strategy

The framework implements a multi-level caching strategy:

1. **Extension Metadata Cache**: Cache extension schemas and metadata (TTL: 60s)
2. **Query Result Cache**: Cache frequently accessed data (TTL: 30s)
3. **Service Discovery Cache**: Cache tool endpoints and capabilities (TTL: 300s)

```python
# Cache configuration
cache_config = {
    "extension_metadata": {"ttl": 60, "max_size": 1000},
    "query_results": {"ttl": 30, "max_size": 500},
    "service_discovery": {"ttl": 300, "max_size": 100}
}
```

## Cache Manager Implementation

```python
import time
from typing import Any, Dict, Optional
import asyncio


class CacheManager:
    """
    TTL-based cache manager for extension metadata and query results.
    
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
```

## Connection Pooling

HTTP connections are pooled to reduce overhead:

```python
import aiohttp


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
                    connector = aiohttp.TCPConnector(
                        limit=100,           # Total connections
                        limit_per_host=10,   # Connections per host
                        ttl_dns_cache=300    # DNS cache TTL
                    )
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
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
```

## Request Coalescing

Duplicate concurrent requests are coalesced:

```python
import asyncio
from typing import Any, Callable, Dict


class RequestCoalescer:
    """Coalesces duplicate concurrent requests."""
    
    def __init__(self):
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def execute(self, key: str, func: Callable) -> Any:
        """Execute request with coalescing."""
        async with self._lock:
            if key in self.pending_requests:
                # Wait for existing request
                return await self.pending_requests[key]
            
            # Create new request
            future = asyncio.Future()
            self.pending_requests[key] = future
        
        try:
            result = await func()
            async with self._lock:
                self.pending_requests[key].set_result(result)
            return result
        except Exception as e:
            async with self._lock:
                self.pending_requests[key].set_exception(e)
            raise
        finally:
            async with self._lock:
                if key in self.pending_requests:
                    del self.pending_requests[key]
```

## Async Optimization

All I/O operations are async to maximize throughput:

```python
import asyncio
from typing import List, Callable


async def process_tools(tools: List[str], func: Callable):
    """Process multiple tools concurrently."""
    tasks = [func(tool) for tool in tools]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def batch_query(queries: List[Dict]):
    """Execute multiple queries in parallel."""
    tasks = [query(q) for q in queries]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

## Performance Configuration

```json
{
  "performance": {
    "cache": {
      "extension_metadata": {
        "ttl": 60,
        "max_size": 1000
      },
      "query_results": {
        "ttl": 30,
        "max_size": 500
      },
      "service_discovery": {
        "ttl": 300,
        "max_size": 100
      }
    },
    "http_client": {
      "timeout": 30,
      "max_connections": 100,
      "max_per_host": 10,
      "keepalive_timeout": 30
    },
    "rate_limit": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_size": 10
    }
  }
}
```

## Monitoring Performance

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
QUERIES_TOTAL = Counter('mcp_queries_total', 'Total queries', ['tool', 'extension'])
QUERY_DURATION = Histogram('mcp_query_duration_seconds', 'Query duration', ['tool', 'extension'])
CACHE_HITS = Counter('mcp_cache_hits_total', 'Cache hits', ['cache_type'])
CACHE_MISSES = Counter('mcp_cache_misses_total', 'Cache misses', ['cache_type'])

# Record metrics
QUERIES_TOTAL.labels(tool="webmcp", extension="api_calls").inc()
QUERY_DURATION.labels(tool="webmcp").observe(duration)

if cache_hit:
    CACHE_HITS.labels(cache_type="query_results").inc()
else:
    CACHE_MISSES.labels(cache_type="query_results").inc()
```

### Performance Tuning Checklist

1. **Enable caching** for frequently accessed data
2. **Use connection pooling** for HTTP clients
3. **Implement request coalescing** for duplicate requests
4. **Use async operations** for all I/O
5. **Monitor cache hit rates** and adjust TTLs
6. **Set appropriate timeouts** to prevent hanging requests
7. **Use batch operations** when possible
