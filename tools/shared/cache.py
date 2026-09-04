"""
Shared caching utilities for FastMCP tools.

ONE implementation of the LRU/TTL/size-bounded cache logic repo-wide (plan C1):
`TTLCache` is a thread-safe, OrderedDict-backed true-LRU cache with per-entry
TTL, optional size bound, periodic expired-entry sweep and hit/miss stats.
Callers needing an async API wrap it — see
`launcher/distributed_registry.py:CacheManager` (thin asyncio facade) and
`tools/webmcp/webmcp_fastmcp.py:SimpleCache` (thin subclass pinning MAX_SIZE).
"""

import time
import threading
import hashlib
from collections import OrderedDict
from typing import Any


class TTLCache:
    """Thread-safe true-LRU + TTL cache.

    Single source of truth for the LRU/TTL/eviction logic:

    - LRU: a fresh ``get`` refreshes recency (OrderedDict.move_to_end);
      at capacity the least-recently-used entry is evicted
      (OrderedDict.popitem(last=False)); overwriting a key deletes and
      re-inserts it, so overwrites count as recent use.
    - TTL: entries expire after ``ttl`` seconds (default ``default_ttl``);
      expired entries are dropped on access and swept periodically on set.
    - Size: bounded by ``max_size`` entries (``None`` = unbounded).  When the
      bound is hit, expired entries are purged before any live LRU entry is
      evicted.  Subclasses may instead pin a ``MAX_SIZE`` class attribute
      (legacy webmcp.SimpleCache name); it wins over ``max_size`` and may be
      overridden per instance.
    """

    MAX_SIZE: int | None = None

    def __init__(self, default_ttl: int = 3600, max_size: int | None = None):
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.lock = threading.RLock()
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._last_cleanup = time.time()
        self._cleanup_interval = 300

    def _maybe_cleanup(self):
        if time.time() - self._last_cleanup >= self._cleanup_interval:
            self.cleanup_expired()
            self._last_cleanup = time.time()

    def _capacity(self) -> int | None:
        """Effective entry bound: MAX_SIZE (subclass hook) wins over max_size."""
        return self.MAX_SIZE if self.MAX_SIZE is not None else self.max_size

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired (refreshes LRU recency)."""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    self.hits += 1
                    self.cache.move_to_end(key)
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with TTL, evicting LRU entries past capacity."""
        with self.lock:
            self._maybe_cleanup()
            capacity = self._capacity()
            if capacity:
                if key in self.cache:
                    # Re-insert below marks the key most-recently-used
                    del self.cache[key]
                elif len(self.cache) >= capacity:
                    # Prefer dropping expired entries over evicting live ones
                    expired_keys = [
                        k for k, (_, exp) in self.cache.items()
                        if time.time() >= exp
                    ]
                    for k in expired_keys:
                        del self.cache[k]
                    if len(self.cache) >= capacity:
                        self.cache.popitem(last=False)  # evict least-recently-used
            expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
            self.cache[key] = (value, expiry)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if current_time >= expiry
            ]
            for key in expired_keys:
                del self.cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_ratio = self.hits / total if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(hit_ratio, 3),
                "total_entries": len(self.cache)
            }


def generate_cache_key(url: str, params: dict) -> str:
    """Generate a cache key from URL and parameters."""
    param_str = str(sorted(params.items(), key=lambda x: str(x[0])))
    combined = f"{url}:{param_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Global cache instance for web operations
_web_cache = TTLCache(default_ttl=3600)


def get_web_cache() -> TTLCache:
    """Get the global web cache instance."""
    return _web_cache
