"""
Shared caching utilities for FastMCP tools.
"""

import time
import threading
import hashlib
from typing import Any


class TTLCache:
    """Thread-safe TTL (Time-To-Live) cache implementation."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: dict[str, tuple[Any, float]] = {}
        self.lock = threading.RLock()
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    self.hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with TTL."""
        with self.lock:
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
    param_str = str(sorted(params.items()))
    combined = f"{url}:{param_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Global cache instance for web operations
_web_cache = TTLCache(default_ttl=3600)


def get_web_cache() -> TTLCache:
    """Get the global web cache instance."""
    return _web_cache
