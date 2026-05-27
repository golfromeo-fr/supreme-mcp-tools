"""
Rate Limiting for FEF V3

Provides token bucket rate limiting for API endpoints.
"""

import threading
import time
import logging
from collections import defaultdict
from functools import wraps

from collections.abc import Callable
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Implements the token bucket algorithm for rate limiting requests.
    Each client (identified by key) has a bucket that refills over time.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int | None = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        
        self.buckets: dict[str, dict] = defaultdict(lambda: {
            "tokens": self.burst_size,
            "last_update": time.time()
        })
        self._lock = threading.Lock()
        self._async_lock = None  # Will be set if asyncio is available
    
    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Client identifier (IP, API key, etc.)
            tokens: Number of tokens to consume
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            bucket = self.buckets[key]
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.burst_size,
                bucket["tokens"] + elapsed * self.refill_rate
            )
            bucket["last_update"] = now
            
            # Check if enough tokens available
            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True
            
            return False
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining tokens for a key.
        
        Args:
            key: Client identifier
            
        Returns:
            Number of remaining tokens
        """
        with self._lock:
            bucket = self.buckets.get(key)
            if bucket is None:
                return self.burst_size
            
            now = time.time()
            elapsed = now - bucket["last_update"]
            tokens = min(
                self.burst_size,
                bucket["tokens"] + elapsed * self.refill_rate
            )
            return int(tokens)
    
    def get_reset_time(self, key: str) -> float:
        """
        Get time until bucket is fully refilled.
        
        Args:
            key: Client identifier
            
        Returns:
            Seconds until full refill
        """
        remaining = self.get_remaining(key)
        if remaining >= self.burst_size:
            return 0.0
        
        tokens_needed = self.burst_size - remaining
        return tokens_needed / self.refill_rate
    
    def reset(self, key: str) -> None:
        """
        Reset the bucket for a key.
        
        Args:
            key: Client identifier
        """
        with self._lock:
            if key in self.buckets:
                self.buckets[key] = {
                    "tokens": self.burst_size,
                    "last_update": time.time()
                }
    
    def clear(self) -> None:
        """Clear all buckets."""
        self.buckets.clear()


class AsyncRateLimiter:
    """
    Async-compatible rate limiter.
    
    Same as RateLimiter but with async lock for thread safety.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int | None = None
    ):
        """
        Initialize the async rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.refill_rate = requests_per_minute / 60.0
        
        self.buckets: dict[str, dict] = {}
        self._lock = None
    
    async def _get_lock(self):
        """Get or create async lock."""
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        return self._lock
    
    async def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed (async).
        
        Args:
            key: Client identifier
            tokens: Number of tokens to consume
            
        Returns:
            True if allowed
        """
        lock = await self._get_lock()
        async with lock:
            if key not in self.buckets:
                self.buckets[key] = {
                    "tokens": self.burst_size,
                    "last_update": time.time()
                }
            
            bucket = self.buckets[key]
            now = time.time()
            
            # Refill tokens
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.burst_size,
                bucket["tokens"] + elapsed * self.refill_rate
            )
            bucket["last_update"] = now
            
            # Check tokens
            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True
            
            return False


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(
    requests_per_minute: int = 60,
    burst_size: int | None = None
) -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size
        )
    return _rate_limiter


def rate_limit(
    requests_per_minute: int = 60,
    key_func: Callable | None = None
):
    """
    Decorator for rate limiting FastAPI endpoints.
    
    Args:
        requests_per_minute: Maximum requests per minute
        key_func: Function to extract key from request (defaults to client IP)
    """
    limiter = RateLimiter(requests_per_minute=requests_per_minute)
    
    def default_key_func(request: Request) -> str:
        """Extract client IP as key."""
        return request.client.host if request.client else "unknown"
    
    get_key = key_func or default_key_func
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            key = get_key(request)
            
            if not limiter.is_allowed(key):
                remaining = limiter.get_remaining(key)
                reset_time = limiter.get_reset_time(key)
                
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(requests_per_minute),
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(int(reset_time)),
                        "Retry-After": str(int(reset_time))
                    }
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def rate_limit_by_api_key(requests_per_minute: int = 60):
    """
    Rate limit decorator that uses API key as the rate limit key.
    
    Args:
        requests_per_minute: Maximum requests per minute per API key
    """
    def extract_api_key(request: Request) -> str:
        """Extract API key from request."""
        api_key = request.headers.get("X-API-Key")
        return api_key or request.client.host or "unknown"
    
    return rate_limit(requests_per_minute, key_func=extract_api_key)
