"""
Retry with Exponential Backoff for FEF V3

Provides configurable retry logic with exponential backoff.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: int = 2
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


# Default config
DEFAULT_CONFIG = RetryConfig()


async def retry_with_backoff(
    func: Callable,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    fallback: Optional[Callable] = None
) -> Any:
    """
    Execute function with exponential backoff retry.
    
    Args:
        func: Async function to execute
        config: Retry configuration (uses default if not provided)
        on_retry: Optional callback called on each retry (attempt, exception, delay)
        fallback: Optional fallback function if all retries fail
        
    Returns:
        Result from func or fallback
        
    Raises:
        RetryExhaustedError: If all retries fail and no fallback provided
    """
    cfg = config or DEFAULT_CONFIG
    last_exception = None
    
    for attempt in range(cfg.max_retries):
        try:
            return await func()
        except cfg.retryable_exceptions as e:
            last_exception = e
            
            if attempt == cfg.max_retries - 1:
                # Last attempt failed
                break
            
            delay = cfg.calculate_delay(attempt)
            
            logger.warning(
                f"Retry {attempt + 1}/{cfg.max_retries} after {delay:.2f}s: {e}"
            )
            
            if on_retry:
                try:
                    on_retry(attempt, e, delay)
                except Exception:
                    pass  # Don't let callback errors break retry logic
            
            await asyncio.sleep(delay)
    
    # All retries exhausted
    if fallback:
        logger.info("All retries exhausted, using fallback")
        return await fallback()
    
    raise RetryExhaustedError(
        f"All {cfg.max_retries} retry attempts failed",
        last_exception=last_exception,
        attempts=cfg.max_retries
    )


def retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None,
    fallback: Optional[Callable] = None
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        config: Retry configuration
        on_retry: Callback on each retry
        fallback: Fallback function if all retries fail
        
    Example:
        @retry(config=RetryConfig(max_retries=5))
        async def fetch_data():
            return await http_client.get(url)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async def call_func():
                return await func(*args, **kwargs)
            
            return await retry_with_backoff(
                call_func,
                config=config,
                on_retry=on_retry,
                fallback=fallback
            )
        return wrapper
    return decorator


class RetryableHTTPClient:
    """
    HTTP client with built-in retry support.
    
    Wraps aiohttp with automatic retry on transient failures.
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the retryable HTTP client.
        
        Args:
            retry_config: Retry configuration
            timeout: Request timeout in seconds
        """
        import aiohttp
        
        self.retry_config = retry_config or RetryConfig(
            max_retries=3,
            retryable_exceptions=(
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError
            )
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self):
        """Get or create HTTP session."""
        import aiohttp
        
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout
            )
        return self._session
    
    async def get(self, url: str, **kwargs) -> Any:
        """
        Execute GET request with retry.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            Response JSON data
        """
        async def do_get():
            session = await self._get_session()
            async with session.get(url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        
        return await retry_with_backoff(do_get, config=self.retry_config)
    
    async def post(self, url: str, data: Optional[dict] = None, **kwargs) -> Any:
        """
        Execute POST request with retry.
        
        Args:
            url: URL to request
            data: Request body data
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            Response JSON data
        """
        async def do_post():
            session = await self._get_session()
            async with session.post(url, json=data, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        
        return await retry_with_backoff(do_post, config=self.retry_config)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
