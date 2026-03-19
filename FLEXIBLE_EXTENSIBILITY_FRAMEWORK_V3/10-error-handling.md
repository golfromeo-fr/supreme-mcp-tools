# Error Handling and Resilience

## Circuit Breaker Pattern

The circuit breaker prevents cascading failures:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Circuit Breaker States                       │
│                                                                 │
│  ┌──────────┐     Failure threshold      ┌──────────┐          │
│  │  CLOSED  │ ──────────────────────────► │   OPEN   │          │
│  │ (Normal) │                             │ (Failing)│          │
│  └──────────┘                             └──────────┘          │
│       ▲                                        │                │
│       │                                        │                │
│       │ Success                    Recovery     │                │
│       │                            timeout      │                │
│       │                                        ▼                │
│       └─────────────────────────────────── ┌──────────┐         │
│                                            │HALF_OPEN │         │
│                                            │(Testing) │         │
│                                            └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Circuit Breaker Implementation

```python
import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


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
        func: Callable,
        fallback: Optional[Callable] = None
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
```

## Retry with Exponential Backoff

```python
import asyncio
import logging

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Execute function with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
            await asyncio.sleep(delay)
```

## Dead Letter Queue

For failed operations that need manual review:

```python
import json
import uuid
import time
from pathlib import Path
from typing import Any, Dict


class DeadLetterQueue:
    """Queue for failed operations requiring manual review."""
    
    def __init__(self, queue_dir: str = "~/.config/supreme-mcp-tools/dlq"):
        self.queue_dir = Path(queue_dir).expanduser()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
    
    def enqueue(
        self,
        operation: str,
        tool_name: str,
        params: Dict[str, Any],
        error: str
    ) -> str:
        """Add failed operation to queue."""
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "operation": operation,
            "tool_name": tool_name,
            "params": params,
            "error": error,
            "timestamp": time.time(),
            "status": "pending"
        }
        
        entry_path = self.queue_dir / f"{entry_id}.json"
        with open(entry_path, "w") as f:
            json.dump(entry, f, indent=2)
        
        return entry_id
    
    def get_pending(self) -> list:
        """Get all pending entries."""
        entries = []
        for path in self.queue_dir.glob("*.json"):
            with open(path, "r") as f:
                entry = json.load(f)
                if entry["status"] == "pending":
                    entries.append(entry)
        return entries
    
    def mark_processed(self, entry_id: str) -> None:
        """Mark an entry as processed."""
        entry_path = self.queue_dir / f"{entry_id}.json"
        if entry_path.exists():
            with open(entry_path, "r") as f:
                entry = json.load(f)
            entry["status"] = "processed"
            entry["processed_at"] = time.time()
            with open(entry_path, "w") as f:
                json.dump(entry, f, indent=2)
```

## Error Handling Best Practices

### 1. Always Use Circuit Breakers for External Calls

```python
# Good
result = await circuit_breaker.execute(
    "webmcp",
    lambda: http_client.post(url, data)
)

# Bad - no circuit breaker
result = await http_client.post(url, data)
```

### 2. Implement Fallbacks for Critical Operations

```python
async def query_with_fallback(tool_name: str, extension: str, params: dict):
    """Query with fallback to cached data."""
    try:
        return await circuit_breaker.execute(
            tool_name,
            lambda: distributed_registry.query(tool_name, extension, params)
        )
    except CircuitBreakerOpenError:
        # Fallback to cached data
        cached = await cache.get(f"query:{tool_name}:{extension}")
        if cached:
            return cached
        raise
```

### 3. Log Errors with Context

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = await process_request(params)
except Exception as e:
    logger.error(
        "Failed to process request",
        extra={
            "tool": tool_name,
            "extension": extension,
            "params": params,
            "error": str(e)
        },
        exc_info=True
    )
    raise
```

### 4. Use Dead Letter Queue for Unrecoverable Errors

```python
try:
    await process_mutation(tool_name, extension, params)
except Exception as e:
    dlq.enqueue(
        operation="mutate",
        tool_name=tool_name,
        params=params,
        error=str(e)
    )
    logger.error(f"Added to dead letter queue: {e}")
```
