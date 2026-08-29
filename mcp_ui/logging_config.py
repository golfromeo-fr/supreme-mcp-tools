"""
Structured logging configuration for MCP UI.

Provides trace ID generation, timing context, and formatted logging.
"""

import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from collections.abc import Generator

# Context variable for trace ID across async boundaries
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class TraceFormatter(logging.Formatter):
    """Custom formatter that includes trace ID in log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp, level, trace_id, module, and message."""
        # Get trace ID from context or use default
        trace_id = _trace_id_var.get("-")

        # Format timestamp as ISO 8601 UTC
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Truncate module name for readability
        module = record.name[-20:] if len(record.name) > 20 else record.name

        # Build formatted message
        return (
            f"[{timestamp}] "
            f"[{record.levelname:7}] "
            f"[{trace_id:8}] "
            f"[{module}] "
            f"{record.getMessage()}"
        )


_configured = False


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the entire application.

    Idempotent: repeated calls (module reload, multi-import) do not add
    duplicate handlers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Create handler with custom formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TraceFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("nicegui").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def generate_trace_id() -> str:
    """
    Generate a short trace ID for request tracking.

    Returns:
        8-character unique identifier
    """
    return str(uuid.uuid4())[:8]


def set_trace_id(trace_id: str) -> None:
    """
    Set the trace ID for the current async context.

    Args:
        trace_id: Trace ID to set
    """
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """
    Get the current trace ID from context.

    Returns:
        Current trace ID or '-' if not set
    """
    return _trace_id_var.get("-")


@contextmanager
def log_timing(
    logger: logging.Logger,
    operation: str,
    level: int = logging.DEBUG
) -> Generator[None, None, None]:
    """
    Context manager to log operation timing.

    Logs start, end, and duration of an operation.

    Args:
        logger: Logger instance to use
        operation: Description of the operation
        level: Log level (default: DEBUG)

    Example:
        >>> logger = get_logger(__name__)
        >>> with log_timing(logger, "API call"):
        ...     await fetch_data()
    """
    start_time = datetime.now(timezone.utc)
    trace_id = get_trace_id()

    logger.log(level, f"[{trace_id}] --> START: {operation}")

    try:
        yield
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.log(level, f"[{trace_id}] <-- END: {operation} ({elapsed_ms:.1f}ms)")
    except Exception as e:
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.error(f"[{trace_id}] <-- FAIL: {operation} ({elapsed_ms:.1f}ms) - {e}")
        raise


@contextmanager
def trace_context(trace_id: str | None = None) -> Generator[str, None, None]:
    """
    Context manager for setting a trace ID.

    Args:
        trace_id: Optional trace ID (generated if not provided)

    Yields:
        The trace ID being used
    """
    actual_id = trace_id or generate_trace_id()
    token = _trace_id_var.set(actual_id)
    try:
        yield actual_id
    finally:
        _trace_id_var.reset(token)
