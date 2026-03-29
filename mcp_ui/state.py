"""
State Management for the Management UI.

Simple state container with change notification and comprehensive logging.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

from .logging_config import get_logger, get_trace_id

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class AppState:
    """
    Simple state container with change notification.

    Thread-safe for concurrent access via asyncio lock.
    All state changes are logged with trace ID for debugging.

    Attributes:
        tools: List of registered tools
        selected_tool: Currently selected tool (if any)
        selected_tool_detail: Cached tool detail for selected tool
        loading_tools: Whether tools list is being loaded
        loading_detail: Whether tool detail is being loaded
        last_error: Last error message (if any)
    """

    # Data state
    tools: list = field(default_factory=list)
    selected_tool: Optional[str] = None
    selected_tool_detail: Optional[Any] = None  # ToolDetail or similar
    tool_detail_cache: dict = field(default_factory=dict)  # tool_name -> ToolDetail
    loading_tools: bool = False
    loading_detail: bool = False

    # Connection state
    connection_status: str = "connected"

    # Error state
    last_error: Optional[str] = None

    # Internal lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def set_tools(self, tools: list) -> None:
        """Update tools list and log."""
        self.tools = tools
        logger.debug(
            f"[{get_trace_id()}] State: set_tools count={len(tools)} "
            f"names={[t.name if hasattr(t, 'name') else str(t) for t in tools]}"
        )

    def select_tool(self, tool_name: Optional[str]) -> None:
        """Select a tool by name."""
        if tool_name is None:
            self.selected_tool = None
            logger.debug(f"[{get_trace_id()}] State: select_tool None")
            return

        self.selected_tool = tool_name
        logger.debug(f"[{get_trace_id()}] State: select_tool name={tool_name}")

    def set_loading(self, what: str, loading: bool) -> None:
        """Update loading state with logging."""
        if what == "tools":
            self.loading_tools = loading
        elif what == "detail":
            self.loading_detail = loading
        else:
            logger.warning(f"[{get_trace_id()}] State: unknown loading type '{what}'")
            return

        logger.debug(f"[{get_trace_id()}] State: loading_{what}={loading}")

    def set_error(self, error: Optional[str]) -> None:
        """Set/clear error with logging."""
        self.last_error = error
        if error:
            logger.error(f"[{get_trace_id()}] State: error='{error}'")
            self.connection_status = "error"
        else:
            logger.debug(f"[{get_trace_id()}] State: error cleared")
            self.connection_status = "connected"

    def clear(self) -> None:
        """Reset state."""
        self.tools = []
        self.selected_tool = None
        self.selected_tool_detail = None
        self.tool_detail_cache.clear()
        self.loading_tools = False
        self.loading_detail = False
        self.last_error = None
        self.connection_status = "connected"
        logger.debug(f"[{get_trace_id()}] State: cleared")

    async def with_lock(self, coro) -> None:
        """Execute async operation with lock for thread safety."""
        async with self._lock:
            await coro


# Global state instance (single instance for the application)
_state: Optional[AppState] = None


def get_state() -> AppState:
    """Get the global state instance, creating if needed."""
    global _state
    if _state is None:
        _state = AppState()
        logger.debug(f"[{get_trace_id()}] State: created global instance")
    return _state


def reset_state() -> None:
    """Reset the global state (useful for testing)."""
    global _state
    _state = AppState()
    logger.debug(f"[{get_trace_id()}] State: reset global instance")
