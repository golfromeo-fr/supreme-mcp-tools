"""
State Management for the Management UI.

NiceGUI handles state per-client automatically. This module provides
a simple state management approach using client-specific instances.
"""

from nicegui import ui, context
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AppState:
    """
    Application state for a single client.
    
    This is NOT a Pydantic model - it's a dataclass that holds
    transient UI state. NiceGUI manages per-client instances.
    """
    # Tool selection
    selected_tool: Optional[str] = None
    
    # Loading states
    tools_loading: bool = False
    extensions_loading: bool = False
    mutation_loading: bool = False
    
    # Connection status
    connection_status: str = "connected"  # connected, disconnected, error
    
    # Cached data
    tools: list = field(default_factory=list)
    extensions: Dict[str, list] = field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    
    # Last refresh time
    last_refresh: Optional[str] = None


# Global state dictionary keyed by client.id
# Note: NiceGUI's context.client.storage is preferred for persistence
_client_states: Dict[int, AppState] = {}


def get_state() -> AppState:
    """Get or create state for current client."""
    try:
        client_id = context.client.id
        if client_id not in _client_states:
            _client_states[client_id] = AppState()
        return _client_states[client_id]
    except Exception:
        # If context is not available, return a global state (for testing/development)
        return _global_state


# Global fallback state for when client context is not available
_global_state = AppState()


def clear_state():
    """Clear state for current client."""
    try:
        client_id = context.client.id
        if client_id in _client_states:
            del _client_states[client_id]
    except Exception:
        pass
