"""
MCP Launcher Package

A unified launcher system for running multiple MCP tools in a single process.
"""

from .config import Config
from .errors import (
    ConfigError,
    DiscoveryError,
    LauncherError,
    PortConflictError,
    ServerRuntimeError,
    ServerStartupError,
    ValidationError,
)
from .port_manager import PortManager
from .server_manager import ServerManager, ServerInstance, run_servers_concurrently
from .tool_discovery import ToolDiscovery, ToolMetadata

__version__ = "1.0.0"
__all__ = [
    "Config",
    "ConfigError",
    "DiscoveryError",
    "LauncherError",
    "PortConflictError",
    "PortManager",
    "ServerInstance",
    "ServerManager",
    "ServerRuntimeError",
    "ServerStartupError",
    "ToolDiscovery",
    "ToolMetadata",
    "ValidationError",
    "run_servers_concurrently",
]
