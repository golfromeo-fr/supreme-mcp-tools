#!/usr/bin/env python3
"""
FEF V3 Integration Helper for MCP Tools

Provides common extension handlers and registration utilities
for integrating FEF V3 into MCP tools.

Environment Variables:
    MCP_MGMT_PORT: Port for the management server (set by launcher)
"""

import logging
import os
import time
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def get_management_port() -> int | None:
    """
    Get the management port from environment variable.
    
    Returns:
        Management port if set, None otherwise
    """
    port_str = os.environ.get("MCP_MGMT_PORT")
    if port_str:
        try:
            return int(port_str)
        except ValueError:
            logger.warning(f"Invalid MCP_MGMT_PORT value: {port_str}")
    return None


@dataclass
class ToolMetrics:
    """Metrics tracker for an MCP tool."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(default_factory=time.time)
    _by_endpoint: dict[str, int] = field(default_factory=dict)
    _by_tool: dict[str, int] = field(default_factory=dict)
    
    def record_request(
        self,
        endpoint: str,
        tool_name: str | None = None,
        success: bool = True,
        duration_ms: float = 0.0
    ):
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_duration_ms += duration_ms
        
        self._by_endpoint[endpoint] = self._by_endpoint.get(endpoint, 0) + 1
        if tool_name:
            self._by_tool[tool_name] = self._by_tool.get(tool_name, 0) + 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        uptime = time.time() - self.start_time
        avg_duration = (
            self.total_duration_ms / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_response_time_ms": round(avg_duration, 2),
            "uptime_seconds": round(uptime, 2),
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(hit_rate, 4)
            },
            "by_endpoint": self._by_endpoint,
            "by_tool": self._by_tool
        }
    
    def reset(self):
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_duration_ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        self._by_endpoint.clear()
        self._by_tool.clear()


class ToolExtensionManager:
    """
    Manages FEF V3 extensions for an MCP tool.
    
    Provides common extension handlers and metrics tracking.
    """
    
    def __init__(self, tool_name: str):
        """
        Initialize the extension manager.
        
        Args:
            tool_name: Name of the MCP tool
        """
        self.tool_name = tool_name
        self.metrics = ToolMetrics()
        self._custom_data: dict[str, Any] = {}
    
    def get_request_stats(self, params: dict[str, Any]) -> dict[str, Any]:
        """Data source: Get request statistics."""
        return self.metrics.to_dict()

    def clear_cache(self, params: dict[str, Any]) -> dict[str, Any]:
        """Action: Clear cache."""
        cache_type = params.get("cache_type", "all")
        cleared = self._custom_data.get("cache_size", 0)
        
        self._custom_data["cache_size"] = 0
        logger.info(f"[{self.tool_name}] Cache cleared: type={cache_type}")
        
        return {
            "success": True,
            "cleared_entries": cleared,
            "cache_type": cache_type
        }
    
    def reset_counters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Action: Reset all counters."""
        self.metrics.reset()
        logger.info(f"[{self.tool_name}] Counters reset")
        
        return {
            "success": True,
            "message": "All counters have been reset"
        }
    
    def get_tool_info(self, params: dict[str, Any]) -> dict[str, Any]:
        """Data source: Get tool information."""
        return {
            "name": self.tool_name,
            "uptime_seconds": round(time.time() - self.metrics.start_time, 2),
            "total_requests": self.metrics.total_requests,
        }


def register_common_extensions(
    tool_name: str,
    registry,
    manager: ToolExtensionManager
):
    """
    Register common FEF V3 extensions for a tool.
    
    Args:
        tool_name: Name of the tool
        registry: ExtensionRegistry instance
        manager: ToolExtensionManager instance
    """
    from launcher.tool_extensions import Extension, ExtensionType
    
    # Data Sources
    registry.register(tool_name, Extension(
        name="request_stats",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "enum": ["1h", "24h", "7d"]}
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "total_requests": {"type": "integer"},
                    "successful_requests": {"type": "integer"},
                    "failed_requests": {"type": "integer"},
                    "avg_response_time_ms": {"type": "number"}
                }
            }
        },
        handler=manager.get_request_stats,
        metadata={
            "description": "Request statistics and performance metrics",
            "category": "metrics"
        }
    ))

    registry.register(tool_name, Extension(
        name="tool_info",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={
            "input": {"type": "object", "properties": {}},
            "output": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "uptime_seconds": {"type": "number"},
                    "total_requests": {"type": "integer"}
                }
            }
        },
        handler=manager.get_tool_info,
        metadata={
            "description": "Tool information and status",
            "category": "info"
        }
    ))

    # Actions
    registry.register(tool_name, Extension(
        name="clear_cache",
        ext_type=ExtensionType.ACTION,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "cache_type": {
                        "type": "string",
                        "enum": ["all", "search", "fetch", "schema"],
                        "default": "all"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "cleared_entries": {"type": "integer"}
                }
            }
        },
        handler=manager.clear_cache,
        metadata={
            "description": "Clear cached data",
            "category": "maintenance"
        }
    ))
    
    registry.register(tool_name, Extension(
        name="reset_counters",
        ext_type=ExtensionType.ACTION,
        schema={
            "input": {"type": "object", "properties": {}},
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            }
        },
        handler=manager.reset_counters,
        metadata={
            "description": "Reset all metrics counters",
            "category": "maintenance"
        }
    ))
    
    logger.info(f"[{tool_name}] Registered {4} common FEF V3 extensions")


def setup_tool_extensions(
    tool_name: str,
    mgmt_port: int | None = None,
    custom_extensions: list | None = None
) -> tuple:
    """
    Set up FEF V3 extensions for a tool.
    
    This is the main entry point for tool integration.
    
    When called from a tool launched by the launcher, the MCP_MGMT_PORT
    environment variable will be set and the launcher will already have
    an ExtensionHTTPServer running on that port. In this case, we register
    extensions directly with that server's registry.
    
    When called standalone (no MCP_MGMT_PORT), a new registry and server
    are created.
    
    Args:
        tool_name: Name of the tool
        mgmt_port: Port for the management server (optional, uses env var if not set)
        custom_extensions: Optional list of custom Extension objects
        
    Returns:
        Tuple of (extension_manager, registry, http_server)
        Note: http_server will be None if using launcher's existing server
    """
    from launcher.tool_extensions import ExtensionRegistry, ExtensionHTTPServer
    
    # Determine the management port
    env_port = get_management_port()
    if mgmt_port is None:
        mgmt_port = env_port
    elif env_port is not None and mgmt_port != env_port:
        logger.warning(
            f"[{tool_name}] mgmt_port ({mgmt_port}) differs from MCP_MGMT_PORT ({env_port}). "
            f"Using MCP_MGMT_PORT to connect to launcher's registry."
        )
        mgmt_port = env_port
    
    # Create extension manager
    manager = ToolExtensionManager(tool_name)
    
    # Check if launcher already has a registry server running
    # If MCP_MGMT_PORT is set, the launcher started an ExtensionHTTPServer
    # We need to get the registry from that server and register extensions with it
    registry = None
    http_server = None
    
    if mgmt_port is not None:
        from launcher.tool_extensions.registry import _global_registries

        registry = _global_registries.get(tool_name)
        if registry is not None:
            logger.info(f"[{tool_name}] Using existing registry from launcher")
        else:
            registry = ExtensionRegistry()
            http_server = ExtensionHTTPServer(
                tool_name=tool_name,
                registry=registry,
                port=mgmt_port
            )
            logger.info(f"[{tool_name}] Created new registry and server on port {mgmt_port}")
    else:
        # No management port specified - create everything
        registry = ExtensionRegistry()
        # Get port from ports.json or raise error
        from launcher.launcher_config import load_ports_config
        try:
            ports_config = load_ports_config()
            port = ports_config.get("assignments", {}).get("mgmt", {}).get(tool_name)
            if port is None:
                raise ValueError(
                    f"Port for {tool_name} not found in ports.json assignments.mgmt"
                )
        except Exception as e:
            raise ValueError(
                f"Could not determine management port for {tool_name}: {e}. "
                "Please configure ports.json with the tool's management port."
            ) from e
        http_server = ExtensionHTTPServer(
            tool_name=tool_name,
            registry=registry,
            port=port
        )
        logger.info(f"[{tool_name}] FEF V3 standalone mode on port {port}")
    
    # Register common extensions
    register_common_extensions(tool_name, registry, manager)
    
    # Register custom extensions if provided
    if custom_extensions:
        for ext in custom_extensions:
            registry.register(tool_name, ext)
            logger.info(f"[{tool_name}] Registered custom extension: {ext.name}")
    
    logger.info(f"[{tool_name}] FEF V3 extensions configured (port={mgmt_port}, server={'new' if http_server else 'existing'})")
    
    return manager, registry, http_server
