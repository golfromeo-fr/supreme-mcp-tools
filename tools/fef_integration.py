#!/usr/bin/env python3
"""
FEF V3 Integration Helper for MCP Tools

Provides common extension handlers and registration utilities
for integrating FEF V3 into MCP tools.

Environment Variables:
    MCP_MGMT_PORT: Port for the management server (set by launcher)
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def get_management_port() -> Optional[int]:
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
    _by_endpoint: Dict[str, int] = field(default_factory=dict)
    _by_tool: Dict[str, int] = field(default_factory=dict)
    
    def record_request(
        self,
        endpoint: str,
        tool_name: Optional[str] = None,
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
    
    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class CacheConfig:
    """Cache configuration."""
    max_size: int = 1000
    ttl_seconds: int = 300
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "enabled": self.enabled
        }


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
        self.cache_config = CacheConfig()
        self._api_key: Optional[str] = None
        self._custom_data: Dict[str, Any] = {}
    
    def get_request_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Data source: Get request statistics."""
        return self.metrics.to_dict()
    
    def get_cache_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Data source: Get cache statistics."""
        total = self.metrics.cache_hits + self.metrics.cache_misses
        return {
            "size": self._custom_data.get("cache_size", 0),
            "max_size": self.cache_config.max_size,
            "hit_rate": round(
                self.metrics.cache_hits / total if total > 0 else 0.0, 4
            ),
            "hits": self.metrics.cache_hits,
            "misses": self.metrics.cache_misses,
            "ttl_seconds": self.cache_config.ttl_seconds,
            "enabled": self.cache_config.enabled
        }
    
    def set_cache_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutator: Update cache configuration."""
        previous = self.cache_config.to_dict()
        
        if "max_size" in params:
            try:
                self.cache_config.max_size = int(params["max_size"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid max_size value: {params['max_size']!r}. Must be an integer.")
        if "ttl" in params:
            try:
                self.cache_config.ttl_seconds = int(params["ttl"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid ttl value: {params['ttl']!r}. Must be an integer.")
        if "enabled" in params:
            self.cache_config.enabled = bool(params["enabled"])
        
        logger.info(f"[{self.tool_name}] Cache config updated: {self.cache_config.to_dict()}")
        
        return {
            "success": True,
            "message": "Cache configuration updated",
            "previous": previous,
            "new": self.cache_config.to_dict()
        }
    
    def set_api_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutator: Set API key."""
        new_key = params.get("key")
        if not new_key:
            return {"success": False, "message": "API key is required"}
        
        self._api_key = new_key
        logger.info(f"[{self.tool_name}] API key updated")
        
        return {
            "success": True,
            "message": "API key updated successfully"
        }
    
    def clear_cache(self, params: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def reset_counters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Action: Reset all counters."""
        self.metrics.reset()
        logger.info(f"[{self.tool_name}] Counters reset")
        
        return {
            "success": True,
            "message": "All counters have been reset"
        }
    
    def get_tool_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Data source: Get tool information."""
        return {
            "name": self.tool_name,
            "uptime_seconds": round(time.time() - self.metrics.start_time, 2),
            "total_requests": self.metrics.total_requests,
            "cache_config": self.cache_config.to_dict()
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
        name="cache_stats",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={
            "input": {"type": "object", "properties": {}},
            "output": {
                "type": "object",
                "properties": {
                    "size": {"type": "integer"},
                    "max_size": {"type": "integer"},
                    "hit_rate": {"type": "number"},
                    "hits": {"type": "integer"},
                    "misses": {"type": "integer"}
                }
            }
        },
        handler=manager.get_cache_stats,
        metadata={
            "description": "Cache statistics and hit rates",
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
    
    # Mutators
    registry.register(tool_name, Extension(
        name="cache_config",
        ext_type=ExtensionType.MUTATOR,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "max_size": {"type": "integer", "minimum": 1},
                    "ttl": {"type": "integer", "minimum": 0},
                    "enabled": {"type": "boolean"}
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            }
        },
        handler=manager.set_cache_config,
        metadata={
            "description": "Update cache configuration",
            "category": "configuration"
        }
    ))
    
    registry.register(tool_name, Extension(
        name="api_key",
        ext_type=ExtensionType.MUTATOR,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"}
                },
                "required": ["key"]
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            }
        },
        handler=manager.set_api_key,
        metadata={
            "description": "Set API key for authentication",
            "category": "configuration"
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
    
    logger.info(f"[{tool_name}] Registered {7} common FEF V3 extensions")


def setup_tool_extensions(
    tool_name: str,
    mgmt_port: Optional[int] = None,
    custom_extensions: Optional[List] = None
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
        # Try to get the existing registry from the launcher's server
        # The launcher sets MCP_MGMT_PORT and creates a server before starting the tool
        # We need to access that registry directly (same process)
        from launcher.tool_extensions.registry import _global_registries
        
        # Look for an existing registry for this tool
        if tool_name in _global_registries:
            registry = _global_registries[tool_name]
            logger.info(f"[{tool_name}] Using existing registry from launcher")
        else:
            # Create a new registry and server (standalone mode)
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
            )
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
