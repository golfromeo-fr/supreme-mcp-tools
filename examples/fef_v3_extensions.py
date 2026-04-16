#!/usr/bin/env python3
"""
Example: Adding FEF V3 Extensions to MCP Tools

This example demonstrates how to add extensions to existing MCP tools
using the Flexible Extensibility Framework V3.
"""

import asyncio
import logging
import time
from typing import Any

# Add parent directory to path
import sys
import pathlib
sys.path.insert(0, (pathlib.Path(__file__).parent / "..").resolve())

from launcher.tool_extensions import Extension, ExtensionRegistry, ExtensionType, ExtensionHTTPServer

logger = logging.getLogger(__name__)


# ============================================================================
# Example Extension Handlers
# ============================================================================

def get_request_stats(params: dict[str, Any]) -> dict[str, Any]:
    """
    Data source: Get request statistics.
    
    Args:
        params: Query parameters (time_range, group_by)
        
    Returns:
        Request statistics
    """
    time_range = params.get("time_range", "1h")
    
    # In a real implementation, this would query actual metrics
    return {
        "total": 1234,
        "success": 1200,
        "errors": 34,
        "avg_response_time_ms": 45.2,
        "time_range": time_range,
        "by_endpoint": {
            "/api/search": 500,
            "/api/fetch": 400,
            "/api/health": 334
        }
    }


def get_cache_stats(params: dict[str, Any]) -> dict[str, Any]:
    """
    Data source: Get cache statistics.
    
    Args:
        params: Query parameters
        
    Returns:
        Cache statistics
    """
    return {
        "size": 150,
        "max_size": 1000,
        "hit_rate": 0.85,
        "hits": 8500,
        "misses": 1500,
        "evictions": 50
    }


def set_cache_config(params: dict[str, Any]) -> dict[str, Any]:
    """
    Mutator: Update cache configuration.
    
    Args:
        params: Configuration parameters (max_size, ttl)
        
    Returns:
        Update result
    """
    max_size = params.get("max_size")
    ttl = params.get("ttl")
    
    # In a real implementation, this would update the actual cache config
    logger.info(f"Updating cache config: max_size={max_size}, ttl={ttl}")
    
    return {
        "success": True,
        "message": "Cache configuration updated",
        "previous": {"max_size": 1000, "ttl": 300},
        "new": {"max_size": max_size or 1000, "ttl": ttl or 300}
    }


def set_api_key(params: dict[str, Any]) -> dict[str, Any]:
    """
    Mutator: Update API key.
    
    Args:
        params: New API key
        
    Returns:
        Update result
    """
    new_key = params.get("key")
    
    if not new_key:
        return {"success": False, "message": "API key is required"}
    
    # In a real implementation, this would update the actual API key
    logger.info("API key updated")
    
    return {
        "success": True,
        "message": "API key updated successfully"
    }


def clear_cache(params: dict[str, Any]) -> dict[str, Any]:
    """
    Action: Clear the cache.
    
    Args:
        params: Cache type to clear (all, api, query)
        
    Returns:
        Clear result
    """
    cache_type = params.get("cache_type", "all")
    
    # In a real implementation, this would clear the actual cache
    cleared_count = 150  # Example
    
    logger.info(f"Cleared {cache_type} cache ({cleared_count} entries)")
    
    return {
        "success": True,
        "cleared": cleared_count,
        "cache_type": cache_type
    }


def reset_counters(params: dict[str, Any]) -> dict[str, Any]:
    """
    Action: Reset all counters.
    
    Args:
        params: Counter names to reset (or "all")
        
    Returns:
        Reset result
    """
    counters = params.get("counters", "all")
    
    # In a real implementation, this would reset actual counters
    logger.info(f"Reset counters: {counters}")
    
    return {
        "success": True,
        "reset": counters,
        "timestamp": time.time()
    }


# ============================================================================
# Extension Registration Helper
# ============================================================================

def register_tool_extensions(tool_name: str, registry: ExtensionRegistry) -> None:
    """
    Register all extensions for a tool.
    
    Args:
        tool_name: Name of the tool
        registry: Extension registry instance
    """
    # Data Sources
    registry.register(tool_name, Extension(
        name="request_stats",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "string",
                        "enum": ["1h", "24h", "7d"],
                        "description": "Time range for statistics"
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["endpoint", "status", "hour"],
                        "description": "Group results by field"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer"},
                    "success": {"type": "integer"},
                    "errors": {"type": "integer"},
                    "avg_response_time_ms": {"type": "number"}
                }
            }
        },
        handler=get_request_stats,
        metadata={
            "description": "Request statistics and metrics",
            "category": "metrics",
            "tags": ["performance", "monitoring"]
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
                    "hit_rate": {"type": "number"}
                }
            }
        },
        handler=get_cache_stats,
        metadata={
            "description": "Cache statistics",
            "category": "metrics",
            "tags": ["cache", "performance"]
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
                    "max_size": {
                        "type": "integer",
                        "description": "Maximum cache size"
                    },
                    "ttl": {
                        "type": "integer",
                        "description": "Time to live in seconds"
                    }
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
        handler=set_cache_config,
        metadata={
            "description": "Update cache configuration",
            "category": "configuration",
            "tags": ["cache", "config"]
        }
    ))
    
    registry.register(tool_name, Extension(
        name="api_key",
        ext_type=ExtensionType.MUTATOR,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "New API key"
                    }
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
        handler=set_api_key,
        metadata={
            "description": "Update API key",
            "category": "configuration",
            "tags": ["auth", "config"]
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
                        "enum": ["all", "api", "query"],
                        "description": "Type of cache to clear"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "cleared": {"type": "integer"}
                }
            }
        },
        handler=clear_cache,
        metadata={
            "description": "Clear application cache",
            "category": "maintenance",
            "tags": ["cache", "maintenance"]
        }
    ))
    
    registry.register(tool_name, Extension(
        name="reset_counters",
        ext_type=ExtensionType.ACTION,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "counters": {
                        "type": "string",
                        "description": "Counter names to reset (or 'all')"
                    }
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "reset": {"type": "string"}
                }
            }
        },
        handler=reset_counters,
        metadata={
            "description": "Reset all counters",
            "category": "maintenance",
            "tags": ["counters", "maintenance"]
        }
    ))
    
    logger.info(f"Registered 6 extensions for {tool_name}")


# ============================================================================
# Example: Running a Tool with Management Server
# ============================================================================

def _get_example_mgmt_port(tool_name: str) -> int:
    """Get management port for example tool from ports.json.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Port number
        
    Raises:
        ValueError: If port not found in ports.json
    """
    try:
        from launcher.launcher_config import load_ports_config
        ports_config = load_ports_config()
        assignments = ports_config.get("assignments", {})
        mgmt_ports = assignments.get("mgmt", {})
        port = mgmt_ports.get(tool_name)
        if port is None:
            raise ValueError(
                f"Port for {tool_name} not found in ports.json assignments.mgmt"
            )
        return port
    except Exception as e:
        raise ValueError(
            f"Could not load port from ports.json: {e}. "
            "Please ensure ports.json is properly configured."
        )


async def run_example():
    """Run an example tool with management server."""
    tool_name = "example_tool"
    mgmt_port = _get_example_mgmt_port(tool_name)
    
    # Create extension registry
    registry = ExtensionRegistry()
    
    # Register extensions
    register_tool_extensions(tool_name, registry)
    
    # Create and start management server
    server = ExtensionHTTPServer(
        tool_name=tool_name,
        registry=registry,
        port=mgmt_port
    )
    
    print(f"\n{'='*60}")
    print(f"Example Tool with FEF V3 Management Server")
    print(f"{'='*60}")
    print(f"Management API: http://localhost:{mgmt_port}")
    print(f"\nTry these commands:")
    print(f"  curl http://localhost:{mgmt_port}/health")
    print(f"  curl http://localhost:{mgmt_port}/extensions")
    print(f"  curl -X POST http://localhost:{mgmt_port}/extensions/request_stats/query")
    print(f"  curl -X POST http://localhost:{mgmt_port}/extensions/clear_cache/execute")
    print(f"{'='*60}\n")
    
    await server.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_example())
