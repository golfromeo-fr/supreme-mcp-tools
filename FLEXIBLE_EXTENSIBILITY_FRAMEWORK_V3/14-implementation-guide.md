# Implementation Guide

## Adding Extensions to a Tool

### Step 1: Import Required Modules

```python
# In your tool's main module
from launcher.tool_extensions.registry import ExtensionRegistry, Extension, ExtensionType
```

### Step 2: Create Extension Handlers

```python
# Data source handler
def get_api_calls(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for API call statistics."""
    time_range = params.get("time_range", "1h")
    # ... implementation
    return {
        "total": 1234,
        "by_endpoint": {"/api/users": 500}
    }

# Mutator handler
def set_api_key(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for setting API key."""
    new_key = params.get("key")
    # ... implementation
    return {"success": True, "message": "API key updated"}

# Action handler
def clear_cache(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for clearing cache."""
    cache_type = params.get("cache_type", "all")
    # ... implementation
    return {"cleared": 150}
```

### Step 3: Register Extensions

```python
# In your tool's initialization
registry = ExtensionRegistry.get_instance()

# Register data source
registry.register("webmcp", Extension(
    name="api_calls",
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
                "total": {"type": "integer"},
                "by_endpoint": {"type": "object"}
            }
        }
    },
    handler=get_api_calls,
    metadata={
        "description": "API call statistics",
        "category": "metrics"
    }
))

# Register mutator
registry.register("webmcp", Extension(
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
    handler=set_api_key,
    metadata={
        "description": "Set API key for authentication",
        "category": "configuration"
    }
))

# Register action
registry.register("webmcp", Extension(
    name="clear_cache",
    ext_type=ExtensionType.ACTION,
    schema={
        "input": {
            "type": "object",
            "properties": {
                "cache_type": {"type": "string", "enum": ["all", "api", "query"]}
            }
        },
        "output": {
            "type": "object",
            "properties": {
                "cleared": {"type": "integer"}
            }
        }
    },
    handler=clear_cache,
    metadata={
        "description": "Clear application cache",
        "category": "maintenance"
    }
))
```

### Step 4: Start Management Server

```python
# In your tool's main entry point
from launcher.tool_extensions.http_server import ExtensionHTTPServer
from launcher.config.manager import ConfigManager

async def main():
    # Initialize components
    registry = ExtensionRegistry.get_instance()
    config_manager = ConfigManager("webmcp")
    
    # Create and start HTTP server
    server = ExtensionHTTPServer(
        tool_name="webmcp",
        registry=registry,
        config_manager=config_manager,
        port=9001
    )
    
    await server.start()
```

## Integrating with Launcher

### Step 1: Update ServerManager

```python
# launcher/server_manager.py

class ServerManager:
    """Manages tool server lifecycle."""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.port_manager = PortManager()
    
    async def start_tool(self, tool_name: str):
        """Start a tool and register it."""
        # Allocate ports
        mcp_port = self.port_manager.allocate()
        mgmt_port = self.port_manager.allocate()
        
        # Start tool process with both ports
        process = await self._start_process(
            tool_name,
            mcp_port=mcp_port,
            mgmt_port=mgmt_port
        )
        
        # Wait for tool to be ready
        await self._wait_for_health(f"http://127.0.0.1:{mgmt_port}/health")
        
        # Register with service registry
        self.service_registry.register(
            name=tool_name,
            management_url=f"http://127.0.0.1:{mgmt_port}",
            mcp_port=mcp_port
        )
        
        return process
```

### Step 2: Update Management Server

```python
# launcher/management_server.py

from launcher.distributed_registry import DistributedExtensionRegistry

class ManagementServer:
    """Main management server."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.registry = DistributedExtensionRegistry(service_registry)
        self.app = FastAPI(title="Supreme MCP Tools Management")
        self._register_routes()
    
    def _register_routes(self):
        @self.app.get("/api/tools")
        async def list_tools():
            return {"tools": await self.registry.list_tools()}
        
        @self.app.post("/api/tools/{tool}/extensions/{ext}/query")
        async def query_extension(tool: str, ext: str, request: Dict):
            return await self.registry.query(tool, ext, request.get("params"))
        
        # ... more routes
```

## Testing Extensions

### Unit Testing

```python
import pytest
from launcher.tool_extensions.registry import ExtensionRegistry, Extension, ExtensionType

def test_extension_registration():
    """Test that extensions can be registered."""
    registry = ExtensionRegistry()
    registry.reset()
    
    ext = Extension(
        name="test_ext",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={"input": {}, "output": {}},
        handler=lambda params: {"result": "ok"}
    )
    
    registry.register("test_tool", ext)
    
    retrieved = registry.get_extension("test_tool", "test_ext")
    assert retrieved is not None
    assert retrieved.name == "test_ext"

def test_extension_query():
    """Test querying a data source extension."""
    registry = ExtensionRegistry()
    registry.reset()
    
    def handler(params):
        return {"value": params.get("key", "default")}
    
    ext = Extension(
        name="test_data",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={},
        handler=handler
    )
    
    registry.register("test_tool", ext)
    
    result = registry.query("test_tool", "test_data", {"key": "test"})
    assert result == {"value": "test"}
```

### Integration Testing

```python
import pytest
import asyncio
from httpx import AsyncClient
from launcher.tool_extensions.http_server import ExtensionHTTPServer

@pytest.mark.asyncio
async def test_http_api():
    """Test the HTTP API endpoints."""
    registry = ExtensionRegistry()
    registry.reset()
    
    # Register test extension
    registry.register("test_tool", Extension(
        name="test",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={},
        handler=lambda params: {"status": "ok"}
    ))
    
    # Create server
    server = ExtensionHTTPServer(
        tool_name="test_tool",
        registry=registry,
        config_manager=MockConfigManager(),
        port=9999
    )
    
    # Start server in background
    task = asyncio.create_task(server.start())
    await asyncio.sleep(0.5)  # Wait for server to start
    
    # Test endpoint
    async with AsyncClient() as client:
        response = await client.get("http://localhost:9999/extensions")
        assert response.status_code == 200

## Using the FEF Integration Helper

A simpler integration path is available using `tools/fef_integration.py`:

```python
# In your tool's main module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.fef_integration import (
    ToolExtensionManager,
    register_common_extensions,
    setup_tool_extensions
)
from launcher.tool_extensions import Extension, ExtensionType

# Create custom extensions for your tool
def get_custom_stats(params):
    return {"custom_metric": 123}

custom_extensions = [
    Extension(
        name="custom_stats",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={"input": {}, "output": {"type": "object"}},
        handler=get_custom_stats,
        metadata={"description": "Custom tool statistics"}
    )
]

# Set up FEF V3 with common + custom extensions
manager, registry, http_server = setup_tool_extensions(
    tool_name="mytool",
    mgmt_port=9005,
    custom_extensions=custom_extensions
)

# Start management server
if __name__ == "__main__":
    import asyncio
    asyncio.run(http_server.start())
```

## Tool-Specific Extensions

Each MCP tool in the repository has its own set of FEF V3 extensions:

| Tool | MCP Port | Mgmt Port | Data Sources | Mutators | Actions |
|------|----------|-----------|--------------|----------|---------|
| webmcp | 8001 | 9001 | request_stats, cache_stats, search_stats, fetch_stats | cache_config, api_key, search_config | clear_cache, reset_counters |
| oraclemcp | 8000 | 9000 | query_stats, connection_pool, schema_cache | pool_config | reset_connections |
| simplemcp | 8002 | 9002 | request_stats, cache_stats, tool_usage | cache_config, api_key | clear_cache, reset_counters |
| convertermcp | 8003 | 9003 | conversion_stats, format_usage | output_config | clear_cache |
| ragmcp | 8004 | 9004 | vector_db_stats, embedding_stats, collection_stats | collection_config | reindex |
        assert len(response.json()) == 1
    
    # Cleanup
    await server.stop()
    task.cancel()
```

## Best Practices

### Extension Naming

- Use snake_case for extension names
- Be descriptive: `api_call_stats` not `stats`
- Prefix related extensions: `cache_size`, `cache_clear`, `cache_config`

### Schema Design

- Always define input and output schemas
- Use enums for constrained values
- Include descriptions for all properties
- Mark required fields

### Error Handling

```python
def safe_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler with proper error handling."""
    try:
        # Validate input
        if "required_field" not in params:
            raise ValueError("Missing required_field")
        
        # Process
        result = process(params)
        
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"success": False, "error": "Internal error"}
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def logged_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler with structured logging."""
    logger.info(f"Executing handler with params: {params}")
    
    start_time = time.time()
    result = process(params)
    duration = time.time() - start_time
    
    logger.info(f"Handler completed in {duration:.3f}s")
    
    return result
```
