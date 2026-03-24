# Core Components

This directory contains the core components of the Flexible Extensibility Framework V3.

## Files

| File | Description |
|------|-------------|
| [README.md](./README.md) | This file - overview of core components |
| [05-core-components-completion.md](./05-core-components-completion.md) | Complete code implementations for all core classes |

## Components Overview

### 1. Local ExtensionRegistry

Each tool runs its own `ExtensionRegistry` instance for local extension management.

**Location**: `launcher/tool_extensions/registry.py`

**Key Classes**:
- `ExtensionType` - Enum for extension types (DATA_SOURCE, MUTATOR, ACTION, EVENT, STREAM)
- `Extension` - Dataclass representing a registered extension
- `ExtensionRegistry` - Main registry class for managing extensions

**Key Methods**:
- `register(extension)` - Register a new extension
- `unregister(extension_name)` - Remove an extension
- `list_extensions(ext_type)` - List all extensions, optionally filtered by type
- `query(extension_name, params)` - Query a data source extension
- `mutate(extension_name, params)` - Execute a mutator extension
- `execute(extension_name, params)` - Execute an action extension
- `emit_event(extension_name, event_data)` - Emit an event
- `subscribe(event_type, callback)` - Subscribe to events

### 2. DistributedExtensionRegistry

The distributed registry acts as a proxy that routes requests to appropriate tool processes.

**Location**: `launcher/distributed_registry.py`

**Key Classes**:
- `CircuitBreaker` - Circuit breaker for resilient HTTP communication
- `CacheManager` - TTL-based cache for extension metadata and query results
- `EventAggregator` - Aggregates events from tools and distributes to subscribers
- `HTTPClient` - Async HTTP client with connection pooling
- `ConfigManager` - Manages configuration loading and persistence
- `ConfigPersistence` - File-based configuration persistence
- `DistributedExtensionRegistry` - Main distributed registry class

**Key Methods**:
- `list_tools()` - List all available tools
- `list_extensions(tool_name, ext_type)` - List extensions from tools with caching
- `query(tool_name, extension_name, params)` - Query a data source extension in a tool process
- `mutate(tool_name, extension_name, params)` - Mutate configuration in a tool process
- `execute(tool_name, extension_name, params)` - Execute an action in a tool process
- `subscribe(tool_name, extension_name)` - Subscribe to events from a tool process

### 3. ExtensionHTTPServer

Each tool exposes an HTTP API for extension management.

**Location**: `launcher/tool_extensions/http_server.py`

**Key Classes**:
- `ExtensionHTTPServer` - HTTP server for extension management in tool processes

**Key Endpoints**:
- `GET /health` - Health check
- `GET /extensions` - List all extensions
- `GET /extensions/{name}` - Get extension details
- `POST /extensions/{name}/query` - Query a data source extension
- `POST /extensions/{name}/mutate` - Mutate configuration via extension
- `POST /extensions/{name}/execute` - Execute an action extension
- `WS /extensions/{name}/events` - WebSocket for real-time events

### 4. ServiceRegistry

The service registry manages tool discovery and health monitoring.

**Location**: `launcher/service_registry.py`

**Key Classes**:
- `ServiceInfo` - Dataclass containing service information
- `ServiceRegistry` - Service registry for tool discovery and health monitoring

**Key Methods**:
- `start()` - Start the health check background task
- `stop()` - Stop the health check background task
- `register(name, management_url, mcp_port)` - Register a tool service
- `unregister(name)` - Unregister a tool service
- `list_tools()` - List all registered tool names
- `get_endpoint(name)` - Get endpoint information for a tool

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MANAGEMENT SERVER (Port 9091)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              DistributedExtensionRegistry                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │   Service    │  │   Circuit    │  │    HTTP      │  │  Config  │ │  │
│  │  │  Discovery   │  │   Breaker    │  │   Client     │  │ Persist  │ │  │
│  │  │   Client     │  │   Manager    │  │    Pool      │  │  Events  │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │ HTTP/WebSocket
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     webmcp      │  │    simplemcp    │  │     ragmcp      │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ Extension │  │  │  │ Extension │  │  │  │ Extension │  │
│  │ Registry  │  │  │  │ Registry  │  │  │  │ Registry  │  │
│  │ HTTP Srv  │  │  │  │ HTTP Srv  │  │  │  │ HTTP Srv  │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Usage Example

```python
# In a tool's main module
from launcher.tool_extensions.registry import ExtensionRegistry, Extension, ExtensionType
from launcher.tool_extensions.http_server import ExtensionHTTPServer
from launcher.config.manager import ConfigManager

# Initialize registry
registry = ExtensionRegistry.get_instance()

# Register an extension
registry.register("webmcp", Extension(
    name="api_calls",
    ext_type=ExtensionType.DATA_SOURCE,
    schema={
        "input": {"type": "object", "properties": {"time_range": {"type": "string"}}},
        "output": {"type": "object", "properties": {"total": {"type": "integer"}}}
    },
    handler=lambda params: {"total": 1234},
    metadata={"description": "API call statistics"}
))

# Start HTTP server
async def main():
    config_manager = ConfigManager("webmcp")
    server = ExtensionHTTPServer(
        tool_name="webmcp",
        registry=registry,
        config_manager=config_manager,
        port=9001
    )
    await server.start()
```
