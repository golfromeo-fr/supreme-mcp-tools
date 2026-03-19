# Distributed Architecture Design

## System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MANAGEMENT SERVER (Port 9091)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              DistributedExtensionRegistry                            │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Core Components                             │ │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │   │   │
│  │  │  │   Service    │  │   Circuit    │  │    HTTP      │        │   │   │
│  │  │  │  Discovery   │  │   Breaker    │  │   Client     │        │   │   │
│  │  │  │   Client     │  │   Manager    │  │    Pool      │        │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘        │   │   │
│  │  │                                                               │ │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │   │   │
│  │  │  │    Cache     │  │    Event     │  │    Config    │        │   │   │
│  │  │  │   Manager    │  │  Aggregator  │  │  Persistence │        │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘        │   │   │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    API Layer                                   │ │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │   │   │
│  │  │  │   REST API   │  │  WebSocket   │  │   Web UI     │        │   │   │
│  │  │  │  Endpoints   │  │   Handler    │  │   Server     │        │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘        │   │   │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOOL PROCESSES                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │     webmcp (Process P1)                                            │   │
│  │     MCP Server: Port 8001 | Management Server: Port 9001          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │  Extension   │  │  HTTP API    │  │  WebSocket   │      │   │   │
│  │  │  │  Registry    │  │  Server      │  │  Server      │      │   │   │
│  │  │  │  (Local)     │  │              │  │              │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │     simplemcp (Process P2)                                          │   │
│  │     MCP Server: Port 8002 | Management Server: Port 9002          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │  Extension   │  │  HTTP API    │  │  WebSocket   │      │   │   │
│  │  │  │  Registry    │  │  Server      │  │  Server      │      │   │   │
│  │  │  │  (Local)     │  │              │  │              │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │     ragmcp (Process P3)                                             │   │
│  │     MCP Server: Port 8004 | Management Server: Port 9004          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │  Extension   │  │  HTTP API    │  │  WebSocket   │      │   │   │
│  │  │  │  Registry    │  │  Server      │  │  Server      │      │   │   │
│  │  │  │  (Local)     │  │              │  │              │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Health Checks
                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SERVICE REGISTRY (Launcher)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Tool Registry:                                                      │  │
│  │  ┌────────────┬─────────────────────┬────────────┬──────────────┐   │  │
│  │  │ Tool Name  │ Management Endpoint │ MCP Port   │ Status       │   │  │
│  │  ├────────────┼─────────────────────┼────────────┼──────────────┤   │  │
│  │  │ webmcp     │ http://localhost:9001│ 8001       │ healthy      │   │  │
│  │  │ simplemcp  │ http://localhost:9002│ 8002       │ healthy      │   │  │
│  │  │ ragmcp     │ http://localhost:9004│ 8004       │ degraded     │   │  │
│  │  └────────────┴─────────────────────┴────────────┴──────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Management Server (Port 9091)

The Management Server acts as the central hub for all client interactions:

1. **DistributedExtensionRegistry**: Routes requests to appropriate tool processes
2. **ServiceDiscovery**: Maintains registry of available tools and their endpoints
3. **CircuitBreaker**: Prevents cascading failures when tools are unavailable
4. **CacheManager**: Caches extension metadata and query results for performance
5. **EventAggregator**: Distributes events from tools to WebSocket subscribers
6. **ConfigPersistence**: Persists configuration changes to disk

### Tool Processes (Ports 800x, 900x)

Each tool runs in its own process with:

1. **MCP Server** (Port 800x): Traditional JSON-RPC interface for LLM clients
2. **Management Server** (Port 900x): HTTP API for extension management
3. **ExtensionRegistry**: Local registry for tool's extensions
4. **ConfigManager**: Manages tool configuration

### Service Registry (Launcher)

The Launcher manages tool lifecycle:

1. **Port Management**: Allocates ports for MCP and Management servers
2. **Process Management**: Starts and stops tool processes
3. **Health Monitoring**: Periodic health checks for all tools
4. **Service Discovery**: Maintains registry of available tools

---

## Design Principles

### 1. Distributed by Design

Each tool maintains its own registry; communication via IPC:

```python
# Each tool has its own registry
registry = ExtensionRegistry.get_instance()

# Tools register their own extensions
registry.register("webmcp", Extension(...))

# Management server routes requests to tools
result = await distributed_registry.query("webmcp", "api_calls", params)
```

### 2. Service-Oriented

Tools expose HTTP APIs for management operations:

```python
# Tool exposes HTTP API
@app.get("/extensions")
async def list_extensions():
    return registry.list_extensions()

# Management server calls tool API
extensions = await http_client.get("http://localhost:9001/extensions")
```

### 3. Resilient Communication

Circuit breakers, retries, and fallbacks for reliability:

```python
# Circuit breaker prevents cascading failures
try:
    result = await circuit_breaker.execute(
        "webmcp",
        lambda: http_client.post(url, data)
    )
except CircuitBreakerOpenError:
    # Use cached data or return error
    result = await cache.get(f"extensions:{tool_name}")
```

### 4. Automatic Discovery

Tools register themselves with the launcher on startup:

```python
# Launcher registers tools on startup
service_registry.register(
    name="webmcp",
    management_url="http://localhost:9001",
    mcp_port=8001
)
```

### 5. Persistent Configuration

All changes saved to disk; survive restarts:

```python
# Configuration changes are persisted
await config_persistence.save(
    tool_name="webmcp",
    extension_name="api_key",
    params={"key": "new-key"}
)
```

### 6. Secure by Default

Authentication, authorization, and audit logging:

```python
# API key authentication
@app.get("/api/tools")
async def list_tools(permissions: dict = Depends(verify_api_key)):
    return await distributed_registry.list_tools()

# Audit logging
audit_logger.log(
    action="mutate",
    user="admin",
    tool_name="webmcp",
    details={"extension": "api_key"}
)
```

### 7. Observable

Comprehensive metrics, logs, and traces:

```python
# Prometheus metrics
QUERIES_TOTAL.labels(tool="webmcp", extension="api_calls").inc()
QUERY_DURATION.labels(tool="webmcp").observe(duration)

# Structured logging
logger.info("Query executed", tool="webmcp", extension="api_calls", duration=0.045)
```

### 8. Scalable

Can extend to distributed deployment across machines:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Management Server (Machine A)                         │
└─────────────────────────────┬───────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Machine B     │  │   Machine C     │  │   Machine D     │
│   webmcp        │  │   simplemcp     │  │   ragmcp        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```
