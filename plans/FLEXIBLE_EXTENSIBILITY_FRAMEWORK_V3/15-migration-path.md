# Migration Path

## From V1 to V2

### Overview

The migration from V1 to V2 involves transitioning from a singleton, single-process architecture to a distributed, multi-process architecture. This guide provides a phased approach to minimize disruption.

### Architecture Comparison

| Aspect | V1 (Current) | V2 (Target) |
|--------|--------------|-------------|
| Registry | Singleton in Management Server | Distributed - one per tool |
| Communication | Direct function calls | HTTP/WebSocket IPC |
| Discovery | Manual registration | Automatic via launcher |
| Persistence | In-memory only | File-based + SQLite |
| Security | None | API keys, rate limiting |
| Resilience | Basic error handling | Circuit breakers, retries |

## Phase 1: Add Management Servers to Tools

### Objective

Add an HTTP management server to each tool that exposes extensions via REST API.

### Steps

#### 1.1 Add Dependencies

```bash
pip install fastapi uvicorn aiohttp
```

Or add to `requirements.txt`:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
aiohttp>=3.9.0
```

#### 1.2 Create Local Extension Registry

Each tool gets its own `ExtensionRegistry` instance:

```python
# In each tool's main module
from launcher.tool_extensions.registry import ExtensionRegistry, Extension, ExtensionType

# Initialize registry
registry = ExtensionRegistry.get_instance()

# Register existing extensions
registry.register("webmcp", Extension(
    name="api_calls",
    ext_type=ExtensionType.DATA_SOURCE,
    schema={...},
    handler=api_calls_handler,
    metadata={...}
))
```

#### 1.3 Add Management Server

```python
# In each tool's main entry point
from launcher.tool_extensions.http_server import ExtensionHTTPServer
from launcher.config.manager import ConfigManager

async def start_management_server(tool_name: str, port: int):
    """Start the management server for this tool."""
    registry = ExtensionRegistry.get_instance()
    config_manager = ConfigManager(tool_name)
    
    server = ExtensionHTTPServer(
        tool_name=tool_name,
        registry=registry,
        config_manager=config_manager,
        port=port
    )
    
    await server.start()
```

#### 1.4 Test Local API

```bash
# Start tool with management server
python -m webmcp --mgmt-port 9001

# Test endpoints
curl http://localhost:9001/health
curl http://localhost:9001/extensions
curl -X POST http://localhost:9001/extensions/api_calls/query \
  -H "Content-Type: application/json" \
  -d '{"params": {"time_range": "1h"}}'
```

### Verification Checklist

- [ ] Management server starts on allocated port
- [ ] `/health` endpoint returns 200
- [ ] `/extensions` lists all registered extensions
- [ ] Query endpoint returns correct data
- [ ] Mutate endpoint updates configuration
- [ ] Execute endpoint runs actions

## Phase 2: Update Launcher

### Objective

Add service registry to launcher and update tool startup to register with it.

### Steps

#### 2.1 Add Service Registry

```python
# launcher/service_registry.py (new file)
from launcher.service_registry import ServiceRegistry

# Initialize in launcher
service_registry = ServiceRegistry()
await service_registry.start()
```

#### 2.2 Update ServerManager

```python
# launcher/server_manager.py

class ServerManager:
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.port_manager = PortManager()
    
    async def start_tool(self, tool_name: str):
        # Allocate ports
        mcp_port = self.port_manager.allocate()
        mgmt_port = self.port_manager.allocate()
        
        # Start tool process
        process = await self._start_process(
            tool_name,
            mcp_port=mcp_port,
            mgmt_port=mgmt_port
        )
        
        # Wait for health
        await self._wait_for_health(f"http://127.0.0.1:{mgmt_port}/health")
        
        # Register with service registry
        self.service_registry.register(
            name=tool_name,
            management_url=f"http://127.0.0.1:{mgmt_port}",
            mcp_port=mcp_port
        )
        
        return process
```

#### 2.3 Test Service Discovery

```bash
# Start launcher
python -m launcher

# Check registered tools
curl http://localhost:9091/api/tools
```

### Verification Checklist

- [ ] Service registry starts with launcher
- [ ] Tools register on startup
- [ ] Service registry shows all running tools
- [ ] Health checks run periodically
- [ ] Tools unregister on shutdown

## Phase 3: Update Management Server

### Objective

Replace singleton ExtensionRegistry with DistributedExtensionRegistry.

### Steps

#### 3.1 Create Distributed Registry

```python
# launcher/distributed_registry.py (new file)
from launcher.distributed_registry import DistributedExtensionRegistry

# Initialize in management server
distributed_registry = DistributedExtensionRegistry(service_registry)
```

#### 3.2 Update API Endpoints

```python
# launcher/management_server.py

@app.get("/api/tools")
async def list_tools():
    return {"tools": await distributed_registry.list_tools()}

@app.get("/api/tools/{tool}/extensions")
async def list_extensions(tool: str):
    return await distributed_registry.list_extensions(tool_name=tool)

@app.post("/api/tools/{tool}/extensions/{ext}/query")
async def query_extension(tool: str, ext: str, request: dict):
    return await distributed_registry.query(tool, ext, request.get("params"))

@app.post("/api/tools/{tool}/extensions/{ext}/mutate")
async def mutate_extension(tool: str, ext: str, request: dict):
    return await distributed_registry.mutate(tool, ext, request.get("params"))

@app.post("/api/tools/{tool}/extensions/{ext}/execute")
async def execute_extension(tool: str, ext: str, request: dict):
    return await distributed_registry.execute(tool, ext, request.get("params"))
```

#### 3.3 Test Distributed Operations

```bash
# List all extensions across tools
curl http://localhost:9091/api/tools/webmcp/extensions

# Query a tool's extension
curl -X POST http://localhost:9091/api/tools/webmcp/extensions/api_calls/query \
  -H "Content-Type: application/json" \
  -d '{"params": {"time_range": "1h"}}'

# Mutate a tool's configuration
curl -X POST http://localhost:9091/api/tools/webmcp/extensions/api_key/mutate \
  -H "Content-Type: application/json" \
  -d '{"params": {"key": "new-key"}}'
```

### Verification Checklist

- [ ] Distributed registry routes requests to correct tools
- [ ] Circuit breakers activate on tool failures
- [ ] Caching reduces redundant requests
- [ ] Configuration changes persist to disk
- [ ] Events propagate to subscribers

## Phase 4: Add Security and Persistence

### Objective

Add authentication, authorization, and persistent configuration storage.

### Steps

#### 4.1 Add API Key Authentication

```python
# launcher/security/auth.py (new file)
from launcher.security.auth import verify_api_key

# Add to management server
@app.get("/api/tools")
async def list_tools(permissions: dict = Depends(verify_api_key)):
    return {"tools": await distributed_registry.list_tools()}
```

#### 4.2 Add Configuration Persistence

```python
# launcher/config/persistence.py (new file)
from launcher.config.persistence import ConfigPersistence

# Initialize persistence layer
config_persistence = ConfigPersistence()
```

#### 4.3 Add Audit Logging

```python
# launcher/security/audit.py (new file)
from launcher.security.audit import AuditLogger

# Log configuration changes
audit_logger = AuditLogger()
audit_logger.log(
    action="mutate",
    user="admin",
    tool_name="webmcp",
    details={"extension": "api_key", "params": {...}}
)
```

#### 4.4 Test Security Features

```bash
# Test without API key (should fail)
curl http://localhost:9091/api/tools
# Returns: 401 Unauthorized

# Test with API key
curl http://localhost:9091/api/tools \
  -H "Authorization: Bearer admin-key-xxx"
# Returns: 200 OK with tool list

# Check audit log
cat ~/.config/supreme-mcp-tools/audit.log
```

### Verification Checklist

- [ ] API key required for all endpoints
- [ ] Invalid keys return 403
- [ ] Configuration changes persist across restarts
- [ ] Audit log records all mutations
- [ ] Rate limiting prevents abuse

## Backward Compatibility

### Maintaining V1 Compatibility

The V2 architecture can maintain backward compatibility with V1 clients:

1. **Local Registry**: Tools can still use local registry for direct access
2. **HTTP API**: Management server API remains compatible
3. **Extension Schema**: Extension definitions unchanged
4. **Event System**: Event subscription pattern preserved

### Deprecation Timeline

| Version | Status | Notes |
|---------|--------|-------|
| V1 | Deprecated | Will be removed in future release |
| V2 | Current | Recommended for all new development |
| V1+V2 | Transitional | Both available during migration |

## Rollback Plan

If issues arise during migration:

1. **Phase 1 Rollback**: Remove management server from tools
2. **Phase 2 Rollback**: Disable service registry in launcher
3. **Phase 3 Rollback**: Revert to singleton ExtensionRegistry
4. **Phase 4 Rollback**: Disable authentication and persistence

### Rollback Commands

```bash
# Stop all tools
pkill -f "python -m webmcp"
pkill -f "python -m simplemcp"
pkill -f "python -m ragmcp"

# Stop launcher
pkill -f "python -m launcher"

# Revert to V1 configuration
git checkout main -- launcher/
```

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | 1-2 weeks | None |
| Phase 2 | 1 week | Phase 1 complete |
| Phase 3 | 1-2 weeks | Phase 2 complete |
| Phase 4 | 1 week | Phase 3 complete |
| **Total** | **4-6 weeks** | |

## Support

For migration assistance:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API Reference](#api-reference) for endpoint details
3. File an issue on the project repository
