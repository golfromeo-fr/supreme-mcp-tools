# Architecture Evolution

## From V1 to V2/V3: Key Changes

| Aspect | V1 (Original) | V2/V3 (Enhanced) |
|--------|---------------|------------------|
| **Registry Pattern** | Singleton (single-process) | Distributed (multi-process) |
| **Communication** | Direct function calls | HTTP/WebSocket IPC |
| **Discovery** | Manual registration | Automatic service discovery |
| **Resilience** | Basic error handling | Circuit breakers, retries, fallbacks |
| **Persistence** | In-memory only | File-based + SQLite options |
| **Security** | None | API keys, rate limiting, audit |
| **Performance** | No caching | TTL caching, connection pooling |
| **Events** | Callback-based | WebSocket streaming |
| **Cross-Platform** | Unix sockets only | HTTP (universal) |
| **Documentation** | Single monolithic file | Modular multi-file structure |

---

## Problem Statement (V1 Limitations)

The original FEF specification contained a critical architectural flaw: it assumed a singleton `ExtensionRegistry` running in the Management Server could directly access extensions registered in tool processes. This is fundamentally infeasible because:

### Core Issues

1. **Memory Isolation**: Each tool process has its own memory space
2. **No Cross-Process Access**: The Management Server cannot access tool process memory
3. **Synchronous Design**: IPC requires async operations
4. **No Persistence**: Configuration changes were lost on restart

### Impact

```
Current Architecture (Infeasible):
┌─────────────────────────────────────────────────────────────────────────┐
│                    Management Server (Port 9091)                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    ExtensionRegistry                            │  │
│  │                    (Singleton - IN-MEMORY)                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │ ❌ Cannot access tools in other processes
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOLS (Separate Processes)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │     webmcp      │  │    simplemcp     │  │     ragmcp      │        │
│  │   Port: 8001    │  │   Port: 8002    │  │   Port: 8004    │        │
│  │   Process: P1   │  │   Process: P2   │  │   Process: P3   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Solution: Distributed Architecture

V2/V3 implements a distributed architecture where:

1. Each tool runs its own `ExtensionRegistry` locally
2. Tools expose HTTP APIs for cross-process communication
3. The Management Server acts as a proxy/router to tool processes
4. Service discovery enables automatic tool registration
5. Circuit breakers provide resilience against tool failures

### Recommended Architecture (Feasible)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Management Server (Port 9091)                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │           DistributedExtensionRegistry (Proxy)                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Service      │  │ Circuit      │  │ HTTP Client  │  │  │
│  │  │ Discovery    │  │ Breaker      │  │ Pool         │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  │                                                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │ Config       │  │ Event        │  │ Cache        │  │  │
│  │  │ Persistence  │  │ Aggregator   │  │ Manager      │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────┘
                               │
                               │ HTTP/WebSocket
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     webmcp      │  │    simplemcp     │  │     ragmcp      │
│   Port: 8001    │  │   Port: 8002    │  │   Port: 8004    │
│   (MCP Server)  │  │   (MCP Server)  │  │   (MCP Server)  │
│   Port: 9001    │  │   Port: 9002    │  │   Port: 9004    │
│   (Mgmt Server) │  │   (Mgmt Server) │  │   (Mgmt Server) │
│                 │  │                 │  │                 │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ HTTP API  │  │  │  │ HTTP API  │  │  │  │ HTTP API  │  │
│  │ ExtReg    │  │  │  │ ExtReg    │  │  │  │ ExtReg    │  │
│  │ WebSocket │  │  │  │ WebSocket │  │  │  │ WebSocket │  │
│  │ ConfigMgr │  │  │  │ ConfigMgr │  │  │  │ ConfigMgr │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          ▲                    ▲                    ▲
          │                    │                    │
          │ Health Checks      │ Health Checks      │
          │                    │                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                    Service Registry (Launcher)                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  webmcp: {endpoint: "http://localhost:9001", ...}      │  │
│  │  simplemcp: {endpoint: "http://localhost:9002", ...}    │  │
│  │  ragmcp: {endpoint: "http://localhost:9004", ...}       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## IPC Approach Comparison

| Criterion | Unix Sockets | Redis | Embedded HTTP |
|-----------|--------------|-------|---------------|
| **Latency** | Best (0.1-0.5ms) | Good (0.5-2ms) | Fair (1-5ms) |
| **Complexity** | Low | Medium | Medium |
| **Dependencies** | None | Redis | None |
| **Cross-Platform** | No (Unix only) | Yes | Yes |
| **Distributed** | No | Yes | Yes |
| **Debugging** | Medium | Easy | Easy |
| **Scalability** | Single machine | Excellent | Good |

### Rationale for HTTP-Based Solution

1. **Cross-Platform Compatibility**: Works on Linux, macOS, and Windows without modification
2. **No External Dependencies**: Leverages existing HTTP infrastructure, no Redis required
3. **Integration with Existing System**: Builds on `ServerManager` and `PortManager`
4. **Production-Ready**: Circuit breakers, health checks, retries built-in
5. **Easy Debugging**: Standard HTTP tools (curl, Postman, browser)
6. **Scalable**: Can extend to distributed deployment with minimal changes

---

## Design Principles (V2/V3)

1. **Distributed by Design**: Each tool maintains its own registry; communication via IPC
2. **Service-Oriented**: Tools expose HTTP APIs for management operations
3. **Resilient Communication**: Circuit breakers, retries, and fallbacks for reliability
4. **Automatic Discovery**: Tools register themselves with the launcher on startup
5. **Persistent Configuration**: All changes saved to disk; survive restarts
6. **Secure by Default**: Authentication, authorization, and audit logging
7. **Observable**: Comprehensive metrics, logs, and traces
8. **Scalable**: Can extend to distributed deployment across machines
