# Flexible Extensibility Framework V3

## Enhanced Two-Way Tool Management System with Distributed Architecture

---

## Table of Contents

| # | Section | File | Description |
|---|---------|------|-------------|
| 1 | [Executive Summary](./01-executive-summary.md) | `01-executive-summary.md` | What's new in V2/V3, vision, key improvements |
| 2 | [Dependencies and Requirements](./02-dependencies.md) | `02-dependencies.md` | Python packages, system requirements, installation |
| 3 | [Architecture Evolution](./03-architecture-evolution.md) | `03-architecture-evolution.md` | V1 to V2/V3 changes, problem statement, solution |
| 4 | [Distributed Architecture Design](./04-distributed-architecture.md) | `04-distributed-architecture.md` | System components, design principles, diagrams |
| 5 | [Core Components](./05-core-components/README.md) | `05-core-components/` | ExtensionRegistry, HTTP Server, Service Registry |
| 6 | [Extension Types and Data Models](./06-extension-types.md) | `06-extension-types.md` | Extension types, data types, schema definitions |
| 7 | [Cross-Process Communication](./07-cross-process-communication.md) | `07-cross-process-communication.md` | Communication patterns, HTTP API specification |
| 8 | [Service Mesh and Discovery](./08-service-mesh.md) | `08-service-mesh.md` | Service discovery integration, health monitoring |
| 9 | [Configuration Persistence](./09-configuration-persistence.md) | `09-configuration-persistence.md` | JSON and SQLite persistence strategies |
| 10 | [Error Handling and Resilience](./10-error-handling.md) | `10-error-handling.md` | Circuit breakers, retries, dead letter queues |
| 11 | [Security Framework](./11-security.md) | `11-security.md` | API keys, rate limiting, audit logging |
| 12 | [Performance Optimization](./12-performance.md) | `12-performance.md` | Caching, connection pooling, async optimization |
| 13 | [API Reference](./13-api-reference.md) | `13-api-reference.md` | REST endpoints, WebSocket events, error responses |
| 14 | [Implementation Guide](./14-implementation-guide.md) | `14-implementation-guide.md` | Step-by-step guides, best practices |
| 15 | [Migration Path](./15-migration-path.md) | `15-migration-path.md` | V1 to V2/V3 migration, rollback procedures |
| 16 | [Advanced Features](./16-advanced-features.md) | `16-advanced-features.md` | Distributed deployment, HA, plugins, monitoring |
| 17 | [Troubleshooting](./17-troubleshooting.md) | `17-troubleshooting.md` | Common issues, debug mode, performance tuning |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start a tool with management server
python -m webmcp --mgmt-port 9001

# Query extensions
curl http://localhost:9001/extensions

# Start management server
python -m launcher --management-port 9091

# List all tools
curl http://localhost:9091/api/tools
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   VSCode/Claude  │  │   Web Dashboard  │  │  Python Script   │          │
│  │   (MCP Client)   │  │    (Browser)     │  │   (API Client)   │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                    │
│           │ MCP JSON-RPC        │ HTTP/WebSocket       │ HTTP               │
│           ▼                     ▼                     ▼                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                    MANAGEMENT SERVER (Port 9091)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              DistributedExtensionRegistry                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │ HTTP/WebSocket
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     webmcp      │  │    simplemcp    │  │     ragmcp      │
│  MCP: Port 8001 │  │  MCP: Port 8002 │  │  MCP: Port 8004 │
│  Mgmt: Port 9001│  │  Mgmt: Port 9002│  │  Mgmt: Port 9004│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Key Features

1. **Distributed Extension Registry**: Works across multiple processes
2. **Embedded HTTP Service Mesh**: Each tool exposes an HTTP management server
3. **Service Discovery**: Automatic tool discovery via launcher
4. **Circuit Breaker Pattern**: Resilient communication with failure recovery
5. **Configuration Persistence**: File-based and SQLite options
6. **Enhanced Security**: API keys, rate limiting, audit logging
7. **Performance Optimization**: Caching, connection pooling
8. **Real-Time Events**: WebSocket-based event streaming

---

## License

See [LICENSE](../LICENSE) for details.
