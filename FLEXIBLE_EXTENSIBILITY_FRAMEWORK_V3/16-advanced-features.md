# Advanced Features

## Distributed Deployment

The V2 architecture supports distributed deployment across multiple machines:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Management Server (Machine A)                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │           DistributedExtensionRegistry                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Machine B     │  │   Machine C     │  │   Machine D     │
│   webmcp        │  │   simplemcp     │  │   ragmcp        │
│   Port: 8001    │  │   Port: 8002    │  │   Port: 8004    │
│   Mgmt: 9001    │  │   Mgmt: 9002    │  │   Mgmt: 9004    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Configuration for Distributed Deployment

```python
# launcher/config/distributed.py

DISTRIBUTED_CONFIG = {
    "management_server": {
        "host": "0.0.0.0",  # Listen on all interfaces
        "port": 9091,
        "advertised_url": "https://mgmt.example.com"  # Public URL
    },
    "tools": {
        "webmcp": {
            "host": "machine-b.internal",
            "mcp_port": 8001,
            "mgmt_port": 9001
        },
        "simplemcp": {
            "host": "machine-c.internal",
            "mcp_port": 8002,
            "mgmt_port": 9002
        },
        "ragmcp": {
            "host": "machine-d.internal",
            "mcp_port": 8004,
            "mgmt_port": 9004
        }
    }
}
```

### Network Considerations

1. **Firewall Rules**: Open ports 800x (MCP) and 900x (Management) between machines
2. **DNS Resolution**: Ensure all machines can resolve each other's hostnames
3. **TLS/SSL**: Use HTTPS for all inter-machine communication
4. **Latency**: Keep tools and management server in same data center for low latency

## High Availability

For production deployments requiring high availability:

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Load Balancer                                    │
│                    (nginx, HAProxy, or cloud LB)                         │
└─────────────────────────────┬───────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Mgmt Server  │  │   Mgmt Server  │  │   Mgmt Server  │
│   Instance 1   │  │   Instance 2   │  │   Instance 3   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐  ┌─────────────────┐
          │  Redis Cluster  │  │  Service Reg    │
          │  (Shared State) │  │  (Consul/etcd)  │
          └─────────────────┘  └─────────────────┘
```

### Implementation

```python
# launcher/config/ha.py

HA_CONFIG = {
    "replicas": 3,
    "session_store": "redis",
    "redis_url": "redis://redis-cluster:6379",
    "service_registry": "consul",
    "consul_url": "http://consul:8500",
    "health_check_interval": 10,
    "failover_timeout": 30
}
```

### Failover Behavior

1. **Detection**: Health checks detect failed management server
2. **Routing**: Load balancer routes to healthy instances
3. **State Recovery**: New instance loads state from Redis
4. **Tool Reconnection**: Tools reconnect to new instance

## Event Sourcing

For audit trail and time-travel debugging:

### Event Store

```python
# launcher/events/sourcing.py

class EventStore:
    """Stores all configuration changes as events."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    extension_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    params TEXT NOT NULL,
                    result TEXT,
                    user TEXT,
                    correlation_id TEXT
                )
            """)
    
    def append(self, event: Dict[str, Any]):
        """Append an event to the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO events 
                   (timestamp, tool_name, extension_name, operation, params, result, user, correlation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event["timestamp"], event["tool_name"], event["extension_name"],
                 event["operation"], json.dumps(event["params"]), 
                 json.dumps(event.get("result")), event.get("user"),
                 event.get("correlation_id"))
            )
    
    def get_history(self, tool_name: str, start_time: float, end_time: float) -> List[Dict]:
        """Get event history for a tool."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM events 
                   WHERE tool_name = ? AND timestamp BETWEEN ? AND ?
                   ORDER BY timestamp""",
                (tool_name, start_time, end_time)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def replay(self, tool_name: str, target_time: float) -> Dict[str, Any]:
        """Replay events to reconstruct state at a specific time."""
        events = self.get_history(tool_name, 0, target_time)
        state = {}
        for event in events:
            # Apply event to state
            key = f"{event['extension_name']}:{event['operation']}"
            state[key] = json.loads(event["params"])
        return state
```

### Time-Travel Debugging

```python
# Replay configuration at time of bug
event_store = EventStore("~/.config/supreme-mcp-tools/events.db")
state_at_bug = event_store.replay("webmcp", bug_timestamp)
print(f"Configuration at bug time: {state_at_bug}")
```

## Custom Extension Types

### Creating Custom Extension Types

```python
from launcher.tool_extensions.registry import ExtensionType

# Add custom extension type
class CustomExtensionType(ExtensionType):
    TRANSFORMER = "transformer"  # Data transformation
    VALIDATOR = "validator"      # Input validation
    ENRICHER = "enricher"        # Data enrichment
```

### Transformer Extension

```python
def register_transformer(registry: ExtensionRegistry, tool_name: str):
    """Register a data transformer extension."""
    
    def transform_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform input data."""
        data = params.get("data", {})
        transform_type = params.get("transform_type", "uppercase")
        
        if transform_type == "uppercase":
            return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}
        elif transform_type == "lowercase":
            return {k: v.lower() if isinstance(v, str) else v for k, v in data.items()}
        else:
            return data
    
    registry.register(tool_name, Extension(
        name="data_transformer",
        ext_type=CustomExtensionType.TRANSFORMER,
        schema={
            "input": {
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "transform_type": {"type": "string", "enum": ["uppercase", "lowercase", "identity"]}
                }
            },
            "output": {"type": "object"}
        },
        handler=transform_handler,
        metadata={"description": "Transform data", "category": "processing"}
    ))
```

## Plugin System

### Loading External Plugins

```python
# launcher/plugins/loader.py

import importlib
import sys
from pathlib import Path
from typing import Dict, Any

class PluginLoader:
    """Loads extensions from external plugin packages."""
    
    def __init__(self, plugin_dir: str = "~/.supreme-mcp-tools/plugins"):
        self.plugin_dir = Path(plugin_dir).expanduser()
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins."""
        plugins = []
        for path in self.plugin_dir.glob("*.py"):
            plugins.append(path.stem)
        return plugins
    
    def load_plugin(self, plugin_name: str, registry: ExtensionRegistry, tool_name: str):
        """Load a plugin and register its extensions."""
        plugin_path = self.plugin_dir / f"{plugin_name}.py"
        
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin not found: {plugin_name}")
        
        # Load module
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_name] = module
        spec.loader.exec_module(module)
        
        # Call plugin's register function
        if hasattr(module, "register"):
            module.register(registry, tool_name)
            self.loaded_plugins[plugin_name] = module
            logger.info(f"Loaded plugin: {plugin_name}")
        else:
            raise ValueError(f"Plugin {plugin_name} has no register function")
    
    def unload_plugin(self, plugin_name: str):
        """Unload a plugin."""
        if plugin_name in self.loaded_plugins:
            del self.loaded_plugins[plugin_name]
            del sys.modules[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
```

### Example Plugin

```python
# ~/.supreme-mcp-tools/plugins/custom_metrics.py

from launcher.tool_extensions.registry import ExtensionRegistry, Extension, ExtensionType

def register(registry: ExtensionRegistry, tool_name: str):
    """Register custom metrics extensions."""
    
    def get_custom_metric(params):
        return {"custom_value": 42}
    
    registry.register(tool_name, Extension(
        name="custom_metric",
        ext_type=ExtensionType.DATA_SOURCE,
        schema={},
        handler=get_custom_metric,
        metadata={"description": "Custom metric from plugin"}
    ))
```

## Advanced Monitoring

### Prometheus Integration

```python
# launcher/monitoring/prometheus.py

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
EXTENSIONS_TOTAL = Gauge('mcp_extensions_total', 'Total extensions', ['tool', 'type'])
QUERIES_TOTAL = Counter('mcp_queries_total', 'Total queries', ['tool', 'extension'])
QUERY_DURATION = Histogram('mcp_query_duration_seconds', 'Query duration', ['tool', 'extension'])
MUTATIONS_TOTAL = Counter('mcp_mutations_total', 'Total mutations', ['tool', 'extension'])
ERRORS_TOTAL = Counter('mcp_errors_total', 'Total errors', ['tool', 'extension', 'error_type'])

class PrometheusMetrics:
    """Prometheus metrics collector."""
    
    def __init__(self, port: int = 9090):
        start_http_server(port)
    
    def record_query(self, tool: str, extension: str, duration: float, success: bool):
        """Record a query operation."""
        QUERIES_TOTAL.labels(tool=tool, extension=extension).inc()
        QUERY_DURATION.labels(tool=tool, extension=extension).observe(duration)
        if not success:
            ERRORS_TOTAL.labels(tool=tool, extension=extension, error_type="query").inc()
    
    def record_mutation(self, tool: str, extension: str, success: bool):
        """Record a mutation operation."""
        MUTATIONS_TOTAL.labels(tool=tool, extension=extension).inc()
        if not success:
            ERRORS_TOTAL.labels(tool=tool, extension=extension, error_type="mutation").inc()
    
    def update_extension_count(self, tool: str, extensions: Dict[str, List]):
        """Update extension count metrics."""
        for ext_type, exts in extensions.items():
            EXTENSIONS_TOTAL.labels(tool=tool, type=ext_type).set(len(exts))
```

### Grafana Dashboards

Example Grafana dashboard JSON for monitoring:

```json
{
  "title": "Supreme MCP Tools Dashboard",
  "panels": [
    {
      "title": "Queries per Second",
      "targets": [
        {
          "expr": "rate(mcp_queries_total[5m])",
          "legendFormat": "{{tool}} - {{extension}}"
        }
      ]
    },
    {
      "title": "Query Latency (95th percentile)",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(mcp_query_duration_seconds_bucket[5m]))",
          "legendFormat": "{{tool}} - {{extension}}"
        }
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {
          "expr": "rate(mcp_errors_total[5m])",
          "legendFormat": "{{tool}} - {{error_type}}"
        }
      ]
    }
  ]
}
```
