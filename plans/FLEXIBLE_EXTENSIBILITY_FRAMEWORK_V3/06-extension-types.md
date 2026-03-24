# Extension Types and Data Models

## Extension Types

| Type | Direction | Purpose | Example |
|------|-----------|---------|---------|
| `DATA_SOURCE` | Read | Expose metrics, state, logs | API call counts, cache size |
| `MUTATOR` | Write | Change configuration | Set API key, enable function |
| `ACTION` | Execute | One-off operations | Clear cache, reset counters |
| `EVENT` | Stream | Discrete event emission | Error alerts, state changes |
| `STREAM` | Stream | Continuous data streams | Live metrics, log streaming |

## Data Types

| Type | Description | Example |
|------|-------------|---------|
| `COUNTER` | Cumulative value | Total API calls |
| `GAUGE` | Current value | Cache size |
| `HISTOGRAM` | Distribution | Response times |
| `MAP` | Key-value pairs | Errors by type |
| `LIST` | Array of items | Recent logs |
| `JSON` | Structured data | Complex state |
| `TEXT` | String data | Configuration |
| `BINARY` | Binary data | Export files |

## Schema Definition

Extensions use JSON Schema for self-documenting interfaces:

```json
{
  "name": "api_calls",
  "type": "data_source",
  "schema": {
    "input": {
      "type": "object",
      "properties": {
        "time_range": {
          "type": "string",
          "enum": ["1h", "24h", "7d"],
          "description": "Time range for query"
        }
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
  "metadata": {
    "description": "API call statistics",
    "category": "metrics",
    "tags": ["performance", "monitoring"]
  }
}
```

## Extension Class

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ExtensionType(Enum):
    """Types of extensions a tool can register."""
    DATA_SOURCE = "data_source"      # Read-only data exposure
    MUTATOR = "mutator"              # Configuration changes
    ACTION = "action"                # One-off operations
    EVENT = "event"                  # Event emission
    STREAM = "stream"                # Continuous data streams


@dataclass
class Extension:
    """
    Represents a registered extension.
    
    Attributes:
        name: Unique identifier within the tool
        ext_type: Type of extension
        schema: JSON schema for parameters and returns
        handler: Callable that implements the extension
        metadata: Additional information (description, category, etc.)
    """
    name: str
    ext_type: ExtensionType
    schema: Dict[str, Any]
    handler: Callable
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
```

## Extension Examples

### Data Source Extension

```python
def get_api_calls(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for API call statistics."""
    time_range = params.get("time_range", "1h")
    # ... implementation
    return {
        "total": 1234,
        "by_endpoint": {"/api/users": 500}
    }

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
```

### Mutator Extension

```python
def set_api_key(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for setting API key."""
    new_key = params.get("key")
    # ... implementation
    return {"success": True, "message": "API key updated"}

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
```

### Action Extension

```python
def clear_cache(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for clearing cache."""
    cache_type = params.get("cache_type", "all")
    # ... implementation
    return {"cleared": 150}

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
