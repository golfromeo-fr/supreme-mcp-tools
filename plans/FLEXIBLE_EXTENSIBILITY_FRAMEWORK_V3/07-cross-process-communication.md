# Cross-Process Communication

## Communication Patterns

### 1. Query Flow

```
Management Server → ServiceRegistry → Get tool endpoint
Management Server → CircuitBreaker → HTTP GET /extensions/{name}/query
Tool HTTP Server → ExtensionRegistry → Execute handler → Return data
```

### 2. Mutation Flow

```
Management Server → ServiceRegistry → Get tool endpoint
Management Server → CircuitBreaker → HTTP POST /extensions/{name}/mutate
Tool HTTP Server → ExtensionRegistry → Execute handler → Apply change
Management Server → ConfigPersistence → Save to disk
Management Server → CacheManager → Invalidate cache
Management Server → EventAggregator → Notify subscribers
```

### 3. Event Streaming

```
Tool ExtensionRegistry → Emit event
Tool HTTP Server → WebSocket → Management Server
Management Server → EventAggregator → WebSocket → Web UI
```

## HTTP API Specification

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/extensions` | List all extensions |
| GET | `/extensions/{name}` | Get extension details |
| POST | `/extensions/{name}/query` | Query data source |
| POST | `/extensions/{name}/mutate` | Mutate configuration |
| POST | `/extensions/{name}/execute` | Execute action |
| WS | `/extensions/{name}/events` | WebSocket for events |

### Request/Response Examples

#### Query Extension

```bash
POST /extensions/api_calls/query
Content-Type: application/json

{
  "params": {
    "time_range": "1h"
  }
}

Response:
{
  "data": {
    "total": 1234,
    "by_endpoint": {
      "/api/users": 500,
      "/api/posts": 734
    }
  }
}
```

#### Mutate Extension

```bash
POST /extensions/api_key/mutate
Content-Type: application/json

{
  "params": {
    "key": "sk-xxx"
  }
}

Response:
{
  "result": {
    "success": true,
    "message": "API key updated"
  }
}
```

#### Execute Action

```bash
POST /extensions/clear_cache/execute
Content-Type: application/json

{
  "params": {
    "cache_type": "all"
  }
}

Response:
{
  "result": {
    "success": true,
    "cleared": {
      "api_cache": 150,
      "query_cache": 75
    }
  }
}
```

## WebSocket Events

### Subscribe to Events

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:9001/extensions/errors/events');

// Handle events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event received:', data);
  // {
  //   "extension": "errors",
  //   "data": {
  //     "message": "API rate limit exceeded",
  //     "endpoint": "/api/users",
  //     "timestamp": 1679064000
  //   }
  // }
};
```

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "EXTENSION_NOT_FOUND",
    "message": "Extension 'unknown' not found",
    "details": {
      "extension_name": "unknown",
      "available_extensions": ["api_calls", "api_key", "clear_cache"]
    }
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Extension not found |
| 408 | Request Timeout - Operation timed out |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Tool unhealthy |
