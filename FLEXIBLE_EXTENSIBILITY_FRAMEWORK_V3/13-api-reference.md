# API Reference

## REST API Endpoints

### Management Server API (Port 9091)

| Method | Path | Description | Auth Required |
|--------|------|-------------|---------------|
| GET | `/health` | Health check | No |
| GET | `/api/tools` | List all tools | Yes |
| GET | `/api/tools/{name}` | Get tool details | Yes |
| GET | `/api/tools/{name}/extensions` | List tool extensions | Yes |
| POST | `/api/tools/{name}/extensions/{ext}/query` | Query data source | Yes |
| POST | `/api/tools/{name}/extensions/{ext}/mutate` | Mutate configuration | Yes |
| POST | `/api/tools/{name}/extensions/{ext}/execute` | Execute action | Yes |
| WS | `/api/tools/{name}/extensions/{ext}/events` | Subscribe to events | Yes |

### Tool Management API (Port 900x)

| Method | Path | Description | Auth Required |
|--------|------|-------------|---------------|
| GET | `/health` | Health check | No |
| GET | `/extensions` | List all extensions | No |
| GET | `/extensions/{name}` | Get extension details | No |
| POST | `/extensions/{name}/query` | Query data source | Optional |
| POST | `/extensions/{name}/mutate` | Mutate configuration | Optional |
| POST | `/extensions/{name}/execute` | Execute action | Optional |
| WS | `/extensions/{name}/events` | Subscribe to events | Optional |

## Request/Response Examples

### List Tools

```bash
GET /api/tools
Authorization: Bearer <api-key>

Response:
{
  "tools": [
    {
      "name": "webmcp",
      "status": "healthy",
      "management_url": "http://localhost:9001",
      "mcp_port": 8001,
      "capabilities": {
        "total_extensions": 15,
        "data_sources": 8,
        "mutators": 4,
        "actions": 3
      }
    },
    {
      "name": "simplemcp",
      "status": "healthy",
      "management_url": "http://localhost:9002",
      "mcp_port": 8002,
      "capabilities": {
        "total_extensions": 12,
        "data_sources": 6,
        "mutators": 3,
        "actions": 3
      }
    }
  ]
}
```

### Query Extension

```bash
POST /api/tools/webmcp/extensions/api_calls/query
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "params": {
    "time_range": "1h",
    "group_by": "endpoint"
  }
}

Response:
{
  "data": {
    "total": 1234,
    "by_endpoint": {
      "/api/users": 500,
      "/api/posts": 734
    },
    "time_range": "1h"
  }
}
```

### Mutate Configuration

```bash
POST /api/tools/webmcp/extensions/api_key/mutate
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "params": {
    "key": "sk-xxx-new-key"
  }
}

Response:
{
  "result": {
    "success": true,
    "message": "API key updated",
    "previous_key": "sk-xxx-old-key"
  }
}
```

### Execute Action

```bash
POST /api/tools/webmcp/extensions/clear_cache/execute
Authorization: Bearer <api-key>
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
const ws = new WebSocket('ws://localhost:9091/api/tools/webmcp/extensions/errors/events');
ws.setRequestHeader('Authorization', 'Bearer <api-key>');

// Handle events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event received:', data);
  // {
  //   "tool": "webmcp",
  //   "type": "error",
  //   "extension": "errors",
  //   "data": {
  //     "message": "API rate limit exceeded",
  //     "endpoint": "/api/users",
  //     "timestamp": 1679064000
  //   },
  //   "timestamp": 1679064000
  // }
};
```

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "TOOL_NOT_FOUND",
    "message": "Tool 'unknown_tool' not found",
    "details": {
      "tool_name": "unknown_tool",
      "available_tools": ["webmcp", "simplemcp", "ragmcp"]
    }
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Tool or extension not found |
| 408 | Request Timeout - Operation timed out |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Tool unhealthy |
