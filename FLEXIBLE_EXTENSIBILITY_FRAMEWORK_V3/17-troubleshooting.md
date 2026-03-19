# Troubleshooting

## Common Issues

### 1. Tool Not Discoverable

**Symptoms**: Tool not appearing in management server's tool list

**Causes**:
- Management server not started
- Port allocation conflict
- Service registry not updated
- Network connectivity issues

**Diagnosis**:

```bash
# Check if tool management server is running
curl http://localhost:9001/health

# Check if tool is registered with service registry
curl http://localhost:9091/api/tools

# Check tool logs
tail -f ~/.config/supreme-mcp-tools/logs/webmcp.log

# Check port availability
netstat -tlnp | grep 9001
```

**Solutions**:

```bash
# Restart tool with management server
python -m webmcp --mgmt-port 9001

# Manually register tool (if service registry is down)
curl -X POST http://localhost:9091/api/tools/register \
  -H "Content-Type: application/json" \
  -d '{"name": "webmcp", "management_url": "http://localhost:9001", "mcp_port": 8001}'

# Check firewall rules
sudo ufw status
sudo iptables -L
```

### 2. Circuit Breaker Open

**Symptoms**: Requests failing with "Circuit breaker open" error

**Causes**:
- Tool process crashed
- Network connectivity issues
- Tool overloaded/unresponsive
- Too many consecutive failures

**Diagnosis**:

```bash
# Check tool health
curl http://localhost:9001/health

# Check circuit breaker state
curl http://localhost:9091/api/circuit-breakers

# Check tool process
ps aux | grep webmcp

# Check tool resource usage
top -p $(pgrep -f webmcp)
```

**Solutions**:

```bash
# Restart the tool
./scripts/restart_tool.sh webmcp

# Reset circuit breaker manually
curl -X POST http://localhost:9091/api/circuit-breakers/webmcp/reset

# Adjust circuit breaker settings
# In launcher_config.json:
{
  "circuit_breaker": {
    "failure_threshold": 10,  # Increase from default 5
    "recovery_timeout": 60    # Increase from default 30
  }
}
```

### 3. Configuration Not Persisted

**Symptoms**: Configuration changes lost after tool restart

**Causes**:
- Persistence layer not configured
- File permissions issue
- Disk full
- Config directory doesn't exist

**Diagnosis**:

```bash
# Check config directory
ls -la ~/.config/supreme-mcp-tools/

# Check disk space
df -h ~/.config/supreme-mcp-tools/

# Check config file
cat ~/.config/supreme-mcp-tools/webmcp.json

# Check file permissions
stat ~/.config/supreme-mcp-tools/webmcp.json
```

**Solutions**:

```bash
# Create config directory
mkdir -p ~/.config/supreme-mcp-tools

# Fix permissions
chmod 755 ~/.config/supreme-mcp-tools
chmod 644 ~/.config/supreme-mcp-tools/*.json

# Clean up disk space
du -sh ~/.config/supreme-mcp-tools/*
rm ~/.config/supreme-mcp-tools/logs/*.log.old

# Enable persistence in config
# In launcher_config.json:
{
  "persistence": {
    "enabled": true,
    "type": "json",  # or "sqlite"
    "directory": "~/.config/supreme-mcp-tools"
  }
}
```

### 4. WebSocket Connection Dropped

**Symptoms**: Event stream stops receiving events

**Causes**:
- Network timeout
- Tool process restarted
- Server overloaded
- Proxy/load balancer timeout

**Diagnosis**:

```bash
# Test WebSocket connection
wscat -c ws://localhost:9091/api/tools/webmcp/extensions/events/events

# Check server logs
tail -f ~/.config/supreme-mcp-tools/logs/management.log

# Check for proxy timeout settings
nginx -T | grep proxy_read_timeout
```

**Solutions**:

```javascript
// Implement reconnection logic in client
function connectWebSocket(url, authToken) {
    const ws = new WebSocket(url);
    ws.setRequestHeader('Authorization', `Bearer ${authToken}`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleEvent(data);
    };
    
    ws.onclose = (event) => {
        console.log(`WebSocket closed: ${event.code} ${event.reason}`);
        // Reconnect after delay
        setTimeout(() => connectWebSocket(url, authToken), 5000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    return ws;
}
```

```bash
# Increase proxy timeout (nginx)
proxy_read_timeout 300s;
proxy_send_timeout 300s;

# Increase uvicorn timeout
uvicorn --timeout-keep-alive 300
```

### 5. Authentication Failures

**Symptoms**: 401 Unauthorized or 403 Forbidden errors

**Causes**:
- Missing API key
- Invalid API key
- Expired API key
- Insufficient permissions

**Diagnosis**:

```bash
# Test without API key
curl -v http://localhost:9091/api/tools

# Test with API key
curl -v http://localhost:9091/api/tools \
  -H "Authorization: Bearer admin-key-xxx"

# Check API key configuration
cat ~/.config/supreme-mcp-tools/auth.json

# Check audit log for auth failures
grep "401\|403" ~/.config/supreme-mcp-tools/audit.log
```

**Solutions**:

```bash
# Generate new API key
python -m launcher.auth generate-key --role admin

# Update API key in config
# In auth.json:
{
  "api_keys": {
    "new-key-xxx": {"role": "admin", "tools": ["*"]}
  }
}

# Check key format (should be Bearer token)
# Correct: Authorization: Bearer admin-key-xxx
# Wrong: Authorization: admin-key-xxx
```

### 6. High Latency

**Symptoms**: Requests taking longer than expected

**Causes**:
- Network latency between machines
- Tool overloaded
- Cache misses
- Large data transfers

**Diagnosis**:

```bash
# Measure request latency
time curl http://localhost:9091/api/tools/webmcp/extensions/api_calls/query \
  -H "Content-Type: application/json" \
  -d '{"params": {"time_range": "1h"}}'

# Check cache hit rate
curl http://localhost:9091/api/metrics/cache

# Check tool resource usage
top -p $(pgrep -f webmcp)

# Network latency
ping machine-b.internal
traceroute machine-b.internal
```

**Solutions**:

```bash
# Increase cache TTL
# In launcher_config.json:
{
  "cache": {
    "extension_metadata_ttl": 300,  # Increase from 60
    "query_results_ttl": 120,       # Increase from 30
    "max_size": 10000               # Increase from 1000
  }
}

# Enable connection pooling
{
  "http_client": {
    "max_connections": 200,
    "max_per_host": 20,
    "timeout": 60
  }
}

# Optimize tool queries
# - Add indexes to database queries
# - Implement pagination for large results
# - Use streaming for large data transfers
```

### 7. Memory Leaks

**Symptoms**: Tool memory usage growing over time

**Causes**:
- Unclosed connections
- Event subscriber accumulation
- Cache not evicting old entries
- Circular references

**Diagnosis**:

```bash
# Monitor memory usage
watch -n 1 "ps aux | grep webmcp | awk '{print \$6}'"

# Check for open connections
lsof -p $(pgrep -f webmcp) | wc -l

# Python memory profiling
pip install memory_profiler
python -m memory_profiler -m webmcp
```

**Solutions**:

```python
# Fix common memory leak patterns

# 1. Close HTTP sessions
async def make_request():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 2. Limit cache size
cache = CacheManager(max_size=1000)

# 3. Clean up event subscribers
def cleanup_subscribers():
    registry._subscribers = {
        k: [s for s in v if s.is_alive()]
        for k, v in registry._subscribers.items()
    }

# 4. Use weak references for callbacks
import weakref
callback = weakref.ref(my_callback)
```

## Debug Mode

### Enable Debug Logging

```bash
# Environment variable
export MCP_DEBUG=1
export MCP_LOG_LEVEL=DEBUG

# Or in Python
import os
os.environ['MCP_DEBUG'] = '1'
os.environ['MCP_LOG_LEVEL'] = 'DEBUG'
```

### Debug Configuration

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "~/.config/supreme-mcp-tools/logs/debug.log"
  },
  "debug": {
    "enable_profiling": true,
    "profile_dir": "~/.config/supreme-mcp-tools/profiles",
    "trace_requests": true
  }
}
```

### Profiling

```python
# Enable profiling
import cProfile
import pstats

def profile_tool():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run tool
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

## Performance Tuning

### Cache Configuration

```json
{
  "cache": {
    "extension_metadata": {
      "ttl": 300,
      "max_size": 5000
    },
    "query_results": {
      "ttl": 60,
      "max_size": 2000
    },
    "service_discovery": {
      "ttl": 600,
      "max_size": 100
    }
  }
}
```

### Connection Pooling

```json
{
  "http_client": {
    "timeout": 60,
    "max_connections": 500,
    "max_per_host": 50,
    "keepalive_timeout": 30
  }
}
```

### Rate Limiting

```json
{
  "rate_limit": {
    "enabled": true,
    "requests_per_minute": 120,
    "burst_size": 20,
    "by_ip": true,
    "by_api_key": true
  }
}
```

### Async Optimization

```python
# Optimize async operations
import asyncio

# Use connection pooling
connector = aiohttp.TCPConnector(limit=500, limit_per_host=50)

# Use semaphore for concurrency control
semaphore = asyncio.Semaphore(100)

async def limited_request(url):
    async with semaphore:
        return await make_request(url)

# Batch operations
async def batch_query(queries):
    tasks = [query(q) for q in queries]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

## Getting Help

### Log Locations

- Tool logs: `~/.config/supreme-mcp-tools/logs/{tool_name}.log`
- Management server logs: `~/.config/supreme-mcp-tools/logs/management.log`
- Audit logs: `~/.config/supreme-mcp-tools/audit.log`
- Debug logs: `~/.config/supreme-mcp-tools/logs/debug.log`

### Useful Commands

```bash
# View recent errors
grep -i error ~/.config/supreme-mcp-tools/logs/*.log | tail -50

# Check system status
curl http://localhost:9091/health | jq

# List all extensions
curl http://localhost:9091/api/tools | jq '.tools[].capabilities'

# Export configuration
curl http://localhost:9091/api/config > backup.json
```

### Support Channels

1. Check this troubleshooting guide
2. Review the [API Reference](#api-reference)
3. Search existing issues on GitHub
4. File a new issue with:
   - Error messages
   - Log excerpts
   - Configuration (redacted)
   - Steps to reproduce
