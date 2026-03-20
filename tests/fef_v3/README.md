# FEF V3 Automated Test Suite

Comprehensive test suite for the Flexible Extensibility Framework V3 integration across all MCP tools.

## Overview

This test suite validates the FEF V3 implementation for all MCP tools:
- **webmcp** - Web search and URL fetch capabilities (port 9001)
- **simplemcp** - Simple demonstration tools (port 9012)
- **ragmcp** - Retrieval-Augmented Generation (port 9014)
- **convertermcp** - Document conversion tools (port 9013)
- **oraclemcp** - Oracle database access (port 9010)

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MCP Tools

Ensure your MCP tools are running with FEF V3 enabled:

```bash
# Using launchmcp.py
./launchmcp.py simplemcp ragmcp webmcp

# Or start individual tools
python tools/webmcp/webmcp_sse.py &
python tools/simplemcp/simplemcp_sse.py &
python tools/ragmcp/ragmcp_sse.py &
```

### 3. Verify Management Servers

Check that management servers are accessible:

```bash
curl http://localhost:9001/extensions  # webmcp
curl http://localhost:9012/extensions  # simplemcp
curl http://localhost:9014/extensions  # ragmcp
```

## Quick Start

### Run All Tests

```bash
python tests/fef_v3/test_runner.py
```

### Run Specific Tools

```bash
# Test only webmcp
python tests/fef_v3/test_runner.py --tools webmcp

# Test multiple tools (comma-separated)
python tests/fef_v3/test_runner.py --tools webmcp,simplemcp,ragmcp

# Or use the quick start script
./tests/fef_v3/run_tests.sh --tools webmcp,simplemcp,ragmcp
```

### Verbose Output

```bash
python tests/fef_v3/test_runner.py --verbose
```

## Test Structure

### Test Phases

The test suite runs tests in 5 phases for each tool:

1. **Extension Listing** - Verifies all extensions are registered
2. **Common Extensions** - Tests standard FEF V3 extensions:
   - `request_stats` - Request statistics and performance metrics
   - `cache_stats` - Cache statistics and hit rates
   - `tool_info` - Tool information and status
   - `cache_config` - Update cache configuration
   - `api_key` - Set API key for authentication
   - `clear_cache` - Clear cached data
   - `reset_counters` - Reset all metrics counters

3. **Tool-Specific Extensions** - Tests custom extensions for each tool:
   - **webmcp**: `search_stats`, `fetch_stats`, `search_config`, `search_history`, `fetch_cache_hits`
   - **simplemcp**: `tool_usage`, `api_response_times`, `timeout_config`
   - **ragmcp**: `vector_db_stats`, `embedding_stats`, `collection_stats`, `collection_config`
   - **convertermcp**: `conversion_stats`, `format_usage`, `output_config`
   - **oraclemcp**: `query_stats`, `connection_pool`, `schema_cache`, `pool_config`

4. **Error Handling** - Tests error handling for invalid requests:
   - Non-existent extensions
   - Invalid parameters
   - Missing required parameters

5. **Performance** - Tests performance metrics:
   - Average response time (< 100ms)
   - Maximum response time (< 500ms)
   - Concurrent request handling (10 concurrent requests)

### Test Results

Each test produces a result with:
- **Status**: PASSED, FAILED, SKIPPED, or ERROR
- **Duration**: Execution time in milliseconds
- **Message**: Description of the test outcome
- **Details**: Additional information for debugging

## Configuration

### Test Configuration File

Edit [`test_config.json`](test_config.json) to customize tests:

```json
{
  "tools": {
    "webmcp": {
      "enabled": true,
      "mgmt_port": 9001,
      "extensions": {
        "common": [...],
        "custom": [...]
      }
    }
  },
  "performance": {
    "enabled": true,
    "thresholds": {
      "avg_response_time_ms": 100,
      "max_response_time_ms": 500,
      "concurrent_requests": 10
    }
  }
}
```

### Disabling Tools

To skip testing a specific tool, set `enabled: false` in the configuration:

```json
{
  "tools": {
    "oraclemcp": {
      "enabled": false
    }
  }
}
```

## Manual Testing

### Test Individual Extensions

```bash
# Test request_stats
curl -X POST http://localhost:9001/extensions/request_stats \
  -H "Content-Type: application/json" \
  -d '{"time_range": "1h"}'

# Test cache_config update
curl -X POST http://localhost:9001/extensions/cache_config \
  -H "Content-Type: application/json" \
  -d '{"max_size": 2000, "ttl": 600, "enabled": true}'

# Test clear_cache
curl -X POST http://localhost:9001/extensions/clear_cache \
  -H "Content-Type: application/json" \
  -d '{"cache_type": "all"}'
```

### List All Extensions

```bash
curl http://localhost:9001/extensions | jq '.extensions[].name'
```

## Test Output

### Console Output

```
============================================================
FEF V3 Automated Test Suite
============================================================

Testing tools: webmcp, simplemcp, ragmcp

============================================================
Testing webmcp
Management Server: http://localhost:9001
============================================================

[1/5] Testing extension listing...
  PASSED: Found 11 extensions (expected >= 11)

[2/5] Testing common extensions...
  PASSED: request_stats_execution
  PASSED: cache_stats_execution
  PASSED: tool_info_execution
  ...

[3/5] Testing tool-specific extensions...
  PASSED: search_stats_execution
  PASSED: fetch_stats_execution
  ...

[4/5] Testing error handling...
  PASSED: error_handling_nonexistent
  PASSED: error_handling_invalid_params

[5/5] Testing performance...
  PASSED: Performance acceptable (avg: 45.23ms, max: 120.45ms)

============================================================
Test Summary
============================================================

webmcp:
  Total: 15 tests
  Passed: 15
  Failed: 0
  Errors: 0
  Duration: 1234.56ms

============================================================
Overall Results:
  Total Tests: 45
  Passed: 45
  Failed: 0
  Errors: 0
  Success Rate: 100.0%

✓ All tests passed!
```

### JSON Output

Test results can be saved to JSON for CI/CD integration:

```python
from tests.fef_v3.test_utils import ReportGenerator

# Save results to file
ReportGenerator.save_results(results, "test_results.json")
```

## Troubleshooting

### Connection Refused

If you see connection errors, ensure the MCP tools are running:

```bash
# Check if management servers are listening
netstat -tlnp | grep -E ':(9010|9012|9013|9014|9001)'
```

### Timeout Errors

The test runner uses curl for HTTP requests with a 30-second default timeout. This is sufficient for most extensions, but ragmcp's Qdrant-based extensions may take longer. The timeout is configured in the `curl_request()` function in [`test_runner.py`](test_runner.py).<parameter>
<parameter name="expected_replacements">1

### Extension Not Found

If an extension is not found, verify it's registered:

```bash
curl http://localhost:9001/extensions | jq '.extensions[].name'
```

Check the tool's FEF V3 setup code for proper extension registration.

### Missing Dependencies

Install required packages (the test runner uses curl for HTTP requests):

```bash
pip install rich
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: FEF V3 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/fef_v3/requirements.txt
      
      - name: Start MCP tools
        run: |
          ./launchmcp.py simplemcp ragmcp webmcp &
          sleep 10
      
      - name: Run tests
        run: python tests/fef_v3/test_runner.py
      
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_results.json
```

## Advanced Usage

### Custom Test Scripts

Create custom test scripts using the utilities:

```python
from tests.fef_v3.test_runner import curl_request

# Make HTTP requests directly
status, data = curl_request("GET", "http://localhost:9001/extensions")
print(f"Extensions: {data}")

# Test an extension
status, data = curl_request(
    "POST",
    "http://localhost:9001/extensions/request_stats/query",
    data={"params": {}}
)
print(f"Response time: {data}")
```

### Batch Testing

Test multiple tools in sequence:

```python
from tests.fef_v3.test_runner import FEFTestRunner

# Create runner with specific tools
runner = FEFTestRunner(tools=["webmcp", "simplemcp", "ragmcp"])
runner.run_all_tests()
runner.print_summary()

# Check results
for result in runner.all_results:
    print(f"{result.tool_name}: {result.passed_count}/{len(result.results)} passed")
```

## Contributing

When adding new extensions to MCP tools:

1. Update [`test_config.json`](test_config.json) with the new extension
2. Add test cases to the appropriate tool section
3. Update the expected fields list
4. Run the test suite to verify

## Support

For issues or questions:
- Check the [FEF V3 Documentation](../../FLEXIBLE_EXTENSIBILITY_FRAMEWORK_V3/)
- Review the [Implementation Guide](../../FLEXIBLE_EXTENSIBILITY_FRAMEWORK_V3/14-implementation-guide.md)
- Open an issue in the project repository

## License

Same as the main project.
