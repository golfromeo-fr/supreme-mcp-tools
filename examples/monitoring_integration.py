"""
Example Integration of Monitoring into MCP Tools

This file demonstrates how to integrate the monitoring system into MCP tools.
It shows various ways to record metrics, use middleware, and instrument tool calls.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from fastapi import Request, Response

# Import monitoring components
from monitoring.collector import (
    MetricsRegistry,
    record_request_start,
    record_request_end,
    record_tool_call,
    record_error,
    track_duration
)
from monitoring.middleware import (
    MetricsMiddleware,
    MetricsContext,
    track_metrics
)
from monitoring.exporters import create_metrics_app


# Example 1: Basic tool with manual metric recording
class BasicToolExample:
    """Example of a basic tool with manual metric recording."""
    
    def __init__(self, tool_name: str = "example_tool"):
        self.tool_name = tool_name
        # Get the default collector
        self.collector = MetricsRegistry.get_instance().get_default_collector()
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request with manual metric recording.
        
        This demonstrates the basic pattern for recording metrics.
        """
        # Record request start
        record_request_start(
            tool=self.tool_name,
            method="process"
        )
        
        start_time = time.time()
        status = "success"
        
        try:
            # Simulate some work
            await asyncio.sleep(0.1)
            
            # Process the request (example logic)
            result = {
                "echo": request_data.get("message", "Hello"),
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            # Record error
            record_error(
                tool=self.tool_name,
                error_type=type(e).__name__
            )
            status = "error"
            raise
            
        finally:
            # Record request end
            duration = time.time() - start_time
            record_request_end(
                tool=self.tool_name,
                method="process",
                status=status,
                duration=duration
            )
            
            # Record tool call
            record_tool_call(
                tool=self.tool_name,
                tool_name="process_request",
                status=status,
                duration=duration
            )


# Example 2: Tool using context manager for automatic timing
class ContextManagerToolExample:
    """Example using context manager for automatic duration tracking."""
    
    def __init__(self, tool_name: str = "context_tool"):
        self.tool_name = tool_name
        self.collector = MetricsRegistry.get_instance().get_default_collector()
    
    async def calculate_fibonacci(self, n: int) -> int:
        """
        Calculate Fibonacci number with automatic timing.
        
        Uses the track_duration context manager.
        """
        # Use context manager for automatic timing and error handling
        with track_duration(
            collector=self.collector,
            tool=self.tool_name,
            tool_name="calculate_fibonacci"
        ) as status:
            # Simulate work
            await asyncio.sleep(0.01 * n)
            
            # Calculate Fibonacci
            if n <= 1:
                return n
            return await self.calculate_fibonacci(n-1) + await self.calculate_fibonacci(n-2)
        
        # This won't be reached due to recursion, but shows the pattern
        return 0


# Example 3: FastAPI endpoint with metrics middleware
def create_example_fastapi_app():
    """
    Create a FastAPI example showing metrics middleware integration.
    
    This demonstrates how to add metrics middleware to an existing FastAPI app.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        # Return a mock if FastAPI is not available
        return None
    
    app = FastAPI(title="MCP Tool Example", version="1.0.0")
    
    # Add metrics middleware
    from monitoring.middleware import add_metrics_middleware
    add_metrics_middleware(app)
    
    # Add metrics endpoints
    from monitoring.exporters import add_metrics_routes
    add_metrics_routes(app)
    
    class QueryRequest(BaseModel):
        query: str
        max_results: Optional[int] = 10
    
    class QueryResponse(BaseModel):
        results: list
        count: int
    
    @app.post("/tools/search", response_model=QueryResponse)
    @track_metrics(tool="search_tool", operation="search")
    async def search(request: QueryRequest, req: Request):
        """
        Search endpoint with automatic metrics tracking.
        
        The @track_metrics decorator automatically records:
        - Request start/end
        - Duration
        - Errors
        """
        # Simulate search work
        await asyncio.sleep(0.05)
        
        # Example results
        results = [
            {"id": i, "title": f"Result {i} for '{request.query}'", "score": 1.0 - (i * 0.1)}
            for i in range(min(request.max_results, 5))
        ]
        
        return QueryResponse(results=results, count=len(results))
    
    @app.get("/tools/status")
    async def get_status():
        """Get tool status endpoint."""
        return {"status": "healthy", "tool": "search_tool"}
    
    return app


# Example 4: Manual middleware usage in Streamable HTTP transport
def instrument_streamable_http_base():
    """
    Show how to instrument StreamableHttpTransportBase._handle_tool_call.
    
    This is the exact modification needed for the base class.
    """
    # This is the code that would be added to launcher/streamable_http/streamable_http_base.py
    # in the _handle_tool_call method:
    
    example_code = '''
    async def _handle_tool_call(
        self,
        params: Dict[str, Any],
        session: Dict[str, Any],
        request_id: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle tools/call request with metrics."""
        # Extract tool name from params
        tool_name = params.get("name", "unknown")
        
        # Record tool call start
        from monitoring.collector import record_tool_call
        import time
        start_time = time.time()
        
        try:
            # Call the actual tool implementation (to be implemented by subclasses)
            async for response in self._actual_tool_call(params, session, request_id):
                yield response
            
            # Record successful tool call
            duration = time.time() - start_time
            record_tool_call(
                tool=self.server_name,
                tool_name=tool_name,
                status="success",
                duration=duration
            )
            
        except Exception as e:
            # Record failed tool call
            duration = time.time() - start_time
            from monitoring.collector import record_error
            record_error(
                tool=self.server_name,
                error_type=type(e).__name__
            )
            record_tool_call(
                tool=self.server_name,
                tool_name=tool_name,
                status="error",
                duration=duration
            )
            raise
    '''
    
    return example_code


# Example 5: Integration with launchmcp.py
def show_launchmcp_integration():
    """
    Show how to integrate monitoring with launchmcp.py.
    
    This demonstrates the initialization and setup needed in the main launcher.
    """
    integration_code = '''
    # In launchp.py, after loading configuration:
    
    # 1. Initialize monitoring system
    from monitoring.config import get_monitoring_config, is_monitoring_enabled
    from monitoring.collector import MetricsRegistry
    from monitoring.exporters import create_metrics_app
    
    # Check if monitoring is enabled
    monitoring_config = get_monitoring_config()
    if is_monitoring_enabled():
        # Initialize the metrics registry
        registry = MetricsRegistry.get_instance()
        
        # Create and start metrics exporter app (if needed)
        # This would typically run on a separate port
        metrics_app = create_metrics_app()
        # Note: In practice, you'd mount this or run it separately
    
    # 2. Add metrics middleware to FastAPI apps in tools
    # This would be done in each tool's server setup
    
    # 3. Instrument StreamableHttpTransportBase
    # This modification is done in launcher/streamable_http/streamable_http_base.py
    '''
    
    return integration_code


# Example usage demonstrations
async def demonstrate_usage():
    """Demonstrate various ways to use the monitoring system."""
    
    print("=== Monitoring Integration Examples ===\n")
    
    # Example 1: Basic tool
    print("1. Basic Tool Example:")
    tool = BasicToolExample("demo_tool")
    result = await tool.process_request({"message": "Hello World"})
    print(f"   Result: {result}\n")
    
    # Example 2: Context manager
    print("2. Context Manager Example:")
    ctx_tool = ContextManagerToolExample("fib_tool")
    # Note: This is recursive and would need adjustment for real use
    print("   Context manager tool ready for use\n")
    
    # Example 3: FastAPI app
    print("3. FastAPI Integration:")
    app = create_example_fastapi_app()
    if app:
        print("   FastAPI app with metrics middleware created")
        print("   Endpoints: /tools/search, /tools/status, /metrics, /health, /stats\n")
    else:
        print("   FastAPI not available, skipping app creation\n")
    
    # Show metrics
    print("4. Current Metrics:")
    collector = MetricsRegistry.get_instance().get_default_collector()
    if collector:
        metrics = collector.get_all_metrics()
        print(f"   Total metrics collected: {len(metrics)}")
        for name, metric in list(metrics.items())[:5]:  # Show first 5
            print(f"   - {name}: {metric.description}")
        if len(metrics) > 5:
            print(f"   ... and {len(metrics) - 5} more")
    else:
        print("   No collector available")
    
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_usage())