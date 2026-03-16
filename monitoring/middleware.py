"""
FastAPI Middleware for MCP Launcher Monitoring

This module provides FastAPI middleware for collecting request/response metrics
and integrating with the metrics collection system.
"""

import asyncio
import logging
import time
import uuid
from typing import Callable, List, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .collector import (
    MetricsRegistry
)

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for collecting request/response metrics.
    
    Records metrics for:
    - Request count
    - Request duration
    - Response status codes
    - Active requests
    - Errors
    """
    
    def __init__(
        self,
        app: ASGIApp,
        registry: Optional[MetricsRegistry] = None,
        collector_name: str = "mcp",
        exclude_paths: Optional[List[str]] = None,
        include_paths: Optional[List[str]] = None
    ):
        """
        Initialize the metrics middleware.
        
        Args:
            app: ASGI application
            registry: MetricsRegistry instance
            collector_name: Name of the collector to use
            exclude_paths: List of paths to exclude from metrics
            include_paths: List of paths to include (if None, include all)
        """
        super().__init__(app)
        self.registry = registry or MetricsRegistry.get_instance()
        self.collector_name = collector_name
        self.exclude_paths = set(exclude_paths or [])
        self.include_paths = set(include_paths) if include_paths else None
        
        # Default exclusions
        self.exclude_paths.update({
            "/metrics",
            "/metrics/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        })
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], asyncio.Future[Response]]
    ) -> Response:
        """
        Process the request and collect metrics.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response from the application
        """
        # Check if we should collect metrics for this path
        if not self._should_collect(request.url.path):
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Extract tool name from path if possible
        tool_name = self._extract_tool_name(request)
        
        # Record request start
        start_time = time.time()
        self.registry.record_request_start(
            tool=tool_name,
            method=request.method,
            collector_name=self.collector_name
        )
        
        # Add request ID to request state for tracking
        request.state.request_id = request_id
        request.state.start_time = start_time
        request.state.tool_name = tool_name
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record request end
            self.registry.record_request_end(
                tool=tool_name,
                method=request.method,
                status="success",  # Successful response
                duration=duration,
                collector_name=self.collector_name
            )
            
            return response
            
        except Exception as exc:
            # Calculate duration
            duration = time.time() - start_time
            
            # Record error
            self.registry.record_error(
                tool=tool_name,
                error_type=type(exc).__name__,
                collector_name=self.collector_name
            )
            
            # Record request end with error status
            self.registry.record_request_end(
                tool=tool_name,
                method=request.method,
                status="error",  # Internal server error
                duration=duration,
                collector_name=self.collector_name
            )
            
            # Re-raise the exception
            raise
    
    def _should_collect(self, path: str) -> bool:
        """
        Determine if metrics should be collected for this path.
        
        Args:
            path: Request path
            
        Returns:
            True if metrics should be collected
        """
        # Check exclusions
        if path in self.exclude_paths:
            return False
        
        # Check if path starts with any excluded prefix
        for excluded in self.exclude_paths:
            if path.startswith(excluded):
                return False
        
        # Check inclusions (if specified)
        if self.include_paths is not None:
            # Check if path starts with any included prefix
            for included in self.include_paths:
                if path.startswith(included):
                    return True
            return False
        
        return True
    
    def _extract_tool_name(self, request: Request) -> str:
        """
        Extract tool name from the request.
        
        Args:
            request: Incoming request
            
        Returns:
            Tool name or "unknown"
        """
        # Try to get from path parameters
        path_params = request.path_params
        if "tool" in path_params:
            return str(path_params["tool"])
        
        # Try to get from query parameters
        tool_param = request.query_params.get("tool")
        if tool_param:
            return str(tool_param)
        
        # Try to extract from path
        # Example: /tools/webmcp/messages/ -> webmcp
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "tools":
            return str(path_parts[1])
        
        # Try to get from headers
        tool_header = request.headers.get("X-MCP-Tool")
        if tool_header:
            return str(tool_header)
        
        # Default to unknown
        return "unknown"


def create_metrics_middleware(
    app: ASGIApp,
    registry: Optional[MetricsRegistry] = None,
    collector_name: str = "mcp",
    exclude_paths: Optional[List[str]] = None,
    include_paths: Optional[List[str]] = None
) -> MetricsMiddleware:
    """
    Create and configure metrics middleware.
    
    Args:
        app: ASGI application
        registry: MetricsRegistry instance
        collector_name: Name of the collector to use
        exclude_paths: List of paths to exclude from metrics
        include_paths: List of paths to include (if None, include all)
        
    Returns:
        Configured MetricsMiddleware instance
    """
    return MetricsMiddleware(
        app=app,
        registry=registry,
        collector_name=collector_name,
        exclude_paths=exclude_paths,
        include_paths=include_paths
    )


# Convenience function for adding middleware to FastAPI app
def add_metrics_middleware(
    app: ASGIApp,
    registry: Optional[MetricsRegistry] = None,
    collector_name: str = "mcp",
    exclude_paths: Optional[List[str]] = None,
    include_paths: Optional[List[str]] = None
) -> None:
    """
    Add metrics middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        registry: MetricsRegistry instance
        collector_name: Name of the collector to use
        exclude_paths: List of paths to exclude from metrics
        include_paths: List of paths to include (if None, include all)
    """
    # Pass just the class and let FastAPI handle instantiation
    app.add_middleware(
        MetricsMiddleware,
        registry=registry,
        collector_name=collector_name,
        exclude_paths=exclude_paths,
        include_paths=include_paths
    )
    logger.info("Added metrics middleware to application")


# Context manager for manual metric recording in endpoints
class MetricsContext:
    """
    Context manager for manual metric recording in endpoint functions.
    
    Usage:
        async def my_endpoint(request: Request):
            async with MetricsContext(request, "webmcp", "search") as ctx:
                # Your endpoint logic here
                ctx.set_status("success")
                return {"result": "data"}
    """
    
    def __init__(
        self,
        request: Request,
        tool: str,
        operation: str,
        registry: Optional[MetricsRegistry] = None,
        collector_name: str = "mcp"
    ):
        """
        Initialize the metrics context.
        
        Args:
            request: FastAPI request object
            tool: Tool name
            operation: Operation name
            registry: MetricsRegistry instance
            collector_name: Name of the collector to use
        """
        self.request = request
        self.tool = tool
        self.operation = operation
        self.registry = registry or MetricsRegistry.get_instance()
        self.collector_name = collector_name
        self.start_time = None
        self.status = "error"
    
    async def __aenter__(self):
        """Enter the context and record request start."""
        self.start_time = time.time()
        self.registry.record_request_start(
            tool=self.tool,
            method=self.request.method,
            collector_name=self.collector_name
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and record request end."""
        duration = self.start_time - time.time() if self.start_time else 0.0
        
        if exc_type is not None:
            # An exception occurred
            self.registry.record_error(
                tool=self.tool,
                error_type=exc_type.__name__,
                collector_name=self.collector_name
            )
            self.status = "error"
        else:
            # No exception
            self.status = getattr(self, '_status', 'success')
        
        self.registry.record_request_end(
            tool=self.tool,
            method=self.request.method,
            status=self.status,
            duration=duration,
            collector_name=self.collector_name
        )
        
        # Don't suppress exceptions
        return False
    
    def set_status(self, status: str) -> None:
        """Set the status for successful completion."""
        self._status = status


# Decorator for automatic metric recording
def track_metrics(
    tool: str,
    operation: Optional[str] = None,
    registry: Optional[MetricsRegistry] = None,
    collector_name: str = "mcp"
):
    """
    Decorator for automatic metric recording of async functions.
    
    Usage:
        @track_metrics(tool="webmcp", operation="search")
        async def search_tool(params):
            # Your implementation here
            return result
    
    Args:
        tool: Tool name
        operation: Operation name (defaults to function name)
        registry: MetricsRegistry instance
        collector_name: Name of the collector to use
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Determine operation name
            op_name = operation or func.__name__
            
            # Try to extract request from args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                # No request found, call function without metrics
                return await func(*args, **kwargs)
            
            # Use metrics context
            async with MetricsContext(
                request=request,
                tool=tool,
                operation=op_name,
                registry=registry,
                collector_name=collector_name
            ):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator
