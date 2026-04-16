"""
FEF V3 Prometheus Metrics

Provides Prometheus metrics integration for the Flexible Extensibility Framework V3.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to stubs if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics will be disabled")


class FEFMetrics:
    """
    Prometheus metrics for FEF V3.
    
    Collects and exposes metrics for:
    - Extension queries, mutations, and executions
    - Circuit breaker states
    - Cache hit/miss rates
    - Service registry health
    - HTTP request latencies
    """
    
    def __init__(self, port: int | None = None):
        """
        Initialize FEF V3 metrics.
        
        Args:
            port: Optional port to start Prometheus HTTP server
        """
        self.port = port
        self._metrics: dict[str, Any] = {}
        
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()
            
            if port:
                start_http_server(port)
                logger.info(f"Prometheus metrics server started on port {port}")
        else:
            logger.warning("Prometheus metrics disabled (prometheus_client not installed)")
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Extension operations
        self._metrics["queries_total"] = Counter(
            "mcp_queries_total",
            "Total number of extension queries",
            ["tool", "extension"]
        )
        
        self._metrics["mutations_total"] = Counter(
            "mcp_mutations_total",
            "Total number of extension mutations",
            ["tool", "extension"]
        )
        
        self._metrics["executions_total"] = Counter(
            "mcp_executions_total",
            "Total number of extension executions",
            ["tool", "extension"]
        )
        
        self._metrics["errors_total"] = Counter(
            "mcp_errors_total",
            "Total number of errors",
            ["tool", "extension", "error_type"]
        )
        
        # Query duration
        self._metrics["query_duration"] = Histogram(
            "mcp_query_duration_seconds",
            "Extension query duration in seconds",
            ["tool", "extension"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Circuit breaker
        self._metrics["circuit_breaker_state"] = Gauge(
            "mcp_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["tool"]
        )
        
        self._metrics["circuit_breaker_failures"] = Counter(
            "mcp_circuit_breaker_failures_total",
            "Total circuit breaker failures",
            ["tool"]
        )
        
        # Cache
        self._metrics["cache_hits"] = Counter(
            "mcp_cache_hits_total",
            "Total cache hits",
            ["cache_type"]
        )
        
        self._metrics["cache_misses"] = Counter(
            "mcp_cache_misses_total",
            "Total cache misses",
            ["cache_type"]
        )
        
        self._metrics["cache_size"] = Gauge(
            "mcp_cache_size",
            "Current cache size",
            ["cache_type"]
        )
        
        # Service registry
        self._metrics["tools_total"] = Gauge(
            "mcp_tools_total",
            "Total number of registered tools"
        )
        
        self._metrics["tools_healthy"] = Gauge(
            "mcp_tools_healthy",
            "Number of healthy tools"
        )
        
        self._metrics["extensions_total"] = Gauge(
            "mcp_extensions_total",
            "Total number of registered extensions",
            ["tool", "type"]
        )
        
        # HTTP
        self._metrics["http_requests_total"] = Counter(
            "mcp_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        
        self._metrics["http_request_duration"] = Histogram(
            "mcp_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Events
        self._metrics["events_total"] = Counter(
            "mcp_events_total",
            "Total events published",
            ["tool", "event_type"]
        )
        
        # Dead letter queue
        self._metrics["dlq_entries"] = Gauge(
            "mcp_dlq_entries",
            "Number of entries in dead letter queue",
            ["status"]
        )
        
        # System info
        self._metrics["info"] = Info(
            "mcp_fef_v3",
            "FEF V3 system information"
        )
        self._metrics["info"].info({
            "version": "3.0.0",
            "framework": "FEF V3"
        })
    
    def record_query(
        self,
        tool: str,
        extension: str,
        duration: float,
        success: bool = True
    ) -> None:
        """
        Record an extension query.
        
        Args:
            tool: Tool name
            extension: Extension name
            duration: Query duration in seconds
            success: Whether the query succeeded
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["queries_total"].labels(tool=tool, extension=extension).inc()
        self._metrics["query_duration"].labels(tool=tool, extension=extension).observe(duration)
        
        if not success:
            self._metrics["errors_total"].labels(
                tool=tool, extension=extension, error_type="query"
            ).inc()
    
    def record_mutation(
        self,
        tool: str,
        extension: str,
        success: bool = True
    ) -> None:
        """
        Record an extension mutation.
        
        Args:
            tool: Tool name
            extension: Extension name
            success: Whether the mutation succeeded
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["mutations_total"].labels(tool=tool, extension=extension).inc()
        
        if not success:
            self._metrics["errors_total"].labels(
                tool=tool, extension=extension, error_type="mutation"
            ).inc()
    
    def record_execution(
        self,
        tool: str,
        extension: str,
        success: bool = True
    ) -> None:
        """
        Record an extension execution.
        
        Args:
            tool: Tool name
            extension: Extension name
            success: Whether the execution succeeded
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["executions_total"].labels(tool=tool, extension=extension).inc()
        
        if not success:
            self._metrics["errors_total"].labels(
                tool=tool, extension=extension, error_type="execution"
            ).inc()
    
    def record_circuit_breaker_state(self, tool: str, state: str) -> None:
        """
        Record circuit breaker state.
        
        Args:
            tool: Tool name
            state: Circuit breaker state (closed, open, half_open)
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
        self._metrics["circuit_breaker_state"].labels(tool=tool).set(state_value)
    
    def record_circuit_breaker_failure(self, tool: str) -> None:
        """
        Record a circuit breaker failure.
        
        Args:
            tool: Tool name
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["circuit_breaker_failures"].labels(tool=tool).inc()
    
    def record_cache_hit(self, cache_type: str) -> None:
        """
        Record a cache hit.
        
        Args:
            cache_type: Type of cache
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["cache_hits"].labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """
        Record a cache miss.
        
        Args:
            cache_type: Type of cache
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["cache_misses"].labels(cache_type=cache_type).inc()
    
    def update_cache_size(self, cache_type: str, size: int) -> None:
        """
        Update cache size metric.
        
        Args:
            cache_type: Type of cache
            size: Current cache size
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["cache_size"].labels(cache_type=cache_type).set(size)
    
    def update_tools_total(self, total: int, healthy: int) -> None:
        """
        Update tools metrics.
        
        Args:
            total: Total number of tools
            healthy: Number of healthy tools
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["tools_total"].set(total)
        self._metrics["tools_healthy"].set(healthy)
    
    def update_extensions_total(
        self,
        tool: str,
        ext_type: str,
        count: int
    ) -> None:
        """
        Update extensions count metric.
        
        Args:
            tool: Tool name
            ext_type: Extension type
            count: Number of extensions
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["extensions_total"].labels(tool=tool, type=ext_type).set(count)
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ) -> None:
        """
        Record an HTTP request.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint
            status: Response status code
            duration: Request duration in seconds
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["http_requests_total"].labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc()
        
        self._metrics["http_request_duration"].labels(
            method=method, endpoint=endpoint
        ).observe(duration)
    
    def record_event(self, tool: str, event_type: str) -> None:
        """
        Record an event publication.
        
        Args:
            tool: Tool name
            event_type: Event type
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["events_total"].labels(tool=tool, event_type=event_type).inc()
    
    def update_dlq_entries(self, status: str, count: int) -> None:
        """
        Update dead letter queue entries metric.
        
        Args:
            status: Entry status (pending, processed, failed)
            count: Number of entries
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._metrics["dlq_entries"].labels(status=status).set(count)


# Global metrics instance
_fef_metrics: FEFMetrics | None = None


def get_fef_metrics(port: int | None = None) -> FEFMetrics:
    """
    Get or create the global FEF metrics instance.
    
    Args:
        port: Optional port for Prometheus HTTP server
        
    Returns:
        FEFMetrics instance
    """
    global _fef_metrics
    if _fef_metrics is None:
        _fef_metrics = FEFMetrics(port=port)
    return _fef_metrics
