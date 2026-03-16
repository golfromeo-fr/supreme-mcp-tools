"""
Prometheus Exporter for MCP Launcher Metrics

This module provides Prometheus-compatible metrics export functionality.
It formats metrics in Prometheus text format and provides HTTP endpoints.
"""

import logging
import time
import hmac
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, Response, Depends, Header, HTTPException, APIRouter
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader

from .collector import (
    MetricsCollector,
    MetricsRegistry,
    MetricType,
    BaseMetric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricSample
)

logger = logging.getLogger(__name__)

# Optional API key for metrics endpoint authentication
API_KEY_NAME = "X-Metrics-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


class PrometheusExporter:
    """
    Prometheus metrics exporter.
    
    Formats metrics in Prometheus text format and provides methods
    for generating the /metrics endpoint response.
    """
    
    # Prometheus content type
    CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
    
    def __init__(
        self,
        collector: Optional[MetricsCollector] = None,
        include_start_time: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Prometheus exporter.
        
        Args:
            collector: MetricsCollector instance to export from
            include_start_time: Whether to include process start time
            api_key: Optional API key for securing the metrics endpoint
        """
        self.collector = collector
        self.include_start_time = include_start_time
        self.api_key = api_key
    
    def _check_auth(self, api_key_header: Optional[str]) -> bool:
        """
        Check if the provided API key matches the configured key.
        If no API key is configured, authentication is disabled.
        
        Args:
            api_key_header: The API key from the request header
            
        Returns:
            True if authentication passes or is not required
        """
        # If no API key is configured, authentication is disabled
        if not self.api_key:
            return True
            
        # If API key is configured, check if it matches using constant-time comparison
        if api_key_header is None:
            return False
        return hmac.compare_digest(api_key_header, self.api_key)
    
    def generate_metrics(self) -> str:
        """
        Generate Prometheus-formatted metrics output.
        
        Returns:
            Metrics in Prometheus text format
        """
        lines: List[str] = []
        
        if not self.collector:
            return ""
        
        metrics = self.collector.get_all_metrics()
        
        for name, metric in metrics.items():
            # Add help text
            if metric.description:
                lines.append(f"# HELP {metric.name} {metric.description}")
            
            # Add type
            metric_type = metric._get_metric_type()
            lines.append(f"# TYPE {metric.name} {metric_type.value}")
            
            # Add samples based on metric type
            if isinstance(metric, Counter):
                lines.extend(self._format_counter(metric))
            elif isinstance(metric, Gauge):
                lines.extend(self._format_gauge(metric))
            elif isinstance(metric, Histogram):
                lines.extend(self._format_histogram(metric))
            elif isinstance(metric, Summary):
                lines.extend(self._format_summary(metric))
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def _format_counter(self, counter: Counter) -> List[str]:
        """Format a counter metric."""
        lines = []
        
        # Get all current values
        for label_key, value in counter.get_all_values().items():
            # Convert label key back to labels dictionary
            labels = {}
            if label_key:
                for i, label_name in enumerate(counter.label_names):
                    labels[label_name] = label_key[i]
            
            labels_str = self._format_labels(labels)
            lines.append(f"{counter.name}{labels_str} {value}")
        
        return lines
    
    def _format_gauge(self, gauge: Gauge) -> List[str]:
        """Format a gauge metric."""
        lines = []
        
        # Get all current values
        for label_key, value in gauge.get_all_values().items():
            # Convert label key back to labels dictionary
            labels = {}
            if label_key:
                for i, label_name in enumerate(gauge.label_names):
                    labels[label_name] = label_key[i]
            
            labels_str = self._format_labels(labels)
            lines.append(f"{gauge.name}{labels_str} {value}")
        
        return lines
    
    def _format_histogram(self, histogram: Histogram) -> List[str]:
        """Format a histogram metric."""
        lines = []
        
        # Get all samples including buckets
        for sample in histogram.get_samples():
            labels_str = self._format_labels(sample.labels)
            lines.append(f"{sample.name}{labels_str} {sample.value}")
        
        return lines
    
    def _format_summary(self, summary: Summary) -> List[str]:
        """Format a summary metric."""
        lines = []
        
        # Get all current values
        for label_key, stats in summary.get_all_values().items():
            # Convert label key back to labels dictionary
            labels = {}
            if label_key:
                for i, label_name in enumerate(summary.label_names):
                    labels[label_name] = label_key[i]
            
            labels_str = self._format_labels(labels)
            
            # Add count
            lines.append(f"{summary.name}_count{labels_str} {stats['count']}")
            
            # Add sum
            lines.append(f"{summary.name}_sum{labels_str} {stats['sum']}")
            
            # Add percentiles
            for percentile, value in stats['percentiles'].items():
                labels[f'quantile'] = str(percentile)
                percentile_labels_str = self._format_labels(labels)
                lines.append(f"{summary.name}{percentile_labels_str} {value}")
                # Remove the quantile label for next iteration
                if 'quantile' in labels:
                    del labels['quantile']
        
        return lines
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        
        label_parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_parts) + "}"


def create_metrics_app(
    collector: Optional[MetricsCollector] = None,
    app_name: str = "mcp-monitoring",
    api_key: Optional[str] = None
) -> FastAPI:
    """
    Create a FastAPI application with metrics endpoints.
    
    Args:
        collector: MetricsCollector instance
        app_name: Name for the FastAPI app
        api_key: Optional API key for securing the metrics endpoints
        
    Returns:
        FastAPI application with /metrics endpoint
    """
    app = FastAPI(
        title=f"{app_name} Monitoring",
        description="MCP Launcher Metrics API",
        version="1.0.0"
    )
    
    exporter = PrometheusExporter(collector, api_key=api_key)
    
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(api_key_header: Optional[str] = Depends(api_key_header)) -> str:
        """Get Prometheus-formatted metrics."""
        if not exporter._check_auth(api_key_header):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if collector:
            return exporter.generate_metrics()
        return ""
    
    @app.get("/health")
    async def health(api_key_header: Optional[str] = Depends(api_key_header)) -> Dict[str, Any]:
        """Get health status."""
        if not exporter._check_auth(api_key_header):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {
            "status": "healthy",
            "collector": collector is not None,
            "timestamp": time.time()
        }
    
    @app.get("/stats")
    async def stats(api_key_header: Optional[str] = Depends(api_key_header)) -> Dict[str, Any]:
        """Get basic statistics."""
        if not exporter._check_auth(api_key_header):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not collector:
            return {"error": "No collector configured"}
        
        metrics = collector.get_all_metrics()
        
        stats = {
            "metrics_count": len(metrics),
            "uptime_seconds": collector.get_uptime()
        }
        
        # Add counts for each metric type
        for name, metric in metrics.items():
            if isinstance(metric, Counter):
                stats[f"{name}_total"] = metric.get_value()
            elif isinstance(metric, Gauge):
                stats[name] = metric.get_value()
        
        return stats
    
    return app


class MetricsEndpoint:
    """
    Standalone metrics endpoint handler.
    
    Can be added to existing FastAPI applications.
    """
    
    def __init__(
        self,
        registry: Optional[MetricsRegistry] = None,
        collector_name: str = "mcp",
        api_key: Optional[str] = None
    ):
        """
        Initialize the metrics endpoint.
        
        Args:
            registry: MetricsRegistry instance
            collector_name: Name of the collector to use
            api_key: Optional API key for securing the metrics endpoints
        """
        self.registry = registry or MetricsRegistry.get_instance()
        self.collector_name = collector_name
        self.api_key = api_key
        self._exporter: Optional[PrometheusExporter] = None
    
    def _check_auth(self, api_key: Optional[str]) -> bool:
        """
        Check if the provided API key matches the configured key.
        If no API key is configured, authentication is disabled.
        
        Args:
            api_key: The API key from the request header
            
        Returns:
            True if authentication passes or is not required
        """
        # If no API key is configured, authentication is disabled
        if not self.api_key:
            return True
            
        # If API key is configured, check if it matches using constant-time comparison
        if api_key is None:
            return False
        return hmac.compare_digest(api_key, self.api_key)
    
    @property
    def exporter(self) -> PrometheusExporter:
        """Get or create the exporter."""
        if self._exporter is None:
            collector = self.registry.get_or_create_collector(self.collector_name)
            self._exporter = PrometheusExporter(collector, api_key=self.api_key)
        return self._exporter
    
    async def get_metrics(self, request: Request) -> PlainTextResponse:
        """Get Prometheus metrics."""
        # Extract API key from header
        api_key = request.headers.get(API_KEY_NAME)
        # Check auth BEFORE accessing collector to prevent endpoint probing
        if not self._check_auth(api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return PlainTextResponse(
            content=self.exporter.generate_metrics(),
            media_type=PrometheusExporter.CONTENT_TYPE
        )
    
    async def get_health(self, request: Request) -> Dict[str, Any]:
        """Get health status."""
        # Extract API key from header
        api_key = request.headers.get(API_KEY_NAME)
        # Check auth BEFORE accessing collector to prevent endpoint probing
        if not self._check_auth(api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")
        collector = self.registry.get_or_create_collector(self.collector_name)
        return {
            "status": "healthy" if collector else "unhealthy",
            "enabled": self.registry.is_enabled,
            "timestamp": time.time()
        }
    
    async def get_stats(self, request: Request) -> Dict[str, Any]:
        """Get statistics."""
        # Extract API key from header
        api_key = request.headers.get(API_KEY_NAME)
        # Check auth BEFORE accessing collector to prevent endpoint probing
        if not self._check_auth(api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")
        collector = self.registry.get_or_create_collector(self.collector_name)
        
        if not collector:
            return {"error": "No collector available"}
        
        metrics = collector.get_all_metrics()
        
        stats = {
            "metrics_count": len(metrics),
            "uptime_seconds": collector.get_uptime()
        }
        
        # Add current values for key metrics
        key_metrics = [
            "mcp_requests_total",
            "mcp_tool_calls_total", 
            "mcp_errors_total",
            "mcp_active_requests",
            "mcp_server_up"
        ]
        
        for name in key_metrics:
            if name in metrics:
                metric = metrics[name]
                if isinstance(metric, (Counter, Gauge)):
                    stats[name] = metric.get_value()
        
        return stats


def create_metrics_router(
    registry: Optional[MetricsRegistry] = None,
    collector_name: str = "mcp",
    api_key: Optional[str] = None
):
    """
    Create a metrics router for FastAPI.
    
    Args:
        registry: MetricsRegistry instance
        collector_name: Name of the collector
        api_key: Optional API key for securing the metrics endpoints
        
    Returns:
        FastAPI router with metrics endpoints
    """
    router = APIRouter(prefix="", tags=["metrics"])
    endpoint = MetricsEndpoint(registry, collector_name, api_key=api_key)
    
    @router.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """Get Prometheus-formatted metrics."""
        return await endpoint.get_metrics(None)
    
    @router.get("/health")
    async def health():
        """Get health status."""
        return await endpoint.get_health(None)
    
    @router.get("/stats")
    async def stats():
        """Get basic statistics."""
        return await endpoint.get_stats(None)
    
    return router


# Utility function to add metrics to an existing FastAPI app
def add_metrics_routes(
    app: FastAPI,
    registry: Optional[MetricsRegistry] = None,
    collector_name: str = "mcp",
    path: str = "/metrics",
    api_key: Optional[str] = None
) -> None:
    """
    Add metrics routes to an existing FastAPI application.
    
    Args:
        app: FastAPI application
        registry: MetricsRegistry instance
        collector_name: Name of the collector
        path: URL path for metrics endpoint
        api_key: Optional API key for securing the metrics endpoints
    """
    # Check if routes already exist to prevent duplicates
    existing_paths = {route.path for route in app.routes if hasattr(route, 'path')}
    
    metrics_path = path
    health_path = f"{path}/health"
    stats_path = f"{path}/stats"
    
    if metrics_path in existing_paths:
        logger.warning(
            f"Metrics routes already exist at {path}. "
            f"Skipping route addition to prevent duplicates."
        )
        return
    
    endpoint = MetricsEndpoint(registry, collector_name, api_key=api_key)
    
    @app.get(f"{path}", response_class=PlainTextResponse)
    async def metrics(request: Request):
        """Get Prometheus-formatted metrics."""
        return await endpoint.get_metrics(request)
    
    @app.get(f"{path}/health")
    async def health(request: Request):
        """Get health status."""
        return await endpoint.get_health(request)
    
    @app.get(f"{path}/stats")
    async def stats(request: Request):
        """Get basic statistics."""
        return await endpoint.get_stats(request)
    
    logger.info(f"Added metrics routes at {path}")
