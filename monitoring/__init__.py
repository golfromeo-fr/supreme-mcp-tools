"""
Monitoring Package for MCP Launcher

This package provides comprehensive monitoring and observability for the MCP Launcher system.
It includes metrics collection, Prometheus export, and FastAPI middleware integration.

Phase 1 Features:
- Core metrics collection framework
- Request/response metrics
- Tool execution metrics
- Basic metrics API
- Prometheus exporter

Phase 2 (planned):
- System metrics (CPU, memory, I/O)
- Session metrics
- Custom metrics support
- Elasticsearch exporter

Phase 3 (planned):
- Alerting engine
- Dashboard templates
- Grafana integration

Phase 4 (planned):
- Plugin system
- Middleware framework
- Dynamic configuration
"""

from .collector import MetricsCollector, MetricsRegistry
from .exporters import PrometheusExporter
from .middleware import MetricsMiddleware
from .config import MonitoringConfig, get_monitoring_config

__version__ = "1.0.0"

__all__ = [
    "MetricsCollector",
    "MetricsRegistry",
    "PrometheusExporter",
    "MetricsMiddleware",
    "MonitoringConfig",
    "get_monitoring_config",
]
