"""
FEF V3 Monitoring Module

Provides Prometheus metrics integration for the Flexible Extensibility Framework V3.
"""

from .fef_metrics import FEFMetrics, get_fef_metrics

__all__ = [
    "FEFMetrics",
    "get_fef_metrics",
]
