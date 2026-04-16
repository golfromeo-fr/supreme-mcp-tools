"""
Core Metrics Collector for MCP Launcher

This module provides the core metrics collection infrastructure including:
- MetricsCollector: Main class for collecting and storing metrics
- MetricsRegistry: Singleton registry for managing metrics across the application
- Metric types: Counter, Gauge, Histogram, Summary

Follows Prometheus best practices for metric naming and labeling.
"""

import bisect
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Valid status values for tool calls and requests
VALID_STATUSES = {"success", "error", "timeout"}


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabel:
    """Represents a metric label."""
    name: str
    value: str


@dataclass
class MetricSample:
    """Represents a single metric sample."""
    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.COUNTER


class BaseMetric(ABC):
    """Base class for all metric types."""
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = ""
    ):
        """
        Initialize a metric.
        
        Args:
            name: Metric name (should follow Prometheus naming conventions)
            description: Human-readable description
            label_names: List of label names
            unit: Unit of measurement (e.g., "seconds", "bytes")
        """
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.unit = unit
        self._samples: deque[MetricSample] = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    @abstractmethod
    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a value for this metric."""
        pass
    
    @abstractmethod
    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get the current value of this metric."""
        pass
    
    def get_all_values(self) -> dict[tuple, Any]:
        """
        Get all current values for this metric, keyed by label tuple.
        
        Returns:
            Dictionary mapping label keys to current values or statistics
        """
        raise NotImplementedError("Subclasses must implement get_all_values")
    
    def _create_sample(
        self,
        value: float,
        labels: dict[str, str] | None = None,
        timestamp: float | None = None
    ) -> MetricSample:
        """Create a metric sample."""
        # Filter to only include defined label names
        filtered_labels = {}
        if labels:
            for label_name in self.label_names:
                if label_name in labels:
                    filtered_labels[label_name] = labels[label_name]
        
        return MetricSample(
            name=self.name,
            value=value,
            timestamp=timestamp or time.time(),
            labels=filtered_labels,
            metric_type=self._get_metric_type()
        )
    
    @abstractmethod
    def _get_metric_type(self) -> MetricType:
        """Get the metric type."""
        pass
    
    def get_samples(self) -> list[MetricSample]:
        """Get all samples for this metric."""
        with self._lock:
            return list(self._samples)
    
    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._samples.clear()


class Counter(BaseMetric):
    """
    Counter metric type.
    
    A counter is a cumulative metric that only increases.
    Use for tracking request counts, error counts, etc.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = ""
    ):
        super().__init__(name, description, label_names, unit)
        self._values: dict[tuple, float] = defaultdict(float)
    
    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Increment the counter by the given value.
        
        Args:
            value: Value to add to counter (should be non-negative)
            labels: Label values
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")
        
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] += value
            sample = self._create_sample(self._values[label_key], labels)
            self._samples.append(sample)
    
    def increment(self, labels: dict[str, str] | None = None) -> None:
        """Increment the counter by 1."""
        self.record(1.0, labels)
    
    def get_all_values(self) -> dict[tuple, float]:
        """
        Get all current counter values, keyed by label tuple.
        
        Returns:
            Dictionary mapping label keys to current counter values
        """
        with self._lock:
            return dict(self._values)
    
    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get the current counter value."""
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)
    
    def _get_metric_type(self) -> MetricType:
        return MetricType.COUNTER
    
    def _get_label_key(self, labels: dict[str, str] | None) -> tuple:
        """Get a hashable key for the given labels."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)


class Gauge(BaseMetric):
    """
    Gauge metric type.
    
    A gauge is a metric that can go up and down.
    Use for tracking current values like memory usage, queue size, etc.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = ""
    ):
        super().__init__(name, description, label_names, unit)
        self._values: dict[tuple, float] = defaultdict(float)
    
    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set the gauge to the given value."""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = value
            sample = self._create_sample(value, labels)
            self._samples.append(sample)
    
    def increment(self, labels: dict[str, str] | None = None) -> None:
        """Increment the gauge by 1."""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] += 1
            sample = self._create_sample(self._values[label_key], labels)
            self._samples.append(sample)
    
    def decrement(self, labels: dict[str, str] | None = None) -> None:
        """Decrement the gauge by 1."""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] -= 1
            sample = self._create_sample(self._values[label_key], labels)
            self._samples.append(sample)
    
    def get_all_values(self) -> dict[tuple, float]:
        """
        Get all current gauge values, keyed by label tuple.
        
        Returns:
            Dictionary mapping label keys to current gauge values
        """
        with self._lock:
            return dict(self._values)
    
    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get the current gauge value."""
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)
    
    def _get_metric_type(self) -> MetricType:
        return MetricType.GAUGE
    
    def _get_label_key(self, labels: dict[str, str] | None) -> tuple:
        """Get a hashable key for the given labels."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)


class Histogram(BaseMetric):
    """
    Histogram metric type.
    
    A histogram samples observations and counts them in configurable buckets.
    Use for tracking request durations, response sizes, etc.
    """
    
    # Default bucket boundaries
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = "",
        buckets: tuple[float, ...] | None = None
    ):
        super().__init__(name, description, label_names, unit)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)
        self._bucket_counts: dict[tuple, dict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )
    
    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation in the histogram."""
        if value < 0:
            raise ValueError("Histogram values must be non-negative")
        
        label_key = self._get_label_key(labels)
        with self._lock:
            # Update sum and count
            self._sums[label_key] += value
            self._counts[label_key] += 1
            
            # Use bisect to find the first bucket that value is <=
            # This reduces from O(n) to O(log n + n-m) where m is the bucket index
            bucket_idx = bisect.bisect_left(self.buckets, value)
            
            # Update bucket counts from the found index onwards
            for bucket in self.buckets[bucket_idx:]:
                self._bucket_counts[label_key][bucket] += 1
            # Always count in +inf bucket
            self._bucket_counts[label_key][float('inf')] += 1
            
            # Create sample for the main value
            sample = self._create_sample(value, labels)
            self._samples.append(sample)
    
    def get_value(self, labels: dict[str, str] | None = None) -> dict[str, Any]:
        """Get histogram statistics."""
        label_key = self._get_label_key(labels)
        with self._lock:
            count = self._counts.get(label_key, 0)
            sum_value = self._sums.get(label_key, 0.0)
            buckets = dict(self._bucket_counts.get(label_key, {}))
            
            return {
                "count": count,
                "sum": sum_value,
                "buckets": buckets,
                "mean": sum_value / count if count > 0 else 0.0
            }
    
    def _get_metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM
    
    def _get_label_key(self, labels: dict[str, str] | None) -> tuple:
        """Get a hashable key for the given labels."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)
    
    def get_samples(self) -> list[MetricSample]:
        """Get all samples including bucket samples."""
        samples = []
        with self._lock:
            for label_key, count in self._counts.items():
                labels = self._labels_from_key(label_key)
                
                # Sum sample
                samples.append(MetricSample(
                    name=f"{self.name}_sum",
                    value=self._sums.get(label_key, 0.0),
                    timestamp=time.time(),
                    labels=labels,
                    metric_type=MetricType.HISTOGRAM
                ))
                
                # Count sample
                samples.append(MetricSample(
                    name=f"{self.name}_count",
                    value=float(count),
                    timestamp=time.time(),
                    labels=labels,
                    metric_type=MetricType.HISTOGRAM
                ))
                
                # Bucket samples
                buckets = self._bucket_counts.get(label_key, {})
                for bucket, bucket_count in buckets.items():
                    bucket_labels = labels.copy()
                    if bucket == float('inf'):
                        bucket_labels["le"] = "+Inf"
                    else:
                        bucket_labels["le"] = str(bucket)
                    
                    samples.append(MetricSample(
                        name=f"{self.name}_bucket",
                        value=float(bucket_count),
                        timestamp=time.time(),
                        labels=bucket_labels,
                        metric_type=MetricType.HISTOGRAM
                    ))
        
        return samples
    
    def _labels_from_key(self, label_key: tuple) -> dict[str, str]:
        """Convert label key back to labels dictionary."""
        if not label_key:
            return {}
        return {name: value for name, value in zip(self.label_names, label_key)}
    
    def get_all_values(self) -> dict[tuple, dict[str, Any]]:
        """
        Get all current histogram statistics, keyed by label tuple.
        
        Returns:
            Dictionary mapping label keys to histogram statistics (count, sum, mean, buckets)
        """
        with self._lock:
            result = {}
            for label_key in self._counts.keys():
                count = self._counts.get(label_key, 0)
                sum_value = self._sums.get(label_key, 0.0)
                buckets = dict(self._bucket_counts.get(label_key, {}))
                
                result[label_key] = {
                    "count": count,
                    "sum": sum_value,
                    "buckets": buckets,
                    "mean": sum_value / count if count > 0 else 0.0
                }
            return result


class Summary(BaseMetric):
    """
    Summary metric type.
    
    A summary samples observations and provides total count and sum of values.
    Use for tracking request durations when percentiles are needed.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = "",
        percentiles: list[float] | None = None,
        max_values: int = 1000
    ):
        super().__init__(name, description, label_names, unit)
        self.percentiles = percentiles or [0.5, 0.9, 0.95, 0.99]
        self.max_values = max_values
        self._values: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=max_values))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._counts: dict[tuple, int] = defaultdict(int)
    
    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation in the summary."""
        if value < 0:
            raise ValueError("Summary values must be non-negative")
        
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key].append(value)
            self._sums[label_key] += value
            self._counts[label_key] += 1
            
            sample = self._create_sample(value, labels)
            self._samples.append(sample)
    
    def get_value(self, labels: dict[str, str] | None = None) -> dict[str, Any]:
        """Get summary statistics including percentiles."""
        label_key = self._get_label_key(labels)
        with self._lock:
            values = sorted(self._values.get(label_key, []))
            count = self._counts.get(label_key, 0)
            sum_value = self._sums.get(label_key, 0.0)
            
            percentiles_result = {}
            for p in self.percentiles:
                if values:
                    idx = int(len(values) * p)
                    if idx >= len(values):
                        idx = len(values) - 1
                    percentiles_result[p] = values[idx]
                else:
                    percentiles_result[p] = 0.0
            
            return {
                "count": count,
                "sum": sum_value,
                "mean": sum_value / count if count > 0 else 0.0,
                "percentiles": percentiles_result
            }
    
    def _get_metric_type(self) -> MetricType:
        return MetricType.SUMMARY
    
    def get_all_values(self) -> dict[tuple, dict[str, Any]]:
        """
        Get all current summary statistics, keyed by label tuple.
        
        Returns:
            Dictionary mapping label keys to summary statistics (count, sum, mean, percentiles)
        """
        with self._lock:
            result = {}
            for label_key, values in self._values.items():
                if values:
                    sorted_values = sorted(values)
                    count = self._counts.get(label_key, 0)
                    sum_value = self._sums.get(label_key, 0.0)
                    
                    percentiles_result = {}
                    for p in self.percentiles:
                        if values:
                            idx = int(len(sorted_values) * p)
                            if idx >= len(sorted_values):
                                idx = len(sorted_values) - 1
                            percentiles_result[p] = sorted_values[idx]
                        else:
                            percentiles_result[p] = 0.0
                    
                    result[label_key] = {
                        "count": count,
                        "sum": sum_value,
                        "mean": sum_value / count if count > 0 else 0.0,
                        "percentiles": percentiles_result
                    }
                else:
                    result[label_key] = {
                        "count": 0,
                        "sum": 0.0,
                        "mean": 0.0,
                        "percentiles": {p: 0.0 for p in self.percentiles}
                    }
            return result
    
    def _get_label_key(self, labels: dict[str, str] | None) -> tuple:
        """Get a hashable key for the given labels."""
        if not labels:
            return ()
        return tuple(labels.get(name, "") for name in self.label_names)


class MetricsCollector:
    """
    Main metrics collector class.
    
    Provides methods for creating and recording metrics.
    Metrics are stored in memory and can be exported via various exporters.
    """
    
    def __init__(self, name: str = "mcp"):
        """
        Initialize the metrics collector.
        
        Args:
            name: Prefix for all metric names
        """
        self.name = name
        self._metrics: dict[str, BaseMetric] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()
        
        # Create default metrics
        self._create_default_metrics()
    
    def _create_default_metrics(self) -> None:
        """Create default metrics for MCP."""
        # Request metrics
        self._metrics["requests_total"] = Counter(
            name=f"{self.name}_requests_total",
            description="Total number of requests",
            label_names=["tool", "method", "status"]
        )
        
        self._metrics["request_duration_seconds"] = Histogram(
            name=f"{self.name}_request_duration_seconds",
            description="Request duration in seconds",
            label_names=["tool", "method"],
            unit="seconds"
        )
        
        # Tool metrics
        self._metrics["tool_calls_total"] = Counter(
            name=f"{self.name}_tool_calls_total",
            description="Total number of tool calls",
            label_names=["tool", "tool_name", "status"]
        )
        
        self._metrics["tool_call_duration_seconds"] = Histogram(
            name=f"{self.name}_tool_call_duration_seconds",
            description="Tool call duration in seconds",
            label_names=["tool", "tool_name"],
            unit="seconds"
        )
        
        # Active requests
        self._metrics["active_requests"] = Gauge(
            name=f"{self.name}_active_requests",
            description="Number of currently active requests",
            label_names=["tool"]
        )
        
        # Error metrics
        self._metrics["errors_total"] = Counter(
            name=f"{self.name}_errors_total",
            description="Total number of errors",
            label_names=["tool", "error_type"]
        )
        
        # Server metrics
        self._metrics["server_up"] = Gauge(
            name=f"{self.name}_server_up",
            description="Whether the server is up (1) or down (0)",
            label_names=["tool"]
        )
        
        self._metrics["server_start_time_seconds"] = Gauge(
            name=f"{self.name}_server_start_time_seconds",
            description="Server start time in seconds since epoch",
            label_names=["tool"]
        )
    
    def counter(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = ""
    ) -> Counter:
        """
        Create or get a counter metric.
        
        Args:
            name: Metric name
            description: Human-readable description
            label_names: List of label names
            unit: Unit of measurement
            
        Returns:
            Counter metric instance
        """
        full_name = f"{self.name}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(
                    name=full_name,
                    description=description,
                    label_names=label_names,
                    unit=unit
                )
            return self._metrics[full_name]
    
    def gauge(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = ""
    ) -> Gauge:
        """
        Create or get a gauge metric.
        
        Args:
            name: Metric name
            description: Human-readable description
            label_names: List of label names
            unit: Unit of measurement
            
        Returns:
            Gauge metric instance
        """
        full_name = f"{self.name}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(
                    name=full_name,
                    description=description,
                    label_names=label_names,
                    unit=unit
                )
            return self._metrics[full_name]
    
    def histogram(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = "",
        buckets: tuple[float, ...] | None = None
    ) -> Histogram:
        """
        Create or get a histogram metric.
        
        Args:
            name: Metric name
            description: Human-readable description
            label_names: List of label names
            unit: Unit of measurement
            buckets: Bucket boundaries
            
        Returns:
            Histogram metric instance
        """
        full_name = f"{self.name}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(
                    name=full_name,
                    description=description,
                    label_names=label_names,
                    unit=unit,
                    buckets=buckets
                )
            return self._metrics[full_name]
    
    def summary(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        unit: str = "",
        percentiles: list[float] | None = None
    ) -> Summary:
        """
        Create or get a summary metric.
        
        Args:
            name: Metric name
            description: Human-readable description
            label_names: List of label names
            unit: Unit of measurement
            percentiles: Percentiles to calculate
            
        Returns:
            Summary metric instance
        """
        full_name = f"{self.name}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(
                    name=full_name,
                    description=description,
                    label_names=label_names,
                    unit=unit,
                    percentiles=percentiles
                )
            return self._metrics[full_name]
    
    def record_request_start(
        self,
        tool: str,
        method: str = "unknown"
    ) -> None:
        """Record the start of a request."""
        with self._lock:
            self._metrics["active_requests"].record(
                1,
                {"tool": tool}
            )
    
    def record_request_end(
        self,
        tool: str,
        method: str = "unknown",
        status: str = "success",
        duration: float = 0.0
    ) -> None:
        """Record the end of a request."""
        # Validate inputs
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        
        with self._lock:
            # Decrement active requests
            self._metrics["active_requests"].decrement({"tool": tool})
            
            # Record total requests
            self._metrics["requests_total"].record(
                1,
                {"tool": tool, "method": method, "status": status}
            )
            
            # Record duration
            if duration > 0:
                self._metrics["request_duration_seconds"].record(
                    duration,
                    {"tool": tool, "method": method}
                )
    
    def record_tool_call(
        self,
        tool: str,
        tool_name: str,
        status: str = "success",
        duration: float = 0.0
    ) -> None:
        """Record a tool call."""
        # Validate inputs
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
        
        with self._lock:
            # Record total tool calls
            self._metrics["tool_calls_total"].record(
                1,
                {"tool": tool, "tool_name": tool_name, "status": status}
            )
            
            # Record duration
            if duration > 0:
                self._metrics["tool_call_duration_seconds"].record(
                    duration,
                    {"tool": tool, "tool_name": tool_name}
                )
    
    def record_error(
        self,
        tool: str,
        error_type: str = "unknown"
    ) -> None:
        """Record an error."""
        # Validate inputs
        if not error_type or not error_type.strip():
            raise ValueError("error_type cannot be empty")
        
        with self._lock:
            self._metrics["errors_total"].record(
                1,
                {"tool": tool, "error_type": error_type}
            )
    
    def set_server_status(self, tool: str, up: bool) -> None:
        """Set server status."""
        with self._lock:
            self._metrics["server_up"].record(
                1 if up else 0,
                {"tool": tool}
            )
    
    def set_server_start_time(self, tool: str, timestamp: float | None = None) -> None:
        """Set server start time."""
        with self._lock:
            self._metrics["server_start_time_seconds"].record(
                timestamp or time.time(),
                {"tool": tool}
            )
    
    def get_all_metrics(self) -> dict[str, BaseMetric]:
        """Get all metrics."""
        with self._lock:
            return self._metrics.copy()
    
    def get_metrics_for_tool(self, tool: str) -> dict[str, Any]:
        """Get metrics for a specific tool."""
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, (Counter, Gauge, Histogram, Summary)):
                    # Get values for this specific tool by filtering labels
                    tool_values = {}
                    all_values = metric.get_all_values()
                    
                    for label_key, value in all_values.items():
                        # Convert label key back to labels dictionary
                        labels = {}
                        if label_key and hasattr(metric, 'label_names'):
                            for i, label_name in enumerate(metric.label_names):
                                if i < len(label_key):
                                    labels[label_name] = label_key[i]
                        
                        # Check if this label set belongs to the specified tool
                        if labels.get("tool") == tool:
                            # For Counter and Gauge, value is the current value
                            # For Histogram and Summary, value is already a dict of stats
                            if isinstance(metric, (Counter, Gauge)):
                                tool_values[name] = value
                            else:
                                # For Histogram/Summary, we need to format appropriately
                                if isinstance(metric, Histogram):
                                    tool_values[f"{name}_count"] = value.get("count", 0)
                                    tool_values[f"{name}_sum"] = value.get("sum", 0.0)
                                    tool_values[f"{name}_mean"] = value.get("mean", 0.0)
                                    # Add bucket values
                                    for bucket, count in value.get("buckets", {}).items():
                                        tool_values[f"{name}_bucket{{le=\"{bucket}\"}}"] = count
                                elif isinstance(metric, Summary):
                                    tool_values[f"{name}_count"] = value.get("count", 0)
                                    tool_values[f"{name}_sum"] = value.get("sum", 0.0)
                                    tool_values[f"{name}_mean"] = value.get("mean", 0.0)
                                    # Add percentile values
                                    for percentile, val in value.get("percentiles", {}).items():
                                        tool_values[f"{name}{{quantile={percentile}}}"] = val
                    
                    # If we found values for this tool, add them to result
                    if tool_values:
                        result[name] = tool_values
                    else:
                        # No specific tool values, return general info
                        result[name] = {
                            "description": metric.description,
                            "type": metric._get_metric_type().value
                        }
        return result
    
    def get_uptime(self) -> float:
        """Get collector uptime in seconds."""
        return time.time() - self._start_time
    
    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            for metric in self._metrics.values():
                metric.clear()
            self._start_time = time.time()


class MetricsRegistry:
    """
    Singleton registry for metrics collectors.
    
    Provides a global instance for collecting metrics across the application.
    """
    
    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.RLock()
    
    def __new__(cls) -> "MetricsRegistry":
        """Create or get the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the registry. Called exactly once from __new__."""
        self._collectors: dict[str, MetricsCollector] = {}
        self._default_collector: MetricsCollector | None = None
        self._enabled = True
    
    def __init__(self):
        """Initialize the registry."""
        # Initialization is handled in _initialize() which is called from __new__
        pass
    
    @classmethod
    def get_instance(cls) -> "MetricsRegistry":
        """Get the singleton instance."""
        return cls()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance._collectors.clear()
                cls._instance._default_collector = None
            cls._instance = None
    
    def enable(self) -> None:
        """Enable metrics collection."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable metrics collection."""
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled
    
    def get_or_create_collector(
        self,
        name: str = "mcp",
        create_if_not_exists: bool = True
    ) -> MetricsCollector | None:
        """
        Get or create a metrics collector.
        
        Args:
            name: Collector name
            create_if_not_exists: Create collector if it doesn't exist
            
        Returns:
            MetricsCollector instance or None if disabled
        """
        if not self._enabled:
            return None
        
        with self._lock:
            if name not in self._collectors:
                if create_if_not_exists:
                    self._collectors[name] = MetricsCollector(name)
                else:
                    return None
            
            return self._collectors[name]
    
    def get_default_collector(self) -> MetricsCollector | None:
        """Get or create the default collector."""
        if not self._enabled:
            return None
        
        with self._lock:
            if self._default_collector is None:
                self._default_collector = MetricsCollector("mcp")
            return self._default_collector
    
    def record_request_start(
        self,
        tool: str,
        method: str = "unknown",
        collector_name: str = "mcp"
    ) -> None:
        """Record request start across all collectors."""
        if not self._enabled:
            return
        
        collector = self.get_or_create_collector(collector_name)
        if collector:
            collector.record_request_start(tool, method)
    
    def record_request_end(
        self,
        tool: str,
        method: str = "unknown",
        status: str = "success",
        duration: float = 0.0,
        collector_name: str = "mcp"
    ) -> None:
        """Record request end across all collectors."""
        if not self._enabled:
            return
        
        collector = self.get_or_create_collector(collector_name)
        if collector:
            collector.record_request_end(tool, method, status, duration)
    
    def record_tool_call(
        self,
        tool: str,
        tool_name: str,
        status: str = "success",
        duration: float = 0.0,
        collector_name: str = "mcp"
    ) -> None:
        """Record tool call across all collectors."""
        if not self._enabled:
            return
        
        collector = self.get_or_create_collector(collector_name)
        if collector:
            collector.record_tool_call(tool, tool_name, status, duration)
    
    def record_error(
        self,
        tool: str,
        error_type: str = "unknown",
        collector_name: str = "mcp"
    ) -> None:
        """Record error across all collectors."""
        if not self._enabled:
            return
        
        collector = self.get_or_create_collector(collector_name)
        if collector:
            collector.record_error(tool, error_type)
    
    def get_all_collectors(self) -> dict[str, MetricsCollector]:
        """Get all collectors."""
        with self._lock:
            return self._collectors.copy()


# Context manager for timing operations
@contextmanager
def track_duration(
    collector: MetricsCollector,
    tool: str,
    tool_name: str,
    status_var: dict[str, str] | None = None
):
    """
    Context manager for tracking operation duration.
    
    Usage:
        with track_duration(collector, "webmcp", "search") as status:
            # do work
            status["status"] = "success"
    
    Args:
        collector: MetricsCollector instance
        tool: Tool name
        tool_name: Operation name
        status_var: Dictionary to store status (modified in place)
    
    Yields:
        Dictionary that can be updated with status
    """
    status = {"status": "success"}
    start_time = time.time()
    try:
        yield status
    except Exception as e:
        status["status"] = "error"
        collector.record_error(tool, type(e).__name__)
        raise
    finally:
        duration = time.time() - start_time
        collector.record_tool_call(
            tool=tool,
            tool_name=tool_name,
            status=status.get("status", "success"),
            duration=duration
        )


# Convenience functions
def get_collector(name: str = "mcp") -> MetricsCollector | None:
    """Get a metrics collector by name."""
    return MetricsRegistry.get_instance().get_or_create_collector(name)


def get_default_collector() -> MetricsCollector | None:
    """Get the default metrics collector."""
    return MetricsRegistry.get_instance().get_default_collector()


def record_request_start(tool: str, method: str = "unknown") -> None:
    """Record request start."""
    MetricsRegistry.get_instance().record_request_start(tool, method)


def record_request_end(
    tool: str,
    method: str = "unknown",
    status: str = "success",
    duration: float = 0.0
) -> None:
    """Record request end."""
    MetricsRegistry.get_instance().record_request_end(tool, method, status, duration)


def record_tool_call(
    tool: str,
    tool_name: str,
    status: str = "success",
    duration: float = 0.0
) -> None:
    """Record tool call."""
    MetricsRegistry.get_instance().record_tool_call(tool, tool_name, status, duration)


def record_error(tool: str, error_type: str = "unknown") -> None:
    """Record error."""
    MetricsRegistry.get_instance().record_error(tool, error_type)
