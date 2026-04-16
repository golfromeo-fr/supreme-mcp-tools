"""
Monitoring Configuration for MCP Launcher

This module provides configuration management for the monitoring system.
It handles enabling/disabling monitoring, configuring exporters,
and setting collection intervals.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends."""
    PROMETHEUS = "prometheus"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"
    CONSOLE = "console"


class MetricType(Enum):
    """Types of metrics to collect."""
    REQUEST = "request"
    TOOL = "tool"
    SERVER = "server"
    SESSION = "session"
    SYSTEM = "system"


@dataclass
class ExporterConfig:
    """Configuration for a metrics exporter."""
    type: StorageBackend
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate exporter configuration."""
        if self.type == StorageBackend.PROMETHEUS:
            # Set defaults for Prometheus (port must come from ports.json or be set explicitly)
            self.config.setdefault("endpoint", "/metrics")
            # Do NOT set default port - it must come from ports.json
        elif self.type == StorageBackend.ELASTICSEARCH:
            # Set defaults for Elasticsearch
            self.config.setdefault("url", "http://localhost:9200")
            self.config.setdefault("index", "mcp-metrics")
        elif self.type == StorageBackend.INFLUXDB:
            # Set defaults for InfluxDB
            self.config.setdefault("url", "http://localhost:8086")
            self.config.setdefault("database", "mcp")
            self.config.setdefault("organization", "mcp")
            self.config.setdefault("bucket", "metrics")


@dataclass
class CollectorConfig:
    """Configuration for a metrics collector."""
    type: MetricType
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate collector configuration."""
        if self.type == MetricType.REQUEST:
            self.config.setdefault("sample_rate", 1.0)
            self.config.setdefault("track_arguments", False)
        elif self.type == MetricType.TOOL:
            self.config.setdefault("track_arguments", False)
            self.config.setdefault("track_return_values", False)
        elif self.type == MetricType.SERVER:
            self.config.setdefault("interval", 30000)  # 30 seconds
        elif self.type == MetricType.SESSION:
            self.config.setdefault("track_activity", True)
            self.config.setdefault("timeout", 300)  # 5 minutes
        elif self.type == MetricType.SYSTEM:
            self.config.setdefault("interval", 10000)  # 10 seconds
            self.config.setdefault("collect_cpu", True)
            self.config.setdefault("collect_memory", True)
            self.config.setdefault("collect_disk", True)
            self.config.setdefault("collect_network", True)


@dataclass
class AlertingConfig:
    """Configuration for alerting system."""
    enabled: bool = False
    evaluation_interval: int = 60000  # 60 seconds
    rules: list[dict[str, Any]] = field(default_factory=list)
    notifiers: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default alerting rules if none provided."""
        if not self.rules and self.enabled:
            self.rules = [
                {
                    "name": "high_error_rate",
                    "expression": "rate(mcp_errors_total[5m]) > 0.05",
                    "duration": "5m",
                    "severity": "warning",
                    "annotations": {
                        "summary": "High error rate detected",
                        "description": "Error rate is above 5% for the last 5 minutes"
                    }
                }
            ]


@dataclass
class MonitoringConfig:
    """Main monitoring configuration."""
    enabled: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    collectors: dict[str, Any] = field(default_factory=dict)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    
    def __post_init__(self):
        """Set default configuration values."""
        if not self.metrics:
            self.metrics = {
                "collection_interval": 1000,  # 1 second
                "buffer_size": 10000,
                "flush_interval": 5000,  # 5 seconds
                "retention_days": 90,
                "storage_backend": StorageBackend.PROMETHEUS.value,
                "exporters": [
                    asdict(ExporterConfig(
                        type=StorageBackend.PROMETHEUS,
                        enabled=True
                    ))
                ]
            }
        
        if not self.collectors:
            self.collectors = {
                MetricType.REQUEST.value: asdict(CollectorConfig(
                    type=MetricType.REQUEST,
                    enabled=True
                )),
                MetricType.TOOL.value: asdict(CollectorConfig(
                    type=MetricType.TOOL,
                    enabled=True
                )),
                MetricType.SERVER.value: asdict(CollectorConfig(
                    type=MetricType.SERVER,
                    enabled=True
                )),
                MetricType.SESSION.value: asdict(CollectorConfig(
                    type=MetricType.SESSION,
                    enabled=True
                )),
                MetricType.SYSTEM.value: asdict(CollectorConfig(
                    type=MetricType.SYSTEM,
                    enabled=False  # Disabled by default in Phase 1
                ))
            }


def load_monitoring_config(config_path: str | Path | None = None) -> MonitoringConfig:
    """
    Load monitoring configuration from file.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        MonitoringConfig instance
    """
    # Start with default configuration
    config = MonitoringConfig()
    
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with Path(config_path).open('r') as f:
                    file_config = json.load(f)
                
                # Update config with file values
                config = _update_config_from_dict(config, file_config)
                logger.info(f"Loaded monitoring configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load monitoring config from {config_path}: {e}")
                logger.info("Using default monitoring configuration")
        else:
            logger.warning(f"Monitoring config file not found: {config_path}")
            logger.info("Using default monitoring configuration")
    else:
        # Try to load from default locations
        default_paths = [
            Path("config/monitoring.json"),
            Path("monitoring_config.json"),
            Path("./monitoring/config.json")
        ]
        
        for path in default_paths:
            logger.debug(f"Checking for monitoring config at: {path}")
            if path.exists():
                try:
                    with Path(path).open('r') as f:
                        file_config = json.load(f)
                    
                    config = _update_config_from_dict(config, file_config)
                    logger.info(f"Loaded monitoring configuration from {path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load monitoring config from {path}: {e}")
                    continue
            else:
                logger.debug(f"Config file not found: {path}")
        else:
            logger.warning(
                "No monitoring config file found in default locations. "
                "Using default configuration. Create a config file to customize monitoring settings."
            )
    
    return config


def _update_config_from_dict(config: MonitoringConfig, updates: dict[str, Any]) -> MonitoringConfig:
    """
    Update configuration from dictionary.
    
    Args:
        config: Current configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    # Update top-level fields
    for key, value in updates.items():
        if hasattr(config, key):
            if key == "alerting" and isinstance(value, dict):
                # Merge alerting config
                alerting_config = asdict(config.alerting)
                alerting_config.update(value)
                config.alerting = AlertingConfig(**alerting_config)
            elif key in ["metrics", "collectors"] and isinstance(value, dict):
                # Merge nested dicts
                current_value = getattr(config, key)
                current_value.update(value)
                setattr(config, key, current_value)
            else:
                setattr(config, key, value)
    
    return config


def save_monitoring_config(config: MonitoringConfig, config_path: str | Path) -> None:
    """
    Save monitoring configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration to
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for JSON serialization
    config_dict = asdict(config)
    
    # Handle enum serialization recursively
    def convert_enums(obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_enums(i) for i in obj]
        return obj
    
    config_dict = convert_enums(config_dict)
    
    with Path(config_path).open('w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Saved monitoring configuration to {config_path}")


def get_monitoring_config(launcher_config: Any | None = None) -> MonitoringConfig:
    """
    Get monitoring configuration from launcher config.
    
    Args:
        launcher_config: Optional launcher config object. If not provided,
                        defaults will be used.
                        
    Returns:
        MonitoringConfig instance
    """
    try:
        if launcher_config is None:
            # Try to get from a global registry if available
            # Import here to avoid circular import at module level
            try:
                from launcher.config import Config
                launcher_config = Config()
            except ImportError:
                logger.info("No launcher config available, using defaults")
                return MonitoringConfig()
        
        monitoring_dict = launcher_config.get("monitoring", {})
        
        if monitoring_dict:
            config = MonitoringConfig()
            return _update_config_from_dict(config, monitoring_dict)
        else:
            logger.info("No monitoring configuration found in launcher config, using defaults")
            return MonitoringConfig()
    except Exception as e:
        logger.warning(f"Failed to get monitoring config from launcher: {e}")
        return MonitoringConfig()


def is_monitoring_enabled() -> bool:
    """
    Check if monitoring is enabled.
    
    Returns:
        True if monitoring is enabled
    """
    config = get_monitoring_config()
    return config.enabled


def create_default_monitoring_config() -> MonitoringConfig:
    """
    Create a default monitoring configuration.
    
    Returns:
        Default MonitoringConfig instance
    """
    return MonitoringConfig()


# Example configuration generator
def generate_example_config() -> dict[str, Any]:
    """
    Generate an example monitoring configuration.
    
    Returns:
        Dictionary representing example configuration
    """
    config = MonitoringConfig()
    
    # Enable all collectors for the example
    for collector_type in config.collectors:
        config.collectors[collector_type]["enabled"] = True
    
    # Enable system metrics for the example
    config.collectors[MetricType.SYSTEM.value]["enabled"] = True
    
    # Add multiple exporters
    # NOTE: Port should NOT be set here - it comes from config/ports.json
    # Example: "port": 8300  # from ports.json reserved.metrics_server
    config.metrics["exporters"] = [
        asdict(ExporterConfig(
            type=StorageBackend.PROMETHEUS,
            enabled=True,
            config={
                "endpoint": "/metrics"
                # port comes from ports.json: reserved.metrics_server
            }
        )),
        asdict(ExporterConfig(
            type=StorageBackend.ELASTICSEARCH,
            enabled=False,  # Disabled by default
            config={
                "url": "http://localhost:9200",
                "index": "mcp-metrics"
            }
        ))
    ]
    
    # Enable alerting with example rules
    config.alerting.enabled = True
    config.alerting.rules = [
        {
            "name": "high_error_rate",
            "expression": "rate(mcp_errors_total[5m]) > 0.05",
            "duration": "5m",
            "severity": "warning",
            "annotations": {
                "summary": "High error rate detected",
                "description": "Error rate is above 5% for the last 5 minutes"
            }
        },
        {
            "name": "high_latency",
            "expression": "histogram_quantile(0.95, mcp_request_duration_seconds) > 1.0",
            "duration": "10m",
            "severity": "warning",
            "annotations": {
                "summary": "High latency detected",
                "description": "P95 latency is above 1 second for the last 10 minutes"
            }
        },
        {
            "name": "server_down",
            "expression": "mcp_server_up == 0",
            "duration": "1m",
            "severity": "critical",
            "annotations": {
                "summary": "Server is down",
                "description": "Server {{ $labels.tool }} has been down for more than 1 minute"
            }
        }
    ]
    
    config.alerting.notifiers = [
        {
            "type": "email",
            "enabled": False,
            "config": {
                "to": ["admin@example.com"],
                "from": "alerts@example.com",
                "smtp_server": "localhost",
                "smtp_port": 587
            }
        },
        {
            "type": "slack",
            "enabled": False,
            "config": {
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                "channel": "#alerts",
                "username": "MCP Monitor"
            }
        }
    ]
    
    return asdict(config)
