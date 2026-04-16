"""
MCP Launcher Package

A unified launcher system for running multiple MCP tools in a single process.
Supports the Flexible Extensibility Framework V3.
"""

from .launcher_config import Config
from .errors import (
    ConfigError,
    DiscoveryError,
    LauncherError,
    PortConflictError,
    ServerRuntimeError,
    ServerStartupError,
    ValidationError,
)
from .port_manager import PortManager
from .server_manager import ServerManager, ServerInstance, run_servers_concurrently
from .tool_discovery import ToolDiscovery, ToolMetadata

# FEF V3 - Core
from .service_registry import ServiceRegistry, ServiceInfo
from .distributed_registry import (
    DistributedExtensionRegistry,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    CacheManager,
    EventAggregator,
    HTTPClient,
    RequestCoalescer,
)
from .distributed_registry import ConfigPersistence as DistributedConfigPersistence
from .management_server import ManagementServer
from .tool_extensions import Extension, ExtensionRegistry, ExtensionType, ExtensionHTTPServer

# FEF V3 - Security
from .security import (
    APIKeyAuth,
    verify_api_key,
    require_permission,
    RateLimiter,
    AuditLogger,
)

# FEF V3 - Resilience
from .resilience import (
    retry_with_backoff,
    RetryConfig,
    RetryExhaustedError,
    DeadLetterQueue,
)

# FEF V3 - Configuration
from .config.persistence import ConfigPersistence as FileConfigPersistence
from .config.sqlite_persistence import SQLitePersistence
from .config.manager import ConfigManager

# FEF V3 - Events
from .events import EventStore, Event

# FEF V3 - Plugins
from .plugins import PluginLoader

__version__ = "3.0.0"
__all__ = [
    # Core
    "Config",
    "ConfigError",
    "DiscoveryError",
    "LauncherError",
    "PortConflictError",
    "PortManager",
    "ServerInstance",
    "ServerManager",
    "ServerRuntimeError",
    "ServerStartupError",
    "ToolDiscovery",
    "ToolMetadata",
    "ValidationError",
    "run_servers_concurrently",
    # FEF V3 - Core
    "ServiceRegistry",
    "ServiceInfo",
    "DistributedExtensionRegistry",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerState",
    "CacheManager",
    "EventAggregator",
    "HTTPClient",
    "RequestCoalescer",
    "ManagementServer",
    "Extension",
    "ExtensionRegistry",
    "ExtensionType",
    "ExtensionHTTPServer",
    # FEF V3 - Security
    "APIKeyAuth",
    "verify_api_key",
    "require_permission",
    "RateLimiter",
    "AuditLogger",
    # FEF V3 - Resilience
    "retry_with_backoff",
    "RetryConfig",
    "RetryExhaustedError",
    "DeadLetterQueue",
    # FEF V3 - Configuration
    "ConfigPersistence",
    "SQLitePersistence",
    "ConfigManager",
    # FEF V3 - Events
    "EventStore",
    "Event",
    # FEF V3 - Plugins
    "PluginLoader", "DistributedConfigPersistence", "FileConfigPersistence",
]
