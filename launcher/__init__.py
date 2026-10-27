"""
MCP Launcher Package

A unified launcher system for running multiple MCP tools in a single process.
Supports the Flexible Extensibility Framework V3.

(The FEF V3 persistence backends — SQLitePersistence, FileConfigPersistence,
ConfigManager, EventStore, DeadLetterQueue, AuditLogger, HA/distributed
config — were deleted 2026-09-05; they never ran in production. Design pass
D1: plans/launcher-persistence-design-2026-09-05.md.)
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
)
from .distributed_registry import ConfigPersistence as DistributedConfigPersistence
from .management_server import ManagementServer
from .tool_extensions import Extension, ExtensionRegistry, ExtensionType, ExtensionHTTPServer

# FEF V3 - Security
# (APIKeyAuth / verify_api_key / require_permission were deleted 2026-10-25,
# review 2026-10-24 L3: re-export-only dead code — every served surface uses
# the DualHeaderVerifier in tools/shared/server_factory.py; the module's
# dict-membership verify was also timing-unsafe. RateLimiter stays — tested.)
from .security import RateLimiter

# FEF V3 - Resilience
from .resilience import (
    retry_with_backoff,
    RetryConfig,
    RetryExhaustedError,
)

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
    "ManagementServer",
    "Extension",
    "ExtensionRegistry",
    "ExtensionType",
    "ExtensionHTTPServer",
    # FEF V3 - Security
    "RateLimiter",
    # FEF V3 - Resilience
    "retry_with_backoff",
    "RetryConfig",
    "RetryExhaustedError",
    # FEF V3 - Plugins
    "PluginLoader", "DistributedConfigPersistence",
]
