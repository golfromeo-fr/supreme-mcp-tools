"""
Configuration handling for the MCP launcher system.

This module provides functionality to load, validate, and manage
configuration from JSON files and environment variables.

ARCHITECTURAL NOTE:
    Config now delegates to focused dataclasses (LoggingConfig, ServerConfig,
    ErrorHandlingConfig, PortConfig, ToolDirectoryConfig) while maintaining
    full backward compatibility via get_* methods.
    Call sites can use the new typed configs directly when isolation is needed.
"""

import json
import os
from pathlib import Path
from typing import Any
import logging

from .errors import ConfigError
from .config_types import (
    LoggingConfig,
    ServerConfig,
    ErrorHandlingConfig,
    PortConfig,
    ToolDirectoryConfig,
)


logger = logging.getLogger(__name__)

# Default config directory
_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_ports_config(config_dir: Path | None = None) -> dict[str, Any]:
    """
    Load ports configuration from ports.json.
    
    This is the ONLY source of truth for port configuration.
    Fails with clear error if ports.json is missing.
    
    Args:
        config_dir: Optional directory containing ports.json
        
    Returns:
        Dictionary with ranges, reserved, and assignments
        
    Raises:
        ConfigError: If ports.json is missing or invalid
    """
    if config_dir is None:
        config_dir = _DEFAULT_CONFIG_DIR
    
    ports_path = config_dir / "ports.json"
    if not ports_path.exists():
        raise ConfigError(
            f"ports.json not found at {ports_path}. "
            "This is the ONLY source of truth for port configuration. "
            "Please create it from ports.example.json."
        )
    
    try:
        with Path(ports_path).open('r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid ports.json: {e}")
    
    # Validate required sections exist
    if "ranges" not in config:
        raise ConfigError("ports.json missing 'ranges' section")
    if "reserved" not in config:
        raise ConfigError("ports.json missing 'reserved' section")
    
    return config


class Config:
    """
    Configuration manager for the MCP launcher.

    ARCHITECTURAL DEBT: This class is a god object with 37+ edges and ~32 getter methods.
    It knows about tool directories, port allocation, server settings, logging, and error handling.
    Subsystems that only need logging config must import the entire Config to get it.

    Recommended refactor (when ready for a dedicated sprint):
    1. Create focused config dataclasses: LoggingConfig, PortConfig, ServerConfig, ErrorHandlingConfig
    2. Keep Config as a thin facade that delegates to the focused configs
    3. Update call sites one subsystem at a time
    4. Eventually Config becomes a simple aggregator with no get_* methods of its own

    This is NOT a bug - Config works correctly. It's a maintainability concern for future growth.
    """
    
    # Default configuration values
    # Use dynamic path resolution to avoid hardcoded absolute paths
    DEFAULT_CONFIG = {
        "toolDirectories": [
            # Relative paths from supreme-mcp-tools root
            "tools/webmcp",
            "tools/oraclemcp",
            "tools/simplemcp",
            "tools/convertermcp",
            "tools/ragmcp",
            "tools/memorymcp"
        ],
        "portAllocation": {
            "mode": "manual",
            # Legacy format for backward compatibility
            "basePort": 8000,
            "portRange": [8000, 8099],
            "ports": {
                "oraclemcp": 8000,
                "webmcp": 8001,
                "simplemcp": 8002,
                "convertermcp": 8003,
                "ragmcp": 8004
            },
            "managementPorts": {
                "oraclemcp": 8100,
                "webmcp": 8101,
                "simplemcp": 8102,
                "convertermcp": 8103,
                "ragmcp": 8104
            },
            # New format (preferred) - populated lazily from port_config.json
            "ranges": None,  # Populated by _ensure_port_defaults()
            "reservedPorts": None,  # Populated by _ensure_port_defaults()
            "manualPorts": {
                "mcp": {
                    "oraclemcp": 8000,
                    "webmcp": 8001,
                    "simplemcp": 8002,
                    "convertermcp": 8003,
                    "ragmcp": 8004
                }
            }
        },
        "server": {
            "host": "0.0.0.0",
            "logLevel": "info"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None
        },
        "errorHandling": {
            "continueOnError": True,
            "failFast": False
        }
    }
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config: dict[str, Any] = {}
        self.config_path = config_path

        # Focused typed configs (lazy-initialized on first access)
        self._logging_config: LoggingConfig | None = None
        self._server_config: ServerConfig | None = None
        self._error_handling_config: ErrorHandlingConfig | None = None
        self._port_config: PortConfig | None = None
        self._tool_directory_config: ToolDirectoryConfig | None = None

        self._load_config()

    def _init_focused_configs(self) -> None:
        """Initialize frozen focused configs from loaded dict config."""
        if self._logging_config is None:
            self._logging_config = LoggingConfig.from_dict(self.config.get("logging", {}))
        if self._server_config is None:
            self._server_config = ServerConfig.from_dict(self.config.get("server", {}))
        if self._error_handling_config is None:
            self._error_handling_config = ErrorHandlingConfig.from_dict(self.config.get("errorHandling", {}))
        if self._port_config is None:
            port_data = self.config.get("portAllocation", {})
            self._port_config = PortConfig.from_dict(port_data)
        if self._tool_directory_config is None:
            self._tool_directory_config = ToolDirectoryConfig.from_list(
                self.config.get("toolDirectories", [])
            )

    @property
    def logging_config(self) -> LoggingConfig:
        """Get typed logging configuration."""
        self._init_focused_configs()
        return self._logging_config

    @property
    def server_config(self) -> ServerConfig:
        """Get typed server configuration."""
        self._init_focused_configs()
        return self._server_config

    @property
    def error_handling_config(self) -> ErrorHandlingConfig:
        """Get typed error handling configuration."""
        self._init_focused_configs()
        return self._error_handling_config

    @property
    def port_config(self) -> PortConfig:
        """Get typed port configuration."""
        self._init_focused_configs()
        return self._port_config

    @property
    def tool_directory_config(self) -> ToolDirectoryConfig:
        """Get typed tool directory configuration."""
        self._init_focused_configs()
        return self._tool_directory_config
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Start with defaults
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Ensure port defaults are loaded from port_config.json
        self._ensure_port_defaults()
        
        # Load from file if provided
        if self.config_path:
            self._load_from_file(self.config_path)
        else:
            # Resolve relative paths in DEFAULT_CONFIG when no config file is provided
            # Use the directory of this file as base for resolution
            self._resolve_tool_directories()
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded from {self.config_path or 'defaults'}")
    
    def _ensure_port_defaults(self) -> None:
        """Ensure port allocation defaults are loaded from ports.json."""
        port_alloc = self.config.get("portAllocation", {})
        
        # Load from ports.json - this is the ONLY source of truth
        ports_config = load_ports_config()
        
        # Set ranges from ports.json
        if port_alloc.get("ranges") is None:
            port_alloc["ranges"] = {k: tuple(v) for k, v in ports_config.get("ranges", {}).items()}
        
        # Set reserved ports from ports.json
        if port_alloc.get("reservedPorts") is None:
            port_alloc["reservedPorts"] = ports_config.get("reserved", {})
        
        # Set assignments from ports.json (ports.json is authoritative, override defaults)
        assignments = ports_config.get("assignments", {})
        if assignments:
            port_alloc["manualPorts"] = assignments
        
        self.config["portAllocation"] = port_alloc
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigError: If file cannot be read or parsed
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return
            
            with Path(path).open('r') as f:
                file_config = json.load(f)
            
            # Resolve relative paths in toolDirectories
            if "toolDirectories" in file_config:
                # Resolve relative to project root (parent of config directory)
                # config/launcher_config.json -> project root (where launchmcp.py lives)
                project_root = path.parent.parent
                resolved_dirs = []
                for dir_path in file_config["toolDirectories"]:
                    p = Path(dir_path)
                    if p.is_absolute():
                        resolved_dirs.append(dir_path)
                    else:
                        # Resolve relative to project root
                        resolved = (project_root / p).resolve()
                        resolved_dirs.append(str(resolved))
                file_config["toolDirectories"] = resolved_dirs
            
            # Migrate legacy port allocation config to new format
            self._migrate_port_config(file_config)
            
            # Merge file config with defaults
            self._merge_config(self.config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config file: {e}")
    
    def _migrate_port_config(self, config: dict[str, Any]) -> None:
        """
        Migrate legacy port allocation config to new format.
        
        Args:
            config: Configuration dictionary (modified in place)
        """
        port_alloc = config.get("portAllocation", {})
        
        # Check if already using new format
        if "ranges" in port_alloc and "manualPorts" in port_alloc:
            return
        
        # Migrate legacy format to new format
        logger.info("Migrating legacy port allocation config to new format")
        
        # Load from ports.json - this is the ONLY source of truth
        try:
            ports_config = load_ports_config()
        except ConfigError:
            logger.warning("ports.json not found, using legacy defaults")
            ports_config = {"ranges": {}, "reserved": {}, "assignments": {}}
        
        # Set ranges from ports.json
        if ports_config.get("ranges"):
            port_alloc.setdefault("ranges", {k: tuple(v) for k, v in ports_config["ranges"].items()})
        port_alloc.setdefault("reservedPorts", ports_config.get("reserved", {}))
        port_alloc.setdefault("manualPorts", ports_config.get("assignments", {}))
        
        # Convert legacy ports to new manualPorts format
        legacy_ports = port_alloc.get("ports", {})
        if legacy_ports and "manualPorts" not in port_alloc:
            port_alloc["manualPorts"] = {"mcp": legacy_ports.copy()}
        
        # Convert legacy managementPorts to new format (mgmt type)
        legacy_mgmt = port_alloc.get("managementPorts", {})
        if legacy_mgmt:
            mgmt_manual = {}
            for tool_name, port in legacy_mgmt.items():
                mgmt_manual[f"{tool_name}_mgmt"] = port
            
            # Merge into existing manualPorts if present
            if "manualPorts" not in port_alloc:
                port_alloc["manualPorts"] = {}
            if "mgmt" not in port_alloc["manualPorts"]:
                port_alloc["manualPorts"]["mgmt"] = {}
            port_alloc["manualPorts"]["mgmt"].update(mgmt_manual)
        
        config["portAllocation"] = port_alloc
    
    def _resolve_tool_directories(self) -> None:
        """
        Resolve relative paths in toolDirectories to absolute paths.
        
        Uses the directory containing this config.py file as the base for resolution.
        """
        if "toolDirectories" not in self.config:
            return
        
        # Use the directory containing this file as the base for resolution
        # config.py is in launcher/, so parent is supreme-mcp-tools/
        config_dir = Path(__file__).parent.parent.resolve()
        
        resolved_dirs = []
        for dir_path in self.config["toolDirectories"]:
            p = Path(dir_path)
            if p.is_absolute():
                resolved_dirs.append(dir_path)
            else:
                # Resolve relative to supreme-mcp-tools directory
                resolved = (config_dir / p).resolve()
                resolved_dirs.append(str(resolved))
        
        self.config["toolDirectories"] = resolved_dirs
        logger.debug(f"Resolved tool directories: {resolved_dirs}")
    
    def _load_from_env(self) -> None:
        """Load configuration overrides from environment variables."""
        env_mappings = {
            "LAUNCHER_TOOL_DIRECTORIES": ("toolDirectories", "list"),
            "LAUNCHER_PORT_MODE": ("portAllocation.mode", "string"),
            "LAUNCHER_BASE_PORT": ("portAllocation.basePort", "int"),
            "LAUNCHER_PORT_RANGE": ("portAllocation.portRange", "list"),
            "LAUNCHER_SERVER_HOST": ("server.host", "string"),
            "LAUNCHER_LOG_LEVEL": ("server.logLevel", "string"),
            "LAUNCHER_LOGGING_LEVEL": ("logging.level", "string"),
            "LAUNCHER_CONTINUE_ON_ERROR": ("errorHandling.continueOnError", "bool"),
            "LAUNCHER_FAIL_FAST": ("errorHandling.failFast", "bool"),
            # New port type environment variables
            "LAUNCHER_CENTRAL_MGMT_PORT": ("portAllocation.reservedPorts.central_management", "int"),
            "LAUNCHER_METRICS_PORT": ("portAllocation.reservedPorts.metrics_server", "int"),
            "LAUNCHER_UI_PORT": ("portAllocation.reservedPorts.management_ui", "int"),
            "LAUNCHER_MCP_RANGE": ("portAllocation.ranges.mcp", "list"),
            "LAUNCHER_MGMT_RANGE": ("portAllocation.ranges.mgmt", "list"),
            "LAUNCHER_SYSTEM_RANGE": ("portAllocation.ranges.system", "list"),
            "LAUNCHER_METRICS_RANGE": ("portAllocation.ranges.metrics", "list"),
            "LAUNCHER_UI_RANGE": ("portAllocation.ranges.ui", "list"),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    parsed_value = self._parse_env_value(value, value_type)
                    self._set_nested_value(self.config, config_path, parsed_value)
                    logger.debug(f"Loaded {env_var}={value}")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse {env_var}: {e}")
    
    def _parse_env_value(self, value: str, value_type: str) -> Any:
        """
        Parse environment variable value based on type.
        
        Args:
            value: String value from environment
            value_type: Type to parse to (string, int, bool, list)
            
        Returns:
            Parsed value
            
        Raises:
            ValueError: If value cannot be parsed
        """
        if value_type == "string":
            return value
        elif value_type == "int":
            return int(value)
        elif value_type == "bool":
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == "list":
            return [item.strip() for item in value.split(",")]
        else:
            raise ValueError(f"Unknown value type: {value_type}")
    
    def _merge_config(self, base: dict, override: dict) -> None:
        """
        Recursively merge override config into base config.
        
        Args:
            base: Base configuration dictionary (modified in place)
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: dict, path: str, value: Any) -> None:
        """
        Set a nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path to the value
            value: Value to set
        """
        keys = path.split(".")
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _validate_config(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ConfigError: If configuration is invalid
        """
        # Validate tool directories
        tool_dirs = self.get_tool_directories()
        for tool_dir in tool_dirs:
            path = Path(tool_dir)
            if not path.exists():
                logger.warning(f"Tool directory does not exist: {tool_dir}")
        
        # Validate port allocation mode
        port_mode = self.get_port_mode()
        if port_mode not in ("auto", "manual"):
            raise ConfigError(f"Invalid port allocation mode: {port_mode}")
        
        # Validate port range
        port_range = self.get_port_range()
        if len(port_range) != 2 or port_range[0] >= port_range[1]:
            raise ConfigError(f"Invalid port range: {port_range}")
        
        # Validate port numbers are within valid range (1-65535)
        if port_range[0] < 1 or port_range[1] > 65535:
            raise ConfigError(f"Port range must be between 1-65535, got: {port_range}")
        
        # Validate base port
        base_port = self.get_base_port()
        if not (1 <= base_port <= 65535):
            raise ConfigError(f"Base port must be between 1-65535, got: {base_port}")
        if not (port_range[0] <= base_port <= port_range[1]):
            raise ConfigError(f"Base port {base_port} outside range {port_range}")
        
        # Validate manual port assignments
        manual_ports = self.get_manual_ports()
        for tool_name, port in manual_ports.items():
            if not (1 <= port <= 65535):
                raise ConfigError(f"Manual port for {tool_name} must be between 1-65535, got: {port}")
        
        # Validate server host
        host = self.get_server_host()
        if not isinstance(host, str) or not host:
            raise ConfigError(f"Invalid server host: {host}")
        
        # Validate log level
        log_level = self.get_log_level()
        valid_levels = ("debug", "info", "warning", "error", "critical")
        if log_level.lower() not in valid_levels:
            raise ConfigError(f"Invalid log level: {log_level}")
        
        # Validate error handling settings
        continue_on_error = self.get_continue_on_error()
        fail_fast = self.get_fail_fast()
        if not isinstance(continue_on_error, bool):
            raise ConfigError(f"continueOnError must be a boolean, got: {continue_on_error}")
        if not isinstance(fail_fast, bool):
            raise ConfigError(f"failFast must be a boolean, got: {fail_fast}")
    
    def get_tool_directories(self) -> list[str]:
        """Get list of tool directories."""
        return self.config.get("toolDirectories", [])
    
    def get_port_mode(self) -> str:
        """Get port allocation mode."""
        return self.config.get("portAllocation", {}).get("mode", "auto")
    
    def get_base_port(self) -> int:
        """Get base port for auto allocation.
        
        Returns:
            Base port number
            
        Raises:
            ConfigError: If basePort is not configured in ports.json
        """
        port_alloc = self.config.get("portAllocation", {})
        base_port = port_alloc.get("basePort") if port_alloc else None
        if base_port is None:
            raise ConfigError(
                "basePort not configured. Please set portAllocation.basePort in config/launcher_config.json "
                "or ensure ports.json is properly configured."
            )
        return base_port
    
    def get_port_range(self) -> list[int]:
        """Get port range for allocation.
        
        Returns:
            List of [min_port, max_port]
            
        Raises:
            ConfigError: If portRange is not configured in ports.json
        """
        port_alloc = self.config.get("portAllocation", {})
        port_range = port_alloc.get("portRange") if port_alloc else None
        if port_range is None:
            raise ConfigError(
                "portRange not configured. Please set portAllocation.portRange in config/launcher_config.json "
                "or ensure ports.json is properly configured."
            )
        return port_range
    
    def get_manual_ports(self) -> dict[str, int]:
        """Get manual port assignments for backward compatibility.
        
        Returns a flat dictionary with MCP endpoint ports.
        For type-aware allocation, use get_manual_ports_by_type() instead.
        
        Returns:
            Dictionary of tool_name -> port for MCP endpoints
        """
        # Try new format first
        manual_by_type = self.get_manual_ports_by_type()
        if "mcp" in manual_by_type:
            return manual_by_type["mcp"].copy()
        
        # Fallback to legacy format
        return self.config.get("portAllocation", {}).get("ports", {}).copy()
    
    def get_all_manual_ports(self) -> dict[str, int]:
        """Get all manual port assignments with type suffixes.
        
        Returns a dictionary with both MCP endpoint ports and management ports.
        Management port names are suffixed with '_mgmt' (e.g., 'simplemcp_mgmt').
        
        Returns:
            Dictionary of service_name -> port (all types)
        """
        ports = {}
        
        # Get MCP ports
        mcp_ports = self.get_manual_ports()
        ports.update(mcp_ports)
        
        # Add management ports with _mgmt suffix
        management_ports = self.get_management_ports()
        for tool_name, port in management_ports.items():
            ports[f"{tool_name}_mgmt"] = port
        
        return ports
    
    def get_management_ports(self) -> dict[str, int]:
        """Get management port assignments from config.
        
        These are pre-configured management ports but tools get their
        actual management ports auto-allocated by the port manager.
        """
        return self.config.get("portAllocation", {}).get("managementPorts", {}).copy()
    
    def get_port_ranges(self) -> dict[str, tuple[int, int]]:
        """Get port ranges per type.
        
        Returns:
            Dictionary mapping port type to (min, max) tuple
        """
        ranges = self.config.get("portAllocation", {}).get("ranges")
        if ranges is None:
            # This should never happen if _ensure_port_defaults was called
            raise ConfigError("Port ranges not configured. Ensure ports.json exists.")
        return ranges
    
    def get_reserved_ports(self) -> dict[str, int]:
        """Get reserved system service ports.
        
        Returns:
            Dictionary mapping service name to port number
        """
        ports = self.config.get("portAllocation", {}).get("reservedPorts")
        if ports is None:
            # This should never happen if _ensure_port_defaults was called
            raise ConfigError("Reserved ports not configured. Ensure ports.json exists.")
        return ports
    
    def get_manual_ports_by_type(self) -> dict[str, dict[str, int]]:
        """Get manual port assignments organized by type.
        
        Returns:
            Dictionary mapping port type to {service_name: port}
        """
        manual_ports = self.config.get("portAllocation", {}).get("manualPorts", {})
        if manual_ports:
            return manual_ports
        
        # Backward compatibility: convert legacy format
        legacy_ports = self.config.get("portAllocation", {}).get("ports", {})
        if legacy_ports:
            return {"mcp": legacy_ports}
        
        return {}
    
    def get_server_host(self) -> str:
        """Get server host address."""
        return self.config.get("server", {}).get("host", "0.0.0.0")
    
    def get_server_log_level(self) -> str:
        """Get server log level."""
        return self.config.get("server", {}).get("logLevel", "info")
    
    def get_log_level(self) -> str:
        """Get launcher log level."""
        return self.config.get("logging", {}).get("level", "INFO")
    
    def get_log_format(self) -> str:
        """Get log format string."""
        return self.config.get("logging", {}).get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def get_log_file(self) -> str | None:
        """Get log file path (None for console only)."""
        return self.config.get("logging", {}).get("file")
    
    def get_continue_on_error(self) -> bool:
        """Get whether to continue on errors."""
        return self.config.get("errorHandling", {}).get("continueOnError", True)
    
    def get_fail_fast(self) -> bool:
        """Get whether to fail fast on errors."""
        return self.config.get("errorHandling", {}).get("failFast", False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def to_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
