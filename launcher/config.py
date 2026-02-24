"""
Configuration handling for the MCP launcher system.

This module provides functionality to load, validate, and manage
configuration from JSON files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .errors import ConfigError


logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the MCP launcher."""
    
    # Default configuration values
    # Use dynamic path resolution to avoid hardcoded absolute paths
    DEFAULT_CONFIG = {
        "toolDirectories": [
            # Relative paths from supreme-mcp-tools root
            "tools/webmcp",
            "tools/oraclemcp",
            "tools/simplemcp8",
            "tools/convertermcp",
            "tools/ragmcp"
        ],
        "portAllocation": {
            "mode": "auto",
            "basePort": 8000,
            "portRange": [8000, 9000],
            "ports": {
                "oracleMCP": 8000,
                "web_mcp": 8001,
                "simplemcp8": 8002
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
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Start with defaults
        self.config = self.DEFAULT_CONFIG.copy()
        
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
            
            with open(path, 'r') as f:
                file_config = json.load(f)
            
            # Resolve relative paths in toolDirectories
            if "toolDirectories" in file_config:
                config_dir = path.parent
                resolved_dirs = []
                for dir_path in file_config["toolDirectories"]:
                    p = Path(dir_path)
                    if p.is_absolute():
                        resolved_dirs.append(dir_path)
                    else:
                        # Resolve relative to config file directory
                        resolved = (config_dir / p).resolve()
                        resolved_dirs.append(str(resolved))
                file_config["toolDirectories"] = resolved_dirs
            
            # Merge file config with defaults
            self._merge_config(self.config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
        
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config file: {e}")
    
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
            "LAUNCHER_FAIL_FAST": ("errorHandling.failFast", "bool")
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
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
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
    
    def _set_nested_value(self, config: Dict, path: str, value: Any) -> None:
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
    
    def get_tool_directories(self) -> List[str]:
        """Get list of tool directories."""
        return self.config.get("toolDirectories", [])
    
    def get_port_mode(self) -> str:
        """Get port allocation mode."""
        return self.config.get("portAllocation", {}).get("mode", "auto")
    
    def get_base_port(self) -> int:
        """Get base port for auto allocation."""
        return self.config.get("portAllocation", {}).get("basePort", 8000)
    
    def get_port_range(self) -> List[int]:
        """Get port range for allocation."""
        return self.config.get("portAllocation", {}).get("portRange", [8000, 9000])
    
    def get_manual_ports(self) -> Dict[str, int]:
        """Get manual port assignments."""
        return self.config.get("portAllocation", {}).get("ports", {})
    
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
    
    def get_log_file(self) -> Optional[str]:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
