"""
Distributed Deployment Configuration for FEF V3

Provides configuration for deploying tools across multiple machines.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..config_types import DEFAULT_HOST

logger = logging.getLogger(__name__)


def _get_port_from_config(key: str, subkey: str = None) -> int:
    """
    Get a port from ports.json configuration.
    
    Args:
        key: Top-level key in ports.json (e.g., "reserved", "assignments")
        subkey: Second-level key (e.g., "central_management", "mcp")
        
    Returns:
        Port number from config
        
    Raises:
        ConfigError: If ports.json is missing or key not found
    """
    from launcher.launcher_config import load_ports_config, ConfigError
    ports_config = load_ports_config()
    
    value = ports_config.get(key, {})
    if subkey:
        value = value.get(subkey)
    
    if value is None:
        raise ConfigError(f"Port not found in ports.json: {key}.{subkey}")
    
    return int(value)


def _get_central_management_port() -> int:
    """Get the central management port from ports.json."""
    return _get_port_from_config("reserved", "central_management")


@dataclass
class ToolEndpoint:
    """Configuration for a remote tool endpoint."""
    host: str
    mcp_port: int
    mgmt_port: int
    protocol: str = "http"
    
    @property
    def management_url(self) -> str:
        """Get the management server URL."""
        return f"{self.protocol}://{self.host}:{self.mgmt_port}"
    
    @property
    def mcp_url(self) -> str:
        """Get the MCP server URL."""
        return f"{self.protocol}://{self.host}:{self.mcp_port}"


def _default_management_server_config() -> "ManagementServerConfig":
    """Factory function to create ManagementServerConfig with port from ports.json."""
    return ManagementServerConfig(port=_get_central_management_port())


@dataclass
class ManagementServerConfig:
    """Configuration for the management server."""
    host: str = DEFAULT_HOST
    port: int = field(default_factory=_get_central_management_port)
    advertised_url: str | None = None
    api_key: str | None = None
    
    @property
    def url(self) -> str:
        """Get the management server URL."""
        if self.advertised_url:
            return self.advertised_url
        return f"http://{self.host}:{self.port}"


@dataclass
class DistributedConfig:
    """
    Configuration for distributed deployment.
    
    Supports deploying tools across multiple machines with a
    centralized management server.
    """
    management_server: ManagementServerConfig = field(
        default_factory=ManagementServerConfig
    )
    tools: dict[str, ToolEndpoint] = field(default_factory=dict)
    tls_enabled: bool = False
    tls_cert_file: str | None = None
    tls_key_file: str | None = None
    
    def add_tool(
        self,
        name: str,
        host: str,
        mcp_port: int,
        mgmt_port: int,
        protocol: str = "http"
    ) -> None:
        """
        Add a tool endpoint.
        
        Args:
            name: Tool name
            host: Host address
            mcp_port: MCP server port
            mgmt_port: Management server port
            protocol: Protocol (http/https)
        """
        self.tools[name] = ToolEndpoint(
            host=host,
            mcp_port=mcp_port,
            mgmt_port=mgmt_port,
            protocol=protocol
        )
        logger.info(f"Added tool endpoint: {name} at {host}")
    
    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool endpoint.
        
        Args:
            name: Tool name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Removed tool endpoint: {name}")
            return True
        return False
    
    def get_tool(self, name: str) -> ToolEndpoint | None:
        """
        Get a tool endpoint.
        
        Args:
            name: Tool name
            
        Returns:
            ToolEndpoint if found
        """
        return self.tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all configured tool names."""
        return list(self.tools.keys())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "management_server": asdict(self.management_server),
            "tools": {
                name: asdict(endpoint)
                for name, endpoint in self.tools.items()
            },
            "tls_enabled": self.tls_enabled,
            "tls_cert_file": self.tls_cert_file,
            "tls_key_file": self.tls_key_file
        }
        return result
    
    def save(self, config_file: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to config file
        """
        path = Path(config_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved distributed config to {path}")
    
    @classmethod
    def load(cls, config_file: str) -> "DistributedConfig":
        """
        Load configuration from file.
        
        Args:
            config_file: Path to config file
            
        Returns:
            DistributedConfig instance
        """
        path = Path(config_file).expanduser()
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return cls()
        
        with Path(path).open("r") as f:
            data = json.load(f)
        
        config = cls()
        
        # Load management server config
        if "management_server" in data:
            ms_data = data["management_server"]
            config.management_server = ManagementServerConfig(**ms_data)
        
        # Load tool endpoints
        if "tools" in data:
            for name, tool_data in data["tools"].items():
                config.tools[name] = ToolEndpoint(**tool_data)
        
        # Load TLS config
        config.tls_enabled = data.get("tls_enabled", False)
        config.tls_cert_file = data.get("tls_cert_file")
        config.tls_key_file = data.get("tls_key_file")
        
        logger.info(f"Loaded distributed config from {path}")
        return config


def _build_default_distributed_config() -> DistributedConfig:
    """
    Build default distributed configuration from ports.json.
    
    All ports are loaded from config/ports.json - no hardcoded ports.
    """
    config = DistributedConfig()
    
    # Management server port from ports.json
    config.management_server = ManagementServerConfig()
    
    # Tool endpoints from ports.json assignments
    try:
        mcp_assignments = _get_port_from_config("assignments", "mcp")
        mgmt_assignments = _get_port_from_config("assignments", "mgmt")
        
        for tool_name in mcp_assignments:
            if tool_name in mgmt_assignments:
                config.tools[tool_name] = ToolEndpoint(
                    host="localhost",
                    mcp_port=mcp_assignments[tool_name],
                    mgmt_port=mgmt_assignments[tool_name]
                )
    except Exception as e:
        logger.warning(f"Could not load tool ports from ports.json: {e}")
    
    return config


# Default distributed configuration - built lazily from ports.json
DEFAULT_DISTRIBUTED_CONFIG = None

def get_default_distributed_config() -> DistributedConfig:
    """Get the default distributed configuration, building it if needed."""
    global DEFAULT_DISTRIBUTED_CONFIG
    if DEFAULT_DISTRIBUTED_CONFIG is None:
        DEFAULT_DISTRIBUTED_CONFIG = _build_default_distributed_config()
    return DEFAULT_DISTRIBUTED_CONFIG
