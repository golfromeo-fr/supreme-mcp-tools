"""
Distributed Deployment Configuration for FEF V3

Provides configuration for deploying tools across multiple machines.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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


@dataclass
class ManagementServerConfig:
    """Configuration for the management server."""
    host: str = "0.0.0.0"
    port: int = 9091
    advertised_url: Optional[str] = None
    api_key: Optional[str] = None
    
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
    tools: Dict[str, ToolEndpoint] = field(default_factory=dict)
    tls_enabled: bool = False
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None
    
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
    
    def get_tool(self, name: str) -> Optional[ToolEndpoint]:
        """
        Get a tool endpoint.
        
        Args:
            name: Tool name
            
        Returns:
            ToolEndpoint if found
        """
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all configured tool names."""
        return list(self.tools.keys())
    
    def to_dict(self) -> Dict[str, Any]:
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
        
        with open(path, "w") as f:
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
        
        with open(path, "r") as f:
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


# Default distributed configuration
DEFAULT_DISTRIBUTED_CONFIG = DistributedConfig(
    management_server=ManagementServerConfig(
        host="0.0.0.0",
        port=9091
    ),
    tools={
        "webmcp": ToolEndpoint(host="localhost", mcp_port=8001, mgmt_port=9001),
        "simplemcp": ToolEndpoint(host="localhost", mcp_port=8002, mgmt_port=9002),
        "ragmcp": ToolEndpoint(host="localhost", mcp_port=8004, mgmt_port=9004),
    }
)
