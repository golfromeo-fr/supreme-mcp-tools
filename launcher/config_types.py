"""
Focused configuration dataclasses for the MCP launcher.

These dataclasses extract strongly-typed configuration domains from the
god-object Config class. Subsystems can import just what they need.

Each dataclass validates on construction and raises ConfigError on invalid values.
"""
from dataclasses import dataclass, field
from typing import Any
import logging

from .errors import ConfigError

logger = logging.getLogger(__name__)

# ── Single source of truth for launcher bind host ─────────────────────────
# Must match tools/shared/server_factory.DEFAULT_HOST.
# On Debian/Linux, "::" binds IPv6-only despite bindv6only=0, breaking IPv4
# clients (VS Code Copilot). Use "0.0.0.0" which accepts both 127.0.0.1 and
# localhost (::1 → kernel maps to IPv4).
DEFAULT_HOST = "0.0.0.0"


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging subsystem configuration.

    Immutable - modify by creating a new instance.
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None
    component_log_levels: dict[str, str] = field(default_factory=dict)

    VALID_LEVELS = ("debug", "info", "warning", "error", "critical")

    def __post_init__(self):
        if self.level.lower() not in self.VALID_LEVELS:
            raise ConfigError(f"Invalid log level: {self.level}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        """Create from nested config dict."""
        return cls(
            level=data.get("level", "INFO"),
            format=data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=data.get("file"),
            component_log_levels=data.get("componentLogLevels", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return as nested dict."""
        return {
            "level": self.level,
            "format": self.format,
            "file": self.file,
            "componentLogLevels": self.component_log_levels,
        }


@dataclass(frozen=True)
class ServerConfig:
    """
    Server subsystem configuration.

    Immutable - modify by creating a new instance.
    """
    host: str = DEFAULT_HOST
    log_level: str = "info"

    VALID_LEVELS = ("debug", "info", "warning", "error", "critical")

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host:
            raise ConfigError(f"Invalid server host: {self.host}")
        if self.log_level.lower() not in self.VALID_LEVELS:
            raise ConfigError(f"Invalid server log level: {self.log_level}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerConfig":
        return cls(
            host=data.get("host", DEFAULT_HOST),
            log_level=data.get("logLevel", "info"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "logLevel": self.log_level}


@dataclass(frozen=True)
class ErrorHandlingConfig:
    """
    Error handling policy configuration.

    Immutable - modify by creating a new instance.
    """
    continue_on_error: bool = True
    fail_fast: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorHandlingConfig":
        return cls(
            continue_on_error=data.get("continueOnError", True),
            fail_fast=data.get("failFast", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"continueOnError": self.continue_on_error, "failFast": self.fail_fast}


@dataclass(frozen=True)
class PortConfig:
    """
    Port allocation and reservation configuration.

    Immutable - modify by creating a new instance.

    The port system has multiple ranges and reserved system ports:
    - MCP tools: 8000-8099 (configurable)
    - Management: 8100-8199
    - System: 8200-8299
    - Metrics: 8300-8399
    - UI: 8400-8499
    """
    mode: str = "manual"
    base_port: int = 8000
    port_range: tuple[int, int] = (8000, 8099)
    ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    reserved_ports: dict[str, int] = field(default_factory=dict)
    manual_ports: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ("auto", "manual"):
            raise ConfigError(f"Invalid port allocation mode: {self.mode}")
        if len(self.port_range) != 2 or self.port_range[0] >= self.port_range[1]:
            raise ConfigError(f"Invalid port range: {list(self.port_range)}")
        if not (1 <= self.port_range[0] <= 65535 and 1 <= self.port_range[1] <= 65535):
            raise ConfigError(f"Port range must be between 1-65535")
        if not (1 <= self.base_port <= 65535):
            raise ConfigError(f"Base port must be between 1-65535, got: {self.base_port}")
        if not (self.port_range[0] <= self.base_port <= self.port_range[1]):
            raise ConfigError(f"Base port {self.base_port} outside range {list(self.port_range)}")
        for tool_name, port in self.flat_manual_ports.items():
            if not (1 <= port <= 65535):
                raise ConfigError(f"Manual port for {tool_name} must be between 1-65535, got: {port}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortConfig":
        ranges_raw = data.get("ranges", {})
        ranges = {k: tuple(v) for k, v in ranges_raw.items()} if ranges_raw else {}

        manual_raw = data.get("manualPorts", {})
        manual_ports = {}
        for type_name, mapping in manual_raw.items():
            if isinstance(mapping, dict):
                manual_ports[type_name] = mapping

        return cls(
            mode=data.get("mode", "manual"),
            base_port=data.get("basePort", 8000),
            port_range=tuple(data.get("portRange", [8000, 8099])),
            ranges=ranges,
            reserved_ports=data.get("reservedPorts", {}),
            manual_ports=manual_ports,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "basePort": self.base_port,
            "portRange": list(self.port_range),
            "ranges": {k: list(v) for k, v in self.ranges.items()},
            "reservedPorts": self.reserved_ports,
            "manualPorts": self.manual_ports,
        }

    @property
    def flat_manual_ports(self) -> dict[str, int]:
        """Flatten manual ports to tool_name -> port mapping for MCP type only."""
        mcp_ports = self.manual_ports.get("mcp", {})
        result = dict(mcp_ports)
        for type_name, mapping in self.manual_ports.items():
            if type_name != "mcp" and isinstance(mapping, dict):
                for tool_name, port in mapping.items():
                    result[f"{tool_name}_{type_name}"] = port
        return result

    def get_mcp_port(self, tool_name: str) -> int | None:
        """Get MCP port for a specific tool."""
        return self.manual_ports.get("mcp", {}).get(tool_name)

    def get_mgmt_port(self, tool_name: str) -> int | None:
        """Get management port for a specific tool."""
        return self.manual_ports.get("mgmt", {}).get(tool_name)


@dataclass(frozen=True)
class ToolDirectoryConfig:
    """
    Tool discovery directories configuration.

    Immutable - modify by creating a new instance.
    """
    directories: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_list(cls, dirs: list[str]) -> "ToolDirectoryConfig":
        if not all(isinstance(d, str) for d in dirs):
            raise ConfigError("toolDirectories must be a list of strings")
        return cls(directories=tuple(dirs))

    def to_list(self) -> list[str]:
        return list(self.directories)