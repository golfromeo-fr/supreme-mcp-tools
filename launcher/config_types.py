"""
Focused configuration dataclasses for the MCP launcher.

These dataclasses extract strongly-typed configuration domains from the
god-object Config class. Subsystems can import just what they need.

Each dataclass validates on construction and raises ConfigError on invalid values.
"""
from dataclasses import dataclass, field
from typing import Any
import logging
import logging.handlers

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

    Rotation: logs rotate on a time interval (default 30 days, keeping 6
    backups) AND when the active file exceeds ``max_bytes`` (10 MB by default).
    Set ``rotation_when`` to one of 'S' (seconds), 'M' (minutes),
    'H' (hours), 'D' (days), 'W0'-'W6' (weekday), or 'midnight'. Set
    ``rotation_interval`` accordingly. The size cap is a secondary guard so a
    log flood within one rotation interval still rolls over (it does not change
    the file naming — size-triggered rollovers reuse the date suffix and are
    pruned by ``rotation_backup_count``). Set ``file`` to None to disable the
    file handler; set ``max_bytes`` to 0 to disable the size guard.
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str | None = None
    component_log_levels: dict[str, str] = field(default_factory=dict)
    # Time-based rotation (primary)
    rotation_when: str = "D"          # D = days; also S/M/H/W0..W6/midnight
    rotation_interval: int = 30       # rotate every N units (30 days default)
    rotation_backup_count: int = 6    # keep N rotated files (6 months at 30d)
    # Size-based rotation (secondary guard)
    max_bytes: int = 10 * 1024 * 1024  # 10 MB — rollover early if a flood hits

    VALID_LEVELS = ("debug", "info", "warning", "error", "critical")
    VALID_WHEN = ("S", "M", "H", "D", "MIDNIGHT", "W0", "W1", "W2", "W3", "W4", "W5", "W6")

    def __post_init__(self):
        if self.level.lower() not in self.VALID_LEVELS:
            raise ConfigError(f"Invalid log level: {self.level}")
        if self.rotation_when.upper() not in self.VALID_WHEN:
            raise ConfigError(
                f"Invalid rotation_when: {self.rotation_when}. "
                f"Must be one of {self.VALID_WHEN}"
            )
        if self.rotation_interval <= 0:
            raise ConfigError(f"rotation_interval must be positive, got {self.rotation_interval}")
        if self.rotation_backup_count < 0:
            raise ConfigError(f"rotation_backup_count must be >= 0, got {self.rotation_backup_count}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        """Create from nested config dict."""
        return cls(
            level=data.get("level", "INFO"),
            format=data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=data.get("file"),
            component_log_levels=data.get("componentLogLevels", {}),
            rotation_when=data.get("rotationWhen", data.get("rotation_when", "D")),
            rotation_interval=data.get("rotationInterval", data.get("rotation_interval", 30)),
            rotation_backup_count=data.get(
                "rotationBackupCount", data.get("rotation_backup_count", 6)
            ),
            max_bytes=data.get("maxBytes", data.get("max_bytes", 10 * 1024 * 1024)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return as nested dict."""
        return {
            "level": self.level,
            "format": self.format,
            "file": self.file,
            "componentLogLevels": self.component_log_levels,
            "rotationWhen": self.rotation_when,
            "rotationInterval": self.rotation_interval,
            "rotationBackupCount": self.rotation_backup_count,
            "maxBytes": self.max_bytes,
        }


class _SizeAwareTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """TimedRotatingFileHandler that ALSO rolls over on size.

    The stdlib offers time-OR-size rotation, never both. A persistent launcher
    can log a flood inside one rotation interval (e.g. a month), so we layer a
    size check on top of the calendar schedule: ``shouldRollover`` returns true
    when either the next-scheduled time has passed OR the active file is at/over
    ``max_bytes``. ``doRollover`` is inherited unchanged (date-stamped naming +
    ``backupCount`` pruning), so total disk stays bounded by ``backupCount``
    rotated files plus one active file.

    ``max_bytes == 0`` disables the size guard (matching ``RotatingFileHandler``).
    """

    def __init__(self, filename, max_bytes: int = 0, **kwargs):
        super().__init__(filename, **kwargs)
        self.max_bytes = max_bytes

    def shouldRollover(self, record):
        # 1) calendar schedule (also opens the stream on first use)
        if super().shouldRollover(record):
            return True
        # 2) size cap — after super() returned False, self.stream is open
        if self.max_bytes > 0 and self.stream is not None:
            try:
                return self.stream.tell() >= self.max_bytes
            except Exception:
                # Never let rotation logic raise into the logging call site.
                return False
        return False


def make_file_handler(cfg: "LoggingConfig") -> logging.Handler | None:
    """Build a rotating file handler from a LoggingConfig.

    Returns a :class:`_SizeAwareTimedRotatingFileHandler` so logs roll over on a
    calendar schedule (default monthly) AND on the ``max_bytes`` size cap — both
    conditions trigger a rollover, and rotated files are pruned to
    ``rotation_backup_count``. Returns None if ``cfg.file`` is unset. Both
    launcher entry points (launchmcp.py legacy and launcher/__main__.py new) use
    this so they share identical rotation behaviour.
    """
    if not cfg.file:
        return None
    from pathlib import Path

    log_path = Path(cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = _SizeAwareTimedRotatingFileHandler(
        log_path,
        when=cfg.rotation_when,
        interval=cfg.rotation_interval,
        backupCount=cfg.rotation_backup_count,
        max_bytes=cfg.max_bytes,
        encoding="utf-8",
        utc=False,
    )
    return handler


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