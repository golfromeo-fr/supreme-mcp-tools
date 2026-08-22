"""
Port allocation manager for the MCP launcher system.

This module provides functionality to allocate and manage ports
for multiple MCP tools running concurrently.

Port Types:
    - mcp: MCP tool endpoints (default range: 8000-8099)
    - mgmt: Tool management servers (default range: 8100-8199)
    - system: Central system services (default range: 8200-8299)
    - metrics: Monitoring/metrics endpoints (default range: 8300-8399)
    - ui: Web UI services (default range: 8400-8499)
"""

import logging
import socket
import time
from typing import Any

from .errors import PortConflictError
from .config_types import DEFAULT_HOST


logger = logging.getLogger(__name__)

# A busy manual port right after a restart is usually the previous launcher
# instance still shutting down (~10s to release 4 servers). Retry briefly
# instead of failing the tool for the whole run.
PORT_BUSY_RETRY_SECS = 20.0
PORT_BUSY_RETRY_INTERVAL = 1.5


def management_endpoint_occupied(mgmt_port: int | None = None) -> bool:
    """True if something listens on the central management port — a launcher
    instance is probably already running. Pre-flight check so a second launcher
    fails fast with guidance instead of fighting (and losing) over tool ports.
    """
    if mgmt_port is None:
        import json
        from pathlib import Path
        try:
            cfg = json.loads(
                (Path(__file__).resolve().parent.parent / "config" / "ports.json").read_text()
            )
            mgmt_port = int(cfg["assignments"]["system"]["central_management"])
        except Exception:
            mgmt_port = 8200
    try:
        with socket.create_connection(("127.0.0.1", int(mgmt_port)), timeout=1.0):
            return True
    except OSError:
        return False


class PortType:
    """Port type constants for categorizing port allocations."""
    MCP = "mcp"              # MCP tool endpoints
    MANAGEMENT = "mgmt"       # Tool management servers
    SYSTEM = "system"         # Central system services
    MONITORING = "metrics"    # Monitoring/metrics endpoints
    UI = "ui"                 # Web UI services
    CUSTOM = "custom"         # Tool-specific needs


class PortManager:
    """Manage port allocation for all service types with type-aware ranges."""
    
    def __init__(
        self,
        ports_config: dict[str, Any],
        mode: str = "auto",
        base_port: int | None = None,
        port_range: list[int] | None = None,
        manual_ports: dict[str, int] | None = None,
        port_ranges: dict[str, tuple[int, int]] | None = None,
        reserved_ports: dict[str, int] | None = None,
        manual_ports_by_type: dict[str, dict[str, int]] | None = None
    ):
        """
        Initialize the port manager.
        
        Args:
            ports_config: Port configuration from ports.json (required)
            mode: Port allocation mode ("auto" or "manual")
            base_port: Starting port for auto allocation (for backward compatibility, 
                      should be configured in launcher_config.json or ports.json)
            port_range: Legacy port range for allocation [min, max] (for backward compatibility)
            manual_ports: Legacy dictionary of tool name -> port (for backward compatibility)
            port_ranges: Port ranges per type {port_type: (min, max)} - overrides ports_config
            reserved_ports: Pre-assigned system ports {service_name: port} - overrides ports_config
            manual_ports_by_type: Manual assignments per type {port_type: {name: port}} - overrides ports_config
        """
        self.mode = mode
        
        # Primary source: ranges and reserved from ports_config
        if port_ranges:
            self.port_ranges = port_ranges
        else:
            self.port_ranges = {
                k: tuple(v) for k, v in ports_config.get("ranges", {}).items()
            }
        
        # Use provided base_port or derive from mcp range for legacy support
        if base_port is not None:
            self.base_port = base_port
        elif PortType.MCP in self.port_ranges:
            self.base_port = self.port_ranges[PortType.MCP][0]
        else:
            raise ValueError(
                "base_port is required. Either provide it explicitly or ensure "
                "ports.json contains an mcp range."
            )
        
        if reserved_ports:
            self.reserved_ports = reserved_ports
        else:
            self.reserved_ports = ports_config.get("reserved", {}).copy()
        
        if manual_ports_by_type:
            self.manual_ports_by_type = manual_ports_by_type
        else:
            self.manual_ports_by_type = ports_config.get("assignments", {}).copy()
        
        # Legacy support: convert old format to new format
        if manual_ports and not manual_ports_by_type:
            # Old format: manual_ports = {"tool": 8000}
            # Convert to new format assuming MCP type
            self.manual_ports_by_type[PortType.MCP] = manual_ports
        
        # Legacy port_range for backward compatibility
        self.legacy_port_range = port_range or [8000, 9000]
        
        self.allocated_ports: set[int] = set()
        self.tool_ports: dict[str, int] = {}
        # Track next available port per type for auto-allocation
        self._next_port_by_type: dict[str, int] = {
            ptype: ranges[0] for ptype, ranges in self.port_ranges.items()
        }
        self.next_port = self.base_port  # Legacy support

        range_errors = self.validate_ranges()
        if range_errors:
            raise ValueError("Port range validation failed: " + "; ".join(range_errors))
        reserved_errors = self.validate_reserved_ports_in_ranges()
        if reserved_errors:
            raise ValueError("Reserved port validation failed: " + "; ".join(reserved_errors))
    
    def _get_range_for_type(self, port_type: str) -> tuple[int, int]:
        """Get the port range for a given port type.
        
        Raises:
            ValueError: If port_type is not recognized
        """
        if port_type not in self.port_ranges:
            raise ValueError(
                f"Unknown port type: '{port_type}'. "
                f"Valid types: {list(self.port_ranges.keys())}"
            )
        return self.port_ranges[port_type]
    
    def _get_manual_ports_for_type(self, port_type: str) -> dict[str, int]:
        """Get manual ports for a given type."""
        return self.manual_ports_by_type.get(port_type, {})
    
    def _is_port_available(self, port: int, port_type: str | None = None) -> bool:
        """
        Check if a port is available for use.
        
        Args:
            port: Port number to check
            port_type: Optional port type for range validation
            
        Returns:
            True if port is available, False otherwise
        """
        # Check if already allocated
        if port in self.allocated_ports:
            return False
        
        # Check if port is in range for its type
        if port_type:
            min_port, max_port = self._get_range_for_type(port_type)
            if not (min_port <= port <= max_port):
                return False
        
        # Check if port is actually available on the system
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((DEFAULT_HOST, port))
                return True
        except OSError:
            return False
    
    def validate_ranges(self) -> list[str]:
        """
        Validate that port ranges don't overlap.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        ranges_list = list(self.port_ranges.items())
        
        for i, (type1, (min1, max1)) in enumerate(ranges_list):
            for type2, (min2, max2) in ranges_list[i+1:]:
                # Check for overlap
                if not (max1 < min2 or max2 < min1):
                    errors.append(
                        f"Port range overlap: {type1} [{min1}, {max1}] and {type2} [{min2}, {max2}]"
                    )
        
        return errors
    
    def validate_reserved_ports_in_ranges(self) -> list[str]:
        """
        Validate that reserved ports fall within their appropriate ranges.
        
        Uses the configured port_ranges from ports.json instead of hardcoded values.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Reserved ports should be in system, metrics, or ui ranges
        # Get these ranges from the configured port_ranges (loaded from ports.json)
        valid_types = {PortType.SYSTEM, PortType.MONITORING, PortType.UI}
        
        for service_name, port in self.reserved_ports.items():
            # Check if port falls within any valid range
            in_valid_range = False
            for ptype in valid_types:
                if ptype in self.port_ranges:
                    min_port, max_port = self.port_ranges[ptype]
                    if min_port <= port <= max_port:
                        in_valid_range = True
                        break
            
            if not in_valid_range:
                # Build a descriptive message with the actual configured ranges
                valid_ranges_desc = []
                for ptype in valid_types:
                    if ptype in self.port_ranges:
                        min_port, max_port = self.port_ranges[ptype]
                        valid_ranges_desc.append(f"{ptype} [{min_port}, {max_port}]")
                
                errors.append(
                    f"Reserved port {port} for {service_name} not in valid ranges: {', '.join(valid_ranges_desc)}"
                )
        
        return errors
    
    def get_manual_port(self, name: str, port_type: str = PortType.MCP) -> int | None:
        """Return the configured manual port for a service, if any."""
        return self._get_manual_ports_for_type(port_type).get(name)

    def wait_for_busy_ports(
        self,
        ports: dict[str, int],
        port_type: str = PortType.MCP,
        timeout: float = PORT_BUSY_RETRY_SECS,
    ) -> set[str]:
        """Wait for busy ports to free, polling the whole set in parallel.

        One shared window for all ports instead of a serial per-port wait
        (4 tools x 20s serial = 80s worst case; parallel = 20s max).
        Returns the names whose ports are still busy when the window closes.
        """
        busy = {n for n, p in ports.items() if not self._is_port_available(p, port_type)}
        if not busy:
            return set()
        listing = ", ".join(f"{n}({ports[n]})" for n in sorted(busy))
        logger.warning(
            f"Busy ports: {listing} — waiting up to {timeout:.0f}s (all in parallel; "
            f"previous launcher still shutting down?)"
        )
        deadline = time.monotonic() + timeout
        while busy and time.monotonic() < deadline:
            time.sleep(PORT_BUSY_RETRY_INTERVAL)
            busy = {n for n in busy if not self._is_port_available(ports[n], port_type)}
        return busy

    def allocate_port(
        self,
        name: str,
        port_type: str = PortType.MCP,
        preferred_port: int | None = None
    ) -> int:
        """
        Allocate a port for a service.
        
        Args:
            name: Name of the service/tool
            port_type: Type of port (mcp, mgmt, system, metrics, ui)
            preferred_port: Optional preferred port number
            
        Returns:
            Allocated port number
            
        Raises:
            PortConflictError: If port allocation fails
        """
        # Check if service already has a port allocated
        if name in self.tool_ports:
            logger.info(f"Service {name} already has port {self.tool_ports[name]}")
            return self.tool_ports[name]
        
        port = None
        range_min, range_max = self._get_range_for_type(port_type)
        manual_ports = self._get_manual_ports_for_type(port_type)
        
        # Try preferred port first
        if preferred_port is not None:
            if self._is_port_available(preferred_port, port_type):
                port = preferred_port
            else:
                logger.warning(f"Preferred port {preferred_port} not available for {name}")
        
        # Try manual port assignment
        if port is None and self.mode == "manual":
            if name in manual_ports:
                manual_port = manual_ports[name]
                if not self._is_port_available(manual_port, port_type):
                    logger.warning(
                        f"Manual port {manual_port} for {name} is busy — retrying for up to "
                        f"{PORT_BUSY_RETRY_SECS:.0f}s (previous launcher still shutting down?)"
                    )
                    deadline = time.monotonic() + PORT_BUSY_RETRY_SECS
                    while time.monotonic() < deadline:
                        time.sleep(PORT_BUSY_RETRY_INTERVAL)
                        if self._is_port_available(manual_port, port_type):
                            break
                    else:
                        raise PortConflictError(
                            f"Manual port {manual_port} for {name} is still not available "
                            f"after {PORT_BUSY_RETRY_SECS:.0f}s",
                            port=manual_port,
                            tool_name=name
                        )
                port = manual_port
            else:
                logger.debug(f"No manual port configured for {name} in {port_type} type, using auto allocation")
        
        # Auto allocate a port
        if port is None:
            port = self._allocate_auto_port(port_type)
        
        # Register the port
        self.allocated_ports.add(port)
        self.tool_ports[name] = port
        
        logger.info(f"Allocated port {port} for service {name} (type: {port_type})")
        return port
    
    def _allocate_auto_port(self, port_type: str = PortType.MCP) -> int:
        """
        Automatically allocate a port from the type's range.
        
        Args:
            port_type: The type of port to allocate
            
        Returns:
            Allocated port number
            
        Raises:
            PortConflictError: If no ports available in range
        """
        min_port, max_port = self._get_range_for_type(port_type)
        
        # Try starting from next available port for this type
        port = self._next_port_by_type.get(port_type, min_port)
        
        # Find next available port
        while port <= max_port:
            if self._is_port_available(port, port_type):
                self._next_port_by_type[port_type] = port + 1
                return port
            port += 1
        
        # Wrap around to base port if needed
        port = min_port
        while port < self._next_port_by_type.get(port_type, min_port):
            if self._is_port_available(port, port_type):
                self._next_port_by_type[port_type] = port + 1
                return port
            port += 1
        
        raise PortConflictError(
            f"No available ports in range for type {port_type}: [{min_port}, {max_port}]"
        )
    
    def _get_port_type_for_port(self, port: int) -> str | None:
        """Infer the port type from the port number."""
        for ptype, (min_port, max_port) in self.port_ranges.items():
            if min_port <= port <= max_port:
                return ptype
        return None
    
    def release_port(self, tool_name: str) -> int | None:
        """
        Release a port allocated to a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Released port number, or None if tool had no port
        """
        if tool_name not in self.tool_ports:
            logger.warning(f"No port allocated for tool {tool_name}")
            return None
        
        port = self.tool_ports[tool_name]
        self.allocated_ports.discard(port)
        del self.tool_ports[tool_name]
        
        # Update next_port tracking if this port is lower than current next for its type
        port_type = self._get_port_type_for_port(port)
        if port_type and port < self._next_port_by_type.get(port_type, float('inf')):
            self._next_port_by_type[port_type] = port
        
        logger.info(f"Released port {port} for tool {tool_name}")
        return port
    
    def release_all_ports(self) -> None:
        """Release all allocated ports."""
        self.allocated_ports.clear()
        self.tool_ports.clear()
        self._next_port_by_type = {
            ptype: ranges[0] for ptype, ranges in self.port_ranges.items()
        }
        logger.info("Released all allocated ports")
    
    def get_port(self, tool_name: str) -> int | None:
        """
        Get the port allocated to a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Port number, or None if tool has no port
        """
        return self.tool_ports.get(tool_name)
    
    def get_all_ports(self, port_type: str | None = None) -> dict[str, int]:
        """
        Get all port allocations, optionally filtered by type.
        
        Args:
            port_type: Optional port type to filter by (mcp, mgmt, system, metrics, ui)
            
        Returns:
            Dictionary of service name -> port
        """
        if port_type is None:
            return self.tool_ports.copy()
        
        # Filter by port type based on which range they fall into
        result = {}
        type_min, type_max = self._get_range_for_type(port_type)
        for name, port in self.tool_ports.items():
            if type_min <= port <= type_max:
                result[name] = port
        return result
    
    def get_ports_by_type(self, port_type: str) -> dict[str, int]:
        """
        Get all ports for a specific type.
        
        Args:
            port_type: Type of ports to retrieve
            
        Returns:
            Dictionary of service name -> port for the specified type
        """
        return self.get_all_ports(port_type=port_type)
    
    def reserve_system_port(self, name: str, port: int) -> bool:
        """
        Reserve a specific port for a system service.
        
        Args:
            name: Name of the system service
            port: Port number to reserve
            
        Returns:
            True if port was reserved, False if already in use
        """
        if not self._is_port_available(port, PortType.SYSTEM):
            logger.warning(f"Cannot reserve port {port} for {name}: already in use")
            return False
        
        self.allocated_ports.add(port)
        self.tool_ports[name] = port
        self.reserved_ports[name] = port
        logger.info(f"Reserved system port {port} for {name}")
        return True
    
    def get_allocated_ports(self) -> set[int]:
        """
        Get all currently allocated ports.
        
        Returns:
            Set of allocated port numbers
        """
        return self.allocated_ports.copy()
    
    def is_port_in_use(self, port: int) -> bool:
        """
        Check if a port is currently allocated.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is allocated, False otherwise
        """
        return port in self.allocated_ports
    
    def reserve_port(self, port: int, port_type: str = PortType.MCP) -> bool:
        """
        Reserve a port without assigning it to a service.
        
        Args:
            port: Port number to reserve
            port_type: Type of port for range validation
            
        Returns:
            True if port was reserved, False if already in use
        """
        if not self._is_port_available(port, port_type):
            return False
        
        self.allocated_ports.add(port)
        logger.info(f"Reserved port {port} (type: {port_type})")
        return True
    
    def unreserve_port(self, port: int) -> bool:
        """
        Unreserve a previously reserved port.
        
        Args:
            port: Port number to unreserve
            
        Returns:
            True if port was unreserved, False if not reserved
        """
        if port in self.allocated_ports:
            self.allocated_ports.discard(port)
            logger.info(f"Unreserved port {port}")
            return True
        return False
    
    def get_next_available_port(self, port_type: str = PortType.MCP) -> int | None:
        """
        Get the next available port without allocating it.
        
        Args:
            port_type: Type of port to find
            
        Returns:
            Next available port number, or None if none available
        """
        min_port, max_port = self._get_range_for_type(port_type)
        port = self._next_port_by_type.get(port_type, min_port)
        
        while port <= max_port:
            if self._is_port_available(port, port_type):
                return port
            port += 1
        
        return None
    
    def get_port_status(self) -> dict[str, any]:
        """
        Get status information about port allocation.
        
        Returns:
            Dictionary with port allocation status
        """
        # Calculate status per type
        type_status = {}
        for ptype, (min_port, max_port) in self.port_ranges.items():
            total = max_port - min_port + 1
            allocated = sum(1 for p in self.allocated_ports if min_port <= p <= max_port)
            type_status[ptype] = {
                "range": (min_port, max_port),
                "total": total,
                "allocated": allocated,
                "available": total - allocated
            }
        
        return {
            "mode": self.mode,
            "base_port": self.base_port,
            "port_ranges": self.port_ranges,
            "reserved_ports": self.reserved_ports.copy(),
            "allocated_ports": len(self.allocated_ports),
            "tools": self.tool_ports.copy(),
            "by_type": type_status
        }
