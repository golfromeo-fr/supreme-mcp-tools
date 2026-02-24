"""
Port allocation manager for the MCP launcher system.

This module provides functionality to allocate and manage ports
for multiple MCP tools running concurrently.
"""

import logging
import socket
from typing import Dict, List, Optional, Set

from .errors import PortConflictError


logger = logging.getLogger(__name__)


class PortManager:
    """Manage port allocation for MCP tools."""
    
    def __init__(
        self,
        mode: str = "auto",
        base_port: int = 8000,
        port_range: Optional[List[int]] = None,
        manual_ports: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the port manager.
        
        Args:
            mode: Port allocation mode ("auto" or "manual")
            base_port: Starting port for auto allocation
            port_range: Port range for allocation [min, max]
            manual_ports: Dictionary of tool name -> port for manual allocation
        """
        self.mode = mode
        self.base_port = base_port
        self.port_range = port_range or [8000, 9000]
        self.manual_ports = manual_ports or {}
        
        self.allocated_ports: Set[int] = set()
        self.tool_ports: Dict[str, int] = {}
        self.next_port = base_port
    
    def allocate_port(
        self,
        tool_name: str,
        preferred_port: Optional[int] = None
    ) -> int:
        """
        Allocate a port for a tool.
        
        Args:
            tool_name: Name of the tool
            preferred_port: Optional preferred port number
            
        Returns:
            Allocated port number
            
        Raises:
            PortConflictError: If port allocation fails
        """
        # Check if tool already has a port allocated
        if tool_name in self.tool_ports:
            logger.info(f"Tool {tool_name} already has port {self.tool_ports[tool_name]}")
            return self.tool_ports[tool_name]
        
        port = None
        
        # Try preferred port first
        if preferred_port is not None:
            if self._is_port_available(preferred_port):
                port = preferred_port
            else:
                logger.warning(f"Preferred port {preferred_port} not available for {tool_name}")
        
        # Try manual port assignment
        if port is None and self.mode == "manual":
            if tool_name in self.manual_ports:
                manual_port = self.manual_ports[tool_name]
                if self._is_port_available(manual_port):
                    port = manual_port
                else:
                    raise PortConflictError(
                        f"Manual port {manual_port} for {tool_name} is not available",
                        port=manual_port,
                        tool_name=tool_name
                    )
            else:
                logger.warning(f"No manual port configured for {tool_name}, using auto allocation")
        
        # Auto allocate a port
        if port is None:
            port = self._allocate_auto_port()
        
        # Register the port
        self.allocated_ports.add(port)
        self.tool_ports[tool_name] = port
        
        logger.info(f"Allocated port {port} for tool {tool_name}")
        return port
    
    def _allocate_auto_port(self) -> int:
        """
        Automatically allocate a port from the range.
        
        Returns:
            Allocated port number
            
        Raises:
            PortConflictError: If no ports available in range
        """
        min_port, max_port = self.port_range
        
        # Try starting from next_port
        port = self.next_port
        
        # Find next available port
        while port <= max_port:
            if self._is_port_available(port):
                self.next_port = port + 1
                return port
            port += 1
        
        # Wrap around to base port if needed
        port = min_port
        while port < self.next_port:
            if self._is_port_available(port):
                self.next_port = port + 1
                return port
            port += 1
        
        raise PortConflictError(
            f"No available ports in range {self.port_range}"
        )
    
    def _is_port_available(self, port: int) -> bool:
        """
        Check if a port is available for use.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available, False otherwise
        """
        # Check if already allocated
        if port in self.allocated_ports:
            return False
        
        # Check if port is in range
        min_port, max_port = self.port_range
        if not (min_port <= port <= max_port):
            return False
        
        # Check if port is actually available on the system
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return True
        except (OSError, socket.error):
            return False
    
    def release_port(self, tool_name: str) -> Optional[int]:
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
        
        logger.info(f"Released port {port} for tool {tool_name}")
        return port
    
    def release_all_ports(self) -> None:
        """Release all allocated ports."""
        self.allocated_ports.clear()
        self.tool_ports.clear()
        self.next_port = self.base_port
        logger.info("Released all allocated ports")
    
    def get_port(self, tool_name: str) -> Optional[int]:
        """
        Get the port allocated to a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Port number, or None if tool has no port
        """
        return self.tool_ports.get(tool_name)
    
    def get_all_ports(self) -> Dict[str, int]:
        """
        Get all tool port allocations.
        
        Returns:
            Dictionary of tool name -> port
        """
        return self.tool_ports.copy()
    
    def get_allocated_ports(self) -> Set[int]:
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
    
    def reserve_port(self, port: int) -> bool:
        """
        Reserve a port without assigning it to a tool.
        
        Args:
            port: Port number to reserve
            
        Returns:
            True if port was reserved, False if already in use
        """
        if not self._is_port_available(port):
            return False
        
        self.allocated_ports.add(port)
        logger.info(f"Reserved port {port}")
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
    
    def get_next_available_port(self) -> Optional[int]:
        """
        Get the next available port without allocating it.
        
        Returns:
            Next available port number, or None if none available
        """
        min_port, max_port = self.port_range
        port = self.next_port
        
        while port <= max_port:
            if self._is_port_available(port):
                return port
            port += 1
        
        return None
    
    def get_port_status(self) -> Dict[str, any]:
        """
        Get status information about port allocation.
        
        Returns:
            Dictionary with port allocation status
        """
        min_port, max_port = self.port_range
        total_ports = max_port - min_port + 1
        available_ports = total_ports - len(self.allocated_ports)
        
        return {
            "mode": self.mode,
            "base_port": self.base_port,
            "port_range": self.port_range,
            "allocated_ports": len(self.allocated_ports),
            "available_ports": available_ports,
            "total_ports": total_ports,
            "tools": self.tool_ports.copy()
        }
