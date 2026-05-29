# Shared utilities for FastMCP tools
"""
Shared utilities and base classes for FastMCP tools to eliminate duplication.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from tools.shared.server_factory import DEFAULT_HOST

try:
    from launcher.launcher_config import load_ports_config
    LAUNCHER_AVAILABLE = True
except ImportError:
    LAUNCHER_AVAILABLE = False

try:
    from tools.fef_integration import (
        ToolExtensionManager,
        register_common_extensions,
        setup_tool_extensions
    )
    from launcher.tool_extensions import Extension, ExtensionType
    FEF_V3_AVAILABLE = True
except ImportError:
    FEF_V3_AVAILABLE = False


class FastMCPBase:
    """Base class for FastMCP tools with common functionality."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.mcp_port: int | None = None
        self.mgmt_port: int | None = None
        self.logger = logging.getLogger(tool_name)
        self.fef_manager = None
        self.fef_registry = None
        self.fef_http_server = None
        self.fef_setup_done = False
        self.mcp_instance: FastMCP | None = None
        
        # Initialize logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
    
    def load_ports_config(self) -> None:
        """Load port configuration from launcher config."""
        if not LAUNCHER_AVAILABLE:
            raise ImportError("Launcher config not available")
            
        try:
            ports_config = load_ports_config()
            self.mcp_port = int(os.environ.get(
                "MCP_PORT",
                ports_config["assignments"]["mcp"][self.tool_name]
            ))
            self.mgmt_port = int(os.environ.get(
                "MCP_MGMT_PORT",
                ports_config["assignments"]["mgmt"][self.tool_name]
            ))
        except Exception as e:
            self.logger.error(f"Failed to load ports.json: {e}")
            raise
    
    def create_mcp_instance(self) -> FastMCP:
        """Create and configure FastMCP instance."""
        if not self.mcp_port:
            self.load_ports_config()
            
        self.mcp_instance = FastMCP(
            self.tool_name,
        )
        return self.mcp_instance
    
    def setup_fef_extensions(self, registry=None, custom_extensions=None) -> None:
        """Setup FEF V3 extensions."""
        if self.fef_setup_done:
            return
            
        if not FEF_V3_AVAILABLE:
            self.fef_setup_done = True
            return
            
        mgmt_port = int(os.environ.get("MCP_MGMT_PORT", self.mgmt_port or 0))
        
        if registry is not None:
            # Use launcher's registry
            self.fef_registry = registry
            self.fef_manager = ToolExtensionManager(self.tool_name)
            register_common_extensions(self.tool_name, self.fef_registry, self.fef_manager)
            if custom_extensions:
                for ext in custom_extensions:
                    self.fef_registry.register(self.tool_name, ext)
            self.fef_http_server = None
            self.logger.info(f"[{self.tool_name}] FEF V3 registered with launcher's registry")
        else:
            # Standalone mode
            if custom_extensions:
                self.fef_manager, self.fef_registry, self.fef_http_server = setup_tool_extensions(
                    tool_name=self.tool_name,
                    mgmt_port=mgmt_port,
                    custom_extensions=custom_extensions
                )
                self.logger.info(f"[{self.tool_name}] FEF V3 standalone mode on port {mgmt_port}")
            else:
                self.fef_manager, self.fef_registry, self.fef_http_server = setup_tool_extensions(
                    tool_name=self.tool_name,
                    mgmt_port=mgmt_port
                )
                self.logger.info(f"[{self.tool_name}] FEF V3 standalone mode on port {mgmt_port}")
        
        self.fef_setup_done = True
    
    @asynccontextmanager
    async def lifespan(self, app):
        """Lifespan context manager for startup/shutdown."""
        self.logger.info(f"{self.tool_name} FastMCP server starting on port {self.mcp_port}...")
        
        # Setup FEF V3 if not done by launcher
        if not self.fef_setup_done:
            self.setup_fef_extensions()
        
        # Start FEF V3 management server if standalone
        if FEF_V3_AVAILABLE and self.fef_http_server:
            try:
                await self.fef_http_server.start()
                self.logger.info("FEF V3 management server started")
            except Exception as e:
                self.logger.warning(f"Failed to start FEF V3 management server: {e}")
        
        yield
        
        self.logger.info(f"{self.tool_name} FastMCP server shutting down...")
        if self.fef_http_server:
            try:
                await self.fef_http_server.stop()
            except Exception:
                pass
    
    def get_app(self):
        """Get the FastMCP app for deployment."""
        if not self.mcp_instance:
            self.create_mcp_instance()
        return self.mcp_instance.http_app()
    
    def run(self):
        """Run the server."""
        if not self.mcp_instance:
            self.create_mcp_instance()
        
        import uvicorn
        
        self.logger.info(f"Starting {self.tool_name} FastMCP server")
        self.logger.info(f"  MCP port: {self.mcp_port}")
        self.logger.info(f"  SSE endpoint: http://{DEFAULT_HOST}:{self.mcp_port}/sse")
        self.logger.info(f"  Streamable HTTP: http://{DEFAULT_HOST}:{self.mcp_port}/mcp")
        if FEF_V3_AVAILABLE:
            self.logger.info(f"  FEF V3 mgmt: http://{DEFAULT_HOST}:{self.mgmt_port}")
        
        uvicorn.run(
            self.get_app(),
            host=DEFAULT_HOST,
            port=self.mcp_port,
            log_level="info",
            lifespan="on",
        )


def setup_common_logging(level: int = logging.INFO) -> None:
    """Setup common logging configuration."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


def get_tool_logger(tool_name: str) -> logging.Logger:
    """Get a logger for a specific tool."""
    return logging.getLogger(tool_name)


def _is_internal_ip(ip) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved


def _check_ipv4_mapped(ip) -> bool:
    if hasattr(ip, 'ipv4_mapped') and ip.ipv4_mapped:
        return _is_internal_ip(ip.ipv4_mapped)
    return False


def is_internal_url(url: str) -> bool:
    """Check if URL is internal (SSRF protection).

    Handles hex-encoded IPv4 (0x7f000001), decimal-encoded (2130706433),
    IPv6-mapped IPv4 (::ffff:127.0.0.1), and plain IPv4/IPv6 literals
    before falling back to DNS resolution for hostnames.
    """
    try:
        import ipaddress
        import re
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()

        if not hostname:
            return True

        if hostname in {'localhost', '::1', '0.0.0.0'}:
            return True

        internal_hosts = {
            '169.254.169.254',
            'metadata.google.internal',
            'metadata.azure.internal',
            'metadata.internal',
        }
        if hostname in internal_hosts:
            return True

        hex_match = re.match(r'^0x([0-9a-f]+)$', hostname)
        if hex_match:
            try:
                ip_int = int(hex_match.group(1), 16)
                ip = ipaddress.IPv4Address(ip_int)
                if _is_internal_ip(ip):
                    return True
                return False
            except (ValueError, OverflowError):
                return True

        if hostname.isdigit():
            try:
                ip_int = int(hostname)
                ip = ipaddress.IPv4Address(ip_int)
                if _is_internal_ip(ip):
                    return True
                return False
            except (ValueError, OverflowError):
                return True

        try:
            ip = ipaddress.ip_address(hostname)
            if _is_internal_ip(ip):
                return True
            if _check_ipv4_mapped(ip):
                return True
            return False
        except ValueError:
            pass

        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except (socket.gaierror, OSError):
            return True

        for family, _, _, _, sockaddr in resolved:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                if _is_internal_ip(ip):
                    return True
                if _check_ipv4_mapped(ip):
                    return True
            except ValueError:
                continue

        return False
    except Exception:
        return True
