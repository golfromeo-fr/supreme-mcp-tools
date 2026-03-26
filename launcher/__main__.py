#!/usr/bin/env python3
"""
MCP Launcher CLI

Command-line interface for the MCP Launcher with FEF V3 support.

Usage:
    python -m launcher [OPTIONS]

Examples:
    # Start management server only
    python -m launcher --management-port 8200

    # Start with specific tools
    python -m launcher --tools webmcp,simplemcp

    # Start with custom config
    python -m launcher --config /path/to/config.json

    # Start with debug logging
    python -m launcher --debug
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .launcher_config import Config
from .server_manager import ServerManager
from .service_registry import ServiceRegistry
from .management_server import ManagementServer
from .port_manager import PortManager, PortType
from .tool_discovery import ToolDiscovery


logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Launcher - Start and manage MCP tools with FEF V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --management-port 8200
  %(prog)s --tools webmcp,simplemcp
  %(prog)s --config /path/to/config.json
  %(prog)s --debug
        """
    )
    
    parser.add_argument(
        "--management-port",
        type=int,
        default=None,
        help="Port for the management server (default: from ports.json)"
    )
    
    parser.add_argument(
        "--management-host",
        type=str,
        default="0.0.0.0",
        help="Host for the management server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--tools",
        type=str,
        help="Comma-separated list of tools to start (default: all discovered)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--tools-dir",
        type=str,
        default="tools",
        help="Directory containing tool modules (default: tools)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for tool servers (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--no-management",
        action="store_true",
        help="Disable management server"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for management server authentication"
    )
    
    return parser.parse_args(args)


class Launcher:
    """Main launcher class."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the launcher.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.running = False
        
        # Load configuration
        if args.config:
            self.config = Config(config_path=args.config)
        else:
            self.config = Config()
        
        # Initialize components
        self.service_registry = ServiceRegistry()
        
        # Build ports_config from self.config (following the pattern in launchmcp.py)
        port_alloc = self.config.config.get("portAllocation", {})
        ports_config = {
            "ranges": port_alloc.get("ranges", {}),
            "reserved": port_alloc.get("reservedPorts", {}),
            "assignments": port_alloc.get("manualPorts", {})
        }
        
        self.port_manager = PortManager(
            ports_config=ports_config,
            mode=self.config.get_port_mode(),
            base_port=self.config.get_base_port(),
        )
        self.server_manager = ServerManager(
            host=args.host,
            service_registry=self.service_registry,
            enable_management=not args.no_management
        )
        
        # Management server
        self.management_server: Optional[ManagementServer] = None
        
        # Tool discovery
        self.tool_discovery = ToolDiscovery(
            tools_dir=Path(args.tools_dir)
        )
        
        # Shutdown event
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the launcher."""
        logger.info("Starting MCP Launcher with FEF V3")
        
        # Start service registry
        await self.service_registry.start()
        logger.info("Service registry started")
        
        # Start management server
        if not self.args.no_management:
            # Get port from CLI arg, config, or ports.json
            mgmt_port = self.args.management_port
            if mgmt_port is None:
                try:
                    mgmt_port = self.config.get_reserved_ports().get("central_management")
                    if mgmt_port is None:
                        raise ValueError("central_management port not found in ports.json")
                except Exception as e:
                    logger.error(f"Could not get management port: {e}")
                    raise
            
            self.management_server = ManagementServer(
                service_registry=self.service_registry,
                port=mgmt_port,
                host=self.args.management_host,
                api_key=self.args.api_key
            )
            await self.management_server.start()
            logger.info(
                f"Management server started on "
                f"{self.args.management_host}:{mgmt_port}"
            )
        
        # Discover tools
        tools = self.tool_discovery.discover()
        logger.info(f"Discovered {len(tools)} tools")
        
        # Filter tools if specified
        if self.args.tools:
            tool_names = [t.strip() for t in self.args.tools.split(",")]
            tools = {name: meta for name, meta in tools.items() if name in tool_names}
            logger.info(f"Filtered to {len(tools)} tools: {list(tools.keys())}")
        
        # Start tools
        for tool_name, tool_metadata in tools.items():
            try:
                # Allocate ports
                mcp_port = self.port_manager.allocate_port(tool_name, port_type=PortType.MCP)
                mgmt_port = self.port_manager.allocate_port(f"{tool_name}_mgmt", port_type=PortType.MANAGEMENT) if not self.args.no_management else None
                
                # Start server
                await self.server_manager.start_server(
                    tool_metadata=tool_metadata,
                    port=mcp_port,
                    mgmt_port=mgmt_port
                )
                
                logger.info(
                    f"Started {tool_name}: MCP={mcp_port}, "
                    f"Management={mgmt_port or 'disabled'}"
                )
            except Exception as e:
                logger.error(f"Failed to start {tool_name}: {e}")
        
        self.running = True
        logger.info("Launcher started successfully")
        
        # Print summary
        self._print_summary()
    
    async def stop(self) -> None:
        """Stop the launcher."""
        logger.info("Stopping MCP Launcher")
        
        self.running = False
        
        # Stop all tool servers
        await self.server_manager.stop_all_servers()
        logger.info("All tool servers stopped")
        
        # Stop management server
        if self.management_server:
            await self.management_server.stop()
            logger.info("Management server stopped")
        
        # Stop service registry
        await self.service_registry.stop()
        logger.info("Service registry stopped")
        
        logger.info("Launcher stopped")
    
    async def run(self) -> None:
        """Run the launcher until shutdown."""
        await self.start()
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        await self.stop()
    
    def shutdown(self) -> None:
        """Trigger shutdown."""
        self._shutdown_event.set()
    
    def _print_summary(self) -> None:
        """Print startup summary."""
        print("\n" + "=" * 60)
        print("MCP Launcher with FEF V3")
        print("=" * 60)
        
        if not self.args.no_management:
            print(f"\nManagement Server: http://{self.args.management_host}:{self.management_server.port}")
            print(f"  Health: http://{self.args.management_host}:{self.management_server.port}/health")
            print(f"  Tools:  http://{self.args.management_host}:{self.management_server.port}/api/tools")
        
        instances = self.server_manager.get_all_instances()
        if instances:
            print(f"\nRunning Tools ({len(instances)}):")
            for name, instance in instances.items():
                mgmt_info = f"Management: {instance.mgmt_port}" if instance.mgmt_port else "Management: disabled"
                print(f"  - {name}: MCP={instance.port}, {mgmt_info}")
        
        print("\nPress Ctrl+C to stop")
        print("=" * 60 + "\n")


async def main(args: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parsed_args = parse_args(args)
    
    # Setup logging
    setup_logging(debug=parsed_args.debug)
    
    # Create launcher
    launcher = Launcher(parsed_args)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        launcher.shutdown()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    # Run launcher
    try:
        await launcher.run()
    except Exception as e:
        logger.error(f"Launcher error: {e}", exc_info=True)
        sys.exit(1)


def cli_main() -> None:
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
