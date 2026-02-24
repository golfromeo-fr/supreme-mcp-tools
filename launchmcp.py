#!/usr/bin/env python3
"""
MCP Launcher - Main Entry Point

Launch multiple MCP tools in a single Python process.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from launcher import (
    Config,
    PortManager,
    ServerManager,
    ServerInstance,
    ToolDiscovery,
    ToolMetadata,
    ServerStartupError,
    DiscoveryError,
    PortConflictError,
    LauncherError,
)


# Configure logging
import logging.handlers


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "tool_name"):
            log_data["tool_name"] = record.tool_name
        if hasattr(record, "port"):
            log_data["port"] = record.port
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(config: Config, verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration object
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else getattr(logging, config.get_log_level().upper(), logging.INFO)
    log_format = config.get_log_format()
    log_file = config.get_log_file()
    
    # Check if JSON logging is enabled
    use_json = log_format.lower() == "json"
    
    # Create formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(log_format)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set component-specific log levels from config
    component_levels = config.get("componentLogLevels", {})
    for component, level in component_levels.items():
        comp_logger = logging.getLogger(component)
        comp_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Launch multiple MCP tools in a single process",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launchmcp.py webmcp oraclemcp
  python launchmcp.py --config custom_config.json webmcp
  python launchmcp.py --list-tools
  python launchmcp.py --verbose webmcp oraclemcp simplemcp8
        """
    )
    
    parser.add_argument(
        "tools",
        nargs="*",
        help="Names of tools to launch (if not specified, launches all discovered tools)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available MCP tools and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without actually starting servers"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override server host address"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        help="Override log level"
    )
    
    return parser.parse_args()


def list_tools(discovery: ToolDiscovery) -> None:
    """
    List all discovered MCP tools.
    
    Args:
        discovery: Tool discovery instance
    """
    tools = discovery.get_all_tools()
    
    if not tools:
        print("No MCP tools found.")
        return
    
    print("Available MCP Tools:")
    print("-" * 60)
    
    for tool in tools:
        print(f"Name:        {tool.name}")
        print(f"Path:        {tool.file_path}")
        print(f"Version:     {tool.version}")
        print(f"Description: {tool.description}")
        if tool.dependencies:
            print(f"Dependencies: {', '.join(tool.dependencies)}")
        print("-" * 60)


# Server startup timeout in seconds
SERVER_STARTUP_TIMEOUT = 30


async def start_servers(
    tools: List[ToolMetadata],
    port_manager: PortManager,
    server_manager: ServerManager,
    config: Config
) -> List[ToolMetadata]:
    """
    Start all MCP tool servers.
    
    Args:
        tools: List of tool metadata
        port_manager: Port manager instance
        server_manager: Server manager instance
        config: Configuration object
        
    Returns:
        List of successfully started tools
    """
    # Allocate ports for all tools
    ports = {}
    allocated_tools = []  # Track tools that got ports for cleanup
    
    for tool in tools:
        try:
            port = port_manager.allocate_port(tool.name)
            ports[tool.name] = port
            allocated_tools.append(tool.name)
        except PortConflictError as e:
            logging.error(f"Port allocation failed for {tool.name}: {e}")
            if config.get_continue_on_error():
                continue
            else:
                raise
    
    if not ports:
        raise LauncherError("No ports allocated for any tools")
    
    logging.info(f"Allocated ports: {ports}")
    
    # Start servers concurrently
    logging.info(f"Starting {len(ports)} servers...")
    
    async def start_single_server(tool: ToolMetadata) -> Optional[ServerInstance]:
        """Start a single server with timeout."""
        port = ports.get(tool.name)
        if port is None:
            logging.error(f"No port allocated for {tool.name}")
            return None
        
        try:
            # Apply startup timeout to prevent indefinite hangs
            instance = await asyncio.wait_for(
                server_manager.start_server(tool, port),
                timeout=SERVER_STARTUP_TIMEOUT
            )
            logging.info(f"Server for {tool.name} started on port {port}")
            return instance
        except asyncio.TimeoutError:
            logging.error(f"Server startup timeout for {tool.name} on port {port} "
                        f"(timeout: {SERVER_STARTUP_TIMEOUT}s)")
            if not config.get_continue_on_error():
                raise LauncherError(f"Server startup timeout for {tool.name}")
            return None
        except ServerStartupError as e:
            logging.error(f"Failed to start server for {tool.name}: {e}")
            if not config.get_continue_on_error():
                raise
            return None
    
    # Start all servers concurrently
    start_tasks = [start_single_server(tool) for tool in tools if tool.name in ports]
    results = await asyncio.gather(*start_tasks, return_exceptions=True)
    
    # Check results
    successful = 0
    failed_tools = []
    started_tools = []  # Track successfully started tools
    # Get tools that have allocated ports for accurate reporting
    tools_with_ports = [tool for tool in tools if tool.name in ports]
    for i, result in enumerate(results):
        tool = tools_with_ports[i] if i < len(tools_with_ports) else None
        tool_name = tool.name if tool else f"tool_{i}"
        if isinstance(result, Exception):
            logging.error(f"Server startup error for {tool_name}: {result}")
            failed_tools.append(tool_name)
        elif result is not None:
            successful += 1
            if tool:
                started_tools.append(tool)
        else:
            failed_tools.append(tool_name)
    
    logging.info(f"Successfully started {successful}/{len(tools)} servers")
    
    if successful == 0:
        # Cleanup allocated ports on complete failure
        for tool_name in allocated_tools:
            try:
                port_manager.release_port(tool_name)
                logging.debug(f"Released port for {tool_name} after startup failure")
            except Exception as e:
                logging.warning(f"Failed to release port for {tool_name}: {e}")
        raise LauncherError("No servers started successfully")
    
    return started_tools


async def monitor_servers(server_manager: ServerManager, shutdown_event: asyncio.Event) -> None:
    """
    Monitor running servers until shutdown.
    
    Args:
        server_manager: Server manager instance
        shutdown_event: Event to signal shutdown
    """
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(30)
            
            statuses = server_manager.get_all_statuses()
            running = sum(1 for s in statuses.values() if s == "running")
            errors = sum(1 for s in statuses.values() if s == "error")
            
            if errors > 0:
                error_servers = server_manager.get_error_servers()
                logging.warning(f"Servers with errors: {error_servers}")
            
            logging.debug(f"Server status: {running} running, {errors} errors")
        except asyncio.CancelledError:
            # Clean exit on cancellation
            logging.debug("Monitor task cancelled, exiting")
            raise
        except Exception as e:
            # Log error but continue monitoring
            logging.error(f"Monitor error: {e}", exc_info=True)


# Module-level global for shutdown state
_shutdown_in_progress = False


async def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config_path = args.config or str(Path(__file__).parent / "config.json")
    config = Config(config_path)
    
    # Override config with CLI arguments
    if args.host:
        config.config["server"]["host"] = args.host
    if args.log_level:
        config.config["server"]["logLevel"] = args.log_level
    
    # Set up logging
    setup_logging(config, args.verbose)
    
    logging.info("=" * 60)
    logging.info("MCP Launcher Starting")
    logging.info("=" * 60)
    
    # Get all configured tool directories
    all_tool_dirs = config.get_tool_directories()
    
    # If specific tools are requested, only search their directories
    # Otherwise, search all configured directories
    tool_names = args.tools if args.tools else None
    if tool_names:
        # Map tool names (with possible underscores) to directory names
        # e.g., "web_mcp" -> "webmcp", "ragmcp" -> "ragmcp"
        search_dirs = []
        for tool_name in tool_names:
            # Try both with underscore and without
            tool_dir_name = tool_name.replace("_", "")  # web_mcp -> webmcp
            for dir_path in all_tool_dirs:
                dir_name = Path(dir_path).name  # e.g., "webmcp"
                if dir_name.lower() == tool_dir_name.lower():
                    search_dirs.append(dir_path)
                    break
        # If no matching directory found, use the original directories
        # (discovery will handle the error gracefully)
        if not search_dirs:
            search_dirs = all_tool_dirs
        logging.info(f"Searching only in directories for requested tools: {search_dirs}")
    else:
        search_dirs = all_tool_dirs
    
    # Initialize tool discovery with filtered directories
    discovery = ToolDiscovery(search_dirs)
    
    # Discover tools
    tool_names = args.tools if args.tools else None
    tools = discovery.discover_tools(tool_names=tool_names)
    
    if not tools:
        logging.error("No MCP tools found")
        return 1
    
    logging.info(f"Discovered {len(tools)} MCP tools: {[t.name for t in tools]}")
    
    # List tools and exit if requested
    if args.list_tools:
        list_tools(discovery)
        return 0
    
    # Dry run mode
    if args.dry_run:
        logging.info("Dry run mode - not starting servers")
        logging.info(f"Would launch: {[t.name for t in tools]}")
        return 0
    
    # Initialize port manager
    port_manager = PortManager(
        mode=config.get_port_mode(),
        base_port=config.get_base_port(),
        port_range=config.get_port_range(),
        manual_ports=config.get_manual_ports()
    )
    
    # Initialize server manager
    server_manager = ServerManager(
        host=config.get_server_host(),
        log_level=config.get_server_log_level()
    )
    
    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        global _shutdown_in_progress
        if _shutdown_in_progress:
            logging.debug(f"Already shutting down, ignoring signal {signum}")
            return
        _shutdown_in_progress = True
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start servers
    started_tools = []  # Track successfully started tools
    try:
        started_tools = await start_servers(tools, port_manager, server_manager, config)
        
        # Print server information - only for successfully started servers
        print("\n" + "=" * 60)
        print("MCP Launcher Running")
        print("=" * 60)
        for tool in tools:
            port = port_manager.get_port(tool.name)
            if port:
                # Check if server actually started
                status = server_manager.get_server_status(tool.name)
                if status == "running":
                    print(f"  {tool.name}: http://{config.get_server_host()}:{port}")
                else:
                    print(f"  {tool.name}: FAILED (status: {status})")
        print("=" * 60)
        print("Press Ctrl+C to stop all servers\n")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_servers(server_manager, shutdown_event))
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel monitor task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
    except LauncherError as e:
        logging.error(f"Launcher error: {e}")
        return 1
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return 1
    
    # Graceful shutdown
    logging.info("Stopping all servers...")
    await server_manager.stop_all_servers()
    port_manager.release_all_ports()
    
    # Print summary
    summary = server_manager.get_summary()
    logging.info(f"Shutdown summary: {summary}")
    
    logging.info("MCP Launcher stopped")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)
