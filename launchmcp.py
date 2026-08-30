#!/usr/bin/env python3
"""
MCP Launcher - Main Entry Point

Launch multiple MCP tools in a single Python process.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
import uvicorn

from launcher import (
    Config,
    PortManager,
    ServerManager,
    ServerInstance,
    ToolDiscovery,
    ToolMetadata,
    ServerStartupError,
    PortConflictError,
    LauncherError,
)
from launcher.management_server import ManagementServer
from launcher.service_registry import ServiceRegistry
from launcher.config_types import DEFAULT_HOST, make_file_handler

# Import monitoring modules
from monitoring.config import load_monitoring_config, MonitoringConfig
from monitoring.collector import MetricsRegistry
from monitoring.exporters import create_metrics_app


# Configure logging
import logging.handlers


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone
        
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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
    log_level = logging.DEBUG if verbose else getattr(logging, config.logging_config.level.upper(), logging.INFO)
    log_format = config.logging_config.format
    log_file = config.logging_config.file
    
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
    
    # File handler with rotation (time-based, config-driven).
    # make_file_handler builds a TimedRotatingFileHandler so logs roll over on
    # a calendar schedule (default monthly) rather than only on size. This
    # prevents launcher.log growing unbounded over months/years.
    file_handler = make_file_handler(config.logging_config)
    if file_handler is not None:
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
  python launchmcp.py --verbose webmcp oraclemcp simplemcp
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
    
    parser.add_argument(
        "--no-management",
        action="store_true",
        help="Disable the centralized management API server (enabled by default, port from ports.json)"
    )
    
    parser.add_argument(
        "--management-port",
        type=int,
        default=None,
        help="Port for the management server (default: from ports.json)"
    )

    parser.add_argument(
        "--no-sync-tools",
        action="store_true",
        help="Disable automatic tools config sync after server startup"
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=["streamable-http", "sse"],
        default=None,
        help="Transport protocol for all tools (default: streamable-http; "
             "sse is legacy compatibility for outdated harnesses)"
    )

    parser.add_argument(
        "--health-check-logs",
        type=str,
        choices=["enable", "disable", "errors-only"],
        default="enable",
        help="Health check logging mode (default: enable)"
    )

    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=None,
        help="Health check interval in seconds (default: 30)"
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
    tools: list[ToolMetadata],
    port_manager: PortManager,
    server_manager: ServerManager,
    config: Config,
    service_registry: ServiceRegistry | None = None
) -> list[ToolMetadata]:
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
    # Wait out busy manual ports together (restart race): one parallel 20s
    # window for all tools instead of 20s per tool.
    manual_map = {t.name: port_manager.get_manual_port(t.name) for t in tools}
    still_busy = port_manager.wait_for_busy_ports(
        {n: p for n, p in manual_map.items() if p is not None})
    for name in sorted(still_busy):
        logging.error(f"Port allocation failed for {name}: manual port still busy after retry window")
    if still_busy and not config.error_handling_config.continue_on_error:
        raise LauncherError(f"Ports still busy after retry window: {sorted(still_busy)}")

    # Allocate ports for all tools
    ports = {}
    allocated_tools = []  # Track tools that got ports for cleanup

    for tool in tools:
        if tool.name in still_busy:
            continue
        try:
            port = port_manager.allocate_port(tool.name)
            ports[tool.name] = port
            allocated_tools.append(tool.name)
        except PortConflictError as e:
            logging.error(f"Port allocation failed for {tool.name}: {e}")
            if config.error_handling_config.continue_on_error:
                continue
            else:
                raise
    
    if not ports:
        raise LauncherError("No ports allocated for any tools")
    
    logging.info(f"Allocated ports: {ports}")
    
    # Start servers concurrently
    logging.info(f"Starting {len(ports)} servers...")
    
    async def start_single_server(tool: ToolMetadata) -> ServerInstance | None:
        """Start a single server with timeout."""
        port = ports.get(tool.name)
        if port is None:
            logging.error(f"No port allocated for {tool.name}")
            return None
        
        # Allocate management port if service registry is available
        # Use port_type="mgmt" to allocate from the management port range (8100-8199)
        from launcher.port_manager import PortType
        mgmt_port = port_manager.allocate_port(f"{tool.name}_mgmt", port_type=PortType.MANAGEMENT) if service_registry else None
        
        try:
            # Apply startup timeout to prevent indefinite hangs
            instance = await asyncio.wait_for(
                server_manager.start_server(tool, port, mgmt_port),
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
    failed_startups = []  # Track tools that failed to start so we can release their ports
    # Get tools that have allocated ports for accurate reporting
    tools_with_ports = [tool for tool in tools if tool.name in ports]
    for i, result in enumerate(results):
        tool = tools_with_ports[i] if i < len(tools_with_ports) else None
        tool_name = tool.name if tool else f"tool_{i}"
        if isinstance(result, Exception):
            logging.error(f"Server startup error for {tool_name}: {result}")
            failed_tools.append(tool_name)
            failed_startups.append(tool_name)
        elif result is not None:
            successful += 1
            if tool:
                started_tools.append(tool)
        else:
            failed_tools.append(tool_name)
            failed_startups.append(tool_name)

    logging.info(f"Successfully started {successful}/{len(tools)} servers")

    if successful == 0:
        # Cleanup all allocated ports on complete failure
        for tool_name in allocated_tools:
            try:
                port_manager.release_port(tool_name)
                logging.debug(f"Released port for {tool_name} after startup failure")
            except Exception as e:
                logging.warning(f"Failed to release port for {tool_name}: {e}")
        raise LauncherError("No servers started successfully")

    # Release ports for tools that failed to start (even if some succeeded)
    if failed_startups:
        for tool_name in failed_startups:
            try:
                port_manager.release_port(tool_name)
                logging.debug(f"Released port for failed tool {tool_name}")
            except Exception as e:
                logging.warning(f"Failed to release port for {tool_name}: {e}")

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


# Global variables for monitoring
_metrics_server_task: asyncio.Task | None = None
_monitoring_config: MonitoringConfig | None = None


def get_monitoring_config_path() -> Path | None:
    """
    Get the path to the monitoring configuration file.
    
    Returns:
        Path to monitoring config file or None if not found
    """
    # Get the workspace directory (parent of supreme-mcp-tools)
    workspace_dir = Path(__file__).parent
    
    # Check workspace config directory first
    config_path = workspace_dir / "config" / "monitoring_config.json"
    if config_path.exists():
        return config_path
    
    # Try relative to current working directory
    config_path = Path("config/monitoring_config.json")
    if config_path.exists():
        return config_path
    
    return None


def initialize_monitoring() -> bool:
    """
    Initialize the monitoring system.
    
    Returns:
        True if monitoring was initialized successfully, False otherwise
    """
    global _monitoring_config
    
    try:
        # Get monitoring config path
        config_path = get_monitoring_config_path()
        
        if config_path:
            logging.info(f"Loading monitoring configuration from: {config_path}")
            _monitoring_config = load_monitoring_config(config_path)
        else:
            logging.info("No monitoring configuration file found, using defaults")
            _monitoring_config = MonitoringConfig()
        
        # Check if monitoring is enabled
        if not _monitoring_config.enabled:
            logging.info("Monitoring is disabled in configuration")
            return False
        
        # Initialize the metrics registry
        registry = MetricsRegistry.get_instance()
        registry.enable()
        
        # Get or create the default collector
        collector = registry.get_default_collector()
        
        logging.info(
            f"Monitoring initialized: collectors enabled, "
            f"storage backend: {_monitoring_config.metrics.get('storage_backend', 'prometheus')}"
        )
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize monitoring: {e}", exc_info=True)
        return False


async def start_metrics_server(shutdown_event: asyncio.Event) -> None:
    """
    Start the metrics server in the background.
    
    Args:
        shutdown_event: Event to signal shutdown
    """
    global _monitoring_config
    
    if _monitoring_config is None or not _monitoring_config.enabled:
        return
    
    try:
        # Get port from ports.json (metrics_server) or config
        port = None
        
        # First try ports.json
        try:
            from launcher.launcher_config import load_ports_config
            ports_config = load_ports_config()
            port = ports_config.get("reserved", {}).get("metrics_server")
        except Exception:
            pass
        
        # Then try monitoring config exporters
        exporters_config = _monitoring_config.metrics.get("exporters", [])
        for exporter in exporters_config:
            if exporter.get("type") == "prometheus" and exporter.get("enabled", False):
                port = exporter.get("config", {}).get("port", port)
                break
        
        if port is None:
            logging.warning("Metrics server port not configured, skipping metrics server")
            return
        
        # Get or create the default collector
        registry = MetricsRegistry.get_instance()
        collector = registry.get_default_collector()
        
        # Create the FastAPI app
        app = create_metrics_app(collector, app_name="mcp-monitoring")
        
        logging.info(f"Starting metrics server on port {port}")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=DEFAULT_HOST,
            port=port,
            log_level="info",
            access_log=False
        )
        
        server = uvicorn.Server(config)
        
        # Run the server
        await server.serve()
        
    except Exception as e:
        logging.error(f"Metrics server error: {e}", exc_info=True)
    finally:
        logging.info("Metrics server stopped")


def _get_management_port_from_config() -> int:
    """Get the management port from ports.json."""
    try:
        from launcher.launcher_config import load_ports_config
        ports_config = load_ports_config()
        return ports_config.get("reserved", {}).get("central_management")
    except Exception:
        return None


async def start_management_api_server(
    shutdown_event: asyncio.Event,
    port: int = None,
    service_registry: ServiceRegistry | None = None
) -> None:
    """
    Start the centralized management API server.
    
    Args:
        shutdown_event: Event to signal shutdown
        port: Port for the management server (default: from ports.json)
        service_registry: Optional existing service registry to use
    """
    if port is None:
        port = _get_management_port_from_config()
    if port is None:
        raise ValueError(
            "Management port not configured. Create config/ports.json with "
            "reserved.central_management port or pass --management-port."
        )
    global _service_registry
    
    try:
        # Use existing registry or create new one
        if service_registry is None:
            service_registry = ServiceRegistry()
            await service_registry.start()
            logging.info("Service registry started")
        else:
            logging.info("Using existing service registry")
        
        _service_registry = service_registry
        
        # Create management server
        management_server = ManagementServer(
            service_registry=service_registry,
            port=port,
            host=DEFAULT_HOST
        )
        
        logging.info(f"Starting management API server on port {port}")
        
        # Start management server
        config = uvicorn.Config(
            management_server.app,
            host=DEFAULT_HOST,
            port=port,
            log_level="info",
            access_log=os.environ.get("MCP_HEALTH_CHECK_LOGS", "enable") != "disable"
        )
        
        server = uvicorn.Server(config)
        
        # Run the server
        await server.serve()
        
    except Exception as e:
        logging.error(f"Management API server error: {e}", exc_info=True)
    finally:
        if _service_registry and service_registry is None:
            # Only stop if we created it
            await _service_registry.stop()
        logging.info("Management API server stopped")


# Module-level global for shutdown state
_shutdown_in_progress = False

# Management server task
_management_server_task: asyncio.Task | None = None
_service_registry: ServiceRegistry | None = None


async def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config_path = args.config or str(Path(__file__).parent / "config" / "launcher_config.json")
    config = Config(config_path)
    
    # Override config with CLI arguments
    if args.host:
        config.config["server"]["host"] = args.host
    if args.log_level:
        config.config["server"]["logLevel"] = args.log_level
    
    # Set up logging
    setup_logging(config, args.verbose)
    
    # Set transport protocol for all tools (must be set before tool modules are
    # imported). Default is unset -> tools build the versatile multi-transport
    # app (/mcp, /mcp-stateless, /sse); an explicit value pins all tools.
    transport = args.transport or config.config.get("transport") or os.environ.get("MCP_TRANSPORT")
    if transport:
        if transport not in ("multi", "streamable-http", "streamable-stateless", "sse"):
            raise ValueError(
                f"Unsupported transport {transport!r}: 'multi' (default), "
                "'streamable-http', 'streamable-stateless' or 'sse'"
            )
        os.environ["MCP_TRANSPORT"] = transport
        logging.info(f"Transport protocol: {transport}")
    else:
        logging.info("Transport: multi (default — /mcp stateful+stateless, /sse)")

    # Health check settings
    if args.health_check_logs:
        os.environ["MCP_HEALTH_CHECK_LOGS"] = args.health_check_logs
        logging.info(f"Health check logging: {args.health_check_logs}")
    if args.health_check_interval:
        os.environ["MCP_HEALTH_CHECK_INTERVAL"] = str(args.health_check_interval)
        logging.info(f"Health check interval: {args.health_check_interval}s")
    
    # Fail fast if another launcher instance is already running
    from launcher.port_manager import management_endpoint_occupied
    if management_endpoint_occupied():
        logging.error(
            "Another launcher instance appears to be already running (management API "
            "port 8200 is responding). Stop it first: pkill -INT -f launchmcp.py — "
            "wait for 'MCP Launcher stopped' — then start again. A live launcher "
            "holds all tool ports; retrying cannot win."
        )
        return 1

    logging.info("=" * 60)
    logging.info("MCP Launcher Starting")
    logging.info("=" * 60)
    
    # Initialize monitoring system
    monitoring_initialized = False
    try:
        monitoring_initialized = initialize_monitoring()
    except Exception as e:
        logging.warning(f"Monitoring initialization failed: {e}")
    
    # Get all configured tool directories
    all_tool_dirs = config.tool_directory_config.directories
    
    # If specific tools are requested, only search their directories
    # Otherwise, search all configured directories
    tool_names = args.tools if args.tools else None
    if tool_names:
        # Map tool names (with possible underscores) to directory names
        # e.g., "webmcp" -> "webmcp", "ragmcp" -> "ragmcp"
        search_dirs = []
        for tool_name in tool_names:
            # Try both with underscore and without
            tool_dir_name = tool_name.replace("_", "")  # webmcp -> webmcp
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
    
    # Initialize port manager with typed port ranges from config
    port_config = config.port_config
    ports_config = {
        "ranges": port_config.ranges,
        "reserved": port_config.reserved_ports,
        "assignments": port_config.manual_ports
    }
    port_manager = PortManager(
        ports_config=ports_config,
        mode=port_config.mode,
        base_port=port_config.base_port,
    )
    
    # Create service registry if management is enabled (default)
    # Management server is enabled by default, use --no-management to disable
    service_registry: ServiceRegistry | None = None
    if not args.no_management:
        service_registry = ServiceRegistry()
        await service_registry.start()
        logging.info("Service registry started for centralized management")
    
    # Initialize server manager
    server_manager = ServerManager(
        host=config.server_config.host,
        log_level=config.server_config.log_level,
        service_registry=service_registry,
        enable_management=bool(service_registry)
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
        started_tools = await start_servers(tools, port_manager, server_manager, config, service_registry)
        
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
                    print(f"  {tool.name}: http://{config.server_config.host}:{port}")
                else:
                    print(f"  {tool.name}: FAILED (status: {status})")
        print("=" * 60)
        print("Press Ctrl+C to stop all servers\n")

        # Sync tools config from running servers (unless disabled)
        if not args.no_sync_tools:
            try:
                from launcher.tools_config import save_tools_config, load_tools_config, validate_and_cleanup_config
                logging.info(f"Syncing tools config from {len(started_tools)} servers...")
                validate_and_cleanup_config()
                discovered = {}
                for tool in started_tools:
                    module = discovery.loaded_modules.get(tool.name)
                    if module and hasattr(module, "mcp"):
                        try:
                            # FastMCP 3.x: await mcp.list_tools()
                            # FastMCP 2.x: mcp._tool_manager.list_tools()
                            mcp_instance = module.mcp
                            if hasattr(mcp_instance, 'list_tools'):
                                try:
                                    tool_list = await mcp_instance.list_tools()
                                    discovered[tool.name] = [t.name for t in tool_list]
                                except TypeError:
                                    # list_tools might not be a coroutine (2.x)
                                    tool_list = mcp_instance.list_tools()
                                    discovered[tool.name] = [t.name for t in tool_list]
                            elif hasattr(mcp_instance, '_tool_manager'):
                                tool_list = mcp_instance._tool_manager.list_tools()
                                discovered[tool.name] = [t.name for t in tool_list]
                            else:
                                discovered[tool.name] = []
                        except Exception:
                            discovered[tool.name] = []
                    else:
                        discovered[tool.name] = []
                config_data = load_tools_config()
                if "tools" not in config_data:
                    config_data["tools"] = {}
                for server_name, tool_names in discovered.items():
                    config_data["tools"][server_name] = tool_names
                save_tools_config(config_data)
                for name, tool_names in discovered.items():
                    logging.info(f"  {name}: {len(tool_names)} tools")
                logging.info("Tools config synced")
            except Exception as e:
                logging.warning(f"Failed to sync tools config: {e}")

        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_servers(server_manager, shutdown_event))
        
        # Start metrics server if monitoring was initialized
        metrics_task = None
        if monitoring_initialized:
            metrics_task = asyncio.create_task(start_metrics_server(shutdown_event))
            logging.info("Metrics server started in background")
        
        # Start management API server if enabled (default)
        management_task = None
        if not args.no_management:
            # Get port from CLI override or config (ports.json)
            mgmt_port = args.management_port
            if mgmt_port is None:
                mgmt_port = config.port_config.reserved_ports.get("central_management")
            if mgmt_port is None:
                logging.error(
                    "Management port not configured. Set --management-port or "
                    "create config/ports.json with reserved.central_management port."
                )
                return 1
            management_task = asyncio.create_task(
                start_management_api_server(shutdown_event, mgmt_port, service_registry)
            )
            logging.info(f"Management API server will start on port {mgmt_port}")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel monitor task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Cancel metrics server task
        if metrics_task:
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
        
        # Cancel management API server task
        if management_task:
            management_task.cancel()
            try:
                await management_task
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
