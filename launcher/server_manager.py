"""
Server lifecycle manager for the MCP launcher system.

This module provides functionality to manage the lifecycle of multiple
Uvicorn servers running MCP tools concurrently.

Supports the Flexible Extensibility Framework V3 with management servers.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .errors import ServerStartupError, ServerRuntimeError
from .tool_discovery import ToolMetadata
from .service_registry import ServiceRegistry
from .tool_extensions import ExtensionRegistry, ExtensionHTTPServer
from .env_manager import load_auth_config


logger = logging.getLogger(__name__)


class MCPApiKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key on MCP endpoints."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/mcp":
            provided_key = request.headers.get("x-api-key")
            if not provided_key:
                logger.warning("[AUTH] MCP request rejected: no X-API-Key header provided")
                return JSONResponse(
                    status_code=200,
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32001,
                            "message": "Authentication required",
                            "data": "Missing X-API-Key header"
                        }
                    }
                )
            if provided_key != self.api_key:
                logger.warning(f"[AUTH] MCP request rejected: invalid API key (got={provided_key[:8]}...)")
                return JSONResponse(
                    status_code=200,
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32001,
                            "message": "Authentication required",
                            "data": "Invalid X-API-Key header"
                        }
                    }
                )
            logger.info(f"[AUTH] MCP request authenticated successfully")
        return await call_next(request)


@dataclass
class ServerInstance:
    """Instance of a running MCP tool server."""
    tool_name: str
    port: int
    app: Any
    server_config: uvicorn.Config
    server: uvicorn.Server
    status: str = "stopped"  # stopped, starting, running, error
    start_time: Optional[datetime] = None
    error: Optional[Exception] = None
    # FEF V3 fields
    mgmt_port: Optional[int] = None
    mgmt_server: Optional[ExtensionHTTPServer] = None
    extension_registry: Optional[ExtensionRegistry] = None
    
    def __repr__(self) -> str:
        return f"ServerInstance(tool={self.tool_name}, port={self.port}, status={self.status})"


class ServerManager:
    """Manage lifecycle of multiple Uvicorn servers with FEF V3 support."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        log_level: str = "info",
        service_registry: Optional[ServiceRegistry] = None,
        enable_management: bool = True
    ):
        """
        Initialize the server manager.
        
        Args:
            host: Host address for servers
            log_level: Log level for servers
            service_registry: Optional service registry for FEF V3
            enable_management: Whether to enable management servers for tools
        """
        self.host = host
        self.log_level = log_level
        self.service_registry = service_registry
        self.enable_management = enable_management
        
        self.servers: Dict[str, ServerInstance] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
    
    async def start_server(
        self,
        tool_metadata: ToolMetadata,
        port: int,
        mgmt_port: Optional[int] = None
    ) -> ServerInstance:
        """
        Start a single MCP tool server with optional management server.
        
        Args:
            tool_metadata: Tool metadata object
            port: Port number to use for MCP server
            mgmt_port: Optional port for management server (FEF V3)
            
        Returns:
            ServerInstance for the started server
            
        Raises:
            ServerStartupError: If server fails to start
        """
        tool_name = tool_metadata.name
        app = tool_metadata.exports["app"]

        logger.info(f"Starting server for {tool_name} on port {port}")

        try:
            # Load per-tool auth config
            auth_config = load_auth_config(tool_name)
            api_key = auth_config.get("api_key")  # None if not configured

            # Add MCP auth middleware if API key is configured
            if api_key:
                app.add_middleware(MCPApiKeyMiddleware, api_key=api_key)
                logger.info(f"MCP endpoint auth enabled for {tool_name}")

            # Create Uvicorn config
            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=port,
                log_level=self.log_level,
                access_log=True
            )

            # Create Uvicorn server
            server = uvicorn.Server(config)

            # Create extension registry for FEF V3
            extension_registry = None
            mgmt_server = None

            if self.enable_management and mgmt_port:
                # Set environment variable so tool can use the same management port
                os.environ["MCP_MGMT_PORT"] = str(mgmt_port)
                logger.info(f"Set MCP_MGMT_PORT={mgmt_port} for {tool_name}")

                # Create registry with tool_name for global tracking
                # This allows the tool to find this registry when it starts
                extension_registry = ExtensionRegistry(tool_name=tool_name)

                mgmt_server = ExtensionHTTPServer(
                    tool_name=tool_name,
                    registry=extension_registry,
                    port=mgmt_port,
                    host=self.host,
                    api_key=api_key
                )
                
                # Call tool's setup_extensions() if available to register extensions
                # This must be done AFTER module load but BEFORE server starts
                tool_module = tool_metadata.exports.get("_module")
                if tool_module and hasattr(tool_module, "setup_extensions"):
                    try:
                        logger.info(f"Calling {tool_name}.setup_extensions() with launcher's registry")
                        tool_module.setup_extensions(registry=extension_registry)
                    except Exception as e:
                        logger.warning(f"Failed to call {tool_name}.setup_extensions(): {e}")
            
            # Create server instance
            instance = ServerInstance(
                tool_name=tool_name,
                port=port,
                app=app,
                server_config=config,
                server=server,
                status="starting",
                start_time=datetime.now(),
                mgmt_port=mgmt_port,
                mgmt_server=mgmt_server,
                extension_registry=extension_registry
            )
            
            self.servers[tool_name] = instance
            
            # Start the MCP server in a task
            task = asyncio.create_task(self._run_server(instance))
            self.tasks[tool_name] = task
            
            # Start management server if enabled
            if mgmt_server:
                await mgmt_server.start()
                
                # Register with service registry
                # NOTE: Use "localhost" for the URL even though server binds to self.host (0.0.0.0)
                # Clients cannot connect to 0.0.0.0 - they need localhost or 127.0.0.1
                if self.service_registry:
                    await self.service_registry.register(
                        name=tool_name,
                        management_url=f"http://localhost:{mgmt_port}",
                        mcp_port=port
                    )
            
            logger.info(f"Server for {tool_name} starting on port {port}")
            return instance
        
        except Exception as e:
            if tool_name in self.servers:
                self.servers[tool_name].status = "error"
                self.servers[tool_name].error = e
            raise ServerStartupError(
                f"Failed to start server for {tool_name}",
                tool_name=tool_name,
                port=port
            ) from e
    
    async def start_all_servers(
        self,
        tools_ports: Dict[str, int]
    ) -> Dict[str, ServerInstance]:
        """
        Start all servers concurrently.
        
        Args:
            tools_ports: Dictionary of tool name -> port
            
        Returns:
            Dictionary of tool name -> ServerInstance
        """
        logger.info(f"Starting {len(tools_ports)} servers concurrently")
        
        instances = {}
        startup_tasks = []
        
        # Create startup tasks for all servers
        for tool_name, port in tools_ports.items():
            # Get tool metadata from discovery
            # Note: This assumes tool discovery has been done and metadata is available
            # We'll need to pass the metadata in a real implementation
            # For now, we'll handle this in the main launcher
            pass
        
        return instances
    
    async def _run_server(self, instance: ServerInstance) -> None:
        """
        Run a single Uvicorn server.
        
        Args:
            instance: Server instance to run
        """
        tool_name = instance.tool_name
        
        try:
            logger.info(f"Running server for {tool_name} on port {instance.port}")
            instance.status = "running"
            await instance.server.serve()
            instance.status = "stopped"
            logger.info(f"Server for {tool_name} stopped on port {instance.port}")
        
        except asyncio.CancelledError:
            logger.info(f"Server for {tool_name} was cancelled")
            instance.status = "stopped"
        
        except Exception as e:
            instance.status = "error"
            instance.error = e
            logger.error(f"Server for {tool_name} encountered error: {e}")
            raise ServerRuntimeError(
                f"Server runtime error for {tool_name}",
                tool_name=tool_name,
                port=instance.port
            ) from e
    
    async def stop_server(self, tool_name: str) -> bool:
        """
        Stop a specific server and its management server.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if server was stopped, False if not found
        """
        if tool_name not in self.servers:
            logger.warning(f"No server found for {tool_name}")
            return False
        
        instance = self.servers[tool_name]
        
        logger.info(f"Stopping server for {tool_name}")
        
        try:
            # Stop management server first (FEF V3)
            if instance.mgmt_server:
                await instance.mgmt_server.stop()
            
            # Unregister from service registry
            if self.service_registry:
                await self.service_registry.unregister(tool_name)
            
            # Cancel the task if running
            if tool_name in self.tasks:
                task = self.tasks[tool_name]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self.tasks[tool_name]
            
            # Shutdown the server
            if instance.server.started:
                await instance.server.shutdown()
            
            instance.status = "stopped"
            logger.info(f"Server for {tool_name} stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping server for {tool_name}: {e}")
            instance.status = "error"
            instance.error = e
            return False
    
    async def stop_all_servers(self) -> None:
        """Stop all running servers."""
        logger.info("Stopping all servers")
        
        # Cancel all tasks
        tasks = list(self.tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Shutdown all servers
        for instance in self.servers.values():
            if instance.server.started:
                try:
                    await instance.server.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down server for {instance.tool_name}: {e}")
        
        # Clear all
        self.tasks.clear()
        for instance in self.servers.values():
            instance.status = "stopped"
        
        logger.info("All servers stopped")
    
    def get_server_status(self, tool_name: str) -> Optional[str]:
        """
        Get status of a specific server.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Server status or None if not found
        """
        instance = self.servers.get(tool_name)
        return instance.status if instance else None
    
    def get_all_statuses(self) -> Dict[str, str]:
        """
        Get status of all servers.
        
        Returns:
            Dictionary of tool name -> status
        """
        return {
            name: instance.status
            for name, instance in self.servers.items()
        }
    
    def get_server_instance(self, tool_name: str) -> Optional[ServerInstance]:
        """
        Get a server instance by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ServerInstance or None if not found
        """
        return self.servers.get(tool_name)
    
    def get_all_instances(self) -> Dict[str, ServerInstance]:
        """
        Get all server instances.
        
        Returns:
            Dictionary of tool name -> ServerInstance
        """
        return self.servers.copy()
    
    def is_server_running(self, tool_name: str) -> bool:
        """
        Check if a server is currently running.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if server is running, False otherwise
        """
        status = self.get_server_status(tool_name)
        return status == "running"
    
    def get_running_servers(self) -> List[str]:
        """
        Get list of running server names.
        
        Returns:
            List of tool names with running servers
        """
        return [
            name
            for name, status in self.get_all_statuses().items()
            if status == "running"
        ]
    
    def get_error_servers(self) -> List[str]:
        """
        Get list of servers with errors.
        
        Returns:
            List of tool names with error status
        """
        return [
            name
            for name, status in self.get_all_statuses().items()
            if status == "error"
        ]
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get summary of all servers.
        
        Returns:
            Dictionary with server summary information
        """
        statuses = self.get_all_statuses()
        running = sum(1 for s in statuses.values() if s == "running")
        errors = sum(1 for s in statuses.values() if s == "error")
        stopped = sum(1 for s in statuses.values() if s == "stopped")
        starting = sum(1 for s in statuses.values() if s == "starting")
        
        return {
            "total_servers": len(self.servers),
            "running": running,
            "starting": starting,
            "stopped": stopped,
            "errors": errors,
            "servers": {
                name: {
                    "port": instance.port,
                    "status": instance.status,
                    "start_time": instance.start_time.isoformat() if instance.start_time else None,
                    "error": str(instance.error) if instance.error else None
                }
                for name, instance in self.servers.items()
            }
        }


async def run_servers_concurrently(
    tools_metadata: List[ToolMetadata],
    ports: Dict[str, int],
    host: str = "0.0.0.0",
    log_level: str = "info"
) -> Dict[str, ServerInstance]:
    """
    Run multiple Uvicorn servers concurrently.
    
    Args:
        tools_metadata: List of tool metadata objects
        ports: Dictionary of tool name -> port
        host: Host address for servers
        log_level: Log level for servers
        
    Returns:
        Dictionary of tool name -> ServerInstance
    """
    manager = ServerManager(host=host, log_level=log_level)
    instances = {}
    
    async def start_and_run(tool_metadata: ToolMetadata) -> Optional[ServerInstance]:
        """Start and run a single server."""
        tool_name = tool_metadata.name
        port = ports.get(tool_name)
        
        if port is None:
            logger.error(f"No port allocated for {tool_name}")
            return None
        
        try:
            instance = await manager.start_server(tool_metadata, port)
            return instance
        except Exception as e:
            logger.error(f"Failed to start server for {tool_name}: {e}")
            return None
    
    # Start all servers
    start_tasks = [
        start_and_run(tool_metadata)
        for tool_metadata in tools_metadata
    ]
    
    results = await asyncio.gather(*start_tasks, return_exceptions=True)
    
    # Collect successful instances
    for result in results:
        if isinstance(result, ServerInstance):
            instances[result.tool_name] = result
        elif isinstance(result, Exception):
            logger.error(f"Server startup error: {result}")
    
    return instances
