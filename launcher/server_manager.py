"""
Server lifecycle manager for the MCP launcher system.

This module provides functionality to manage the lifecycle of multiple
Uvicorn servers running MCP tools concurrently.

Supports the Flexible Extensibility Framework V3 with management servers.
"""

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from .errors import ServerStartupError, ServerRuntimeError
from .tool_discovery import ToolMetadata
from .config_types import DEFAULT_HOST
from .service_registry import ServiceRegistry
from .tool_extensions import ExtensionRegistry, ExtensionHTTPServer
from .env_manager import load_auth_config


logger = logging.getLogger(__name__)


class MCPApiKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key on MCP endpoints with auto-approve OAuth.

    Supports two auth methods:
    1. X-API-Key header — direct API key (Kilo Code, GitHub Copilot)
    2. Bearer token via OAuth — Claude Code's MCP SDK requires OAuth flow,
       so this implements a minimal auto-approve OAuth that returns the API
       key as the access token.
    """

    # OAuth paths handled by this middleware
    _OAUTH_METADATA_PATHS = frozenset({
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    })

    def __init__(self, app, api_key: str, server_url: str = None, tool_name: str = "unknown"):
        super().__init__(app)
        self.api_key = api_key
        self.server_url = server_url  # e.g. "http://127.0.0.1:8002"
        self.tool_name = tool_name

        # In-memory stores for OAuth flow (per-tool, per-server process)
        self._pending_codes: dict[str, tuple[str, str, float]] = {}  # code -> (client_id, redirect_uri, created_at)
        self._registered_clients: dict[str, tuple[dict, float]] = {}  # client_id -> (client_info, created_at)
        self._oauth_ttl = 600  # 10 minutes

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        now = time.time()
        expired_codes = [k for k, v in self._pending_codes.items() if now - v[2] > self._oauth_ttl]
        for k in expired_codes:
            del self._pending_codes[k]
        expired_clients = [k for k, v in self._registered_clients.items() if now - v[1] > self._oauth_ttl]
        for k in expired_clients:
            del self._registered_clients[k]

        # --- OAuth endpoints suppressed by oauth_fix — FastMCP now returns 404.
        # The handlers below are now dead code but kept for backwards compatibility
        # with clients that bypass FastMCP's routing (e.g. direct uvicorn access).
        if path in self._OAUTH_METADATA_PATHS:
            return self._handle_oauth_metadata(request)

        # --- Protected Resource Metadata ---
        if path == "/.well-known/oauth-protected-resource" or path == "/.well-known/oauth-protected-resource/mcp":
            return self._handle_protected_resource_metadata(request)

        # --- Dynamic Client Registration ---
        if path == "/register" and request.method == "POST":
            return await self._handle_register(request)

        # --- Authorization endpoint (browser redirect) ---
        if path == "/authorize":
            return await self._handle_authorize(request)

        # --- Token exchange ---
        if path == "/token" and request.method == "POST":
            return await self._handle_token(request)

        # --- Other .well-known paths → 404 with parseable error ---
        if "/.well-known/" in path:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "unsupported_server",
                    "error_description": "This server does not support this endpoint."
                }
            )

        # --- MCP endpoint auth ---
        if path == "/mcp":
            provided_key = request.headers.get("x-api-key")
            bearer = request.headers.get("authorization", "")

            # Accept X-API-Key header
            if provided_key:
                if provided_key != self.api_key:
                    logger.warning(f"[{self.tool_name}] AUTH rejected: invalid API key")
                    return self._jsonrpc_auth_error("Invalid X-API-Key header")
                logger.info(f"[{self.tool_name}] AUTH ok via X-API-Key")
                return await call_next(request)

            # Accept Bearer token (from OAuth flow)
            if bearer.lower().startswith("bearer "):
                token = bearer[7:]
                if token == self.api_key:
                    logger.info(f"[{self.tool_name}] AUTH ok via Bearer token")
                    return await call_next(request)
                logger.warning(f"[{self.tool_name}] AUTH rejected: invalid Bearer token")
                return self._jsonrpc_auth_error("Invalid Bearer token")

            # No auth provided → return 401 without WWW-Authenticate header.
            # The WWW-Authenticate header triggers VS Code Copilot's OAuth flow,
            # which then fails because /.well-known/* now returns 404 (oauth_fix).
            # Stripping this header lets Copilot fall back to the configured header.
            logger.warning(f"[{self.tool_name}] AUTH rejected: no auth header provided")
            return JSONResponse(
                status_code=401,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32001,
                        "message": "Authentication required",
                        "data": "Missing X-API-Key header or Bearer token"
                    }
                }
            )

        return await call_next(request)

    def _get_base_url(self, request: Request) -> str:
        """Get the base URL for this server."""
        if self.server_url:
            return self.server_url
        return f"{request.url.scheme}://{request.url.netloc}"

    def _handle_oauth_metadata(self, request: Request) -> JSONResponse:
        """Return valid OAuth Authorization Server metadata."""
        base = self._get_base_url(request)
        return JSONResponse({
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post", "none"],
            "scopes_supported": ["mcp"],
        })

    def _handle_protected_resource_metadata(self, request: Request) -> JSONResponse:
        """Return valid Protected Resource metadata."""
        base = self._get_base_url(request)
        return JSONResponse({
            "resource": f"{base}/mcp",
            "authorization_servers": [base],
            "scopes_supported": ["mcp"],
            "bearer_methods_supported": ["header"],
        })

    async def _handle_register(self, request: Request) -> JSONResponse:
        """Auto-approve dynamic client registration."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        client_id = body.get("client_id") or f"mcp-client-{secrets.token_hex(8)}"
        client_secret = f"mcp-secret-{secrets.token_hex(16)}"

        client_info = {
            "client_id": client_id,
            "client_secret": client_secret,
            "client_id_issued_at": int(time.time()),
            "redirect_uris": body.get("redirect_uris", []),
            "token_endpoint_auth_method": body.get("token_endpoint_auth_method", "client_secret_post"),
        }
        self._registered_clients[client_id] = (client_info, time.time())
        logger.info(f"[AUTH-OAUTH] Client registered: {client_id}")
        return JSONResponse(client_info, status_code=201)

    async def _handle_authorize(self, request: Request) -> RedirectResponse:
        """Auto-approve authorization — immediately redirect back with a code."""
        params = dict(request.query_params) if request.method == "GET" else {}
        client_id = params.get("client_id", "")
        redirect_uri = params.get("redirect_uri", "")
        state = params.get("state", "")
        code_challenge = params.get("code_challenge", "")

        if not redirect_uri:
            return JSONResponse({"error": "invalid_request", "error_description": "Missing redirect_uri"}, status_code=400)

        # Generate auth code and store pending exchange
        code = f"mcp-code-{secrets.token_hex(16)}"
        self._pending_codes[code] = (client_id, redirect_uri, time.time())

        # Redirect back to client with code and state
        sep = "&" if "?" in redirect_uri else "?"
        redirect_url = f"{redirect_uri}{sep}code={code}&state={state}"
        logger.info(f"[AUTH-OAUTH] Auto-approved authorization for client={client_id}")
        return RedirectResponse(url=redirect_url, status_code=302)

    async def _handle_token(self, request: Request) -> JSONResponse:
        """Exchange authorization code for access token (returns the API key)."""
        try:
            body = await request.json()
        except Exception:
            try:
                form = await request.form()
                body = dict(form)
            except Exception:
                body = {}

        code = body.get("code", "")
        pending = self._pending_codes.pop(code, None)

        if not pending:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Invalid authorization code"},
                status_code=400
            )

        client_id, redirect_uri, _ = pending
        logger.info(f"[AUTH-OAUTH] Token issued for client={client_id}")

        return JSONResponse({
            "access_token": self.api_key,
            "token_type": "Bearer",
            "expires_in": 86400,
            "scope": "mcp",
        })

    def _jsonrpc_auth_error(self, message: str) -> JSONResponse:
        """Return a JSON-RPC auth error with HTTP 401 (no WWW-Authenticate)."""
        return JSONResponse(
            status_code=401,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32001,
                    "message": "Authentication required",
                    "data": message
                }
            }
        )


@dataclass
class ServerInstance:
    """Instance of a running MCP tool server."""
    tool_name: str
    port: int
    app: Any
    server_config: uvicorn.Config
    server: uvicorn.Server
    status: str = "stopped"  # stopped, starting, running, error
    start_time: datetime | None = None
    error: Exception | None = None
    # FEF V3 fields
    mgmt_port: int | None = None
    mgmt_server: ExtensionHTTPServer | None = None
    extension_registry: ExtensionRegistry | None = None
    
    def __repr__(self) -> str:
        return f"ServerInstance(tool={self.tool_name}, port={self.port}, status={self.status})"


class ServerManager:
    """Manage lifecycle of multiple Uvicorn servers with FEF V3 support."""
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        log_level: str = "info",
        service_registry: ServiceRegistry | None = None,
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
        
        self.servers: dict[str, ServerInstance] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.running = False
    
    async def start_server(
        self,
        tool_metadata: ToolMetadata,
        port: int,
        mgmt_port: int | None = None
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
                server_url = f"http://127.0.0.1:{port}"
                app.add_middleware(MCPApiKeyMiddleware, api_key=api_key, server_url=server_url, tool_name=tool_name)
                logger.info(f"MCP endpoint auth enabled for {tool_name}")

            # Create Uvicorn config
            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=port,
                log_level=self.log_level,
                access_log=os.environ.get("MCP_HEALTH_CHECK_LOGS", "enable") != "disable"
            )

            # Create Uvicorn server
            server = uvicorn.Server(config)

            # Create extension registry for FEF V3
            extension_registry = None
            mgmt_server = None

            if self.enable_management and mgmt_port:
                os.environ[f"MCP_MGMT_PORT_{tool_name}"] = str(mgmt_port)
                # Note: global MCP_MGMT_PORT intentionally NOT set — concurrent tool
                # startup races if multiple tools overwrite it. Tool-specific var above
                # is the correct mechanism. Keep only the per-tool var.
                logger.info(f"Set MCP_MGMT_PORT_{tool_name}={mgmt_port} for {tool_name}")

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
                # Clients cannot connect to :: - they need localhost or 127.0.0.1
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
        tools_ports: dict[str, int]
    ) -> dict[str, ServerInstance]:
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
        
        return instances  # pragma: no cover — stub for future use
    
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
                    instance.server.should_exit = True
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self.tasks[tool_name]
            
            # Shutdown the server only if no task was managing it
            if instance.server.started and tool_name not in self.tasks:
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
    
    def get_server_status(self, tool_name: str) -> str | None:
        """
        Get status of a specific server.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Server status or None if not found
        """
        instance = self.servers.get(tool_name)
        return instance.status if instance else None
    
    def get_all_statuses(self) -> dict[str, str]:
        """
        Get status of all servers.
        
        Returns:
            Dictionary of tool name -> status
        """
        return {
            name: instance.status
            for name, instance in self.servers.items()
        }
    
    def get_server_instance(self, tool_name: str) -> ServerInstance | None:
        """
        Get a server instance by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ServerInstance or None if not found
        """
        return self.servers.get(tool_name)
    
    def get_all_instances(self) -> dict[str, ServerInstance]:
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
    
    def get_running_servers(self) -> list[str]:
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
    
    def get_error_servers(self) -> list[str]:
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
    
    def get_summary(self) -> dict[str, any]:
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
    tools_metadata: list[ToolMetadata],
    ports: dict[str, int],
    host: str = DEFAULT_HOST,
    log_level: str = "info"
) -> dict[str, ServerInstance]:
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
    
    async def start_and_run(tool_metadata: ToolMetadata) -> ServerInstance | None:
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
