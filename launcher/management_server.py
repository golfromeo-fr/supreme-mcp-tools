"""
Management Server

The main management server that provides a unified API for managing all MCP tools.
This server acts as the central hub for the Flexible Extensibility Framework V3.
"""

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

from .service_registry import ServiceRegistry
from .distributed_registry import DistributedExtensionRegistry
from .config_types import DEFAULT_HOST
from .tools_config import (
    get_all_disabled_tools,
    get_disabled_tools,
    set_disabled_tools,
    enable_tool,
    disable_tool,
)
from .env_manager import (
    get_env_values,
    get_all_env_values,
    set_env_value,
    delete_env_value,
    load_env_schema,
    get_tool_names,
    load_auth_config,
    mask_value,
)

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying data sources."""
    params: dict[str, Any] | None = None


class MutateRequest(BaseModel):
    """Request model for mutating configuration."""
    params: dict[str, Any]


class ExecuteRequest(BaseModel):
    """Request model for executing actions."""
    params: dict[str, Any] | None = None


class EnvVarUpdate(BaseModel):
    """Request model for updating environment variables."""
    variables: dict[str, str]


class AuthUpdate(BaseModel):
    """Request model for updating tool auth configuration."""
    api_key: str


def _get_default_management_port() -> int:
    """Get the default management port from ports.json."""
    try:
        from launcher.launcher_config import load_ports_config
        ports_config = load_ports_config()
        return ports_config.get("reserved", {}).get("central_management")
    except Exception as e:
        logger.debug(f"Could not load ports.json: {e}")
        return None


class ManagementServer:
    """
    Main management server for the Flexible Extensibility Framework V3.
    
    Provides a unified REST API and WebSocket interface for:
    - Tool discovery and listing
    - Extension management across all tools
    - Querying data sources
    - Mutating configurations
    - Executing actions
    - Real-time event streaming
    """
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        port: int = None,
        host: str = DEFAULT_HOST,
        api_key: str | None = None,
        port_manager: Any | None = None
    ):
        """
        Initialize the management server.
        
        Args:
            service_registry: Service registry for tool discovery
            port: Port to listen on (default: from ports.json reserved.central_management)
            host: Host to bind to
            api_key: Optional API key for authentication
            port_manager: Optional PortManager for port reservation
        """
        self.service_registry = service_registry
        self.registry = DistributedExtensionRegistry(service_registry)
        
        # Get port from ports.json if not specified
        if port is None:
            port = _get_default_management_port()
            if port is None:
                raise ValueError(
                    "Management port not specified and ports.json not found. "
                    "Please create config/ports.json with reserved.central_management port."
                )
        
        # Try to reserve port with PortManager, fall back to specified port
        if port_manager:
            reserved = port_manager.reserve_system_port("central_management", port)
            if not reserved:
                # Port already in use or couldn't reserve, try to get an available one
                actual_port = port_manager.allocate_port("central_management", port_type="system")
                if actual_port != port:
                    logger.warning(
                        f"Requested port {port} unavailable, using {actual_port} instead. "
                        f"Update ports.json reserved.central_management to match."
                    )
                port = actual_port
        
        self.port = port
        self.host = host
        self.api_key = api_key
        
        self.app = FastAPI(
            title="Supreme MCP Tools Management API",
            description="Central management API for all MCP tools",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,  # Not using cookies; header-based auth only
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        
        self._register_routes()
    
    def _verify_api_key(
        self,
        credentials: HTTPAuthorizationCredentials | None = Depends(security)
    ) -> bool:
        """Verify API key if configured."""
        if self.api_key is None:
            return True
        
        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing API key")
        
        if credentials.credentials != self.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return True
    
    def _register_routes(self) -> None:
        """Register all API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            tools = await self.service_registry.list_tools()
            return {
                "status": "healthy",
                "tools_count": len(tools),
                "tools": tools
            }
        
        @self.app.get("/api/tools")
        async def list_tools(_: bool = Depends(self._verify_api_key)):
            """List all available tools with their status."""
            services = await self.service_registry.get_all_services()
            
            tools = []
            for name, service in services.items():
                tools.append({
                    "name": name,
                    "status": service.status,
                    "management_url": service.management_url,
                    "mcp_port": service.mcp_port,
                    "capabilities": service.capabilities,
                    "last_check": service.last_check
                })
            
            return {"tools": tools}
        
        @self.app.get("/api/tools/{tool_name}")
        async def get_tool(
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Get details of a specific tool."""
            service = await self.service_registry.get_endpoint(tool_name)
            if service is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool '{tool_name}' not found"
                )
            
            return {
                "name": service.name,
                "status": service.status,
                "management_url": service.management_url,
                "mcp_port": service.mcp_port,
                "capabilities": service.capabilities,
                "last_check": service.last_check,
                "registered_at": service.registered_at
            }
        
        @self.app.get("/api/tools/{tool_name}/extensions")
        async def list_tool_extensions(
            tool_name: str,
            ext_type: str | None = None,
            _: bool = Depends(self._verify_api_key)
        ):
            """List extensions for a specific tool."""
            try:
                extensions = await self.registry.list_extensions(tool_name, ext_type)
                return {"extensions": extensions.get(tool_name, [])}
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        @self.app.get("/api/extensions")
        async def list_all_extensions(
            ext_type: str | None = None,
            _: bool = Depends(self._verify_api_key)
        ):
            """List all extensions across all tools."""
            extensions = await self.registry.list_extensions(ext_type=ext_type)
            return {"extensions": extensions}
        
        @self.app.post("/api/tools/{tool_name}/extensions/{extension_name}/query")
        async def query_extension(
            tool_name: str,
            extension_name: str,
            request: QueryRequest,
            _: bool = Depends(self._verify_api_key)
        ):
            """Query a data source extension."""
            try:
                result = await self.registry.query(
                    tool_name,
                    extension_name,
                    request.params
                )
                return {"data": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error querying extension: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/tools/{tool_name}/extensions/{extension_name}/mutate")
        async def mutate_extension(
            tool_name: str,
            extension_name: str,
            request: MutateRequest,
            _: bool = Depends(self._verify_api_key)
        ):
            """Mutate configuration via extension."""
            try:
                result = await self.registry.mutate(
                    tool_name,
                    extension_name,
                    request.params
                )
                return {"result": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error mutating extension: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/tools/{tool_name}/extensions/{extension_name}/execute")
        async def execute_extension(
            tool_name: str,
            extension_name: str,
            request: ExecuteRequest,
            _: bool = Depends(self._verify_api_key)
        ):
            """Execute an action extension."""
            try:
                result = await self.registry.execute(
                    tool_name,
                    extension_name,
                    request.params
                )
                return {"result": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error executing extension: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.websocket("/api/tools/{tool_name}/extensions/{extension_name}/events")
        async def websocket_events(
            websocket: WebSocket,
            tool_name: str,
            extension_name: str
        ):
            """WebSocket endpoint for real-time event streaming."""
            if self.api_key is not None:
                auth = websocket.query_params.get("token") or websocket.headers.get("authorization", "")
                if auth.startswith("Bearer "):
                    auth = auth[7:]
                if auth != self.api_key:
                    await websocket.close(code=4001, reason="Unauthorized")
                    return
            
            await websocket.accept()
            
            # Verify tool exists
            service = await self.service_registry.get_endpoint(tool_name)
            if service is None:
                await websocket.close(code=4004, reason=f"Tool '{tool_name}' not found")
                return
            
            # Subscribe to events
            queue = await self.event_aggregator.subscribe(tool_name)
            
            try:
                while True:
                    # Wait for events and send to client
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        # Filter by extension if specified
                        if event.get("data", {}).get("extension") == extension_name:
                            await websocket.send_json(event)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for {tool_name}/{extension_name}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await self.event_aggregator.unsubscribe(tool_name, queue)
        
        @self.app.get("/api/config/{tool_name}")
        async def get_tool_config(
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Get persisted configuration for a tool."""
            config = self.registry.config_persistence.load(tool_name)
            return {"config": config}

        # === Disabled Tools Configuration ===

        @self.app.get("/api/disabled-tools")
        async def get_all_disabled_tools_endpoint(
            _: bool = Depends(self._verify_api_key)
        ):
            """Get all disabled tools configuration."""
            return {"disabled_tools": get_all_disabled_tools()}

        @self.app.get("/api/disabled-tools/{server_name}")
        async def get_disabled_tools_endpoint(
            server_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Get disabled tools for a specific server."""
            return {"server": server_name, "disabled": get_disabled_tools(server_name)}

        @self.app.put("/api/disabled-tools/{server_name}")
        async def set_disabled_tools_endpoint(
            server_name: str,
            disabled_list: list[str],
            _: bool = Depends(self._verify_api_key)
        ):
            """Set disabled tools for a specific server."""
            set_disabled_tools(server_name, disabled_list)
            return {"server": server_name, "disabled": disabled_list}

        @self.app.post("/api/disabled-tools/{server_name}/{tool_name}/disable")
        async def disable_tool_endpoint(
            server_name: str,
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Disable a specific tool for a server."""
            disable_tool(tool_name, server_name)
            return {"server": server_name, "tool": tool_name, "disabled": True}

        @self.app.post("/api/disabled-tools/{server_name}/{tool_name}/enable")
        async def enable_tool_endpoint(
            server_name: str,
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Enable a specific tool for a server."""
            enable_tool(tool_name, server_name)
            return {"server": server_name, "tool": tool_name, "disabled": False}

        # === Environment Variable Management ===

        @self.app.get("/api/tools/{tool_name}/env")
        async def get_tool_env(
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Get environment variables for a specific tool."""
            schema = load_env_schema(tool_name)
            if not schema:
                # Check if tool exists at all
                if tool_name not in get_tool_names():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool '{tool_name}' not found"
                    )
                return {"tool_name": tool_name, "variables": {}}

            values = get_env_values(tool_name)
            return {"tool_name": tool_name, "variables": values}

        @self.app.put("/api/tools/{tool_name}/env")
        async def update_tool_env(
            tool_name: str,
            request: EnvVarUpdate,
            _: bool = Depends(self._verify_api_key)
        ):
            """Update environment variables for a specific tool."""
            schema = load_env_schema(tool_name)
            if not schema:
                if tool_name not in get_tool_names():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool '{tool_name}' not found"
                    )
                raise HTTPException(
                    status_code=400,
                    detail=f"Tool '{tool_name}' has no environment variables configured"
                )

            # Validate that all requested variables are declared in the schema
            unknown_vars = set(request.variables.keys()) - set(schema.keys())
            if unknown_vars:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown environment variables: {', '.join(unknown_vars)}"
                )

            # Set each variable (validate + update)
            for var_name, value in request.variables.items():
                try:
                    set_env_value(var_name, value, persist=True)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))

            # Return updated masked values
            updated = get_env_values(tool_name)
            return {
                "tool_name": tool_name,
                "variables": updated,
                "updated_count": len(request.variables)
            }

        @self.app.delete("/api/tools/{tool_name}/env/{var_name}")
        async def delete_tool_env(
            tool_name: str,
            var_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Remove an environment variable for a specific tool."""
            schema = load_env_schema(tool_name)
            if var_name not in schema:
                raise HTTPException(
                    status_code=404,
                    detail=f"Variable '{var_name}' not found in tool '{tool_name}'"
                )

            delete_env_value(var_name, persist=True)
            return {
                "tool_name": tool_name,
                "variable": var_name,
                "deleted": True
            }

        @self.app.get("/api/env")
        async def get_all_env(
            _: bool = Depends(self._verify_api_key)
        ):
            """Get environment variables for all tools."""
            return {"tools": get_all_env_values()}

        # === Per-Tool Auth Management ===

        @self.app.get("/api/tools/{tool_name}/auth")
        async def get_tool_auth(
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Get auth config for a tool (masked key)."""
            auth_config = load_auth_config(tool_name)
            api_key = auth_config.get("api_key", "")
            return {
                "api_key": {
                    "is_set": bool(api_key),
                    "value_masked": mask_value(api_key) if api_key else None
                }
            }

        @self.app.put("/api/tools/{tool_name}/auth")
        async def update_tool_auth(
            tool_name: str,
            request: AuthUpdate,
            _: bool = Depends(self._verify_api_key)
        ):
            """Update auth config for a tool."""
            import json
            from pathlib import Path

            config_path = Path(__file__).parent.parent / "tools" / tool_name / "config.json"
            if ".." in tool_name or "/" in tool_name or "\\" in tool_name:
                raise HTTPException(status_code=400, detail="Invalid tool name")
            if not config_path.resolve().parent.parent.name == "tools":
                raise HTTPException(status_code=400, detail="Invalid tool name")
            if not config_path.exists():
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

            try:
                with Path(config_path).open() as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid config.json")

            config.setdefault("auth", {})["api_key"] = request.api_key

            with Path(config_path).open("w") as f:
                json.dump(config, f, indent=2)

            return {"success": True}

    @property
    def event_aggregator(self):
        """Get the event aggregator from the distributed registry."""
        return self.registry.event_aggregator
    
    async def start(self) -> None:
        """Start the management server."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=os.environ.get("MCP_HEALTH_CHECK_LOGS", "enable") != "disable"
        )
        self._server = uvicorn.Server(config)
        
        logger.info(f"Starting management server on port {self.port}")
        
        self._task = asyncio.create_task(self._server.serve())
        
        # Wait for server to start
        await asyncio.sleep(0.5)
        
        logger.info(f"Management server started on port {self.port}")
    
    async def stop(self) -> None:
        """Stop the management server."""
        if self._server:
            self._server.should_exit = True
            if self._task:
                await self._task
            await self.registry.close()
            logger.info("Management server stopped")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app
