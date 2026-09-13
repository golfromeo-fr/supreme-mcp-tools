"""
Management Server

The main management server that provides a unified API for managing all MCP tools.
This server acts as the central hub for the Flexible Extensibility Framework V3.
"""

import asyncio
import hmac
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
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
from tools.shared import users_store
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


async def _push_runtime_mask(server_name: str, tool_name: str, masked: bool,
                             fanout: bool = True) -> dict:
    """E1: push a just-persisted mask to the RUNNING server (same process —
    the per-tool registry holds its FastMCP instance). File-first already
    happened; a failed push still converges at the tool's next restart.

    M4: masks persist to the SHARED state (db mode) — after the local
    apply, fan out to every other registered cluster node so the whole
    cluster converges immediately, not at their next boot."""
    from .tool_extensions.registry import get_registry_for_tool
    from tools.shared.function_masks import apply_mask_at_runtime

    registry = await get_registry_for_tool(server_name)
    mcp = getattr(registry, "mcp_instance", None) if registry else None
    result = apply_mask_at_runtime(mcp, server_name, tool_name, masked)

    if fanout:
        _schedule_mask_fanout(server_name, tool_name, masked)

    return {"runtime_applied": result["applied"], "runtime_note": result["reason"]}


def _schedule_mask_fanout(server_name: str, tool_name: str, masked: bool) -> None:
    """Fire-and-forget fan-out to sibling cluster nodes (never blocks the
    API response; failures only log — boot-time adoption converges)."""
    import os

    if os.environ.get("MCP_STATE_BACKEND", "json").strip().lower() != "db":
        return
    node_name = os.environ.get("MCP_NODE_NAME")
    if not node_name:
        return  # single-node / mono deployment

    import asyncio

    async def _fanout():
        import httpx

        from tools.shared import cluster

        siblings = cluster.sibling_nodes(node_name)
        if not siblings:
            return
        key = os.environ.get("MCP_MANAGEMENT_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"server_name": server_name, "tool_name": tool_name,
                "masked": masked, "fanout": True}
        applied, failed = [], []
        async with httpx.AsyncClient(timeout=5.0) as client:
            for peer, url in siblings.items():
                try:
                    r = await client.post(f"{url.rstrip('/')}/api/internal/mask-push",
                                          json=body, headers=headers)
                    (applied if r.status_code == 200 else failed).append(peer)
                except Exception:
                    failed.append(peer)
        if applied or failed:
            logging.getLogger(__name__).info(
                f"[M4] mask fan-out {tool_name}({'masked' if masked else 'enabled'}) "
                f"-> applied: {applied or '[]'} failed: {failed or '[]'}")

    try:
        asyncio.get_running_loop().create_task(_fanout())
    except RuntimeError:
        pass


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying data sources."""
    params: dict[str, Any] | None = None


class MutateRequest(BaseModel):
    """Request model for mutating configuration."""
    params: dict[str, Any]


class MaskPushRequest(BaseModel):
    """M4: sibling mask fan-out."""
    server_name: str
    tool_name: str
    masked: bool
    fanout: bool = False


class UserCreateRequest(BaseModel):
    """E3/M2: create a user (mcp_key returned once)."""
    username: str
    password: str
    role: str = "user"
    servers: list[str] = []
    masked_functions: dict[str, list[str]] = {}


class UserPasswordRequest(BaseModel):
    password: str


class UserServersRequest(BaseModel):
    servers: list[str]


class UserMaskedRequest(BaseModel):
    masked_functions: dict[str, list[str]]


class UserEnabledRequest(BaseModel):
    enabled: bool


class UserPresetsRequest(BaseModel):
    presets: list[str]


class UserCollectionsRequest(BaseModel):
    collections: list[str]


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
        _mgmt_port = _get_default_management_port() or 8400
        _allowed_origins = [
            f"http://localhost:{_mgmt_port}",
            f"http://127.0.0.1:{_mgmt_port}",
        ]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed_origins,
            allow_credentials=False,  # Not using cookies; header-based auth only
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        
        self._register_routes()
    
    def _verify_api_key(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(security)
    ) -> str:
        """Central auth (E3 multi-admin): the env break-glass key
        (attributed "system") OR an enabled admin's user key (attributed to
        that admin, revocable via rotate/disable). Non-admin user keys are
        rejected. Returns the acting identity; every authenticated call is
        audit-logged."""
        if self.api_key is None:
            return "open"
        
        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing API key")
        
        token = credentials.credentials
        if hmac.compare_digest(token, self.api_key):
            self._audit_access(request, "system")
            return "system"
        
        from tools.shared import users_store
        for key, entry in users_store.central_tokens().items():
            if hmac.compare_digest(token, key):
                self._audit_access(request, entry["client_id"])
                return entry["client_id"]
        
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    @staticmethod
    def _audit_access(request: Request, who: str) -> None:
        logger.info(f"central.access user={who} {request.method} {request.url.path}")
    
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
                auth = websocket.headers.get("authorization", "")
                if auth.lower().startswith("bearer "):
                    auth = auth[7:]
                elif auth.startswith("X-API-Key: "):
                    auth = auth[10:]
                else:
                    auth = websocket.query_params.get("token", "")
                if not hmac.compare_digest(auth, self.api_key):
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
            """Set disabled tools for a server (persist, then push the DIFF to
            the running server — E1 runtime masks)."""
            before = set(get_disabled_tools(server_name))
            set_disabled_tools(server_name, disabled_list)
            after = set(disabled_list)
            runtime_results = {}
            for tool_name in sorted(after - before):
                runtime_results[tool_name] = await _push_runtime_mask(server_name, tool_name, True)
            for tool_name in sorted(before - after):
                runtime_results[tool_name] = await _push_runtime_mask(server_name, tool_name, False)
            return {
                "server": server_name,
                "disabled": disabled_list,
                "runtime": runtime_results,
            }

        @self.app.post("/api/internal/mask-push")
        async def internal_mask_push(
            request: MaskPushRequest,
            _: bool = Depends(self._verify_api_key)
        ):
            """M4: receive a fan-out mask push from a sibling node. Applies
            LOCALLY only (fanout=False) — no re-propagation, no loop."""
            result = await _push_runtime_mask(
                request.server_name, request.tool_name, request.masked,
                fanout=False)
            return {"node": os.environ.get("MCP_NODE_NAME", "unknown"),
                    **result}

        @self.app.post("/api/disabled-tools/{server_name}/{tool_name}/disable")
        async def disable_tool_endpoint(
            server_name: str,
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Disable a specific tool for a server (persist + push to the
            running server — E1 runtime masks)."""
            disable_tool(tool_name, server_name)
            runtime = await _push_runtime_mask(server_name, tool_name, True)
            return {"server": server_name, "tool": tool_name, "disabled": True, **runtime}

        # === E3/M2: user management (admin; guard hardens in M3) ===

        @self.app.get("/api/users")
        async def list_users_endpoint(_: bool = Depends(self._verify_api_key)):
            """List users — never returns mcp_key or password_hash."""
            return {"users": users_store.list_users()}

        @self.app.post("/api/users")
        async def create_user_endpoint(request: UserCreateRequest,
                                       _: bool = Depends(self._verify_api_key)):
            """Create a user; the mcp_key is returned ONCE."""
            try:
                created = users_store.create_user(
                    request.username, request.password, role=request.role,
                    servers=request.servers, masked_functions=request.masked_functions,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"user": users_store._public_view(
                users_store.get_user_record(created["username"])), "mcp_key": created["mcp_key"]}

        @self.app.delete("/api/users/{username}")
        async def delete_user_endpoint(username: str,
                                       _: bool = Depends(self._verify_api_key)):
            try:
                users_store.delete_user(username)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"deleted": True}

        @self.app.post("/api/users/{username}/rotate-key")
        async def rotate_user_key_endpoint(username: str,
                                           _: bool = Depends(self._verify_api_key)):
            try:
                return users_store.rotate_key(username)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.put("/api/users/{username}/password")
        async def set_user_password_endpoint(username: str,
                                             request: UserPasswordRequest,
                                             _: bool = Depends(self._verify_api_key)):
            try:
                users_store.set_password(username, request.password)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"ok": True}

        @self.app.put("/api/users/{username}/servers")
        async def set_user_servers_endpoint(username: str,
                                            request: UserServersRequest,
                                            _: bool = Depends(self._verify_api_key)):
            try:
                users_store.set_servers(username, request.servers)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"ok": True}

        @self.app.put("/api/users/{username}/masked-functions")
        async def set_user_masked_endpoint(username: str,
                                           request: UserMaskedRequest,
                                           _: bool = Depends(self._verify_api_key)):
            try:
                users_store.set_masked_functions(username, request.masked_functions)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"ok": True}

        @self.app.post("/api/users/{username}/enabled")
        async def set_user_enabled_endpoint(username: str,
                                            request: UserEnabledRequest,
                                            _: bool = Depends(self._verify_api_key)):
            try:
                users_store.set_enabled(username, request.enabled)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"ok": True}

        @self.app.put("/api/users/{username}/db-presets")
        async def set_user_db_presets_endpoint(username: str,
                                               request: UserPresetsRequest,
                                               _: bool = Depends(self._verify_api_key)):
            """E3.5: set the databasemcp presets granted to a user."""
            users_store.set_db_presets(username, request.presets)
            return {"ok": True, "username": username, "db_presets": request.presets}

        @self.app.put("/api/users/{username}/rag-collections")
        async def set_user_rag_collections_endpoint(username: str,
                                                    request: UserCollectionsRequest,
                                                    _: bool = Depends(self._verify_api_key)):
            """E3.5: set the ragmcp collections granted to a user."""
            users_store.set_rag_collections(username, request.collections)
            return {"ok": True, "username": username, "rag_collections": request.collections}

        @self.app.post("/api/disabled-tools/{server_name}/{tool_name}/enable")
        async def enable_tool_endpoint(
            server_name: str,
            tool_name: str,
            _: bool = Depends(self._verify_api_key)
        ):
            """Enable a specific tool for a server (persist + push to the
            running server — E1 runtime masks)."""
            enable_tool(tool_name, server_name)
            runtime = await _push_runtime_mask(server_name, tool_name, False)
            return {"server": server_name, "tool": tool_name, "disabled": False, **runtime}

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
                with config_path.open() as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid config.json")

            config.setdefault("auth", {})["api_key"] = request.api_key

            from tools.shared.atomic_io import atomic_write_json

            atomic_write_json(config_path, config)

            # M4: mirror to the shared state — every node adopts at boot
            import os as _os
            if _os.environ.get("MCP_STATE_BACKEND", "json").strip().lower() == "db":
                try:
                    from tools.shared import cluster
                    cluster.mirror_auth(tool_name, request.api_key)
                except Exception as e:
                    logger.warning(f"[M4] auth mirror failed for {tool_name}: {e}")

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
