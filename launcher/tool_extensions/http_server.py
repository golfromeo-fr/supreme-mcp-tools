"""
Extension HTTP Server

Each tool exposes an HTTP API for extension management.
This server provides REST endpoints and WebSocket support for cross-process communication.
"""

import asyncio
import hmac
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

from .registry import ExtensionRegistry
from ..config_types import DEFAULT_HOST

logger = logging.getLogger(__name__)


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


# API Key header for authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class ExtensionHTTPServer:
    """
    HTTP server for extension management in tool processes.
    
    Provides REST endpoints for:
    - Health checks
    - Listing extensions
    - Querying data sources
    - Mutating configuration
    - Executing actions
    - WebSocket event streaming
    """
    
    @staticmethod
    def _get_default_mgmt_port(tool_name: str) -> int | None:
        """Get the default management port for a tool from ports.json.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Port number if found, None otherwise
        """
        try:
            from launcher.launcher_config import load_ports_config
            ports_config = load_ports_config()
            # Look in assignments.mgmt.<tool_name>
            assignments = ports_config.get("assignments", {})
            mgmt_ports = assignments.get("mgmt", {})
            return mgmt_ports.get(tool_name)
        except Exception:
            return None
    
    def __init__(
        self,
        tool_name: str,
        registry: ExtensionRegistry,
        config_manager: Any | None = None,
        port: int | None = None,
        host: str = DEFAULT_HOST,
        api_key: str | None = None
    ):
        """
        Initialize the extension HTTP server.
        
        Args:
            tool_name: Name of the tool this server belongs to
            registry: Extension registry instance
            config_manager: Optional configuration manager
            port: Port to listen on (required, loaded from ports.json if not provided)
            host: Host to bind to
            api_key: Optional API key for authentication
            
        Raises:
            ValueError: If port is not provided and cannot be loaded from ports.json
        """
        self.tool_name = tool_name
        self.registry = registry
        self.config_manager = config_manager
        
        # Get port from ports.json if not specified
        if port is None:
            port = self._get_default_mgmt_port(tool_name)
            if port is None:
                raise ValueError(
                    f"Port not specified for {tool_name} and not found in ports.json. "
                    f"Please configure ports.json with a management port for {tool_name}."
                )
        
        self.port = port
        self.host = host
        self.api_key = api_key
        
        self.app = FastAPI(
            title=f"{tool_name} Management API",
            description=f"Extension management API for {tool_name}",
            version="1.0.0"
        )
        
        # Add CORS middleware
        _mgmt_port = self._get_default_mgmt_port(tool_name) or 8400
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

    def _verify_api_key(self, key: str | None = Depends(API_KEY_HEADER)) -> bool:
        """Verify API key if configured. If no key is configured, allow access."""
        if self.api_key is None:
            return True  # No auth configured
        if key is None:
            raise HTTPException(status_code=401, detail="Missing API key")
        if not hmac.compare_digest(key, self.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True
    
    def _register_routes(self) -> None:
        """Register all API routes."""
        
        import os
        self._health_check_logs = os.environ.get("MCP_HEALTH_CHECK_LOGS", "enable")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint - optimized for fast response."""
            import time
            start = time.time()
            import logging
            logger = logging.getLogger(__name__)
            
            # Only log if enabled
            should_log = self._health_check_logs != "disable"
            log_errors_only = self._health_check_logs == "errors-only"
            
            if should_log and not log_errors_only:
                logger.warning(f"[HEALTH_CHECK] Received health check request at {start}")
            try:
                # Use include_data=False to avoid calling slow data source handlers
                extensions_count = sum(
                    len(exts) for exts in self.registry.list_extensions(self.tool_name, include_data=False).values()
                )
                elapsed = time.time() - start
                # Only log completion if not disabled and not errors-only mode
                if should_log and not log_errors_only:
                    logger.warning(f"[HEALTH_CHECK] Completed in {elapsed:.3f}s, count={extensions_count}")
                return {
                    "status": "healthy",
                    "tool": self.tool_name,
                    "extensions_count": extensions_count
                }
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[HEALTH_CHECK] Failed after {elapsed:.3f}s: {e}")
                raise
        
        @self.app.get("/extensions")
        async def list_extensions(_: bool = Depends(self._verify_api_key)):
            """List all extensions registered by this tool."""
            extensions = self.registry.list_extensions(self.tool_name)
            return extensions.get(self.tool_name, [])
        
        @self.app.get("/extensions/{extension_name}")
        async def get_extension(extension_name: str, _: bool = Depends(self._verify_api_key)):
            """Get details of a specific extension."""
            ext = self.registry.get_extension(self.tool_name, extension_name)
            if ext is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Extension '{extension_name}' not found"
                )
            return ext.to_dict()
        
        @self.app.post("/extensions/{extension_name}/query")
        async def query_extension(extension_name: str, request: QueryRequest, _: bool = Depends(self._verify_api_key)):
            """Query a data source extension."""
            try:
                result = self.registry.query(
                    self.tool_name,
                    extension_name,
                    request.params
                )
                return {"data": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error querying extension: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/extensions/{extension_name}/mutate")
        async def mutate_extension(extension_name: str, request: MutateRequest, _: bool = Depends(self._verify_api_key)):
            """Mutate configuration via extension."""
            try:
                result = self.registry.mutate(
                    self.tool_name,
                    extension_name,
                    request.params
                )
                return {"result": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error mutating extension: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/extensions/{extension_name}/execute")
        async def execute_extension(extension_name: str, request: ExecuteRequest, _: bool = Depends(self._verify_api_key)):
            """Execute an action extension."""
            try:
                result = self.registry.execute(
                    self.tool_name,
                    extension_name,
                    request.params
                )
                return {"result": result}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error executing extension: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.websocket("/extensions/{extension_name}/events")
        async def websocket_events(websocket: WebSocket, extension_name: str, _: bool = Depends(self._verify_api_key)):
            """WebSocket endpoint for real-time event streaming."""
            await websocket.accept()

            # Verify extension exists
            ext = self.registry.get_extension(self.tool_name, extension_name)
            if ext is None:
                await websocket.close(code=4004, reason=f"Extension '{extension_name}' not found")
                return

            # Subscribe to events
            queue = self.registry.subscribe_queue(self.tool_name, extension_name)

            try:
                while True:
                    # Wait for events and send to client
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        await websocket.send_json(event)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for {extension_name}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.registry.unsubscribe_queue(self.tool_name, extension_name, queue)

        @self.app.get("/config")
        async def get_config(_: bool = Depends(self._verify_api_key)):
            """Get current configuration."""
            if self.config_manager is None:
                raise HTTPException(
                    status_code=404,
                    detail="Configuration manager not available"
                )
            return self.config_manager.get_all()

        @self.app.post("/config/{key}")
        async def set_config(key: str, value: dict[str, Any], _: bool = Depends(self._verify_api_key)):
            """Set a configuration value."""
            if self.config_manager is None:
                raise HTTPException(
                    status_code=404,
                    detail="Configuration manager not available"
                )
            await self.config_manager.set(key, value.get("value"))
            return {"success": True, "key": key, "value": value.get("value")}
    
    async def start(self) -> None:
        """Start the HTTP server."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=self._health_check_logs != "disable"
        )
        self._server = uvicorn.Server(config)
        
        logger.info(f"Starting management server for {self.tool_name} on port {self.port}")
        
        self._task = asyncio.create_task(self._server.serve())
        
        # Wait for server to start
        await asyncio.sleep(0.5)
        
        logger.info(f"Management server for {self.tool_name} started on port {self.port}")
    
    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.should_exit = True
            if self._task:
                await self._task
            logger.info(f"Management server for {self.tool_name} stopped")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app
