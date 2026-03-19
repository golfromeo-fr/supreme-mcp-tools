"""
Management Server

The main management server that provides a unified API for managing all MCP tools.
This server acts as the central hub for the Flexible Extensibility Framework V3.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

from .service_registry import ServiceRegistry
from .distributed_registry import DistributedExtensionRegistry

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying data sources."""
    params: Optional[Dict[str, Any]] = None


class MutateRequest(BaseModel):
    """Request model for mutating configuration."""
    params: Dict[str, Any]


class ExecuteRequest(BaseModel):
    """Request model for executing actions."""
    params: Optional[Dict[str, Any]] = None


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
        port: int = 9091,
        host: str = "0.0.0.0",
        api_key: Optional[str] = None
    ):
        """
        Initialize the management server.
        
        Args:
            service_registry: Service registry for tool discovery
            port: Port to listen on
            host: Host to bind to
            api_key: Optional API key for authentication
        """
        self.service_registry = service_registry
        self.registry = DistributedExtensionRegistry(service_registry)
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
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._server: Optional[uvicorn.Server] = None
        self._task: Optional[asyncio.Task] = None
        
        self._register_routes()
    
    def _verify_api_key(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
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
            ext_type: Optional[str] = None,
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
            ext_type: Optional[str] = None,
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
            access_log=True
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
