"""
Tool discovery module for the MCP launcher system.

This module provides functionality to discover, load, and validate
MCP tools from configured directories.
"""

import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .errors import DiscoveryError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a discovered MCP tool."""
    name: str
    module_path: str
    file_path: str
    version: str = "unknown"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    exports: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"ToolMetadata(name={self.name}, path={self.module_path})"


class ToolDiscovery:
    """Discover and load MCP tools from directories."""
    
    # Required exports for SSE transport
    REQUIRED_EXPORTS_SSE = ["server", "app", "sse_transport"]
    # Required exports for Streamable HTTP transport (just app needed)
    REQUIRED_EXPORTS_STREAMABLE = ["app"]
    
    # Default patterns to exclude from tool discovery
    # These files are supplementary modules, not standalone MCP tools
    DEFAULT_EXCLUDE_PATTERNS = [
        "_sse",         # SSE transport variants (we prefer streamable)
        "migrate_",     # Migration scripts
        "copilot_context_injector",  # Helper module, not a tool
    ]
    
    def __init__(self, search_paths: List[str]):
        """
        Initialize the tool discovery manager.
        
        Args:
            search_paths: List of directories to search for MCP tools
        """
        self.search_paths = [Path(p) for p in search_paths]
        self.discovered_tools: Dict[str, ToolMetadata] = {}
        self.loaded_modules: Dict[str, Any] = {}
    
    def discover_tools(
        self,
        tool_names: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[ToolMetadata]:
        """
        Discover MCP tools from configured directories.
        
        Args:
            tool_names: Optional list of specific tools to discover
            exclude_patterns: Optional list of patterns to exclude
            
        Returns:
            List of discovered tool metadata
            
        Raises:
            DiscoveryError: If discovery fails
        """
        self.discovered_tools.clear()
        self.loaded_modules.clear()
        
        # Combine default patterns with user-specified patterns
        exclude_set = set(self.DEFAULT_EXCLUDE_PATTERNS)
        if exclude_patterns:
            exclude_set.update(exclude_patterns)
        
        for search_path in self.search_paths:
            if not search_path.exists():
                logger.warning(f"Search path does not exist: {search_path}")
                continue
            
            logger.info(f"Searching for MCP tools in: {search_path}")
            
            # Find all Python files in the directory
            for py_file in search_path.glob("*.py"):
                # Skip __init__.py and test files
                if py_file.name.startswith("_") or py_file.name.startswith("test_"):
                    continue
                
                # Skip excluded patterns
                if any(pattern in py_file.name for pattern in exclude_set):
                    logger.debug(f"Skipping excluded file: {py_file}")
                    continue
                
                try:
                    metadata = self._discover_tool(py_file)
                    
                    # Filter by tool names if specified
                    if tool_names is None or metadata.name in tool_names:
                        self.discovered_tools[metadata.name] = metadata
                        logger.info(f"Discovered tool: {metadata.name} from {py_file}")
                    else:
                        logger.debug(f"Skipping tool not in list: {metadata.name}")
                
                except ValidationError as e:
                    logger.warning(f"Tool validation failed for {py_file}: {e}")
                except Exception as e:
                    logger.error(f"Failed to discover tool from {py_file}: {e}")
        
        # Check if requested tools were found
        if tool_names:
            missing = set(tool_names) - set(self.discovered_tools.keys())
            if missing:
                logger.warning(f"Requested tools not found: {', '.join(missing)}")
        
        return list(self.discovered_tools.values())
    
    def _discover_tool(self, file_path: Path) -> ToolMetadata:
        """
        Discover a single MCP tool from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ToolMetadata for the discovered tool
            
        Raises:
            DiscoveryError: If discovery fails
            ValidationError: If tool validation fails
        """
        try:
            # Load the module
            module = self._load_module(file_path)
            
            # Validate the module
            self._validate_tool(module)
            
            # Extract metadata
            metadata = self._extract_metadata(module, file_path)
            
            # Store the loaded module
            self.loaded_modules[metadata.name] = module
            
            return metadata
        
        except ValidationError:
            raise
        except Exception as e:
            raise DiscoveryError(f"Failed to discover tool: {e}", path=str(file_path))
    
    def _load_module(self, file_path: Path) -> Any:
        """
        Load a Python module from a file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Loaded module object
        """
        # Create module name from file path
        module_name = file_path.stem
        
        # Load the module
        # Note: Individual tool files already add their parent directories to sys.path
        # so we don't need to add anything here
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise DiscoveryError(f"Failed to load module spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
    
    def _validate_tool(self, module: Any) -> None:
        """
        Validate that a module is a valid MCP tool.
        Supports both SSE and Streamable HTTP transport types.
        
        Args:
            module: Module to validate
            
        Raises:
            ValidationError: If module is not a valid MCP tool
        """
        # Check for Streamable HTTP first (just needs app)
        streamable_valid = False
        streamable_exports = {}
        for export_name in self.REQUIRED_EXPORTS_STREAMABLE:
            if hasattr(module, export_name):
                streamable_exports[export_name] = getattr(module, export_name)
        
        # If we have an app, check if it's FastAPI (Streamable HTTP)
        if "app" in streamable_exports:
            try:
                from fastapi import FastAPI
                if isinstance(streamable_exports["app"], FastAPI):
                    streamable_valid = True
                    logger.debug(f"Module validated as Streamable HTTP tool")
            except ImportError as e:
                logger.debug(f"Failed to import FastAPI: {e}")
                pass
        
        # If Streamable HTTP valid, we're done
        if streamable_valid:
            return
        
        # Check for SSE transport (needs server, app, sse_transport)
        missing_exports = []
        exports = {}
        
        for export_name in self.REQUIRED_EXPORTS_SSE:
            if not hasattr(module, export_name):
                missing_exports.append(export_name)
            else:
                exports[export_name] = getattr(module, export_name)
        
        if missing_exports:
            raise ValidationError(
                "Module is missing required exports",
                missing_exports=missing_exports
            )
        
        # Validate export types
        try:
            from mcp.server.lowlevel import Server
            from starlette.applications import Starlette
            from mcp.server.sse import SseServerTransport
            
            if not isinstance(exports["server"], Server):
                raise ValidationError(
                    f"'server' must be an instance of mcp.server.lowlevel.Server, "
                    f"got {type(exports['server']).__name__}"
                )
            
            if not isinstance(exports["app"], Starlette):
                raise ValidationError(
                    f"'app' must be an instance of starlette.applications.Starlette, "
                    f"got {type(exports['app']).__name__}"
                )
            
            if not isinstance(exports["sse_transport"], SseServerTransport):
                raise ValidationError(
                    f"'sse_transport' must be an instance of mcp.server.sse.SseServerTransport, "
                    f"got {type(exports['sse_transport']).__name__}"
                )
        
        except ImportError as e:
            raise ValidationError(f"Failed to import MCP/Starlette types: {e}")
    
    def _extract_metadata(self, module: Any, file_path: Path) -> ToolMetadata:
        """
        Extract metadata from a validated MCP tool module.
        
        Args:
            module: Validated module object
            file_path: Path to the module file
            
        Returns:
            ToolMetadata with extracted information
        """
        # Determine tool name from file name
        name = file_path.stem
        # Remove _streamable suffix to get normalized name
        if name.endswith("_streamable"):
            name = name[:-11]  # Remove "_streamable" (11 chars)
        
        # Extract version from __version__ if available
        version = getattr(module, "__version__", "unknown")
        
        # Extract description from __doc__ if available
        description = getattr(module, "__doc__", "")
        if description:
            description = description.strip().split("\n")[0]
        
        # Extract dependencies from requirements.txt if available
        dependencies = self._extract_dependencies(file_path.parent)
        
        # Extract exports - try both SSE and Streamable HTTP
        exports = {}
        for export_name in self.REQUIRED_EXPORTS_SSE:
            if hasattr(module, export_name):
                exports[export_name] = getattr(module, export_name)
        for export_name in self.REQUIRED_EXPORTS_STREAMABLE:
            if hasattr(module, export_name) and export_name not in exports:
                exports[export_name] = getattr(module, export_name)
        
        return ToolMetadata(
            name=name,
            module_path=name,
            file_path=str(file_path),
            version=version,
            description=description,
            dependencies=dependencies,
            exports=exports
        )
    
    def _extract_dependencies(self, module_dir: Path) -> List[str]:
        """
        Extract dependencies from requirements.txt if available.
        
        Args:
            module_dir: Directory containing the module
            
        Returns:
            List of dependencies
        """
        requirements_file = module_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    dependencies = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                return dependencies
            except Exception as e:
                logger.warning(f"Failed to read requirements.txt: {e}")
        
        return []
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """
        Get a discovered tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            ToolMetadata or None if not found
        """
        return self.discovered_tools.get(name)
    
    def get_tool_module(self, name: str) -> Optional[Any]:
        """
        Get a loaded tool module by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Module object or None if not found
        """
        return self.loaded_modules.get(name)
    
    def list_tools(self) -> List[str]:
        """
        List all discovered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.discovered_tools.keys())
    
    def get_all_tools(self) -> List[ToolMetadata]:
        """
        Get all discovered tools.
        
        Returns:
            List of all tool metadata
        """
        return list(self.discovered_tools.values())
