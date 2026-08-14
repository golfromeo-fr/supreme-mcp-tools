"""
Tool discovery module for the MCP launcher system.

This module provides functionality to discover, load, and validate
MCP tools from configured directories.
"""

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    dependencies: list[str] = field(default_factory=list)
    exports: dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"ToolMetadata(name={self.name}, path={self.module_path})"


class ToolDiscovery:
    """Discover and load MCP tools from directories."""
    
    # Required exports for the streamable HTTP transport (just app needed)
    REQUIRED_EXPORTS_STREAMABLE = ["app"]
    
    # Default patterns to exclude from tool discovery
    # These files are supplementary modules, not standalone MCP tools
    DEFAULT_EXCLUDE_PATTERNS = [
        "_streamable",   # Obsoleted by _fastmcp variants
        "migrate_",     # Migration scripts
        "copilot_context_injector",  # Helper module, not a tool
    ]
    
    def __init__(self, search_paths: list[str]):
        """
        Initialize the tool discovery manager.
        
        Args:
            search_paths: List of directories to search for MCP tools
        """
        self.search_paths = [Path(p) for p in search_paths]
        self.discovered_tools: dict[str, ToolMetadata] = {}
        self.loaded_modules: dict[str, Any] = {}

    def discover(self) -> dict[str, ToolMetadata]:
        """Discover all tools and return as a dict.

        Returns:
            Dictionary mapping tool name to ToolMetadata.
        """
        self.discover_tools()
        return self.discovered_tools

    def discover_tools(
        self,
        tool_names: list[str] | None = None,
        exclude_patterns: list[str] | None = None
    ) -> list[ToolMetadata]:
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
            
            # Determine whether to search this directory directly or expand into subdirectories.
            # A "container" directory (like tools/) has subdirectories but no *_fastmcp.py files.
            # A "tool" directory (like tools/ragmcp/) contains its own *_fastmcp.py entry point.
            has_fastmcp = any(search_path.glob("*_fastmcp.py"))
            subdirs = [d for d in search_path.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
            
            if has_fastmcp:
                # Tool directory — search its .py files directly, don't descend into subdirs
                expanded = [search_path]
            elif subdirs:
                # Container directory — expand into subdirectories
                expanded = subdirs
            else:
                expanded = [search_path]
            
            for tool_dir in expanded:
                logger.info(f"Searching for MCP tools in: {tool_dir}")

                # Find Python files in the tool directory only (not subdirectories)
                # Subdirectories contain support modules (indexer/, shared/, etc.)
                #
                # If any *_fastmcp.py exists in the directory, ONLY consider those
                # as tool candidates — all other .py files are support modules.
                fastmcp_files = list(tool_dir.glob("*_fastmcp.py"))

                for py_file in tool_dir.glob("*.py"):
                    # Skip __init__.py and test files
                    if py_file.name.startswith("_") or py_file.name.startswith("test_"):
                        continue

                    # Skip excluded patterns
                    if any(pattern in py_file.name for pattern in exclude_set):
                        logger.debug(f"Skipping excluded file: {py_file}")
                        continue

                    # If fastmcp files exist, only consider fastmcp files as tools
                    if fastmcp_files and not py_file.name.endswith("_fastmcp.py"):
                        logger.debug(f"Skipping support module: {py_file.name} (fastmcp entry point exists)")
                        continue

                    try:
                        metadata = self._discover_tool(py_file)

                        # Filter by tool names if specified
                        if tool_names is None or metadata.name in tool_names:
                            # When duplicate names exist, prefer fastmcp > non-suffixed > streamable
                            existing = self.discovered_tools.get(metadata.name)
                            if existing:
                                existing_is_fastmcp = existing.file_path.endswith("_fastmcp.py")
                                new_is_fastmcp = str(py_file).endswith("_fastmcp.py")
                                if new_is_fastmcp and not existing_is_fastmcp:
                                    # Replace non-fastmcp with fastmcp version
                                    self.discovered_tools[metadata.name] = metadata
                                    logger.info(f"Discovered tool: {metadata.name} from {py_file} (fastmcp variant)")
                                elif not new_is_fastmcp and existing_is_fastmcp:
                                    # Skip non-fastmcp when fastmcp already exists
                                    logger.debug(f"Skipping {py_file} - fastmcp variant already discovered")
                                else:
                                    logger.debug(f"Skipping duplicate tool: {metadata.name} from {py_file}")
                            else:
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
        Supports FastAPI and FastMCP (Starlette) tool shapes.
        
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

        # Check for FastMCP (Starlette app with mcp attribute)
        fastmcp_valid = False
        if hasattr(module, "app") and hasattr(module, "mcp"):
            from starlette.applications import Starlette
            if isinstance(getattr(module, "app"), Starlette):
                fastmcp_valid = True
                logger.debug(f"Module validated as FastMCP tool")

        if fastmcp_valid:
            return

        # No supported tool shape (SSE-style exports were removed 2026-08, Phase -1)
        raise ValidationError(
            "Module is missing required exports (need FastAPI 'app' or FastMCP 'app' + 'mcp')",
            missing_exports=["app"],
        )
    
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
        # Remove _fastmcp or _streamable suffix to get normalized name
        if name.endswith("_fastmcp"):
            name = name[:-8]   # Remove "_fastmcp" (8 chars)
        elif name.endswith("_streamable"):
            name = name[:-11]  # Remove "_streamable" (11 chars)
        
        # Extract version from __version__ if available
        version = getattr(module, "__version__", "unknown")
        
        # Extract description from __doc__ if available
        description = getattr(module, "__doc__", "")
        if description:
            description = description.strip().split("\n")[0]
        
        # Extract dependencies from requirements.txt if available
        dependencies = self._extract_dependencies(file_path.parent)
        
        # Extract exports
        exports = {}
        for export_name in self.REQUIRED_EXPORTS_STREAMABLE:
            if hasattr(module, export_name):
                exports[export_name] = getattr(module, export_name)
        
        # Store module reference for launcher to call setup_extensions()
        exports["_module"] = module
        
        return ToolMetadata(
            name=name,
            module_path=name,
            file_path=str(file_path),
            version=version,
            description=description,
            dependencies=dependencies,
            exports=exports
        )
    
    def _extract_dependencies(self, module_dir: Path) -> list[str]:
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
                with Path(requirements_file).open('r') as f:
                    dependencies = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                return dependencies
            except Exception as e:
                logger.warning(f"Failed to read requirements.txt: {e}")
        
        return []
    
    def get_tool(self, name: str) -> ToolMetadata | None:
        """
        Get a discovered tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            ToolMetadata or None if not found
        """
        return self.discovered_tools.get(name)
    
    def get_tool_module(self, name: str) -> Any | None:
        """
        Get a loaded tool module by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Module object or None if not found
        """
        return self.loaded_modules.get(name)
    
    def list_tools(self) -> list[str]:
        """
        List all discovered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.discovered_tools.keys())
    
    def get_all_tools(self) -> list[ToolMetadata]:
        """
        Get all discovered tools.
        
        Returns:
            List of all tool metadata
        """
        return list(self.discovered_tools.values())

