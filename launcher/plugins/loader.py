"""
Plugin Loader for FEF V3

Loads extensions from external plugin packages.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Loads extensions from external plugin packages.
    
    Plugins are Python files in a designated directory that expose
    a `register` function to add extensions to the registry.
    """
    
    def __init__(
        self,
        plugin_dir: str = "~/.supreme-mcp-tools/plugins",
        registry: Any | None = None
    ):
        """
        Initialize the plugin loader.
        
        Args:
            plugin_dir: Directory containing plugin files
            registry: Extension registry to register plugins with
        """
        self.plugin_dir = Path(plugin_dir).expanduser()
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.loaded_plugins: dict[str, Any] = {}
    
    def discover_plugins(self) -> list[str]:
        """
        Discover available plugins.
        
        Returns:
            List of plugin names (without .py extension)
        """
        plugins = []
        
        for path in self.plugin_dir.glob("*.py"):
            if not path.name.startswith("_"):
                plugins.append(path.stem)
        
        logger.info(f"Discovered {len(plugins)} plugins")
        return plugins
    
    def load_plugin(
        self,
        plugin_name: str,
        tool_name: str,
        registry: Any | None = None
    ) -> Any:
        """
        Load a plugin and register its extensions.
        
        Args:
            plugin_name: Plugin name (without .py extension)
            tool_name: Tool name to register extensions for
            registry: Optional registry override
            
        Returns:
            Loaded plugin module
            
        Raises:
            FileNotFoundError: If plugin file not found
            ValueError: If plugin has no register function
        """
        plugin_path = self.plugin_dir / f"{plugin_name}.py"
        
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin not found: {plugin_name}")
        
        # Load module
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            if plugin_name in sys.modules:
                del sys.modules[plugin_name]
            raise
        sys.modules[plugin_name] = module
        
        # Call plugin's register function
        if hasattr(module, "register"):
            reg = registry or self.registry
            if reg is None:
                raise ValueError("No registry provided for plugin registration")
            
            module.register(reg, tool_name)
            self.loaded_plugins[plugin_name] = module
            logger.info(f"Loaded plugin: {plugin_name} for {tool_name}")
            return module
        else:
            raise ValueError(f"Plugin {plugin_name} has no register function")
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if unloaded, False if not found
        """
        if plugin_name in self.loaded_plugins:
            del self.loaded_plugins[plugin_name]
            
            if plugin_name in sys.modules:
                del sys.modules[plugin_name]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        
        return False
    
    def reload_plugin(
        self,
        plugin_name: str,
        tool_name: str,
        registry: Any | None = None
    ) -> Any:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Plugin name
            tool_name: Tool name
            registry: Optional registry override
            
        Returns:
            Reloaded plugin module
        """
        self.unload_plugin(plugin_name)
        return self.load_plugin(plugin_name, tool_name, registry)
    
    def load_all_plugins(
        self,
        tool_name: str,
        registry: Any | None = None
    ) -> dict[str, Any]:
        """
        Load all discovered plugins.
        
        Args:
            tool_name: Tool name
            registry: Optional registry override
            
        Returns:
            Dictionary of loaded plugin modules
        """
        plugins = self.discover_plugins()
        loaded = {}
        
        for plugin_name in plugins:
            try:
                module = self.load_plugin(plugin_name, tool_name, registry)
                loaded[plugin_name] = module
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
        
        logger.info(f"Loaded {len(loaded)}/{len(plugins)} plugins")
        return loaded
    
    def get_loaded_plugins(self) -> list[str]:
        """
        Get list of loaded plugin names.
        
        Returns:
            List of loaded plugin names
        """
        return list(self.loaded_plugins.keys())
    
    def get_plugin_info(self, plugin_name: str) -> dict[str, Any] | None:
        """
        Get information about a loaded plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin info dictionary or None
        """
        module = self.loaded_plugins.get(plugin_name)
        if module is None:
            return None
        
        info = {
            "name": plugin_name,
            "file": getattr(module, "__file__", None),
            "has_register": hasattr(module, "register"),
        }
        
        # Get docstring
        if module.__doc__:
            info["description"] = module.__doc__.strip()
        
        # Get version if available
        if hasattr(module, "__version__"):
            info["version"] = module.__version__
        
        return info


class PluginRegistry:
    """
    Registry for managing plugins across multiple tools.
    
    Provides a centralized way to manage plugins for all tools.
    """
    
    def __init__(self, plugin_dir: str = "~/.supreme-mcp-tools/plugins"):
        """
        Initialize the plugin registry.
        
        Args:
            plugin_dir: Directory containing plugin files
        """
        self.plugin_dir = plugin_dir
        self.loaders: dict[str, PluginLoader] = {}
    
    def get_loader(self, tool_name: str, registry: Any) -> PluginLoader:
        """
        Get or create a plugin loader for a tool.
        
        Args:
            tool_name: Tool name
            registry: Extension registry
            
        Returns:
            PluginLoader instance
        """
        if tool_name not in self.loaders:
            self.loaders[tool_name] = PluginLoader(
                plugin_dir=self.plugin_dir,
                registry=registry
            )
        
        return self.loaders[tool_name]
    
    def load_plugins_for_tool(
        self,
        tool_name: str,
        registry: Any
    ) -> dict[str, Any]:
        """
        Load all plugins for a tool.
        
        Args:
            tool_name: Tool name
            registry: Extension registry
            
        Returns:
            Dictionary of loaded plugins
        """
        loader = self.get_loader(tool_name, registry)
        return loader.load_all_plugins(tool_name, registry)
    
    def unload_all_plugins(self, tool_name: str) -> None:
        """
        Unload all plugins for a tool.
        
        Args:
            tool_name: Tool name
        """
        if tool_name in self.loaders:
            loader = self.loaders[tool_name]
            for plugin_name in loader.get_loaded_plugins():
                loader.unload_plugin(plugin_name)
            del self.loaders[tool_name]
    
    def get_all_loaded_plugins(self) -> dict[str, list[str]]:
        """
        Get all loaded plugins across all tools.
        
        Returns:
            Dictionary mapping tool names to plugin lists
        """
        return {
            tool_name: loader.get_loaded_plugins()
            for tool_name, loader in self.loaders.items()
        }
