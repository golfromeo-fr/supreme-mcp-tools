"""
Configuration Manager for FEF V3

Manages configuration loading and persistence for a tool.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading and persistence for a tool.
    
    Loads persisted configuration on startup and provides
    methods to update configuration at runtime.
    """
    
    def __init__(
        self,
        tool_name: str,
        config_dir: str = "~/.config/supreme-mcp-tools"
    ):
        """
        Initialize the configuration manager.
        
        Args:
            tool_name: Tool name
            config_dir: Configuration directory
        """
        self.tool_name = tool_name
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / f"{tool_name}.json"
        self.config: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from disk."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config for {self.tool_name}")
            except Exception as e:
                logger.error(f"Error loading config for {self.tool_name}: {e}")
                self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value and persist to disk.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        async with self._lock:
            self.config[key] = value
            self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to disk."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config for {self.tool_name}: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Copy of all configuration
        """
        return self.config.copy()
    
    def delete(self, key: str) -> bool:
        """
        Delete a configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self.config:
            del self.config[key]
            self._save_config()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all configuration."""
        self.config.clear()
        self._save_config()
