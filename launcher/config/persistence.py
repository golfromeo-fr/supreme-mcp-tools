"""
JSON File-Based Configuration Persistence for FEF V3

Stores configuration changes in JSON files that survive restarts.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigPersistence:
    """
    File-based configuration persistence.
    
    Stores configuration changes in JSON files that survive restarts.
    """
    
    def __init__(self, config_dir: str = "~/.config/supreme-mcp-tools"):
        """
        Initialize the configuration persistence.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_tool_config_path(self, tool_name: str) -> Path:
        """Get path to tool's config file."""
        return self.config_dir / f"{tool_name}.json"
    
    def load(self, tool_name: str) -> dict[str, Any]:
        """
        Load persisted configuration for a tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Configuration dictionary
        """
        config_path = self._get_tool_config_path(tool_name)
        
        if config_path.exists():
            try:
                with Path(config_path).open("r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config for {tool_name}: {e}")
                return {}
        return {}
    
    def save(
        self,
        tool_name: str,
        extension_name: str,
        params: dict[str, Any],
        user: str | None = None
    ) -> None:
        """
        Save a configuration change.
        
        Args:
            tool_name: Tool name
            extension_name: Extension that was mutated
            params: New configuration values
            user: Optional user identifier
        """
        config_path = self._get_tool_config_path(tool_name)
        
        # Load existing config
        config = self.load(tool_name)
        
        # Update with new values
        if "mutations" not in config:
            config["mutations"] = []
        
        config["mutations"].append({
            "extension": extension_name,
            "params": params,
            "timestamp": time.time(),
            "user": user
        })
        
        # Save to file
        with Path(config_path).open("w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved config for {tool_name}.{extension_name}")
    
    def get_history(
        self,
        tool_name: str,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get configuration change history for a tool.
        
        Args:
            tool_name: Tool name
            limit: Maximum number of records
            
        Returns:
            List of configuration changes
        """
        config = self.load(tool_name)
        mutations = config.get("mutations", [])
        
        # Return most recent first
        return list(reversed(mutations[-limit:]))
    
    def delete(self, tool_name: str) -> bool:
        """
        Delete configuration for a tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            True if deleted, False if not found
        """
        config_path = self._get_tool_config_path(tool_name)
        
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Deleted config for {tool_name}")
            return True
        
        return False
    
    def list_tools(self) -> list[str]:
        """
        List all tools with persisted configuration.
        
        Returns:
            List of tool names
        """
        tools = []
        
        for path in self.config_dir.glob("*.json"):
            if path.stem != "audit":  # Skip audit log
                tools.append(path.stem)
        
        return tools
