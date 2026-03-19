# Configuration Persistence

## Persistence Strategies

### 1. JSON File-Based Persistence

Simple file-based storage for configuration changes:

```python
# launcher/config/persistence.py

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)


class ConfigPersistence:
    """
    File-based configuration persistence.
    
    Stores configuration changes in JSON files that survive restarts.
    """
    
    def __init__(self, config_dir: str = "~/.config/supreme-mcp-tools"):
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_tool_config_path(self, tool_name: str) -> Path:
        """Get path to tool's config file."""
        return self.config_dir / f"{tool_name}.json"
    
    def load(self, tool_name: str) -> Dict[str, Any]:
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
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config for {tool_name}: {e}")
                return {}
        return {}
    
    async def save(
        self,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Save a configuration change.
        
        Args:
            tool_name: Tool name
            extension_name: Extension that was mutated
            params: New configuration values
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
            "timestamp": time.time()
        })
        
        # Save to file
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved config for {tool_name}.{extension_name}")
```

### 2. SQLite-Based Persistence (Advanced)

For higher performance and query capabilities:

```python
# launcher/config/sqlite_persistence.py

import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class SQLitePersistence:
    """
    SQLite-based configuration persistence.
    
    Provides better performance and query capabilities for
    configuration history and audit trails.
    """
    
    def __init__(self, db_path: str = "~/.config/supreme-mcp-tools/config.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    extension_name TEXT NOT NULL,
                    params TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    user TEXT,
                    reason TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_name
                ON config_changes(tool_name)
            """)
    
    def save(
        self,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any],
        user: Optional[str] = None,
        reason: Optional[str] = None
    ) -> int:
        """
        Save a configuration change.
        
        Returns:
            Change ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO config_changes
                (tool_name, extension_name, params, timestamp, user, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_name,
                    extension_name,
                    json.dumps(params),
                    time.time(),
                    user,
                    reason
                )
            )
            return cursor.lastrowid
    
    def get_history(
        self,
        tool_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get configuration change history for a tool.
        
        Args:
            tool_name: Tool name
            limit: Maximum number of records
            
        Returns:
            List of configuration changes
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM config_changes
                WHERE tool_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (tool_name, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
```

## Configuration Manager

```python
# launcher/config/manager.py

import json
from pathlib import Path
from typing import Any, Dict
import logging
import threading

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading and persistence for a tool.
    
    Loads persisted configuration on startup and provides
    methods to update configuration at runtime.
    """
    
    def __init__(self, tool_name: str, config_dir: str = "~/.config/supreme-mcp-tools"):
        self.tool_name = tool_name
        self.config_dir = Path(config_dir).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / f"{tool_name}.json"
        self.config: Dict[str, Any] = {}
        self._lock = threading.RLock()
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
        """Get a configuration value."""
        with self._lock:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value and persist to disk."""
        with self._lock:
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
        """Get all configuration values."""
        with self._lock:
            return self.config.copy()
```

## Configuration File Format

```json
{
  "mutations": [
    {
      "extension": "api_key",
      "params": {
        "key": "sk-xxx"
      },
      "timestamp": 1679064000
    },
    {
      "extension": "rate_limit",
      "params": {
        "requests_per_minute": 120
      },
      "timestamp": 1679064100
    }
  ],
  "settings": {
    "cache_ttl": 60,
    "max_connections": 100
  }
}
```
