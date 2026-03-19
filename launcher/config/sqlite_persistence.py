"""
SQLite-Based Configuration Persistence for FEF V3

Provides better performance and query capabilities for configuration history.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLitePersistence:
    """
    SQLite-based configuration persistence.
    
    Provides better performance and query capabilities for
    configuration history and audit trails.
    """
    
    def __init__(self, db_path: str = "~/.config/supreme-mcp-tools/config.db"):
        """
        Initialize SQLite persistence.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
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
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON config_changes(timestamp)
            """)
            
            # Current config state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_state (
                    tool_name TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tool_name, key)
                )
            """)
            
            conn.commit()
        
        logger.info(f"SQLite persistence initialized at {self.db_path}")
    
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
        
        Args:
            tool_name: Tool name
            extension_name: Extension that was mutated
            params: New configuration values
            user: Optional user identifier
            reason: Optional reason for change
            
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
            conn.commit()
            
            change_id = cursor.lastrowid
            logger.info(f"Saved config change {change_id} for {tool_name}.{extension_name}")
            return change_id
    
    def get_history(
        self,
        tool_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get configuration change history for a tool.
        
        Args:
            tool_name: Tool name
            limit: Maximum number of records
            offset: Offset for pagination
            
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
                LIMIT ? OFFSET ?
                """,
                (tool_name, limit, offset)
            )
            
            results = []
            for row in cursor.fetchall():
                entry = dict(row)
                entry["params"] = json.loads(entry["params"])
                results.append(entry)
            
            return results
    
    def get_all_history(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get configuration change history across all tools.
        
        Args:
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            List of configuration changes
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM config_changes
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            
            results = []
            for row in cursor.fetchall():
                entry = dict(row)
                entry["params"] = json.loads(entry["params"])
                results.append(entry)
            
            return results
    
    def set_state(
        self,
        tool_name: str,
        key: str,
        value: Any
    ) -> None:
        """
        Set a configuration state value.
        
        Args:
            tool_name: Tool name
            key: Configuration key
            value: Configuration value
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO config_state
                (tool_name, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (tool_name, key, json.dumps(value), time.time())
            )
            conn.commit()
    
    def get_state(
        self,
        tool_name: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a configuration state value.
        
        Args:
            tool_name: Tool name
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT value FROM config_state
                WHERE tool_name = ? AND key = ?
                """,
                (tool_name, key)
            )
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return default
    
    def get_all_state(self, tool_name: str) -> Dict[str, Any]:
        """
        Get all configuration state for a tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Dictionary of configuration values
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT key, value FROM config_state
                WHERE tool_name = ?
                """,
                (tool_name,)
            )
            
            result = {}
            for row in cursor.fetchall():
                result[row[0]] = json.loads(row[1])
            
            return result
    
    def delete_state(self, tool_name: str, key: str) -> bool:
        """
        Delete a configuration state value.
        
        Args:
            tool_name: Tool name
            key: Configuration key
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM config_state
                WHERE tool_name = ? AND key = ?
                """,
                (tool_name, key)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def list_tools(self) -> List[str]:
        """
        List all tools with configuration.
        
        Returns:
            List of tool names
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT tool_name FROM config_changes
                UNION
                SELECT DISTINCT tool_name FROM config_state
                """
            )
            return [row[0] for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get persistence statistics.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM config_changes")
            total_changes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM config_state")
            total_state = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT tool_name) FROM config_changes"
            )
            total_tools = cursor.fetchone()[0]
            
            return {
                "total_changes": total_changes,
                "total_state_entries": total_state,
                "total_tools": total_tools
            }
