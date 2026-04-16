"""
Event Sourcing for FEF V3

Stores all configuration changes as events for audit trail and time-travel debugging.
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents a single event in the event store."""
    id: str
    timestamp: float
    tool_name: str
    extension_name: str
    operation: str
    params: dict[str, Any]
    result: dict[str, Any] | None = None
    user: str | None = None
    correlation_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EventStore:
    """
    Stores all configuration changes as events.
    
    Provides audit trail and time-travel debugging capabilities.
    Events are immutable and stored in chronological order.
    """
    
    def __init__(
        self,
        db_path: str = "~/.config/supreme-mcp-tools/events.db",
        max_events: int = 1000000
    ):
        """
        Initialize the event store.
        
        Args:
            db_path: Path to SQLite database
            max_events: Maximum number of events to keep
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    extension_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    params TEXT NOT NULL,
                    result TEXT,
                    user TEXT,
                    correlation_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_name
                ON events(tool_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_correlation_id
                ON events(correlation_id)
            """)
            
            conn.commit()
        
        logger.info(f"Event store initialized at {self.db_path}")
    
    def append(
        self,
        tool_name: str,
        extension_name: str,
        operation: str,
        params: dict[str, Any],
        result: dict[str, Any] | None = None,
        user: str | None = None,
        correlation_id: str | None = None
    ) -> str:
        """
        Append an event to the store.
        
        Args:
            tool_name: Tool name
            extension_name: Extension name
            operation: Operation type (query, mutate, execute)
            params: Operation parameters
            result: Operation result
            user: User identifier
            correlation_id: Request correlation ID
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = Event(
            id=event_id,
            timestamp=time.time(),
            tool_name=tool_name,
            extension_name=extension_name,
            operation=operation,
            params=params,
            result=result,
            user=user,
            correlation_id=correlation_id
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO events
                (id, timestamp, tool_name, extension_name, operation, params, result, user, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp,
                    event.tool_name,
                    event.extension_name,
                    event.operation,
                    json.dumps(event.params),
                    json.dumps(event.result) if event.result else None,
                    event.user,
                    event.correlation_id
                )
            )
            conn.commit()
        
        # Cleanup old events if needed
        self._cleanup()
        
        logger.debug(f"Appended event {event_id} for {tool_name}.{extension_name}")
        return event_id
    
    def get_event(self, event_id: str) -> Event | None:
        """
        Get a specific event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Event if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM events WHERE id = ?",
                (event_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return self._row_to_event(row)
        
        return None
    
    def get_history(
        self,
        tool_name: str,
        start_time: float | None = None,
        end_time: float | None = None,
        operation: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Event]:
        """
        Get event history for a tool.
        
        Args:
            tool_name: Tool name
            start_time: Start time filter (Unix timestamp)
            end_time: End time filter (Unix timestamp)
            operation: Operation type filter
            limit: Maximum events to return
            offset: Offset for pagination
            
        Returns:
            List of events
        """
        query = "SELECT * FROM events WHERE tool_name = ?"
        params = [tool_name]
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if operation is not None:
            query += " AND operation = ?"
            params.append(operation)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def get_all_history(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Event]:
        """
        Get event history across all tools.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum events to return
            offset: Offset for pagination
            
        Returns:
            List of events
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def get_by_correlation_id(self, correlation_id: str) -> list[Event]:
        """
        Get all events for a correlation ID.
        
        Args:
            correlation_id: Correlation ID
            
        Returns:
            List of events
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM events WHERE correlation_id = ? ORDER BY timestamp",
                (correlation_id,)
            )
            
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def replay(
        self,
        tool_name: str,
        target_time: float
    ) -> dict[str, Any]:
        """
        Replay events to reconstruct state at a specific time.
        
        Args:
            tool_name: Tool name
            target_time: Target time (Unix timestamp)
            
        Returns:
            Reconstructed state
        """
        events = self.get_history(
            tool_name,
            start_time=0,
            end_time=target_time,
            limit=10000
        )
        
        state = {}
        
        # Replay events in chronological order
        for event in reversed(events):
            key = f"{event.extension_name}"
            
            if event.operation == "mutate":
                # Apply mutation to state
                if key not in state:
                    state[key] = {}
                state[key].update(event.params)
            elif event.operation == "execute":
                # Record action execution
                if "actions" not in state:
                    state["actions"] = []
                state["actions"].append({
                    "extension": event.extension_name,
                    "params": event.params,
                    "timestamp": event.timestamp
                })
        
        return state
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get event store statistics.
        
        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            total_events = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT tool_name) FROM events")
            total_tools = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
            row = cursor.fetchone()
            earliest = row[0]
            latest = row[1]
            
            cursor = conn.execute(
                "SELECT operation, COUNT(*) FROM events GROUP BY operation"
            )
            by_operation = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "total_events": total_events,
                "total_tools": total_tools,
                "earliest_event": earliest,
                "latest_event": latest,
                "by_operation": by_operation
            }
    
    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert a database row to an Event."""
        return Event(
            id=row["id"],
            timestamp=row["timestamp"],
            tool_name=row["tool_name"],
            extension_name=row["extension_name"],
            operation=row["operation"],
            params=json.loads(row["params"]),
            result=json.loads(row["result"]) if row["result"] else None,
            user=row["user"],
            correlation_id=row["correlation_id"]
        )
    
    def _cleanup(self) -> None:
        """Remove oldest events if max_events exceeded."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            count = cursor.fetchone()[0]
            
            if count > self.max_events:
                to_remove = count - self.max_events
                conn.execute(
                    """
                    DELETE FROM events WHERE id IN (
                        SELECT id FROM events ORDER BY timestamp ASC LIMIT ?
                    )
                    """,
                    (to_remove,)
                )
                conn.commit()
                logger.info(f"Cleaned up {to_remove} old events")
    
    def clear(self) -> int:
        """
        Clear all events.
        
        Returns:
            Number of events cleared
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM events")
            conn.commit()
            logger.info(f"Cleared {count} events")
            return count


# Global event store instance
_event_store: EventStore | None = None


def get_event_store(
    db_path: str = "~/.config/supreme-mcp-tools/events.db"
) -> EventStore:
    """Get or create the global event store."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore(db_path=db_path)
    return _event_store
