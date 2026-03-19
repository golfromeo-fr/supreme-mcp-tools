"""
Dead Letter Queue for FEF V3

Stores failed operations for manual review and retry.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeadLetterEntry:
    """Represents a failed operation in the dead letter queue."""
    id: str
    operation: str
    tool_name: str
    extension_name: str
    params: Dict[str, Any]
    error: str
    timestamp: float
    status: str = "pending"  # pending, processed, failed
    attempts: int = 0
    processed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DeadLetterQueue:
    """
    Queue for failed operations requiring manual review.
    
    Stores failed operations as JSON files that can be inspected
    and retried manually or automatically.
    """
    
    def __init__(
        self,
        queue_dir: str = "~/.config/supreme-mcp-tools/dlq",
        max_entries: int = 1000
    ):
        """
        Initialize the dead letter queue.
        
        Args:
            queue_dir: Directory to store queue entries
            max_entries: Maximum number of entries to keep
        """
        self.queue_dir = Path(queue_dir).expanduser()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
    
    def enqueue(
        self,
        operation: str,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any],
        error: str
    ) -> str:
        """
        Add a failed operation to the queue.
        
        Args:
            operation: Operation type (query, mutate, execute)
            tool_name: Tool name
            extension_name: Extension name
            params: Operation parameters
            error: Error message
            
        Returns:
            Entry ID
        """
        entry_id = str(uuid.uuid4())
        entry = DeadLetterEntry(
            id=entry_id,
            operation=operation,
            tool_name=tool_name,
            extension_name=extension_name,
            params=params,
            error=error,
            timestamp=time.time()
        )
        
        entry_path = self.queue_dir / f"{entry_id}.json"
        with open(entry_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        logger.info(f"Added to DLQ: {operation} on {tool_name}/{extension_name}")
        
        # Cleanup old entries if needed
        self._cleanup()
        
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """
        Get a specific entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            DeadLetterEntry if found, None otherwise
        """
        entry_path = self.queue_dir / f"{entry_id}.json"
        
        if not entry_path.exists():
            return None
        
        try:
            with open(entry_path, "r") as f:
                data = json.load(f)
            return DeadLetterEntry(**data)
        except Exception as e:
            logger.error(f"Error reading DLQ entry {entry_id}: {e}")
            return None
    
    def get_pending(
        self,
        tool_name: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100
    ) -> List[DeadLetterEntry]:
        """
        Get pending entries.
        
        Args:
            tool_name: Optional filter by tool name
            operation: Optional filter by operation type
            limit: Maximum entries to return
            
        Returns:
            List of pending entries
        """
        entries = []
        
        for path in sorted(self.queue_dir.glob("*.json"), reverse=True):
            if len(entries) >= limit:
                break
            
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                entry = DeadLetterEntry(**data)
                
                if entry.status != "pending":
                    continue
                
                if tool_name and entry.tool_name != tool_name:
                    continue
                
                if operation and entry.operation != operation:
                    continue
                
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Error reading DLQ entry {path}: {e}")
                continue
        
        return entries
    
    def get_all(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[DeadLetterEntry]:
        """
        Get all entries, optionally filtered by status.
        
        Args:
            status: Optional status filter
            limit: Maximum entries to return
            
        Returns:
            List of entries
        """
        entries = []
        
        for path in sorted(self.queue_dir.glob("*.json"), reverse=True):
            if len(entries) >= limit:
                break
            
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                entry = DeadLetterEntry(**data)
                
                if status and entry.status != status:
                    continue
                
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Error reading DLQ entry {path}: {e}")
                continue
        
        return entries
    
    def mark_processed(
        self,
        entry_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark an entry as processed.
        
        Args:
            entry_id: Entry ID
            result: Optional result data
            
        Returns:
            True if marked, False if not found
        """
        entry_path = self.queue_dir / f"{entry_id}.json"
        
        if not entry_path.exists():
            return False
        
        try:
            with open(entry_path, "r") as f:
                data = json.load(f)
            
            data["status"] = "processed"
            data["processed_at"] = time.time()
            if result:
                data["result"] = result
            
            with open(entry_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Marked DLQ entry {entry_id} as processed")
            return True
        except Exception as e:
            logger.error(f"Error marking DLQ entry {entry_id}: {e}")
            return False
    
    def mark_failed(self, entry_id: str, error: str) -> bool:
        """
        Mark an entry as permanently failed.
        
        Args:
            entry_id: Entry ID
            error: Error message
            
        Returns:
            True if marked, False if not found
        """
        entry_path = self.queue_dir / f"{entry_id}.json"
        
        if not entry_path.exists():
            return False
        
        try:
            with open(entry_path, "r") as f:
                data = json.load(f)
            
            data["status"] = "failed"
            data["processed_at"] = time.time()
            data["error"] = error
            data["attempts"] = data.get("attempts", 0) + 1
            
            with open(entry_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Marked DLQ entry {entry_id} as failed")
            return True
        except Exception as e:
            logger.error(f"Error marking DLQ entry {entry_id}: {e}")
            return False
    
    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry from the queue.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if deleted, False if not found
        """
        entry_path = self.queue_dir / f"{entry_id}.json"
        
        if entry_path.exists():
            entry_path.unlink()
            logger.info(f"Deleted DLQ entry {entry_id}")
            return True
        
        return False
    
    def clear(self, status: Optional[str] = None) -> int:
        """
        Clear entries from the queue.
        
        Args:
            status: Optional status filter (clears all if not specified)
            
        Returns:
            Number of entries cleared
        """
        count = 0
        
        for path in self.queue_dir.glob("*.json"):
            try:
                if status:
                    with open(path, "r") as f:
                        data = json.load(f)
                    if data.get("status") != status:
                        continue
                
                path.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error deleting DLQ entry {path}: {e}")
                continue
        
        logger.info(f"Cleared {count} DLQ entries")
        return count
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with counts by status
        """
        stats = {"pending": 0, "processed": 0, "failed": 0, "total": 0}
        
        for path in self.queue_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                status = data.get("status", "unknown")
                stats[status] = stats.get(status, 0) + 1
                stats["total"] += 1
            except Exception:
                continue
        
        return stats
    
    def _cleanup(self) -> None:
        """Remove oldest entries if max_entries exceeded."""
        entries = list(self.queue_dir.glob("*.json"))
        
        if len(entries) <= self.max_entries:
            return
        
        # Sort by modification time
        entries.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest entries
        to_remove = entries[:len(entries) - self.max_entries]
        for path in to_remove:
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Error removing old DLQ entry {path}: {e}")
        
        logger.info(f"Cleaned up {len(to_remove)} old DLQ entries")
