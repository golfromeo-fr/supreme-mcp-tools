"""
Audit Logging for FEF V3

Provides security event logging for compliance and debugging.
"""

import logging
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("audit")


class AuditAction(Enum):
    """Audit action types."""
    AUTHENTICATE = "authenticate"
    AUTHENTICATE_FAILED = "authenticate_failed"
    QUERY = "query"
    MUTATE = "mutate"
    EXECUTE = "execute"
    CONFIG_CHANGE = "config_change"
    TOOL_START = "tool_start"
    TOOL_STOP = "tool_stop"
    EXTENSION_REGISTER = "extension_register"
    EXTENSION_UNREGISTER = "extension_unregister"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ACCESS_DENIED = "access_denied"


@dataclass
class AuditEntry:
    """Represents a single audit log entry."""
    timestamp: str
    action: str
    user: str
    tool_name: str
    details: Dict[str, Any]
    success: bool = True
    ip_address: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Audit logger for security-critical operations.
    
    Logs all security-relevant events to a file for compliance and debugging.
    """
    
    def __init__(
        self,
        log_file: str = "~/.config/supreme-mcp-tools/audit.log",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        max_files: int = 5,
        buffer_size: int = 100
    ):
        """
        Initialize the audit logger.
        
        Args:
            log_file: Path to audit log file
            max_file_size: Maximum log file size before rotation
            max_files: Maximum number of rotated log files
            buffer_size: Number of entries to buffer before writing
        """
        self.log_file = Path(log_file).expanduser()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.buffer_size = buffer_size
        
        self._buffer: List[AuditEntry] = []
        self._lock = None
    
    async def _get_lock(self):
        """Get or create async lock."""
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        return self._lock
    
    def log(
        self,
        action: str,
        user: str,
        tool_name: str,
        details: Dict[str, Any],
        success: bool = True,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Log an audit event (synchronous).
        
        Args:
            action: Action type (use AuditAction enum values)
            user: User identifier (API key prefix, username, etc.)
            tool_name: Tool name involved
            details: Additional details
            success: Whether the action succeeded
            ip_address: Client IP address
            correlation_id: Request correlation ID
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            user=user,
            tool_name=tool_name,
            details=details,
            success=success,
            ip_address=ip_address,
            correlation_id=correlation_id
        )
        
        self._write_entry(entry)
        
        # Also log to standard logger
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"AUDIT: {action} by {user} on {tool_name} [{status}]")
    
    async def log_async(
        self,
        action: str,
        user: str,
        tool_name: str,
        details: Dict[str, Any],
        success: bool = True,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Log an audit event (asynchronous with buffering).
        
        Args:
            action: Action type
            user: User identifier
            tool_name: Tool name involved
            details: Additional details
            success: Whether the action succeeded
            ip_address: Client IP address
            correlation_id: Request correlation ID
        """
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            user=user,
            tool_name=tool_name,
            details=details,
            success=success,
            ip_address=ip_address,
            correlation_id=correlation_id
        )
        
        lock = await self._get_lock()
        async with lock:
            self._buffer.append(entry)
            
            if len(self._buffer) >= self.buffer_size:
                await self._flush_buffer()
        
        # Also log to standard logger
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"AUDIT: {action} by {user} on {tool_name} [{status}]")
    
    def _write_entry(self, entry: AuditEntry) -> None:
        """Write a single entry to the log file."""
        try:
            self._rotate_if_needed()
            
            with open(self.log_file, "a") as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    async def _flush_buffer(self) -> None:
        """Flush the buffer to the log file."""
        if not self._buffer:
            return
        
        try:
            self._rotate_if_needed()
            
            with open(self.log_file, "a") as f:
                for entry in self._buffer:
                    f.write(entry.to_json() + "\n")
            
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Error flushing audit buffer: {e}")
    
    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if not self.log_file.exists():
            return
        
        if self.log_file.stat().st_size < self.max_file_size:
            return
        
        # Rotate files
        for i in range(self.max_files - 1, 0, -1):
            old_file = self.log_file.with_suffix(f".{i}.log")
            new_file = self.log_file.with_suffix(f".{i + 1}.log")
            
            if old_file.exists():
                if i + 1 >= self.max_files:
                    old_file.unlink()  # Delete oldest
                else:
                    old_file.rename(new_file)
        
        # Rotate current file
        self.log_file.rename(self.log_file.with_suffix(".1.log"))
        
        logger.info(f"Rotated audit log file")
    
    async def flush(self) -> None:
        """Force flush the buffer."""
        lock = await self._get_lock()
        async with lock:
            await self._flush_buffer()
    
    def query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        action: Optional[str] = None,
        user: Optional[str] = None,
        tool_name: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Query audit log entries.
        
        Args:
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            action: Action type filter
            user: User filter
            tool_name: Tool name filter
            success: Success filter
            limit: Maximum entries to return
            
        Returns:
            List of matching audit entries
        """
        entries = []
        
        if not self.log_file.exists():
            return entries
        
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    if len(entries) >= limit:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        entry = AuditEntry(**data)
                        
                        # Apply filters
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if action and entry.action != action:
                            continue
                        if user and entry.user != user:
                            continue
                        if tool_name and entry.tool_name != tool_name:
                            continue
                        if success is not None and entry.success != success:
                            continue
                        
                        entries.append(entry)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Error parsing audit entry: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error querying audit log: {e}")
        
        return entries


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    log_file: str = "~/.config/supreme-mcp-tools/audit.log"
) -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_file=log_file)
    return _audit_logger
