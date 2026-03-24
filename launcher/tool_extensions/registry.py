"""
Local Extension Registry

Each tool runs its own ExtensionRegistry instance for local extension management.
This is the core component of the Flexible Extensibility Framework V3.

Global Registry Tracking:
    The _global_registries dictionary tracks all registry instances by tool name.
    This allows tools to find and use the launcher's registry when running under
    the launcher's management.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global tracking of registry instances by tool name
# This allows tools to find the launcher's registry when running under management
_global_registries: Dict[str, "ExtensionRegistry"] = {}


class ExtensionType(Enum):
    """Types of extensions a tool can register."""
    DATA_SOURCE = "data_source"      # Read-only data exposure
    MUTATOR = "mutator"              # Configuration changes
    ACTION = "action"                # One-off operations
    EVENT = "event"                  # Event emission
    STREAM = "stream"                # Continuous data streams


@dataclass
class Extension:
    """
    Represents a registered extension.
    
    Attributes:
        name: Unique identifier within the tool
        ext_type: Type of extension
        schema: JSON schema for parameters and returns
        handler: Callable that implements the extension
        metadata: Additional information (description, category, etc.)
    """
    name: str
    ext_type: ExtensionType
    schema: Dict[str, Any]
    handler: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """
        Convert extension to dictionary representation.
        
        Args:
            include_data: If True, fetch and include current data for data sources.
        """
        result = {
            "name": self.name,
            "type": self.ext_type.value,
            "schema": self.schema,
            "metadata": self.metadata,
        }
        
        # For data sources, optionally include current data values
        if include_data and self.ext_type == ExtensionType.DATA_SOURCE:
            try:
                result["data"] = self.handler({})
            except Exception as e:
                logger.warning(f"Error fetching data for extension '{self.name}': {e}")
                result["data"] = None
        
        return result
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        """
        Get current data values for a data source extension.
        
        Returns:
            Dictionary with current data values, or None if not a data source
            or if an error occurs.
        """
        if self.ext_type != ExtensionType.DATA_SOURCE:
            return None
        
        try:
            return self.handler({})
        except Exception as e:
            logger.warning(f"Error fetching data for extension '{self.name}': {e}")
            return None


class ExtensionRegistry:
    """
    Local extension registry for managing extensions within a tool process.
    
    Each tool process has its own instance of this registry.
    """
    
    def __init__(self, tool_name: Optional[str] = None):
        """
        Initialize the extension registry.
        
        Args:
            tool_name: Optional tool name to register this registry globally.
                      When set, tools can find this registry via _global_registries.
        """
        self._extensions: Dict[str, Dict[str, Extension]] = {}
        self._event_subscribers: Dict[str, List[Callable]] = {}
        self._event_queues: Dict[str, List[asyncio.Queue]] = {}
        self._tool_name = tool_name
        
        # Register globally if tool_name is provided
        if tool_name:
            _global_registries[tool_name] = self
            logger.info(f"Registered global registry for tool '{tool_name}'")
    
    def register_global(self, tool_name: str) -> None:
        """
        Register this registry in the global tracking.
        
        Args:
            tool_name: Tool name to register under
        """
        self._tool_name = tool_name
        _global_registries[tool_name] = self
        logger.info(f"Registered global registry for tool '{tool_name}'")
    
    def unregister_global(self) -> None:
        """Unregister this registry from global tracking."""
        if self._tool_name and self._tool_name in _global_registries:
            del _global_registries[self._tool_name]
            logger.info(f"Unregistered global registry for tool '{self._tool_name}'")
    
    def register(self, tool_name: str, extension: Extension) -> None:
        """
        Register an extension for a tool.
        
        Args:
            tool_name: Name of the tool registering the extension
            extension: Extension to register
            
        Raises:
            ValueError: If extension name already exists for the tool
        """
        if tool_name not in self._extensions:
            self._extensions[tool_name] = {}
        
        if extension.name in self._extensions[tool_name]:
            raise ValueError(
                f"Extension '{extension.name}' already registered for tool '{tool_name}'"
            )
        
        self._extensions[tool_name][extension.name] = extension
        logger.info(
            f"Registered extension '{extension.name}' "
            f"({extension.ext_type.value}) for tool '{tool_name}'"
        )
    
    def unregister(self, tool_name: str, extension_name: str) -> bool:
        """
        Unregister an extension.
        
        Args:
            tool_name: Name of the tool
            extension_name: Name of the extension to unregister
            
        Returns:
            True if extension was unregistered, False if not found
        """
        if tool_name in self._extensions:
            if extension_name in self._extensions[tool_name]:
                del self._extensions[tool_name][extension_name]
                logger.info(
                    f"Unregistered extension '{extension_name}' for tool '{tool_name}'"
                )
                return True
        return False
    
    def get_extension(self, tool_name: str, extension_name: str) -> Optional[Extension]:
        """
        Get an extension by name.
        
        Args:
            tool_name: Name of the tool
            extension_name: Name of the extension
            
        Returns:
            Extension if found, None otherwise
        """
        if tool_name in self._extensions:
            return self._extensions[tool_name].get(extension_name)
        return None
    
    def list_extensions(
        self,
        tool_name: Optional[str] = None,
        ext_type: Optional[ExtensionType] = None,
        include_data: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all extensions, optionally filtered by tool and type.
        
        Args:
            tool_name: Optional filter by tool name
            ext_type: Optional filter by extension type
            include_data: If True, include current data for data sources (default: True)
            
        Returns:
            Dictionary mapping tool names to lists of extension info
        """
        result = {}
        
        tools = [tool_name] if tool_name else list(self._extensions.keys())
        
        for tool in tools:
            if tool not in self._extensions:
                continue
            
            extensions = []
            for ext in self._extensions[tool].values():
                if ext_type is None or ext.ext_type == ext_type:
                    extensions.append(ext.to_dict(include_data=include_data))
            
            if extensions:
                result[tool] = extensions
        
        return result
    
    def query(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Query a data source extension.
        
        Args:
            tool_name: Name of the tool
            extension_name: Name of the extension
            params: Query parameters
            
        Returns:
            Query result
            
        Raises:
            ValueError: If extension not found or not a data source
        """
        ext = self.get_extension(tool_name, extension_name)
        if ext is None:
            raise ValueError(
                f"Extension '{extension_name}' not found for tool '{tool_name}'"
            )
        
        if ext.ext_type != ExtensionType.DATA_SOURCE:
            raise ValueError(
                f"Extension '{extension_name}' is not a data source "
                f"(type: {ext.ext_type.value})"
            )
        
        try:
            return ext.handler(params or {})
        except Exception as e:
            logger.error(f"Error querying extension '{extension_name}': {e}")
            raise
    
    def mutate(
        self,
        tool_name: str,
        extension_name: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        Execute a mutator extension.
        
        Args:
            tool_name: Name of the tool
            extension_name: Name of the extension
            params: Mutation parameters
            
        Returns:
            Mutation result
            
        Raises:
            ValueError: If extension not found or not a mutator
        """
        ext = self.get_extension(tool_name, extension_name)
        if ext is None:
            raise ValueError(
                f"Extension '{extension_name}' not found for tool '{tool_name}'"
            )
        
        if ext.ext_type != ExtensionType.MUTATOR:
            raise ValueError(
                f"Extension '{extension_name}' is not a mutator "
                f"(type: {ext.ext_type.value})"
            )
        
        try:
            result = ext.handler(params)
            # Emit mutation event
            self._emit_event(tool_name, "mutation", {
                "extension": extension_name,
                "params": params,
                "result": result
            })
            return result
        except ValueError as e:
            # Client input validation errors - log at WARNING level
            logger.warning(f"Validation error in mutation '{extension_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error mutating extension '{extension_name}': {e}")
            raise
    
    def execute(
        self,
        tool_name: str,
        extension_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute an action extension.
        
        Args:
            tool_name: Name of the tool
            extension_name: Name of the extension
            params: Action parameters
            
        Returns:
            Action result
            
        Raises:
            ValueError: If extension not found or not an action
        """
        ext = self.get_extension(tool_name, extension_name)
        if ext is None:
            raise ValueError(
                f"Extension '{extension_name}' not found for tool '{tool_name}'"
            )
        
        if ext.ext_type != ExtensionType.ACTION:
            raise ValueError(
                f"Extension '{extension_name}' is not an action "
                f"(type: {ext.ext_type.value})"
            )
        
        try:
            result = ext.handler(params or {})
            # Emit action event
            self._emit_event(tool_name, "action", {
                "extension": extension_name,
                "params": params,
                "result": result
            })
            return result
        except ValueError as e:
            # Client input validation errors - log at WARNING level
            logger.warning(f"Validation error in action '{extension_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing extension '{extension_name}': {e}")
            raise
    
    def subscribe(self, tool_name: str, event_type: str, callback: Callable) -> None:
        """
        Subscribe to events from a tool.
        
        Args:
            tool_name: Name of the tool
            event_type: Type of event to subscribe to
            callback: Callback function to invoke when event occurs
        """
        key = f"{tool_name}:{event_type}"
        if key not in self._event_subscribers:
            self._event_subscribers[key] = []
        self._event_subscribers[key].append(callback)
        logger.debug(f"Subscribed to {key}")
    
    def subscribe_queue(self, tool_name: str, event_type: str) -> asyncio.Queue:
        """
        Subscribe to events via a queue (for async consumers).
        
        Args:
            tool_name: Name of the tool
            event_type: Type of event to subscribe to
            
        Returns:
            Queue that will receive events
        """
        key = f"{tool_name}:{event_type}"
        if key not in self._event_queues:
            self._event_queues[key] = []
        
        queue = asyncio.Queue()
        self._event_queues[key].append(queue)
        logger.debug(f"Queue subscribed to {key}")
        return queue
    
    def unsubscribe_queue(self, tool_name: str, event_type: str, queue: asyncio.Queue) -> None:
        """
        Unsubscribe a queue from events.
        
        Args:
            tool_name: Name of the tool
            event_type: Type of event
            queue: Queue to unsubscribe
        """
        key = f"{tool_name}:{event_type}"
        if key in self._event_queues:
            if queue in self._event_queues[key]:
                self._event_queues[key].remove(queue)
                logger.debug(f"Queue unsubscribed from {key}")
    
    def _emit_event(self, tool_name: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            tool_name: Name of the tool
            event_type: Type of event
            data: Event data
        """
        import time
        
        event = {
            "tool": tool_name,
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # Notify callback subscribers
        key = f"{tool_name}:{event_type}"
        for callback in self._event_subscribers.get(key, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
        
        # Notify queue subscribers
        for queue in self._event_queues.get(key, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Queue full for {key}, dropping event")
    
    def emit_event(
        self,
        tool_name: str,
        extension_name: str,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Emit an event from an event extension.
        
        Args:
            tool_name: Name of the tool
            extension_name: Name of the event extension
            event_data: Event data to emit
        """
        ext = self.get_extension(tool_name, extension_name)
        if ext is None:
            raise ValueError(
                f"Extension '{extension_name}' not found for tool '{tool_name}'"
            )
        
        if ext.ext_type != ExtensionType.EVENT:
            raise ValueError(
                f"Extension '{extension_name}' is not an event extension "
                f"(type: {ext.ext_type.value})"
            )
        
        self._emit_event(tool_name, extension_name, event_data)
