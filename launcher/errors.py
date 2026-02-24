"""
Error definitions for the MCP launcher system.

This module defines custom exception classes for different error types
that can occur during the launcher's operation.
"""


class LauncherError(Exception):
    """Base exception for all launcher errors."""
    
    def __init__(self, message: str, tool_name: str = None):
        """
        Initialize the launcher error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error (optional)
        """
        self.message = message
        self.tool_name = tool_name
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with tool name if available."""
        if self.tool_name:
            return f"[{self.tool_name}] {self.message}"
        return self.message


class DiscoveryError(LauncherError):
    """Error during tool discovery."""
    
    def __init__(self, message: str, tool_name: str = None, path: str = None):
        """
        Initialize the discovery error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error (optional)
            path: Path where discovery failed (optional)
        """
        self.path = path
        super().__init__(message, tool_name)
    
    def _format_message(self) -> str:
        """Format the error message with path if available."""
        base_message = super()._format_message()
        if self.path:
            return f"{base_message} (path: {self.path})"
        return base_message


class PortConflictError(LauncherError):
    """Error when port allocation fails due to conflict."""
    
    def __init__(self, message: str, port: int = None, tool_name: str = None):
        """
        Initialize the port conflict error.
        
        Args:
            message: Error message
            port: Port number that caused the conflict (optional)
            tool_name: Name of the tool requesting the port (optional)
        """
        self.port = port
        super().__init__(message, tool_name)
    
    def _format_message(self) -> str:
        """Format the error message with port if available."""
        base_message = super()._format_message()
        if self.port:
            return f"{base_message} (port: {self.port})"
        return base_message


class ServerStartupError(LauncherError):
    """Error during server startup."""
    
    def __init__(self, message: str, tool_name: str = None, port: int = None):
        """
        Initialize the server startup error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that failed to start (optional)
            port: Port number where startup failed (optional)
        """
        self.port = port
        super().__init__(message, tool_name)
    
    def _format_message(self) -> str:
        """Format the error message with port if available."""
        base_message = super()._format_message()
        if self.port:
            return f"{base_message} (port: {self.port})"
        return base_message


class ServerRuntimeError(LauncherError):
    """Error during server runtime operation."""
    
    def __init__(self, message: str, tool_name: str = None, port: int = None):
        """
        Initialize the server runtime error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that encountered the error (optional)
            port: Port number where the error occurred (optional)
        """
        self.port = port
        super().__init__(message, tool_name)
    
    def _format_message(self) -> str:
        """Format the error message with port if available."""
        base_message = super()._format_message()
        if self.port:
            return f"{base_message} (port: {self.port})"
        return base_message


class ConfigError(LauncherError):
    """Error in configuration."""
    
    def __init__(self, message: str, config_key: str = None):
        """
        Initialize the configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error (optional)
        """
        self.config_key = config_key
        super().__init__(message)
    
    def _format_message(self) -> str:
        """Format the error message with config key if available."""
        base_message = super()._format_message()
        if self.config_key:
            return f"{base_message} (config: {self.config_key})"
        return base_message


class ValidationError(LauncherError):
    """Error during tool validation."""
    
    def __init__(self, message: str, tool_name: str = None, missing_exports: list = None):
        """
        Initialize the validation error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that failed validation (optional)
            missing_exports: List of missing exports (optional)
        """
        self.missing_exports = missing_exports or []
        super().__init__(message, tool_name)
    
    def _format_message(self) -> str:
        """Format the error message with missing exports if available."""
        base_message = super()._format_message()
        if self.missing_exports:
            return f"{base_message} (missing: {', '.join(self.missing_exports)})"
        return base_message
