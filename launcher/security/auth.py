"""
API Key Authentication for FEF V3

Provides API key verification and permission-based access control.
"""

import os
import secrets
import logging
from typing import Optional, Dict, List, Callable
from functools import wraps

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """
    API Key authentication manager.
    
    Manages API keys with roles and permissions for secure access control.
    """
    
    def __init__(
        self,
        api_keys: Optional[Dict[str, Dict]] = None,
        config_file: Optional[str] = None
    ):
        """
        Initialize the API key auth manager.
        
        Args:
            api_keys: Dictionary of API keys to permissions
            config_file: Optional path to JSON config file
        """
        self.api_keys: Dict[str, Dict] = api_keys or {}
        
        if config_file:
            self._load_from_file(config_file)
    
    def _load_from_file(self, config_file: str) -> None:
        """Load API keys from a JSON config file."""
        import json
        from pathlib import Path
        
        path = Path(config_file).expanduser()
        if path.exists():
            try:
                with open(path, "r") as f:
                    config = json.load(f)
                    self.api_keys.update(config.get("api_keys", {}))
                logger.info(f"Loaded {len(self.api_keys)} API keys from {config_file}")
            except Exception as e:
                logger.error(f"Error loading API keys from {config_file}: {e}")
    
    def add_key(
        self,
        key: str,
        role: str = "readonly",
        tools: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None
    ) -> None:
        """
        Add an API key.
        
        Args:
            key: API key string
            role: Role (admin, readonly, custom)
            tools: List of allowed tools (["*"] for all)
            permissions: List of permissions
        """
        self.api_keys[key] = {
            "role": role,
            "tools": tools or ["*"],
            "permissions": permissions or []
        }
        logger.info(f"Added API key with role '{role}'")
    
    def remove_key(self, key: str) -> bool:
        """
        Remove an API key.
        
        Args:
            key: API key to remove
            
        Returns:
            True if removed, False if not found
        """
        if key in self.api_keys:
            del self.api_keys[key]
            logger.info("Removed API key")
            return True
        return False
    
    def verify(self, key: Optional[str]) -> Dict:
        """
        Verify an API key and return permissions.
        
        Args:
            key: API key to verify
            
        Returns:
            Permissions dictionary
            
        Raises:
            HTTPException: If key is missing or invalid
        """
        if not key:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
        
        if key not in self.api_keys:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
        
        return self.api_keys[key]
    
    def check_permission(
        self,
        permissions: Dict,
        tool_name: Optional[str] = None,
        permission: Optional[str] = None
    ) -> bool:
        """
        Check if permissions allow access.
        
        Args:
            permissions: Permissions dictionary from verify()
            tool_name: Optional tool name to check
            permission: Optional permission to check
            
        Returns:
            True if access allowed
        """
        # Admin has access to everything
        if permissions.get("role") == "admin":
            return True
        
        # Check tool access
        if tool_name:
            allowed_tools = permissions.get("tools", [])
            if "*" not in allowed_tools and tool_name not in allowed_tools:
                return False
        
        # Check permission
        if permission:
            allowed_permissions = permissions.get("permissions", [])
            if permission not in allowed_permissions:
                return False
        
        return True
    
    @staticmethod
    def generate_key() -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)


# Default instance with environment-based keys
def _load_default_keys() -> Dict[str, Dict]:
    """Load default API keys from environment variables."""
    keys = {}
    
    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key:
        keys[admin_key] = {"role": "admin", "tools": ["*"]}
    
    readonly_key = os.getenv("READONLY_API_KEY")
    if readonly_key:
        keys[readonly_key] = {"role": "readonly", "tools": ["*"], "permissions": ["query"]}
    
    return keys


# Global auth instance
_auth_instance: Optional[APIKeyAuth] = None


def get_auth() -> APIKeyAuth:
    """Get or create the global auth instance."""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = APIKeyAuth(api_keys=_load_default_keys())
    return _auth_instance


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> dict:
    """
    FastAPI dependency to verify API key.
    
    Args:
        api_key: API key from header
        
    Returns:
        Permissions dictionary
    """
    auth = get_auth()
    return auth.verify(api_key)


def require_permission(permission: str):
    """
    Decorator to require specific permission.
    
    Args:
        permission: Required permission name
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            permissions: dict = Depends(verify_api_key),
            **kwargs
        ):
            auth = get_auth()
            if not auth.check_permission(permissions, permission=permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
            return await func(*args, permissions=permissions, **kwargs)
        return wrapper
    return decorator


def require_tool_access(tool_name: str):
    """
    Decorator to require access to a specific tool.
    
    Args:
        tool_name: Required tool name
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            permissions: dict = Depends(verify_api_key),
            **kwargs
        ):
            auth = get_auth()
            if not auth.check_permission(permissions, tool_name=tool_name):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access to tool '{tool_name}' not allowed"
                )
            return await func(*args, permissions=permissions, **kwargs)
        return wrapper
    return decorator
