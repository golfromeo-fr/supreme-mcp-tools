"""
Authentication Module for the Management UI.

Handles user authentication and session management.
"""

import os
import logging
import json
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any, Tuple

from nicegui import ui, app

# Set up logging
logger = logging.getLogger(__name__)

# Environment variable names
ENV_USERNAME = "MCP_UI_USERNAME"
ENV_PASSWORD = "MCP_UI_PASSWORD"


def _load_ui_config() -> dict:
    """Load UI configuration from ports.json."""
    possible_paths = [
        Path(__file__).parent.parent / "config" / "ports.json",
        Path(__file__).parent.parent / "ports.json",
    ]
    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _get_default_username() -> str:
    """Get default username from environment or config."""
    # Environment takes priority
    default_username = os.environ.get("MCP_UI_DEFAULT_USERNAME")
    if default_username:
        return default_username

    # Fall back to config
    config = _load_ui_config()
    return config.get("auth", {}).get("default_username", "admin")


# Default password is only from environment (never from config file)
_DEFAULT_PASSWORD = os.environ.get("MCP_UI_PASSWORD", "admin")


# Default username for development
_DEFAULT_USERNAME = _get_default_username()


def get_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Get username and password from environment variables.
    
    Returns:
        Tuple of (username, password) or (None, None) if not set.
    """
    username = os.environ.get(ENV_USERNAME)
    password = os.environ.get(ENV_PASSWORD)
    return username, password


def is_auth_enabled() -> bool:
    """
    Check if authentication is enabled.
    
    Authentication is enabled when MCP_UI_PASSWORD is set.
    
    Returns:
        True if authentication is required, False otherwise.
    """
    password = os.environ.get(ENV_PASSWORD)
    return password is not None and len(password) > 0


def verify_credentials(username: str, password: str) -> bool:
    """
    Verify user credentials.
    
    Args:
        username: The username to verify.
        password: The password to verify.
    
    Returns:
        True if credentials are valid, False otherwise.
    """
    stored_username, stored_password = get_credentials()

    # Fill in defaults for any missing credential
    if stored_username is None:
        stored_username = _DEFAULT_USERNAME
    if stored_password is None:
        stored_password = _DEFAULT_PASSWORD
    
    return username == stored_username and password == stored_password


def is_authenticated() -> bool:
    """
    Check if current client is authenticated.
    
    Uses app.storage.user for persistent storage across page navigations.
    This storage is per-user and persists across page navigations within
    the same session.
    
    Returns:
        True if authenticated, False otherwise.
    """
    try:
        result = app.storage.user.get("authenticated", False)
        logger.debug(f"is_authenticated returning: {result}")
        return result
    except Exception as e:
        # If storage is not available (e.g., outside page context), return False
        logger.debug(f"is_authenticated exception: {e}")
        return False


def set_authenticated(value: bool = True) -> None:
    """
    Set authentication status for current client.
    
    Uses app.storage.user for persistent storage across page navigations.
    This storage is per-user and persists across page navigations within
    the same session.
    
    Args:
        value: True if authenticated, False otherwise.
    """
    try:
        app.storage.user["authenticated"] = value
        logger.debug(f"set_authenticated set to: {value}")
    except Exception as e:
        # If storage is not available, silently fail
        logger.debug(f"set_authenticated exception: {e}")
        pass


def logout() -> None:
    """
    Log out the current client by clearing authentication status.
    """
    set_authenticated(False)


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to require authentication for a page or function.
    
    Usage:
        @ui.page('/protected')
        @require_auth
        async def protected_page():
            ...
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_auth_enabled() and not is_authenticated():
            ui.navigate.to('/login')
            return
        return await func(*args, **kwargs)
    return wrapper


def create_login_page() -> None:
    """
    Create the login page with centered login card.
    
    The login page displays a centered card with username and password
    inputs, and handles authentication verification.
    """
    logger.debug("create_login_page called")
    try:
        # Add a simple label first to verify the page is rendering
        ui.label('Please log in').classes('text-h4 absolute-center')
        logger.debug("Simple label added")
        
        # Now add the card with the login form
        with ui.card().classes('absolute-center w-96'):
            ui.label('Management UI Login').classes('text-h5 mb-4')
            
            username_input = ui.input(
                'Username',
                placeholder='Enter username'
            ).classes('w-full mb-2')
            
            password_input = ui.input(
                'Password',
                password=True,
                placeholder='Enter password'
            ).classes('w-full mb-4')
            
            error_label = ui.label('').classes('text-red-500 mb-2')
            
            async def handle_login() -> None:
                """Handle login button click."""
                logger.debug(f"Login attempt with username: {username_input.value}")
                if verify_credentials(username_input.value, password_input.value):
                    logger.debug("Credentials verified, setting authenticated")
                    set_authenticated(True)
                    ui.navigate.to('/')
                else:
                    logger.debug("Invalid credentials")
                    error_label.text = 'Invalid credentials'
                    password_input.value = ''
            
            ui.button('Login', on_click=handle_login).classes('w-full')
        
        logger.debug("create_login_page completed successfully")
    except Exception as e:
        logger.exception(f"Error in create_login_page: {e}")
