"""
Environment Variable Manager for MCP Tools.

Handles reading/writing environment variables across all tools:
- Discovers .env files (project root primary, per-tool fallback)
- Standardized config.json schema with secret/required/description metadata
- Comment-based history when updating values
- Runtime updates via os.environ (immediate for in-process tools)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Project root: launcher/env_manager.py -> supreme-mcp-tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cache for config.json schemas
_schema_cache: Dict[str, Dict[str, Any]] = {}


def find_env_file() -> Path:
    """
    Find the primary .env file.

    Search order:
    1. Project root .env
    2. Create at project root if nothing found

    Returns:
        Path to the .env file
    """
    root_env = PROJECT_ROOT / ".env"
    if root_env.exists():
        return root_env

    # Will be created on first write
    return root_env


def find_tool_env_file(tool_name: str) -> Optional[Path]:
    """
    Check for a legacy per-tool .env file.

    Args:
        tool_name: Name of the tool

    Returns:
        Path to tool's .env if it exists, None otherwise
    """
    tool_env = PROJECT_ROOT / "tools" / tool_name / ".env"
    if tool_env.exists():
        return tool_env
    return None


def mask_value(value: str) -> str:
    """
    Mask a value for display, showing only the last 4 characters.

    Args:
        value: The raw value to mask

    Returns:
        Masked string like '****' or '****abcd'
    """
    if not value or len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


def load_env_schema(tool_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load environment variable schema from a tool's config.json.

    Handles both the new object format and legacy list format:
    - New: { "VAR_NAME": { "description": "...", "required": false, ... } }
    - Legacy: { "required": ["VAR1"], "optional": ["VAR2"] }

    Args:
        tool_name: Name of the tool

    Returns:
        Dict mapping variable names to their metadata.
        Returns empty dict if tool has no environment_variables section.
    """
    if tool_name in _schema_cache:
        return _schema_cache[tool_name]

    config_path = PROJECT_ROOT / "tools" / tool_name / "config.json"
    if not config_path.exists():
        logger.debug(f"No config.json found for tool {tool_name}")
        return {}

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read config.json for {tool_name}: {e}")
        return {}

    env_section = config.get("environment_variables", {})
    if not env_section:
        return {}

    # Detect format and normalize
    schema = _normalize_env_schema(env_section)

    _schema_cache[tool_name] = schema
    return schema


def _normalize_env_schema(env_section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize env section to the standard object-per-variable format.

    Handles legacy list format: { "required": [...], "optional": [...] }
    """
    # If it has "required" or "optional" keys with list values, it's the old format
    if "required" in env_section and isinstance(env_section.get("required"), list):
        return _migrate_list_format(env_section)

    # Already in object format - ensure all fields have defaults
    normalized = {}
    for var_name, var_meta in env_section.items():
        if isinstance(var_meta, dict):
            normalized[var_name] = {
                "description": var_meta.get("description", ""),
                "required": var_meta.get("required", False),
                "secret": var_meta.get("secret", True),
                "default": var_meta.get("default", ""),
                "options": var_meta.get("options", []),
                "type": var_meta.get("type", "string"),
                "minimum": var_meta.get("minimum"),
                "maximum": var_meta.get("maximum"),
            }
        else:
            # Unexpected format, treat as unknown
            normalized[var_name] = {
                "description": "",
                "required": False,
                "secret": True,
                "default": str(var_meta) if var_meta else "",
                "options": [],
                "type": "string",
                "minimum": None,
                "maximum": None,
            }
    return normalized


def _migrate_list_format(env_section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert legacy list format to standard object format.

    Input:  { "required": ["VAR1"], "optional": ["VAR2"] }
    Output: { "VAR1": { "required": true, ... }, "VAR2": { "required": false, ... } }
    """
    normalized = {}

    for var_name in env_section.get("required", []):
        normalized[var_name] = {
            "description": "",
            "required": True,
            "secret": True,
            "default": "",
            "options": [],
        }

    for var_name in env_section.get("optional", []):
        normalized[var_name] = {
            "description": "",
            "required": False,
            "secret": True,
            "default": "",
            "options": [],
        }

    return normalized


def get_env_values(tool_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get environment variable values for a tool with metadata.

    Returns masked values for secret variables.

    Args:
        tool_name: Name of the tool

    Returns:
        Dict mapping variable names to their full metadata + current value info:
        {
            "VAR_NAME": {
                "description": "...",
                "required": false,
                "secret": true,
                "value_masked": "****abcd",
                "value_raw": "full_value",  # only for non-secret
                "is_set": true,
                "default": "",
                "options": []
            }
        }
    """
    schema = load_env_schema(tool_name)
    result = {}

    for var_name, meta in schema.items():
        raw_value = os.environ.get(var_name, "")
        is_set = bool(raw_value)
        is_secret = meta.get("secret", True)

        entry = {
            "description": meta.get("description", ""),
            "required": meta.get("required", False),
            "secret": is_secret,
            "value_masked": mask_value(raw_value) if is_secret else raw_value,
            "is_set": is_set,
            "default": meta.get("default", ""),
            "options": meta.get("options", []),
            "type": meta.get("type", "string"),
            "minimum": meta.get("minimum"),
            "maximum": meta.get("maximum"),
        }

        if not is_secret:
            entry["value_raw"] = raw_value

        result[var_name] = entry

    return result


def get_all_env_values() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get environment variable values for all tools that have config.json.

    Returns:
        Dict mapping tool names to their env var dicts.
    """
    tools_dir = PROJECT_ROOT / "tools"
    if not tools_dir.exists():
        return {}

    result = {}
    for tool_dir in sorted(tools_dir.iterdir()):
        if not tool_dir.is_dir():
            continue
        config_path = tool_dir / "config.json"
        if not config_path.exists():
            continue

        tool_name = tool_dir.name
        values = get_env_values(tool_name)
        if values:
            result[tool_name] = values

    return result


def _find_var_schema(var_name: str) -> Optional[Dict[str, Any]]:
    """
    Find the schema for a variable by searching all tool config.json files.

    Args:
        var_name: Name of the environment variable

    Returns:
        Schema dict or None if not found
    """
    tools_dir = PROJECT_ROOT / "tools"
    if not tools_dir.exists():
        return None

    for tool_dir in tools_dir.iterdir():
        if not tool_dir.is_dir():
            continue
        config_path = tool_dir / "config.json"
        if not config_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        env_section = config.get("environment_variables", {})
        if var_name in env_section:
            meta = env_section[var_name]
            if isinstance(meta, dict):
                return meta

    return None


def _validate_env_value(var_name: str, value: str, schema: Dict[str, Any]) -> None:
    """
    Validate a value against a variable's schema.

    Args:
        var_name: Name of the variable (for error messages)
        value: The value to validate
        schema: The variable's schema from config.json

    Raises:
        ValueError: If validation fails
    """
    var_type = schema.get("type", "string")

    if var_type == "integer":
        try:
            int_val = int(value)
        except ValueError:
            raise ValueError(f"{var_name} must be an integer, got: {value!r}")

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and int_val < minimum:
            raise ValueError(f"{var_name} must be >= {minimum}, got: {int_val}")
        if maximum is not None and int_val > maximum:
            raise ValueError(f"{var_name} must be <= {maximum}, got: {int_val}")

    elif var_type == "number":
        try:
            float_val = float(value)
        except ValueError:
            raise ValueError(f"{var_name} must be a number, got: {value!r}")

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and float_val < minimum:
            raise ValueError(f"{var_name} must be >= {minimum}, got: {float_val}")
        if maximum is not None and float_val > maximum:
            raise ValueError(f"{var_name} must be <= {maximum}, got: {float_val}")

    elif var_type == "boolean":
        if value.lower() not in ("true", "false"):
            raise ValueError(f"{var_name} must be 'true' or 'false', got: {value!r}")


def set_env_value(var_name: str, value: str, persist: bool = True) -> None:
    """
    Set an environment variable both in memory and optionally in .env file.

    Args:
        var_name: Environment variable name
        value: New value
        persist: Whether to also write to .env file

    Raises:
        ValueError: If value doesn't match the variable's type/constraints
    """
    # Validate value against schema if we can find it
    schema = _find_var_schema(var_name)
    if schema:
        _validate_env_value(var_name, value, schema)

    # Update runtime environment immediately
    os.environ[var_name] = value
    logger.info(f"Set env var {var_name} (persist={persist})")

    if persist:
        env_path = find_env_file()
        _update_env_file(var_name, value, env_path)

    # Invalidate schema cache if this var's tool schema might be affected
    # (schemas don't change, but this is a good practice)


def delete_env_value(var_name: str, persist: bool = True) -> None:
    """
    Remove an environment variable from memory and optionally from .env file.

    Args:
        var_name: Environment variable name
        persist: Whether to also comment out in .env file
    """
    # Remove from runtime environment
    if var_name in os.environ:
        del os.environ[var_name]
        logger.info(f"Removed env var {var_name}")

    if persist:
        env_path = find_env_file()
        _comment_out_env_line(var_name, env_path)


def _update_env_file(var_name: str, value: str, env_path: Path) -> None:
    """
    Update a variable in the .env file with comment-based history.

    On update:
    1. Comment out the existing active line for this var (with date)
    2. Append the new value at the end of the file (or after last comment for this var)

    Args:
        var_name: Variable name
        value: New value
        env_path: Path to .env file
    """
    # Create file if it doesn't exist
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with open(env_path, "w") as f:
            f.write(f"{var_name}={value}\n")
        logger.info(f"Created .env file at {env_path} with {var_name}")
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    # Find the last active (uncommented) line for this variable
    active_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{var_name}="):
            active_line_idx = i

    now = datetime.now().strftime("%Y-%m-%d")
    new_line = f"{var_name}={value}\n"

    if active_line_idx is not None:
        # Comment out the existing active line
        old_line = lines[active_line_idx].rstrip("\n")
        lines[active_line_idx] = f"# {old_line}  # updated {now}\n"

        # Insert new line right after the commented line
        lines.insert(active_line_idx + 1, new_line)
        logger.info(f"Updated {var_name} in .env (old value commented out)")
    else:
        # Variable not in file yet - append at end
        # Ensure there's a blank line before if file doesn't end with one
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        elif lines and lines[-1].strip():
            lines.append("\n")
        lines.append(new_line)
        logger.info(f"Added {var_name} to .env")

    with open(env_path, "w") as f:
        f.writelines(lines)


def _comment_out_env_line(var_name: str, env_path: Path) -> None:
    """
    Comment out (but keep) all active lines for a variable in the .env file.

    Args:
        var_name: Variable name
        env_path: Path to .env file
    """
    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    now = datetime.now().strftime("%Y-%m-%d")
    modified = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{var_name}="):
            lines[i] = f"# {stripped}  # removed {now}\n"
            modified = True

    if modified:
        with open(env_path, "w") as f:
            f.writelines(lines)
        logger.info(f"Commented out {var_name} in .env")


def clear_schema_cache() -> None:
    """Clear the config.json schema cache. Useful after config changes."""
    _schema_cache.clear()


def get_tool_names() -> List[str]:
    """
    Get list of tool names that have config.json files.

    Returns:
        List of tool names
    """
    tools_dir = PROJECT_ROOT / "tools"
    if not tools_dir.exists():
        return []

    return sorted(
        d.name
        for d in tools_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def load_auth_config(tool_name: str) -> Dict[str, Any]:
    """
    Load auth config from tool's config.json.

    Args:
        tool_name: Name of the tool

    Returns:
        Dict with auth config, e.g. {"api_key": "secret"} or {} if not configured
    """
    config_path = PROJECT_ROOT / "tools" / tool_name / "config.json"
    if not config_path.exists():
        logger.debug(f"No config.json found for tool {tool_name}")
        return {}

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read config.json for {tool_name}: {e}")
        return {}

    return config.get("auth", {})
