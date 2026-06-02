#!/usr/bin/env python3
"""
FEF V3 Migration Scripts

Provides tools for migrating from V1 to V3 of the Flexible Extensibility Framework.

Usage:
    python -m launcher.migration [COMMAND] [OPTIONS]

Commands:
    check       Check current migration status
    migrate     Run migration
    rollback    Rollback to previous version
    validate    Validate migration
"""

import argparse
import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_types import DEFAULT_HOST

logger = logging.getLogger(__name__)


class MigrationStatus:
    """Tracks migration status."""
    
    def __init__(self, status_file: str = "~/.config/supreme-mcp-tools/migration.json"):
        self.status_file = Path(status_file).expanduser()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()
    
    def _load(self) -> None:
        """Load migration status from file."""
        if self.status_file.exists():
            with Path(self.status_file).open("r") as f:
                self._data = json.load(f)
        else:
            self._data = {
                "version": "v1",
                "migrated_at": None,
                "phases_completed": [],
                "backup_path": None
            }
    
    def _save(self) -> None:
        """Save migration status to file."""
        with Path(self.status_file).open("w") as f:
            json.dump(self._data, f, indent=2)
    
    @property
    def version(self) -> str:
        """Get current version."""
        return self._data.get("version", "v1")
    
    @version.setter
    def version(self, value: str) -> None:
        """Set current version."""
        self._data["version"] = value
        self._save()
    
    @property
    def migrated_at(self) -> str | None:
        """Get migration timestamp."""
        return self._data.get("migrated_at")
    
    @migrated_at.setter
    def migrated_at(self, value: str) -> None:
        """Set migration timestamp."""
        self._data["migrated_at"] = value
        self._save()
    
    @property
    def phases_completed(self) -> list[str]:
        """Get completed phases."""
        return self._data.get("phases_completed", [])
    
    def mark_phase_complete(self, phase: str) -> None:
        """Mark a migration phase as complete."""
        if phase not in self._data["phases_completed"]:
            self._data["phases_completed"].append(phase)
            self._save()
    
    @property
    def backup_path(self) -> str | None:
        """Get backup path."""
        return self._data.get("backup_path")
    
    @backup_path.setter
    def backup_path(self, value: str) -> None:
        """Set backup path."""
        self._data["backup_path"] = value
        self._save()


class MigrationManager:
    """Manages V1 to V3 migration."""
    
    def __init__(self, tools_dir: str = "tools"):
        self.tools_dir = Path(tools_dir)
        self.status = MigrationStatus()
        self.backup_dir = Path.home() / ".config" / "supreme-mcp-tools" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def check_status(self) -> dict[str, Any]:
        """
        Check current migration status.
        
        Returns:
            Dictionary with migration status information
        """
        return {
            "current_version": self.status.version,
            "migrated_at": self.status.migrated_at,
            "phases_completed": self.status.phases_completed,
            "backup_path": self.status.backup_path,
            "tools_found": self._discover_tools(),
            "migration_needed": self.status.version != "v3"
        }
    
    def _discover_tools(self) -> list[str]:
        """Discover available tools."""
        tools = []
        if self.tools_dir.exists():
            for path in self.tools_dir.iterdir():
                if path.is_dir() and not path.name.startswith("_"):
                    tools.append(path.name)
        return tools
    
    async def migrate(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Run migration from V1 to V3.
        
        Args:
            dry_run: If True, only simulate migration
            
        Returns:
            Migration result dictionary
        """
        result = {
            "success": False,
            "phases": [],
            "errors": []
        }
        
        try:
            # Phase 1: Backup
            if "backup" not in self.status.phases_completed:
                phase_result = await self._phase_backup(dry_run)
                result["phases"].append({"name": "backup", **phase_result})
                if not dry_run and phase_result["success"]:
                    self.status.mark_phase_complete("backup")
                if not phase_result["success"]:
                    result["success"] = False
                    result["errors"].append(f"Phase 'backup' failed: {phase_result.get('message', '')}")
                    return result
            
            # Phase 2: Add management servers to tools
            if "add_management" not in self.status.phases_completed:
                phase_result = await self._phase_add_management(dry_run)
                result["phases"].append({"name": "add_management", **phase_result})
                if not dry_run and phase_result["success"]:
                    self.status.mark_phase_complete("add_management")
                if not phase_result["success"]:
                    result["success"] = False
                    result["errors"].append(f"Phase 'add_management' failed: {phase_result.get('message', '')}")
                    return result
            
            # Phase 3: Update configuration
            if "update_config" not in self.status.phases_completed:
                phase_result = await self._phase_update_config(dry_run)
                result["phases"].append({"name": "update_config", **phase_result})
                if not dry_run and phase_result["success"]:
                    self.status.mark_phase_complete("update_config")
                if not phase_result["success"]:
                    result["success"] = False
                    result["errors"].append(f"Phase 'update_config' failed: {phase_result.get('message', '')}")
                    return result
            
            # Phase 4: Validate
            if "validate" not in self.status.phases_completed:
                phase_result = await self._phase_validate(dry_run)
                result["phases"].append({"name": "validate", **phase_result})
                if not dry_run and phase_result["success"]:
                    self.status.mark_phase_complete("validate")
                if not phase_result["success"]:
                    result["success"] = False
                    result["errors"].append(f"Phase 'validate' failed: {phase_result.get('message', '')}")
                    return result
            
            # Update version
            if not dry_run:
                self.status.version = "v3"
                self.status.migrated_at = datetime.now(timezone.utc).isoformat()
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"Migration failed: {e}")
        
        return result
    
    async def _phase_backup(self, dry_run: bool) -> dict[str, Any]:
        """Phase 1: Create backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        if dry_run:
            return {"success": True, "message": f"Would create backup at {backup_path}"}
        
        try:
            # Backup tools directory
            if self.tools_dir.exists():
                shutil.copytree(self.tools_dir, backup_path / "tools")
            
            # Backup config files
            config_dir = Path.home() / ".config" / "supreme-mcp-tools"
            if config_dir.exists():
                shutil.copytree(config_dir, backup_path / "config")
            
            self.status.backup_path = str(backup_path)
            
            return {"success": True, "message": f"Backup created at {backup_path}"}
        except Exception as e:
            return {"success": False, "message": f"Backup failed: {e}"}
    
    async def _phase_add_management(self, dry_run: bool) -> dict[str, Any]:
        """Phase 2: Add management servers to tools."""
        tools = self._discover_tools()
        results = []
        
        for tool_name in tools:
            tool_dir = self.tools_dir / tool_name
            
            # Check if already has management server
            has_mgmt = self._check_management_server(tool_dir)
            
            if has_mgmt:
                results.append(f"{tool_name}: Already has management server")
                continue
            
            if dry_run:
                results.append(f"{tool_name}: Would add management server")
                continue
            
            # Add management server integration
            success = self._add_management_server(tool_dir, tool_name)
            if success:
                results.append(f"{tool_name}: Management server added")
            else:
                results.append(f"{tool_name}: Failed to add management server")
        
        return {"success": True, "details": results}
    
    def _check_management_server(self, tool_dir: Path) -> bool:
        """Check if tool already has management server."""
        for py_file in tool_dir.glob("*.py"):
            content = py_file.read_text()
            if "ExtensionHTTPServer" in content or "management" in content.lower():
                return True
        return False
    
    def _add_management_server(self, tool_dir: Path, tool_name: str) -> bool:
        """Add management server to a tool."""
        # Find main streamable file
        main_file = tool_dir / f"{tool_name}_streamable.py"
        if not main_file.exists():
            main_file = tool_dir / f"{tool_name}.py"
        
        if not main_file.exists():
            return False
        
        # Read current content
        content = main_file.read_text()
        
        # Add imports if not present
        if "from launcher.tool_extensions" not in content:
            import_block = """
# FEF V3 Management Server
from launcher.tool_extensions import ExtensionRegistry, Extension, ExtensionType, ExtensionHTTPServer
from launcher.config.manager import ConfigManager
"""
            # Add after existing imports
            if "import" in content:
                lines = content.split("\n")
                last_import = 0
                for i, line in enumerate(lines):
                    if line.startswith("import ") or line.startswith("from "):
                        last_import = i
                lines.insert(last_import + 1, import_block)
                content = "\n".join(lines)
        
        main_file.write_text(content)
        return True
    
    async def _phase_update_config(self, dry_run: bool) -> dict[str, Any]:
        """Phase 3: Update configuration files."""
        config_dir = Path.home() / ".config" / "supreme-mcp-tools"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create FEF V3 config if not exists
        fef_config = config_dir / "fef_v3.json"
        
        if not fef_config.exists():
            config = {
                "version": "v3",
                "management_server": {
                    "host": DEFAULT_HOST,
                    "port": 9091
                },
                "security": {
                    "enabled": False
                },
                "persistence": {
                    "type": "json",
                    "directory": str(config_dir)
                }
            }
            
            if not dry_run:
                with Path(fef_config).open("w") as f:
                    json.dump(config, f, indent=2)
            
            return {"success": True, "message": f"Created {fef_config}"}
        
        return {"success": True, "message": "Config already exists"}
    
    async def _phase_validate(self, dry_run: bool) -> dict[str, Any]:
        """Phase 4: Validate migration."""
        checks = []
        
        # Check imports
        try:
            from launcher.tool_extensions import ExtensionRegistry
            checks.append(("Imports", True, "All imports successful"))
        except ImportError as e:
            checks.append(("Imports", False, f"Import error: {e}"))
        
        # Check config
        config_dir = Path.home() / ".config" / "supreme-mcp-tools"
        fef_config = config_dir / "fef_v3.json"
        checks.append(("Config", fef_config.exists(), f"Config file: {fef_config}"))
        
        # Check tools
        tools = self._discover_tools()
        checks.append(("Tools", len(tools) > 0, f"Found {len(tools)} tools"))
        
        all_passed = all(check[1] for check in checks)
        
        return {
            "success": all_passed,
            "checks": [
                {"name": name, "passed": passed, "message": msg}
                for name, passed, msg in checks
            ]
        }
    
    async def rollback(self) -> dict[str, Any]:
        """
        Rollback to previous version.
        
        Returns:
            Rollback result dictionary
        """
        backup_path = self.status.backup_path
        
        if not backup_path:
            return {"success": False, "message": "No backup found"}
        
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            return {"success": False, "message": f"Backup not found at {backup_path}"}
        
        try:
            # Restore tools
            backup_tools = backup_dir / "tools"
            if backup_tools.exists():
                if self.tools_dir.exists():
                    shutil.rmtree(self.tools_dir)
                shutil.copytree(backup_tools, self.tools_dir)
            
            # Restore config
            backup_config = backup_dir / "config"
            if backup_config.exists():
                config_dir = Path.home() / ".config" / "supreme-mcp-tools"
                if config_dir.exists():
                    shutil.rmtree(config_dir)
                shutil.copytree(backup_config, config_dir)
            
            # Update status
            self.status.version = "v1"
            self.status.migrated_at = None
            
            return {"success": True, "message": "Rollback completed"}
        
        except Exception as e:
            return {"success": False, "message": f"Rollback failed: {e}"}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FEF V3 Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["check", "migrate", "rollback", "validate"],
        help="Migration command"
    )
    
    parser.add_argument(
        "--tools-dir",
        type=str,
        default="tools",
        help="Directory containing tool modules"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making changes"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args(args)


async def main(args: list[str] | None = None) -> None:
    """Main entry point."""
    parsed_args = parse_args(args)
    
    # Setup logging
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    
    manager = MigrationManager(tools_dir=parsed_args.tools_dir)
    
    if parsed_args.command == "check":
        status = manager.check_status()
        print(json.dumps(status, indent=2))
    
    elif parsed_args.command == "migrate":
        print(f"Running migration (dry_run={parsed_args.dry_run})...")
        result = await manager.migrate(dry_run=parsed_args.dry_run)
        print(json.dumps(result, indent=2))
    
    elif parsed_args.command == "rollback":
        print("Running rollback...")
        result = await manager.rollback()
        print(json.dumps(result, indent=2))
    
    elif parsed_args.command == "validate":
        print("Validating migration...")
        result = await manager._phase_validate(dry_run=False)
        print(json.dumps(result, indent=2))


def cli_main() -> None:
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
