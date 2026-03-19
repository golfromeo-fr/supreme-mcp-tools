#!/usr/bin/env python3
"""
FEF V3 Debug and Troubleshooting Utilities

Provides diagnostic tools for troubleshooting FEF V3 issues.

Usage:
    python -m launcher.debug [COMMAND] [OPTIONS]

Commands:
    diagnose    Run full diagnostic check
    health      Check health of all services
    circuit     Check circuit breaker states
    cache       Check cache statistics
    config      Check configuration
    logs        Show recent logs
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class DiagnosticTool:
    """Diagnostic tool for FEF V3 troubleshooting."""
    
    def __init__(
        self,
        management_url: str = "http://localhost:9091",
        tools_dir: str = "tools"
    ):
        self.management_url = management_url
        self.tools_dir = Path(tools_dir)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def diagnose(self) -> Dict[str, Any]:
        """
        Run full diagnostic check.
        
        Returns:
            Dictionary with diagnostic results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "management_server": await self._check_management_server(),
            "tools": await self._check_tools(),
            "configuration": self._check_configuration(),
            "dependencies": self._check_dependencies(),
            "disk_space": self._check_disk_space()
        }
        
        # Overall status
        all_ok = all(
            r.get("status") == "ok"
            for r in results.values()
            if isinstance(r, dict) and "status" in r
        )
        results["overall_status"] = "ok" if all_ok else "issues_found"
        
        return results
    
    async def _check_management_server(self) -> Dict[str, Any]:
        """Check management server status."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.management_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "ok",
                        "url": self.management_url,
                        "response": data
                    }
                else:
                    return {
                        "status": "error",
                        "url": self.management_url,
                        "error": f"HTTP {response.status}"
                    }
        except aiohttp.ClientError as e:
            return {
                "status": "error",
                "url": self.management_url,
                "error": f"Connection failed: {e}"
            }
        except Exception as e:
            return {
                "status": "error",
                "url": self.management_url,
                "error": str(e)
            }
    
    async def _check_tools(self) -> Dict[str, Any]:
        """Check tool status."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.management_url}/api/tools") as response:
                if response.status == 200:
                    data = await response.json()
                    tools = data.get("tools", [])
                    
                    tool_status = {}
                    for tool in tools:
                        tool_status[tool["name"]] = {
                            "status": tool.get("status", "unknown"),
                            "management_url": tool.get("management_url"),
                            "mcp_port": tool.get("mcp_port")
                        }
                    
                    return {
                        "status": "ok",
                        "count": len(tools),
                        "tools": tool_status
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration files."""
        config_dir = Path.home() / ".config" / "supreme-mcp-tools"
        
        results = {
            "config_dir": str(config_dir),
            "config_dir_exists": config_dir.exists()
        }
        
        if config_dir.exists():
            config_files = list(config_dir.glob("*.json"))
            results["config_files"] = [str(f.name) for f in config_files]
            
            # Check FEF V3 config
            fef_config = config_dir / "fef_v3.json"
            if fef_config.exists():
                try:
                    with open(fef_config, "r") as f:
                        data = json.load(f)
                    results["fef_v3_config"] = {
                        "status": "ok",
                        "version": data.get("version", "unknown")
                    }
                except Exception as e:
                    results["fef_v3_config"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                results["fef_v3_config"] = {
                    "status": "warning",
                    "message": "FEF V3 config not found"
                }
        
        results["status"] = "ok" if config_dir.exists() else "warning"
        return results
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        dependencies = {
            "fastapi": False,
            "uvicorn": False,
            "aiohttp": False,
            "pydantic": False
        }
        
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
            except ImportError:
                pass
        
        all_installed = all(dependencies.values())
        
        return {
            "status": "ok" if all_installed else "warning",
            "dependencies": dependencies,
            "missing": [k for k, v in dependencies.items() if not v]
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space."""
        import shutil
        
        config_dir = Path.home() / ".config" / "supreme-mcp-tools"
        
        try:
            total, used, free = shutil.disk_usage(config_dir.parent)
            
            return {
                "status": "ok",
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "percent_used": round(used / total * 100, 1)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of all services.
        
        Returns:
            Health check results
        """
        return await self._check_management_server()
    
    async def check_circuit_breakers(self) -> Dict[str, Any]:
        """
        Check circuit breaker states.
        
        Returns:
            Circuit breaker states
        """
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.management_url}/api/circuit-breakers"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def check_cache(self) -> Dict[str, Any]:
        """
        Check cache statistics.
        
        Returns:
            Cache statistics
        """
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.management_url}/api/cache/stats"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    def check_config(self) -> Dict[str, Any]:
        """
        Check configuration.
        
        Returns:
            Configuration check results
        """
        return self._check_configuration()
    
    def check_logs(self, lines: int = 50) -> Dict[str, Any]:
        """
        Show recent logs.
        
        Args:
            lines: Number of lines to show
            
        Returns:
            Recent log entries
        """
        log_dir = Path.home() / ".config" / "supreme-mcp-tools" / "logs"
        
        if not log_dir.exists():
            return {"error": "Log directory not found"}
        
        log_files = list(log_dir.glob("*.log"))
        
        if not log_files:
            return {"error": "No log files found"}
        
        # Get most recent log file
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_log, "r") as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
            
            return {
                "log_file": str(latest_log),
                "total_lines": len(all_lines),
                "showing_lines": len(recent_lines),
                "entries": [line.strip() for line in recent_lines]
            }
        except Exception as e:
            return {"error": str(e)}


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FEF V3 Debug and Troubleshooting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["diagnose", "health", "circuit", "cache", "config", "logs"],
        help="Diagnostic command"
    )
    
    parser.add_argument(
        "--management-url",
        type=str,
        default="http://localhost:9091",
        help="Management server URL"
    )
    
    parser.add_argument(
        "--tools-dir",
        type=str,
        default="tools",
        help="Directory containing tool modules"
    )
    
    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of log lines to show"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args(args)


async def main(args: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parsed_args = parse_args(args)
    
    # Setup logging
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    
    tool = DiagnosticTool(
        management_url=parsed_args.management_url,
        tools_dir=parsed_args.tools_dir
    )
    
    try:
        if parsed_args.command == "diagnose":
            result = await tool.diagnose()
        elif parsed_args.command == "health":
            result = await tool.check_health()
        elif parsed_args.command == "circuit":
            result = await tool.check_circuit_breakers()
        elif parsed_args.command == "cache":
            result = await tool.check_cache()
        elif parsed_args.command == "config":
            result = tool.check_config()
        elif parsed_args.command == "logs":
            result = tool.check_logs(lines=parsed_args.lines)
        else:
            result = {"error": f"Unknown command: {parsed_args.command}"}
        
        if parsed_args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(parsed_args.command, result)
    
    finally:
        await tool.close()


def _print_result(command: str, result: Dict[str, Any]) -> None:
    """Print result in human-readable format."""
    print(f"\n{'=' * 60}")
    print(f"FEF V3 Diagnostic: {command.upper()}")
    print(f"{'=' * 60}\n")
    
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    if command == "diagnose":
        print(f"Overall Status: {result.get('overall_status', 'unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'unknown')}")
        
        print("\nManagement Server:")
        ms = result.get("management_server", {})
        print(f"  Status: {ms.get('status', 'unknown')}")
        print(f"  URL: {ms.get('url', 'unknown')}")
        
        print("\nTools:")
        tools = result.get("tools", {})
        print(f"  Count: {tools.get('count', 0)}")
        for name, info in tools.get("tools", {}).items():
            print(f"  - {name}: {info.get('status', 'unknown')}")
        
        print("\nConfiguration:")
        config = result.get("configuration", {})
        print(f"  Config Dir: {config.get('config_dir', 'unknown')}")
        print(f"  Exists: {config.get('config_dir_exists', False)}")
        
        print("\nDependencies:")
        deps = result.get("dependencies", {})
        for name, installed in deps.get("dependencies", {}).items():
            status = "✓" if installed else "✗"
            print(f"  {status} {name}")
        
        if deps.get("missing"):
            print(f"\n  Missing: {', '.join(deps['missing'])}")
    
    elif command == "health":
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"URL: {result.get('url', 'unknown')}")
        if "response" in result:
            print(f"Response: {json.dumps(result['response'], indent=2)}")
    
    elif command == "config":
        print(f"Config Directory: {result.get('config_dir', 'unknown')}")
        print(f"Exists: {result.get('config_dir_exists', False)}")
        if "config_files" in result:
            print(f"Config Files: {', '.join(result['config_files'])}")
    
    elif command == "logs":
        print(f"Log File: {result.get('log_file', 'unknown')}")
        print(f"Total Lines: {result.get('total_lines', 0)}")
        print(f"Showing: {result.get('showing_lines', 0)} lines")
        print("\nRecent Entries:")
        for entry in result.get("entries", []):
            print(f"  {entry}")
    
    else:
        print(json.dumps(result, indent=2))
    
    print(f"\n{'=' * 60}\n")


def cli_main() -> None:
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
