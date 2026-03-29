#!/usr/bin/env python3
"""
Sync Tools Config Script.

Discovers tools from running MCP servers and updates the tools_config.json.
This should be run after MCP servers are started.

Usage:
    python -m launcher.sync_tools_config           # Auto-discover from running servers
    python -m launcher.sync_tools_config --help    # Show all options
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from launcher.tools_config import (
    update_config_with_discovered_tools,
    validate_and_cleanup_config,
)
from launcher.launcher_config import load_ports_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_server_urls_from_ports():
    """Get server URLs from ports.json."""
    try:
        ports_config = load_ports_config()
    except Exception as e:
        logger.error(f"Failed to load ports.json: {e}")
        return {}

    assignments = ports_config.get("assignments", {}).get("mcp", {})
    server_urls = {}

    for server_name, port in assignments.items():
        server_urls[server_name] = f"http://localhost:{port}/mcp"

    return server_urls


def main():
    parser = argparse.ArgumentParser(description="Sync tools config from running MCP servers")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate and cleanup existing config, don't discover"
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover tools, don't validate"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout for each server query (default: 5.0)"
    )

    args = parser.parse_args()

    # Get server URLs from ports.json
    server_urls = get_server_urls_from_ports()

    if not server_urls:
        logger.error("No server URLs found in ports.json")
        return 1

    logger.info(f"Found {len(server_urls)} servers: {list(server_urls.keys())}")

    if args.validate_only:
        logger.info("Running validation only...")
        results = validate_and_cleanup_config()
        logger.info(f"Validation complete: {results}")
        return 0

    if not args.discover_only:
        # First validate and cleanup
        logger.info("Validating existing config...")
        results = validate_and_cleanup_config()
        if results["removed_invalid"]:
            logger.info(f"Removed invalid disabled tools: {results['removed_invalid']}")

    # Discover tools from servers
    logger.info("Discovering tools from servers...")
    discovered = update_config_with_discovered_tools(server_urls, timeout=args.timeout)

    for server_name, tools in discovered.items():
        if tools:
            logger.info(f"  {server_name}: {len(tools)} tools - {tools}")
        else:
            logger.warning(f"  {server_name}: no tools discovered (server may be offline)")

    logger.info("Tools config updated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
