#!/bin/bash
# FEF V3 Test Runner - Quick Start Script
# This script helps you quickly run the FEF V3 test suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FEF V3 Automated Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if MCP tools are running
echo -e "${YELLOW}Checking if MCP tools are running...${NC}"

TOOLS=("webmcp:9001" "simplemcp:9012" "ragmcp:9014")
RUNNING=0
NOT_RUNNING=()

 for tool in "${TOOLS[@]}"; do
     IFS=':' read -r name port <<< "$tool"
     if curl -s "http://localhost:$port/extensions" > /dev/null 2>&1; then
         echo -e "${GREEN}✓${NC} $name is running (port $port)"
         ((RUNNING++))
     else
         echo -e "${RED}✗${NC} $name is NOT running (port $port)"
         NOT_RUNNING+=("$name")
     fi
 done || true

echo ""

if [ ${#NOT_RUNNING[@]} -gt 0 ]; then
    echo -e "${YELLOW}Warning: The following tools are not running:${NC}"
    for tool in "${NOT_RUNNING[@]}"; do
        echo -e "  - ${RED}$tool${NC}"
    done
    echo ""
    echo -e "${YELLOW}To start the tools, run:${NC}"
    echo -e "  ${BLUE}./launchmcp.py simplemcp ragmcp webmcp${NC}"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Check if dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! python3 -c "import requests" 2>/dev/null; then
    echo -e "${RED}Error: 'requests' is not installed${NC}"
    echo -e "${YELLOW}Install with: pip install requests${NC}"
    exit 1
fi

if ! python3 -c "import rich" 2>/dev/null; then
    echo -e "${YELLOW}Warning: 'rich' is not installed (optional for better output)${NC}"
    echo -e "${YELLOW}Install with: pip install rich${NC}"
    echo ""
fi

# Parse command line arguments
VERBOSE=""
TOOLS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -t|--tools)
            # Convert comma-separated to space-separated for Python
            TOOL_LIST=$(echo "$2" | tr ',' ' ')
            TOOLS="--tools $TOOL_LIST"
            shift 2
            ;;
        -s|--space-tools)
            # Space-separated tools
            TOOLS="--tools $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose        Enable verbose output"
            echo "  -t, --tools TOOLS   Comma-separated list of tools to test"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Test all tools"
            echo "  $0 --tools webmcp           # Test only webmcp"
            echo "  $0 -t webmcp,simplemcp    # Test multiple tools (comma-separated)"
            echo "  $0 --verbose                # Enable verbose output"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run the tests
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running tests...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/test_runner.py" $VERBOSE $TOOLS

# Capture exit code
EXIT_CODE=$?

echo ""

# Print summary based on exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some tests failed${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE
