#!/bin/bash
#
# Migration Script: Web MCP - SSE to Streamable HTTP Transport
# 
# This script helps users migrate from SSE to Streamable HTTP transport
# for the Web MCP server.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Web MCP Migration Script${NC}"
echo -e "${BLUE}SSE to Streamable HTTP Transport${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if FastAPI is installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}Error: FastAPI is not installed.${NC}"
    echo "Please install it with: pip install fastapi>=0.104.0"
    echo "Or run: pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}✓ FastAPI is installed${NC}"
echo ""

# Display current SSE configuration
echo -e "${YELLOW}Current SSE Configuration:${NC}"
echo '{
  "mcpServers": {
    "web_mcp": {
      "type": "sse",
      "url": "http://localhost:8001/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}'
echo ""

# Display new Streamable HTTP configuration
echo -e "${YELLOW}New Streamable HTTP Configuration:${NC}"
echo '{
  "mcpServers": {
    "web_mcp": {
      "type": "streamable-http",
      "url": "http://localhost:8002/mcp",
      "headers": {
        "Content-Type": "application/json"
      },
      "framing": "newline-delimited"
    }
  }
}'
echo ""

# Display migration steps
echo -e "${BLUE}Migration Steps:${NC}"
echo ""
echo "1. Stop the SSE server (if running)"
echo "   - Press Ctrl+C in the terminal running web_mcp.py"
echo ""
echo "2. Update your VSCode configuration:"
echo "   - Open VSCode settings (Ctrl+,)"
echo "   - Search for 'mcpServers'"
echo "   - Replace the web_mcp configuration with the new one shown above"
echo ""
echo "3. Start the Streamable HTTP server:"
echo "   - Run: python web_mcp_streamable.py"
echo "   - Or with custom options: python web_mcp_streamable.py --host 0.0.0.0 --port 8002"
echo ""
echo "4. Verify the server is running:"
echo "   - Run: curl http://localhost:8002/health"
echo "   - Expected output: {\"status\":\"healthy\",\"active_sessions\":0}"
echo ""

# Ask if user wants to proceed
echo -e "${YELLOW}Do you want to proceed with the migration? (y/n)${NC}"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Migration cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting migration...${NC}"
echo ""

# Check if SSE server is running
echo -e "${YELLOW}Checking if SSE server is running...${NC}"
if curl -s http://localhost:8001/sse > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Warning: SSE server appears to be running on port 8001${NC}"
    echo "Please stop the SSE server before starting the Streamable HTTP server."
    echo ""
    echo -e "${YELLOW}Do you want to continue anyway? (y/n)${NC}"
    read -r continue_response
    if [[ ! "$continue_response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Migration cancelled.${NC}"
        exit 0
    fi
else
    echo -e "${GREEN}✓ SSE server is not running${NC}"
fi
echo ""

# Offer to start the Streamable HTTP server
echo -e "${YELLOW}Do you want to start the Streamable HTTP server now? (y/n)${NC}"
read -r start_response

if [[ "$start_response" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting Streamable HTTP server...${NC}"
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Start the server in the foreground
    cd "$SCRIPT_DIR" && python3 web_mcp_streamable.py
else
    echo ""
    echo -e "${GREEN}Migration preparation complete!${NC}"
    echo ""
    echo "To start the Streamable HTTP server later, run:"
    echo "  cd $SCRIPT_DIR"
    echo "  python3 web_mcp_streamable.py"
    echo ""
fi

# Rollback instructions
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Rollback Instructions${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "If you need to rollback to SSE transport:"
echo ""
echo "1. Stop the Streamable HTTP server (if running)"
echo "2. Restore your VSCode configuration to:"
echo '{
  "mcpServers": {
    "web_mcp": {
      "type": "sse",
      "url": "http://localhost:8001/sse",
      "headers": {
        "Content-Type": "application/json"
      }
    }
  }
}'
echo "3. Start the SSE server: python web_mcp.py"
echo ""

echo -e "${GREEN}Migration complete!${NC}"
