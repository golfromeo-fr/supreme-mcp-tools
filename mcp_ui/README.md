# MCP Tools Management UI

A web-based management interface for MCP tools built with NiceGUI.

## Overview

The Management UI provides a user-friendly web interface for monitoring and managing MCP tools, including:
- Viewing tool status and capabilities
- Querying data sources
- Configuring mutators
- Executing actions

## Quick Start

### Installation

Ensure all dependencies are installed:

```bash
pip install nicegui>=1.4.0 httpx>=0.27.0 pydantic>=2.5.0
```

### Launching the UI

```bash
# Method 1: Using uvicorn (recommended - enables all NiceGUI features)
cd /home/gr/supreme-mcp-tools
uvicorn mcp_ui:app --host 0.0.0.0 --port 9092

# Method 2: Using python module (auto-index mode)
cd /home/gr/supreme-mcp-tools
python -m mcp_ui.management_ui

# With authentication
export MCP_UI_PASSWORD="yourpassword"
uvicorn mcp_ui:app --host 0.0.0.0 --port 9092

# Custom configuration
export MCP_UI_PORT=9092
export MCP_UI_THEME="dark"
export MCP_API_URL="http://localhost:9091"
uvicorn mcp_ui:app --host 0.0.0.0 --port 9092
```

The UI will be available at `http://localhost:9092`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_UI_HOST` | `0.0.0.0` | Host to bind the UI server |
| `MCP_UI_PORT` | `9092` | Port for the UI server |
| `MCP_UI_THEME` | `dark` | Theme: `dark`, `light`, or `system` |
| `MCP_UI_USERNAME` | `admin` | Username for login (requires `MCP_UI_PASSWORD`) |
| `MCP_UI_PASSWORD` | (none) | Password for login. If not set, no login required |
| `MCP_API_URL` | `http://localhost:9091` | Management API base URL |
| `MCP_API_KEY` | (none) | API key for Management API authentication |
| `MCP_UI_REFRESH_INTERVAL` | `30` | Auto-refresh interval in seconds |

## Features

### Tool Management
- View all registered MCP tools in the sidebar
- See tool status (Running, Stopped, Error, Unknown)
- Select a tool to view detailed information

### Data Sources
- Query read-only data sources
- Refresh data on demand
- View key-value data in tables

### Configuration (Mutators)
- Modify tool configuration through dynamic forms
- Support for various input types:
  - Integer and number inputs with min/max validation
  - Boolean switches
  - Text inputs and textareas
  - Array inputs (comma-separated values)

### Actions
- Execute tool-specific actions
- Confirmation dialogs before execution
- Parameter input forms for action arguments

## Architecture

```
mcp_ui/
├── __init__.py              # Package initialization
├── management_ui.py         # Main NiceGUI application
├── auth.py                  # Authentication module
├── api_client.py            # Management API HTTP client
├── models.py                # Pydantic data models
├── state.py                 # Client state management
└── components/
    ├── __init__.py          # Component exports
    ├── tool_list.py         # Sidebar tool list
    ├── tool_card.py         # Tool detail card
    ├── data_sources_box.py  # Data sources panel
    ├── mutators_box.py      # Configuration panel
    └── actions_box.py       # Actions panel
```

## Authentication

When `MCP_UI_PASSWORD` is set, the UI requires authentication:

1. Navigate to the UI
2. Enter username and password
3. Click Login

The session is stored per-client using NiceGUI's client storage.

## API Connection

The UI connects to the Management API server (default: `http://localhost:9091`). Ensure the API server is running before launching the UI.

### API Endpoints Used

- `GET /health` - Health check
- `GET /api/tools` - List all tools
- `GET /api/tools/{name}` - Get tool details
- `GET /api/tools/{name}/extensions` - Get tool extensions
- `POST /api/tools/{name}/extensions/{ext}/query` - Query data source
- `POST /api/tools/{name}/extensions/{ext}/mutate` - Submit mutation
- `POST /api/tools/{name}/extensions/{ext}/execute` - Execute action

## Troubleshooting

### Connection Errors
If you see connection errors:
1. Verify the Management API server is running
2. Check the `MCP_API_URL` environment variable
3. Ensure firewall rules allow the connection

### UI Not Loading
1. Check if port 9092 is available
2. Try a different port with `MCP_UI_PORT`
3. Check browser console for JavaScript errors

### Authentication Issues
1. Ensure `MCP_UI_PASSWORD` is set
2. Clear browser cookies and cache
3. Check username defaults to `admin` if not specified

## Development

### Running from Source

```bash
cd /home/gr/supreme-mcp-tools
python -m ui.management_ui
```

### Importing Components

```python
from ui.management_ui import ManagementUI, main
from ui.auth import is_auth_enabled, logout
from ui.api_client import ManagementAPIClient
from ui.components import ToolList, ToolCard
```

## License

Part of the MCP Tools project.
