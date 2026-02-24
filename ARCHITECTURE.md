# Unified MCP Launcher System - Architecture Design

## Directory Structure

### Proposed Layout

```
supreme-mcp-tools/
├── launchmcp.py                 # Main launcher entry point
├── launcher/
│   ├── __init__.py
│   ├── discovery.py              # Tool discovery module
│   ├── port_manager.py           # Port allocation manager
│   ├── server_manager.py        # Server lifecycle manager
│   ├── config.py                # Configuration loader
│   └── exceptions.py            # Custom exceptions
├── config/
│   ├── launcher_config.json     # Default configuration
│   └── launcher_config.example.json
├── tools/                       # MCP tool directories
│   ├── webmcp/
│   │   ├── web_mcp.py
│   │   ├── requirements.txt
│   │   ├── .env
│   │   └── README.md
│   ├── oraclemcp/
│   │   ├── oracleMCP.py
│   │   ├── requirements.txt
│   │   └── README.md
│   └── simplemcp8/
│       ├── simplemcp8.py
│       ├── requirements.txt
│       └── README.md
├── logs/
│   └── launcher.log
├── tests/
│   ├── test_discovery.py
│   ├── test_port_manager.py
│   └── test_server_manager.py
├── requirements.txt             # Launcher dependencies
├── setup.py                     # Installation script
├── ARCHITECTURE.md              # This document
└── README.md                    # User documentation
```

### Directory Structure Rationale

| Directory | Purpose |
|-----------|---------|
| `launchmcp.py` | Entry point - simple to invoke from command line |
| `launcher/` | Core launcher modules - organized by responsibility |
| `config/` | Configuration files - separate from code |
| `tools/` | MCP tools - each in its own directory for independence |
| `logs/` | Log files - centralized logging location |
| `tests/` | Unit tests - ensure reliability |
| `requirements.txt` | Launcher dependencies - separate from tool dependencies |

### Tool Directory Structure

Each MCP tool maintains its existing structure:

```
tools/<tool_name>/
├── <tool_name>.py              # Main tool module
├── requirements.txt            # Tool-specific dependencies
├── .env                        # Tool-specific environment variables
├── README.md                   # Tool documentation
└── (optional submodules)       # Any additional modules
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-20  
**Author**: Architect Mode  
**Status**: Design Complete - Ready for Implementation
