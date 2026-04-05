# Phase 1: Create Unified Config

**Goal**: Create `config/ports.json` as the single source of truth for all port configuration.

---

## Task 1.1: Create `config/ports.json`

Create the main config file with all port ranges, reserved ports, and tool assignments:

```json
{
  "ranges": {
    "mcp": [8000, 8099],
    "mgmt": [8100, 8199],
    "system": [8200, 8299],
    "metrics": [8300, 8399],
    "ui": [8400, 8499]
  },
  "reserved": {
    "central_management": 8200,
    "metrics_server": 8300,
    "management_ui": 8400
  },
  "assignments": {
    "mcp": {
      "oraclemcp": 8000,
      "webmcp": 8001,
      "simplemcp": 8002,
      "convertermcp": 8003,
      "ragmcp": 8004
    },
    "mgmt": {
      "oraclemcp": 8100,
      "webmcp": 8101,
      "simplemcp": 8102,
      "convertermcp": 8103,
      "ragmcp": 8104
    }
  }
}
```

**File**: `config/ports.json`

---

## Task 1.2: Create `config/ports.example.json`

Create documented example file:

```json
{
  "_description": "Port configuration for MCP tools. All ports must be defined here - no hardcoded ports in Python.",
  "ranges": {
    "mcp": [8000, 8099],
    "mgmt": [8100, 8199],
    "system": [8200, 8299],
    "metrics": [8300, 8399],
    "ui": [8400, 8499]
  },
  "reserved": {
    "central_management": 8200,
    "metrics_server": 8300,
    "management_ui": 8400
  },
  "assignments": {
    "mcp": {
      "_comment": "MCP tool endpoints",
      "oraclemcp": 8000,
      "webmcp": 8001,
      "simplemcp": 8002,
      "convertermcp": 8003,
      "ragmcp": 8004
    },
    "mgmt": {
      "_comment": "Management ports per tool",
      "oraclemcp": 8100,
      "webmcp": 8101,
      "simplemcp": 8102,
      "convertermcp": 8103,
      "ragmcp": 8104
    }
  }
}
```

**File**: `config/ports.example.json`

---

## Files Created

| File | Action |
|------|--------|
| `config/ports.json` | CREATE |
| `config/ports.example.json` | CREATE |

---

## Dependencies

- Phase 1 completes before Phase 2 begins
