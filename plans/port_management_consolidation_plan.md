# Port Management Consolidation Plan

**Date**: 2026-03-26  
**Status**: Planning  
**Branch**: feature/addui  
**Goal**: Single `config/ports.json` for ALL port configuration, zero hardcoded ports in Python code

---

## TL;DR

Consolidate 5 overlapping config files into a single `config/ports.json`. Remove all hardcoded port defaults from Python code. Code fails fast if config missing.

---

## Current State (Problem)

| File | Port Config | Issue |
|------|-------------|-------|
| `config.json` (root) | `portAllocation.ports` + `managementPorts` | OUTDATED - conflicts |
| `config/launcher_config.json` | Mixed port + other config | Contains `portAllocation` |
| `config/port_config.json` | `ranges` + `reservedPorts` | DUPLICATE |
| `config/monitoring_config.json` | Hardcoded `port: 9090` | Not centralized |
| `launcher_config.py` | `_get_default_ranges()` | Python defaults |

**Result**: Confusion, conflicts, maintenance burden.

---

## Target State (Solution)

```
config/
├── ports.json              # ALL port config (single source of truth)
├── ports.example.json      # Documented example
├── launcher_config.json    # Non-port config only
├── monitoring_config.json  # No port field
```

**Key Principle**: `config/ports.json` is the ONLY place for port config. No Python defaults.

---

## Phases

| Phase | Description | Files |
|-------|-------------|-------|
| [Phase 1: Config](port_management_phase1_config.md) | Create unified ports.json | `ports.json`, `ports.example.json` |
| [Phase 2: Code](port_management_phase2_code.md) | Update Python to use ports.json | `launcher_config.py`, `port_manager.py`, etc. |
| [Phase 3: Cleanup](port_management_phase3_cleanup.md) | Delete redundant configs | `config.json`, `port_config.json`, etc. |
| [Phase 4: Verification](port_management_phase4_verification.md) | Verify no hardcoded ports | Grep, tests |

---

## `ports.json` Structure

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

---

## Decisions

- **Single config file**: `config/ports.json` is the ONLY source for port configuration
- **No Python defaults**: Code fails with clear error if `ports.json` missing
- **Clean separation**: `launcher_config.json` contains only non-port config
- **Port ranges**: 100 ports per category (mcp: 8000-8099, mgmt: 8100-8199, etc.)
