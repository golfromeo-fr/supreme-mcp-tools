# Phase 3: Clean Up Config Files

**Goal**: Delete redundant config files and remove port config from other config files.

---

## Task 3.1: Delete `config.json` (root)

**File**: `/home/gr/supreme-mcp-tools/config.json`

**Reason**: Outdated, conflicts with new scheme, duplicated by `config/launcher_config.json`

**Action**: DELETE

---

## Task 3.2: Delete `config/port_config.json`

**File**: `/home/gr/supreme-mcp-tools/config/port_config.json`

**Reason**: Merged into `config/ports.json`

**Action**: DELETE

---

## Task 3.3: Update `config/launcher_config.json`

**File**: `config/launcher_config.json`

**Changes**:
- Remove entire `portAllocation` section
- Keep only non-port config (toolDirectories, fefV3, logging, etc.)

**Before** (has port config):
```json
{
  "toolDirectories": [...],
  "portAllocation": {
    "mode": "manual",
    "ranges": {...},
    "reservedPorts": {...},
    "manualPorts": {...}
  },
  "fefV3": {...}
}
```

**After** (port config removed):
```json
{
  "toolDirectories": [...],
  "fefV3": {...}
}
```

---

## Task 3.4: Update `config/launcher_config.example.json`

**File**: `config/launcher_config.example.json`

**Changes**:
- Remove `portAllocation` section (same as 3.3)

---

## Task 3.5: Update `config/monitoring_config.json`

**File**: `config/monitoring_config.json`

**Changes**:
- Remove `port` field from exporters config

**Before**:
```json
{
  "enabled": true,
  "metrics": {
    "exporters": [
      {
        "type": "prometheus",
        "enabled": true,
        "config": {
          "endpoint": "/metrics",
          "port": 9090
        }
      }
    ]
  }
}
```

**After**:
```json
{
  "enabled": true,
  "metrics": {
    "exporters": [
      {
        "type": "prometheus",
        "enabled": true,
        "config": {
          "endpoint": "/metrics"
        }
      }
    ]
  }
}
```

---

## Files Changed/Deleted

| File | Action |
|------|--------|
| `config.json` (root) | DELETE |
| `config/port_config.json` | DELETE |
| `config/launcher_config.json` | EDIT - remove `portAllocation` |
| `config/launcher_config.example.json` | EDIT - remove `portAllocation` |
| `config/monitoring_config.json` | EDIT - remove `port` from exporters |

---

## Dependencies

- Phase 2 should be complete before Phase 3
