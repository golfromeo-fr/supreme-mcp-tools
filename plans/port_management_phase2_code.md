# Phase 2: Update Python Code

**Goal**: Update all Python code to read ports from `config/ports.json` only. Remove all hardcoded defaults.

---

## Task 2.1: Update `launcher/launcher_config.py`

**Changes**:
- Remove `_get_default_ranges()` function
- Remove `_get_default_reserved_ports()` function
- Add `load_ports_config()` function that reads `config/ports.json`
- Add `get_ports_config()` function to retrieve loaded config
- Fail with clear error if `ports.json` is missing

**Functions to remove**:
```python
# DELETE these functions:
def _get_default_ranges()
def _get_default_reserved_ports()
```

**Functions to add**:
```python
def load_ports_config(config_dir: str = None) -> dict:
    """Load ports configuration from ports.json. Fails if missing."""
    # ...
    
def get_ports_config() -> dict:
    """Get loaded ports configuration."""
    # ...
```

---

## Task 2.2: Update `launcher/port_manager.py`

**Changes**:
- Remove `DEFAULT_RANGES` class variable
- Constructor requires `ports_config` parameter (no defaults)
- All range logic reads from passed config

**Before**:
```python
class PortManager:
    DEFAULT_RANGES = {
        PortType.MCP: (8000, 8099),
        ...
    }
    
    def __init__(self, ...):
        # Has default ranges built in
```

**After**:
```python
class PortManager:
    def __init__(self, ports_config: dict, ...):
        # All ranges from ports_config
        self.ranges = ports_config.get("ranges", {})
```

---

## Task 2.3: Update `launchmcp.py`

**Changes**:
- Load ports config first (before PortManager)
- Pass ports_config to PortManager constructor
- Pass ports_config to all services

```python
# Add near top of main():
ports_config = load_ports_config(config_dir)
# Pass to PortManager and all services
```

---

## Task 2.4: Update `launcher/__main__.py`

**Changes**:
- Load `ports.json` at startup, before other initialization
- Pass to all components

---

## Task 2.5: Update `launcher/management_server.py`

**Changes**:
- Remove default port value from constructor
- Require port from config

**Before**:
```python
def __init__(self, ..., port: int = 9091):
```

**After**:
```python
def __init__(self, ..., port: int = None):  # Must be provided
    if port is None:
        raise ValueError("port required from config")
```

---

## Task 2.6: Update `monitoring/exporters.py`

**Changes**:
- Read metrics port from ports config via Config class
- Remove hardcoded `port: 9090` dependency

---

## Task 2.7: Update `mcp_ui/management_ui.py`

**Changes**:
- Remove default port `9092`
- Read port from ports config

**Before**:
```python
port = int(os.environ.get("MCP_UI_PORT", 9092))
```

**After**:
```python
port = int(os.environ.get("MCP_UI_PORT", ports_config["reserved"]["management_ui"]))
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `launcher/launcher_config.py` | Remove `_get_default_*()`, add `load_ports_config()` |
| `launcher/port_manager.py` | Remove `DEFAULT_RANGES`, require `ports_config` |
| `launchmcp.py` | Load ports config first, pass to all |
| `launcher/__main__.py` | Load ports at startup |
| `launcher/management_server.py` | Remove default port |
| `monitoring/exporters.py` | Read port from config |
| `mcp_ui/management_ui.py` | Remove default port |

---

## Dependencies

- Phase 1 must complete before Phase 2
