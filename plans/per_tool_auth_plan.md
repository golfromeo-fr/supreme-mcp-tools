# Per-Tool API Key Authentication

## Context
Currently the system has a single global API key for the central management server. The user wants per-tool authorization keys so each MCP tool can have its own auth credential. This is needed for:
- Securing each tool's management endpoints independently
- Avoiding a single point of auth failure
- Clearer terminology (e.g., "Brave Search API Key" vs generic "api_key")

## Key Concepts

| Server | Port | Purpose | Auth |
|--------|------|---------|------|
| Central Management API | 8200 | Web UI admin | `MCP_API_KEY` env (global) |
| Tool's FEF Extension API | 81xx | Extension management (mutate/query/execute) | **Tool's own key from config.json** (optional) |
| Tool's MCP Endpoint | 80xx | Actual MCP protocol for Claude Code/Kilo Code | None (network-level only) |

The per-tool auth key is **optional** - if not configured in `config.json`, the tool's extension API has no auth (open access).

## Implementation Plan

### 1. Add optional auth section to tool config.json files

Each tool's `config.json` gets a new optional `auth` section:
```json
{
  "name": "webmcp",
  "auth": {
    "api_key": "tool-specific-secret-key"
  }
}
```
If `auth.api_key` is absent or empty, no auth is required.

**Files to modify:**
- `tools/webmcp/config.json`
- `tools/simplemcp/config.json`
- `tools/ragmcp/config.json`

### 2. Modify ExtensionHTTPServer to verify API keys

**File:** `launcher/tool_extensions/http_server.py`

Add auth verification to all routes:
```python
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _verify_api_key(self, key: Optional[str] = Depends(API_KEY_HEADER)) -> bool:
    if self.api_key is None:
        return True  # No auth configured
    if key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if key != self.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
```

Apply `Depends(self._verify_api_key)` to all routes.

### 3. Pass per-tool auth key from launcher to ExtensionHTTPServer

**File:** `launcher/env_manager.py`

Add function to load auth config:
```python
def load_auth_config(tool_name: str) -> Dict[str, Any]:
    """Load auth config from tool's config.json"""
    config_path = PROJECT_ROOT / "tools" / tool_name / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = json.load(f)
    return config.get("auth", {})
```

**File:** `launcher/server_manager.py`

Read `auth.api_key` from config.json and pass to `ExtensionHTTPServer`:
```python
auth_config = load_auth_config(tool_name)
api_key = auth_config.get("api_key")  # None if not configured
mgmt_server = ExtensionHTTPServer(
    tool_name=tool_name,
    registry=extension_registry,
    port=mgmt_port,
    host=self.host,
    api_key=api_key  # Optional - if None, no auth
)
```

### 4. Add auth API endpoints to ManagementServer

**File:** `launcher/management_server.py`

Add GET/PUT endpoints for per-tool auth:
```python
@app.get("/api/tools/{tool_name}/auth")
async def get_tool_auth(tool_name: str, _: bool = Depends(self._verify_api_key)):
    """Get auth config for a tool (masked key)"""
    auth_config = load_auth_config(tool_name)
    api_key = auth_config.get("api_key", "")
    return {
        "api_key": {
            "is_set": bool(api_key),
            "value_masked": mask_value(api_key) if api_key else None
        }
    }

@app.put("/api/tools/{tool_name}/auth")
async def update_tool_auth(tool_name: str, request: AuthUpdate, _: bool = Depends(self._verify_api_key)):
    """Update auth config for a tool"""
    # Update config.json
    config_path = PROJECT_ROOT / "tools" / tool_name / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config.setdefault("auth", {})["api_key"] = request.api_key
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return {"success": True}
```

### 5. Display per-tool auth in UI

**New UI Section:** "Authorization" card in tool detail view, separate from Environment Variables.

- Shows if tool has auth key configured (masked)
- "Update" button opens dialog to set new key
- If no key configured, shows "No authorization key set - tool is open"

**Files to modify:**
- `mcp_ui/api_client.py`: Add `get_tool_auth()`, `update_tool_auth()`
- `mcp_ui/management_ui.py`: Add auth section to tool detail rendering
- Create `mcp_ui/components/auth_box.py` (new component)

## Files to Modify

| File | Change |
|------|--------|
| `launcher/env_manager.py` | Add `load_auth_config()` |
| `launcher/server_manager.py` | Pass `api_key` to ExtensionHTTPServer |
| `launcher/tool_extensions/http_server.py` | Add `_verify_api_key` to routes |
| `launcher/management_server.py` | Add `/api/tools/{name}/auth` GET/PUT |
| `mcp_ui/api_client.py` | Add `get_tool_auth()`, `update_tool_auth()` |
| `mcp_ui/management_ui.py` | Render auth section |
| `mcp_ui/components/auth_box.py` | New component for auth UI |
| `tools/webmcp/config.json` | Add placeholder `auth` section |
| `tools/simplemcp/config.json` | Add placeholder `auth` section |
| `tools/ragmcp/config.json` | Add placeholder `auth` section |

## Verification
1. Start launcher with a tool that has `auth.api_key` set in config.json
2. Without `X-API-Key` header → 401 error
3. With correct `X-API-Key` header → works
4. UI shows auth key status (set/not set)
5. UI can update auth key (writes to config.json)
6. Restart tool → new key takes effect
7. Tools without `auth` section → no auth required (backward compatible)
