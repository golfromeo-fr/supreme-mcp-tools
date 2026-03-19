# Security Framework

## API Key Authentication

```python
# launcher/security/auth.py

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from typing import Optional, Dict

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Store API keys (in production, use secure storage)
VALID_API_KEYS = {
    "admin-key-xxx": {"role": "admin", "tools": ["*"]},
    "readonly-key-xxx": {"role": "readonly", "tools": ["*"]},
}


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> dict:
    """Verify API key and return permissions."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return VALID_API_KEYS[api_key]


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(*args, permissions: dict = Depends(verify_api_key), **kwargs):
            if permissions["role"] != "admin" and permission not in permissions.get("permissions", []):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

## Rate Limiting

```python
# launcher/security/rate_limit.py

import time
from typing import Dict
from collections import defaultdict


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": requests_per_minute,
            "last_update": time.time()
        })
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        bucket = self.buckets[key]
        now = time.time()
        
        # Refill tokens
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + elapsed * (self.requests_per_minute / 60)
        )
        bucket["last_update"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False
```

## Audit Logging

```python
# launcher/security/audit.py

import logging
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

logger = logging.getLogger("audit")


class AuditLogger:
    """Audit logger for security-critical operations."""
    
    def __init__(self, log_file: str = "~/.config/supreme-mcp-tools/audit.log"):
        self.log_file = Path(log_file).expanduser()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        action: str,
        user: str,
        tool_name: str,
        details: Dict[str, Any],
        success: bool = True
    ):
        """Log an audit event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "tool_name": tool_name,
            "details": details,
            "success": success
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Also log to standard logger
        logger.info(f"AUDIT: {action} by {user} on {tool_name}")
```

## Security Configuration

```json
{
  "security": {
    "api_keys": {
      "admin-key-xxx": {
        "role": "admin",
        "tools": ["*"]
      },
      "readonly-key-xxx": {
        "role": "readonly",
        "tools": ["*"],
        "permissions": ["query"]
      }
    },
    "rate_limit": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_size": 10
    },
    "audit": {
      "enabled": true,
      "log_file": "~/.config/supreme-mcp-tools/audit.log"
    }
  }
}
```

## Security Best Practices

### 1. Use Environment Variables for API Keys

```bash
# .env file
ADMIN_API_KEY=admin-key-xxx
READONLY_API_KEY=readonly-key-xxx
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

VALID_API_KEYS = {
    os.getenv("ADMIN_API_KEY"): {"role": "admin", "tools": ["*"]},
    os.getenv("READONLY_API_KEY"): {"role": "readonly", "tools": ["*"]},
}
```

### 2. Rotate API Keys Regularly

```python
# Generate new API key
import secrets

def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)
```

### 3. Log All Security Events

```python
# Log authentication attempts
audit_logger.log(
    action="authenticate",
    user=api_key[:8] + "...",  # Log partial key only
    tool_name="management",
    details={"success": True, "ip": request.client.host}
)
```

### 4. Use HTTPS in Production

```python
# In production, use HTTPS
uvicorn.run(
    app,
    host="0.0.0.0",
    port=9091,
    ssl_keyfile="key.pem",
    ssl_certfile="cert.pem"
)
```
