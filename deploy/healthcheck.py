#!/usr/bin/env python3
"""Container healthcheck: the node's central /health must answer."""
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8200/health", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
