"""
Regression tests for the 2026-08-29 mcp_ui audit repairs.

Covers the pure-logic pieces of the fixes (see docs/mcp_ui_audit_2026-08-29.md):
login credential verification (per-credential defaults + constant-time compare),
post-login redirect guard, ephemeral storage-secret fallback, env-var parsing
(value_raw passthrough for prefill), and the atomic tools_config.json save.

NiceGUI runtime behavior (awaited callbacks, element trees) is covered by the
live browser verification, not here.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("nicegui")
pytest.importorskip("aiohttp")

from mcp_ui.management_ui import (  # noqa: E402
    _get_storage_secret,
    _safe_redirect_target,
    verify_credentials,
)
from mcp_ui.components.env_var_editor import parse_env_vars_from_api  # noqa: E402
from mcp_ui.components import tool_settings  # noqa: E402


# === UI-10: open-redirect guard ===

@pytest.mark.parametrize("target,expected", [
    ("/", "/"),
    ("/?x=1", "/?x=1"),
    ("https://attacker.example", "/"),
    ("//attacker.example", "/"),
    ("javascript:alert(1)", "/"),
    ("", "/"),
])
def test_safe_redirect_target(target, expected):
    assert _safe_redirect_target(target) == expected


# === UI-6 / UI-7: per-credential defaults, constant-time compare ===

def test_verify_credentials_defaults(monkeypatch):
    monkeypatch.delenv("MCP_UI_USERNAME", raising=False)
    monkeypatch.delenv("MCP_UI_PASSWORD", raising=False)
    assert verify_credentials("admin", "admin")
    assert not verify_credentials("admin", "wrong")
    assert not verify_credentials("wrong", "admin")


def test_verify_credentials_per_credential_override(monkeypatch):
    # UI-6 regression: setting only a password used to lock everyone out.
    monkeypatch.setenv("MCP_UI_PASSWORD", "s3cret")
    monkeypatch.delenv("MCP_UI_USERNAME", raising=False)
    assert verify_credentials("admin", "s3cret")
    assert not verify_credentials("admin", "admin")


# === UI-2: no hardcoded storage secret ===

def test_storage_secret_env_override(monkeypatch):
    monkeypatch.setenv("MCP_UI_SECRET", "configured-secret")
    import mcp_ui.management_ui as m
    m._storage_secret = None
    assert _get_storage_secret() == "configured-secret"
    m._storage_secret = None


def test_storage_secret_ephemeral_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MCP_UI_SECRET", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # hide any real ~/.mcp_ui/secrets
    import mcp_ui.management_ui as m
    m._storage_secret = None
    first = _get_storage_secret()
    assert len(first) == 64  # token_hex(32)
    assert first != "dev-only-secret"
    assert _get_storage_secret() == first  # cached per process
    m._storage_secret = None


# === UI-4: value_raw passthrough enables current-value prefill ===

def test_parse_env_vars_keeps_value_raw():
    data = {"variables": {
        "PORT": {"type": "integer", "is_set": True, "secret": False,
                 "value_raw": "8123", "value_masked": "8123", "default": "8000"},
        "SECRET": {"type": "string", "is_set": True, "secret": True,
                   "value_masked": "****abcd", "default": ""},
    }}
    parsed = {v.name: v for v in parse_env_vars_from_api(data)}
    assert parsed["PORT"].value_raw == "8123"
    assert parsed["SECRET"].value_raw == ""  # raw values for secrets are not sent


# === UI-8: atomic, lock-protected tools_config.json save ===

def test_save_tools_config_atomic(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_settings, "TOOLS_CONFIG_FILE",
                        tmp_path / "tools_config.json")
    config = {"disabled_tools": {"webmcp": ["brave_search_web"]}, "tools": {}, "version": 1}
    tool_settings._save_tools_config(config)

    assert json.loads((tmp_path / "tools_config.json").read_text()) == config
    assert not (tmp_path / "tools_config.json.tmp").exists()  # no temp leftover

    # A second save over a complete file stays consistent (read-modify-write base)
    tool_settings._save_tools_config({"disabled_tools": {}, "tools": {}, "version": 1})
    assert json.loads((tmp_path / "tools_config.json").read_text())["disabled_tools"] == {}
