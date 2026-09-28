"""NiceGUI app assembly: ui.run with the shared MCP_UI_* auth env + :8401."""

from __future__ import annotations

import os
import secrets


def main() -> None:
    from dotenv import load_dotenv
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")  # same source as the launcher: MCP_UI_*, MCP_PODUI_PORT

    port = int(os.environ.get("MCP_PODUI_PORT", "8401"))
    # localhost by default (this UI can erase pods); env override still wins
    host = os.environ.get("MCP_PODUI_HOST", "127.0.0.1")
    secret = os.environ.get("MCP_UI_SECRET")
    if not secret:
        # same warning semantics as mcp_ui: ephemeral secret invalidates all
        # sessions on restart; a persistent MCP_UI_SECRET in .env fixes it
        secret = secrets.token_hex(32)
        print("[pod_ui] MCP_UI_SECRET not set — using an ephemeral secret; "
              "logins are invalidated on restart.")
    # same default-credential warning as mcp_ui (auth.check falls back to admin/admin)
    if not os.environ.get("MCP_UI_USERNAME") and not os.environ.get("MCP_UI_PASSWORD"):
        print("[pod_ui] MCP_UI_USERNAME/MCP_UI_PASSWORD are unset — default "
              f"admin/admin credentials are active. UI is bound to {host}; "
              "set credentials before exposing it beyond localhost.")
    elif not os.environ.get("MCP_UI_USERNAME") or not os.environ.get("MCP_UI_PASSWORD"):
        print("[pod_ui] Only one of MCP_UI_USERNAME/MCP_UI_PASSWORD is set — "
              "the other falls back to 'admin'.")

    from nicegui import ui

    from . import auth
    from .pages_v3 import backups as backups_v3, bundles as bundles_v3, \
        home as home_v3, logs as logs_v3, operations as operations_v3  # noqa: F401

    auth.register_login_page()



    # ---- routes ----
    @ui.page("/")
    @ui.page("/v3")
    def v3_home() -> None:
        home_v3.page()

    @ui.page("/v3/logs")
    def v3_logs() -> None:
        logs_v3.page()

    @ui.page("/v3/bundles")
    def v3_bundles() -> None:
        bundles_v3.page()

    @ui.page("/v3/operations")
    def v3_operations() -> None:
        operations_v3.page()

    @ui.page("/v3/backups")
    def v3_backups() -> None:
        backups_v3.page()

    ui.run(title="Pod Manager", host=host, port=port,
           storage_secret=secret, reload=False, show=False,
           favicon="🎛️")
