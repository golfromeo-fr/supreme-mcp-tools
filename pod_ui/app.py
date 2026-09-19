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
    host = os.environ.get("MCP_PODUI_HOST", "0.0.0.0")
    secret = os.environ.get("MCP_UI_SECRET")
    if not secret:
        # same warning semantics as mcp_ui: ephemeral secret invalidates all
        # sessions on restart; a persistent MCP_UI_SECRET in .env fixes it
        secret = secrets.token_hex(32)
        print("[pod_ui] MCP_UI_SECRET not set — using an ephemeral secret; "
              "logins are invalidated on restart.")

    from nicegui import ui

    from . import auth
    from .pages import backups, bundles, dashboard, logs, operations  # noqa: F401 (register pages)
    from .pages_v2 import bundles as bundles_v2, env as env_v2, home as home_v2  # noqa: F401
    from .pages_v3 import bundles as bundles_v3, home as home_v3, logs as logs_v3  # noqa: F401

    auth.register_login_page()

    # ---- v1 (current layout) ----
    @ui.page("/")
    def index() -> None:
        dashboard.page()

    @ui.page("/logs")
    def logs_page() -> None:
        logs.page()

    @ui.page("/operations")
    def operations_page() -> None:
        operations.page()

    @ui.page("/bundles")
    def bundles_page() -> None:
        bundles.page()

    @ui.page("/backups")
    def backups_page() -> None:
        backups.page()

    # ---- v2 (candidate layout, same process so the console sees all ops) ----
    @ui.page("/v2")
    def v2_home() -> None:
        home_v2.page()

    @ui.page("/v2/env/{env_key}")
    def v2_env(env_key: str) -> None:
        env_v2.page(env_key)

    @ui.page("/v2/bundles")
    def v2_bundles() -> None:
        bundles_v2.page()

    # ---- v3 (cluster-map layout) ----
    @ui.page("/v3")
    def v3_home() -> None:
        home_v3.page()

    @ui.page("/v3/logs")
    def v3_logs() -> None:
        logs_v3.page()

    @ui.page("/v3/bundles")
    def v3_bundles() -> None:
        bundles_v3.page()

    ui.run(title="Pod Manager", host=host, port=port,
           storage_secret=secret, reload=False, show=False,
           favicon="🎛️")
