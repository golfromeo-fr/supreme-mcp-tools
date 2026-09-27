"""Login + session gate for pod_ui. Always-on auth (this UI can erase pods):
same MCP_UI_USERNAME/MCP_UI_PASSWORD pair as mcp_ui, MCP_UI_SECRET signs the
session cookies. No password_toggle_button anywhere (NiceGUI 3.16.0 wiring
bug, removed in mcp_ui 4c422c3)."""

from __future__ import annotations

import hmac
import os

from nicegui import app, ui

ENV_USERNAME = "MCP_UI_USERNAME"
ENV_PASSWORD = "MCP_UI_PASSWORD"

NAV = [
    ("/", "Dashboard", "dashboard"),
    ("/logs", "Logs", "article"),
    ("/operations", "Operations", "play_circle"),
    ("/bundles", "Bundles", "inventory_2"),
    ("/backups", "Backups & data", "settings_backup_restore"),
]


def check(username: str, password: str) -> bool:
    stored_u = os.environ.get(ENV_USERNAME) or "admin"
    stored_p = os.environ.get(ENV_PASSWORD) or "admin"
    u_ok = hmac.compare_digest((username or "").encode(), stored_u.encode())
    p_ok = hmac.compare_digest((password or "").encode(), stored_p.encode())
    return u_ok and p_ok


def is_authed() -> bool:
    try:
        return bool(app.storage.user.get("authenticated"))
    except Exception:
        return False


def require() -> bool:
    """Call at the top of every page; navigates to /login when not authed."""
    if not is_authed():
        ui.navigate.to("/login")
        return False
    return True


def frame(title: str):
    """Common chrome. Returns the content container, or None when redirected
    to /login (callers: `if (body := frame(...)) is None: return`)."""
    if not require():
        return None
    ui.dark_mode(True)
    with ui.header().classes("w-full p-3 items-center"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label("Pod Manager").classes("text-h6")
            ui.label(title).classes("text-caption opacity-60")
    # value=True: without it NiceGUI probes drawer state via JS on every
    # client connect (1s timeout) — a reconnecting tab after a server
    # restart spams a TimeoutError traceback into the console
    with ui.drawer(side="left", value=True).props("bordered behavior=desktop"):
        with ui.column().classes("w-full p-3 gap-1"):
            for path, label, icon in NAV:
                ui.button(label, icon=icon, on_click=lambda p=path: ui.navigate.to(p)) \
                    .classes("w-full").props("flat align=left")
            ui.separator()
            ui.button("try the v2 layout", icon="bolt",
                      on_click=lambda: ui.navigate.to("/v2")).classes("w-full") \
                .props("flat align=left").tooltip("candidate redesign — same backend")
            ui.button("try the v3 layout", icon="hub",
                      on_click=lambda: ui.navigate.to("/v3")).classes("w-full") \
                .props("flat align=left").tooltip("cluster-map redesign — current candidate")
            ui.button("Logout", icon="logout", on_click=logout).classes("w-full").props("flat")
    body = ui.column().classes("w-full p-4 gap-3")
    return body


def logout() -> None:
    app.storage.user.clear()
    ui.navigate.to("/login")


def register_login_page() -> None:
    @ui.page("/login")
    async def login_page() -> None:
        if is_authed():
            ui.navigate.to("/")
            return
        ui.dark_mode(True)
        with ui.column().classes("absolute-center gap-3 w-80"):
            ui.label("Pod Manager").classes("text-h5")
            ui.label("podman environments").classes("text-caption opacity-60")
            user = ui.input("Username")
            pwd = ui.input("Password", password=True)
            msg = ui.label("").classes("text-red-8 text-caption")

            def try_login() -> None:
                if check(user.value, pwd.value):
                    app.storage.user.update({"authenticated": True})
                    ui.navigate.to("/")
                else:
                    msg.set_text("Wrong username or password")

            ui.button("Log in", on_click=try_login)
            pwd.on("keydown.enter", try_login)
