"""E3/M2 — Users tab (admin only): manage the deployment's user accounts.

Per-user: server checkboxes (reach) + per-server function-mask grid
(deny-list) — the two dimensions from the user's spec — plus create with
once-only key display, rotate key, set password, enable/disable, delete.
All data through the central /api/users surface; NiceGUI rules:
container.clear() rebuilds, asyncio.create_task for loads, dialogs awaited
in async handlers.
"""

import logging

from nicegui import ui

logger = logging.getLogger(__name__)


def render_users_tab(container) -> None:
    """Entry point: (re)build the tab into `container` (sync — same
    slot-stack rule as the Memory tab: initial build INSIDE the container)."""
    from ..management_ui import get_api_client

    client = get_api_client()

    def _notify_error(message: str):
        ui.notify(message, type="negative", duration=5)

    def _rebuild():
        container.clear()
        with container:
            _build()

    def _build():
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.label("Users").classes("text-h6")
            ui.space()
            ui.button("Add user", icon="person_add", on_click=_add_user_dialog)\
                .props("outline dense")
            ui.button("Refresh", icon="refresh", on_click=_rebuild).props("flat dense")

        loading = ui.label("Loading users…").classes("text-grey")
        users_box = ui.column().classes("w-full")

        async def _load():
            response = await client.list_users()
            loading.set_text("")
            users_box.clear()
            if not response.success:
                _notify_error(f"Failed to load users: {response.error}")
                with users_box:
                    ui.label(f"Error: {response.error}").classes("text-negative")
                return
            users = response.data.get("users", []) if isinstance(response.data, dict) else []
            if not users:
                with users_box:
                    ui.label("No users yet — add one.").classes("text-grey")
                return
            with users_box:
                for user in users:
                    _user_card(user, users_box)

        asyncio_create_task(_load())

    def _user_card(user: dict, users_box):
        username = user.get("username", "?")
        servers = user.get("servers") or []
        masked = user.get("masked_functions") or {}
        enabled = user.get("enabled", True)
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full justify-between items-center gap-2 flex-wrap"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(username, color="blue-7")
                    ui.badge(user.get("role", "user"),
                             color="deep-purple-7" if user.get("role") == "admin" else "grey-8")
                    if not enabled:
                        ui.badge("disabled", color="red-7")
                    for s in servers:
                        ui.badge(s, color="teal-7").props("outline")
                with ui.row().classes("items-center gap-1"):
                    if enabled:
                        ui.button(icon="toggle_off", on_click=lambda _e=False, u=username: _action(
                            lambda: client.set_user_enabled(u, False),
                            f"{u} disabled")).tooltip("Disable").props("flat dense")
                    else:
                        ui.button(icon="toggle_on", on_click=lambda _e=False, u=username: _action(
                            lambda: client.set_user_enabled(u, True),
                            f"{u} enabled")).tooltip("Enable").props("flat dense")
                    ui.button(icon="key", on_click=lambda _e=False, u=username: _rotate_key(u))\
                        .tooltip("Rotate MCP key (old key dies)").props("flat dense")
                    ui.button(icon="lock_reset", on_click=lambda _e=False, u=username: _password_dialog(u))\
                        .tooltip("Set password").props("flat dense")
                    ui.button(icon="delete", color="negative", on_click=lambda _e=False, u=username: _delete_dialog(u))\
                        .tooltip("Delete user").props("flat dense")
            masked_summary = ", ".join(
                f"{srv}: {len(fns)}" for srv, fns in sorted(masked.items())
            ) or "no per-user masks"
            ui.label(f"servers: {', '.join(servers) or 'none'} · masks: {masked_summary}")\
                .classes("text-caption text-grey")
            # per-server function-mask grid (edit)
            ui.button("Edit access", icon="rule",
                      on_click=lambda _e=False, u=user: _access_dialog(u))\
                .props("flat dense")

    def _action(fn, message: str):
        async def _run():
            response = await fn()
            if response.success:
                ui.notify(message, type="positive")
                _rebuild()
            else:
                _notify_error(response.error)
        asyncio_create_task(_run())

    async def _rotate_key(username: str):
        response = await get_api_client().rotate_user_key(username)
        if response.success:
            key = response.data.get("mcp_key") if isinstance(response.data, dict) else None
            with ui.dialog() as dialog, ui.card():
                ui.label(f"New MCP key for {username}").classes("text-subtitle1")
                ui.label(key or "?").classes("font-mono text-sm")
                ui.label("Copy it now — it will not be shown again.").classes("text-caption text-negative")
                ui.button("Close", on_click=dialog.close)
            dialog.open()
        else:
            _notify_error(f"Rotation failed: {response.error}")

    def _password_dialog(username: str):
        with ui.dialog() as dialog, ui.card().classes("w-80"):
            ui.label(f"Set password for {username}").classes("text-subtitle1")
            pw = ui.input("New password", password=True).classes("w-full")

            async def _save():
                response = await get_api_client().set_user_password(username, pw.value or "")
                dialog.close()
                if response.success:
                    ui.notify("Password updated.", type="positive")
                else:
                    _notify_error(response.error or "update failed")

            ui.button("Save", on_click=_save).classes("w-full")
        dialog.open()

    def _delete_dialog(username: str):
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete user {username}? This cannot be undone.")
            with ui.row():
                ui.button("Delete", color="negative",
                          on_click=lambda: dialog.submit("yes"))
                ui.button("Cancel", on_click=lambda: dialog.submit("no"))

        async def _wait():
            if await dialog == "yes":
                response = await get_api_client().delete_user(username)
                if response.success:
                    ui.notify(f"{username} deleted.", type="positive")
                    _rebuild()
                else:
                    _notify_error(response.error or "delete failed")

        asyncio_create_task(_wait())

    def _access_dialog(user: dict):
        """Edit one user's reach (server checkboxes) + per-server function
        masks (comma-separated deny-lists, E1 semantics scoped to the user)."""
        from ..management_ui import get_state

        username = user.get("username", "?")
        current_servers = user.get("servers") or []
        current_masks = user.get("masked_functions") or {}
        known_servers = sorted(
            {t.name for t in get_state().tools} | set(current_servers)
        )
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            ui.label(f"Access for {username}").classes("text-subtitle1")
            checks: dict[str, ui.checkbox] = {}
            mask_inputs: dict[str, ui.input] = {}
            with ui.column().classes("w-full gap-2"):
                for srv in known_servers:
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        checks[srv] = ui.checkbox(srv, value=srv in current_servers)
                        mask_inputs[srv] = ui.input(
                            "masked (comma-sep)",
                            value=", ".join(current_masks.get(srv, [])),
                        ).classes("w-64").tooltip("Functions this user must NOT see on this server")

            async def _save():
                servers = [s for s, cb in checks.items() if cb.value]
                masked = {
                    s: [f.strip() for f in (mi.value or "").split(",") if f.strip()]
                    for s, mi in mask_inputs.items()
                    if (mi.value or "").strip()
                }
                ok1 = await get_api_client().set_user_servers(username, servers)
                ok2 = await get_api_client().set_user_masked_functions(username, masked)
                dialog.close()
                if ok1.success and ok2.success:
                    ui.notify("Access updated.", type="positive")
                    _rebuild()
                else:
                    _notify_error((ok1.error or ok2.error or "update failed"))

            ui.button("Save", on_click=_save).props("outline dense")
        dialog.open()

    def _add_user_dialog():
        from ..management_ui import get_state

        known_servers = sorted({t.name for t in get_state().tools})
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Add user").classes("text-subtitle1")
            username = ui.input("Username (a-z, 0-9, - _)").classes("w-full")
            password = ui.input("Password (min 8)", password=True).classes("w-full")
            role = ui.select({"user": "user", "admin": "admin"}, value="user",
                             label="Role").classes("w-full")
            with ui.column().classes("w-full"):
                ui.label("Servers this user may reach:").classes("text-caption")
                # create the checkboxes INSIDE the column's slot — creating
                # them before the with-block attached them to the card above
                # the label (slot-stack variant, user screenshot 2026-09-08)
                server_checks = {
                    s: ui.checkbox(s, value=False) for s in known_servers
                }
                for cb in server_checks.values():
                    cb.classes("ml-2")
            error = ui.label("").classes("text-negative text-caption")

            async def _create():
                servers = [s for s, cb in server_checks.items() if cb.value]
                response = await get_api_client().create_user(
                    username.value or "", password.value or "", role.value, servers
                )
                if not response.success:
                    error.set_text(response.error or "creation failed")
                    return
                key = response.data.get("mcp_key") if isinstance(response.data, dict) else None
                dialog.close()
                with ui.dialog() as key_dialog, ui.card().classes("w-96"):
                    ui.label(f"MCP key for {username.value}").classes("text-subtitle1")
                    ui.label(key or "?").classes("font-mono text-sm whitespace-pre-wrap")
                    ui.label("Copy it now — it will not be shown again.")\
                        .classes("text-caption text-negative")
                    ui.button("Close", on_click=key_dialog.close)
                key_dialog.open()
                _rebuild()

            with ui.row():
                ui.button("Create", on_click=_create).props("outline dense")
                ui.button("Cancel", on_click=dialog.close).props("flat dense")
        dialog.open()

    _rebuild()


def asyncio_create_task(coro):
    import asyncio
    return asyncio.create_task(coro)


def get_api_client():
    from ..management_ui import get_api_client

    return get_api_client()
