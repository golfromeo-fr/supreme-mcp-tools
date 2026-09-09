"""E3/M2 — Users tab (admin only): manage the deployment's user accounts.

Per-user: server checkboxes (reach) + per-server function-mask grid
(deny-list) — the two dimensions from the user's spec — plus create with
once-only key display, rotate key, set password, enable/disable, delete.
All data through the central /api/users surface; NiceGUI rules:
container.clear() rebuilds, asyncio.create_task for loads, dialogs awaited
in async handlers.
"""

import asyncio
import json
import logging
from pathlib import Path

from nicegui import ui

logger = logging.getLogger(__name__)

PORTS = json.loads(
    (Path(__file__).resolve().parents[2] / "config" / "ports.json")
    .read_text())["assignments"]["mcp"]


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
                        ui.button(icon="toggle_off", on_click=_action(
                            lambda: client.set_user_enabled(username, False),
                            f"{username} disabled")).tooltip("Disable").props("flat dense")
                    else:
                        ui.button(icon="toggle_on", on_click=_action(
                            lambda: client.set_user_enabled(username, True),
                            f"{username} enabled")).tooltip("Enable").props("flat dense")
                    # NOTE: plain lambdas — handle_event CALLS the handler, so
                    # a bare `_rotate_key(username)` coroutine object would
                    # raise, and `def ..._dialog(username)` would run at
                    # button-creation time instead of click time.
                    ui.button(icon="key", on_click=lambda: _rotate_key(username))\
                        .tooltip("Rotate MCP key (old key dies)").props("flat dense")
                    ui.button(icon="lock_reset", on_click=lambda: _password_dialog(username))\
                        .tooltip("Set password").props("flat dense")
                    ui.button(icon="delete", color="negative", on_click=lambda: _delete_dialog(username))\
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
        # async handler (NOT a background task): ui.notify needs the event's
        # slot context — notify inside asyncio_create_task dies with
        # "slot stack for this task is empty" (UI test 2026-09-09).
        async def _run():
            response = await fn()
            if response.success:
                ui.notify(message, type="positive")
                _rebuild()
            else:
                _notify_error(response.error)
        return _run

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

            async def _do_delete():
                # async handler keeps the event's slot context — ui.notify
                # works here (a create_task task has none: RuntimeError).
                response = await get_api_client().delete_user(username)
                dialog.close()
                if response.success:
                    ui.notify(f"{username} deleted.", type="positive")
                    _rebuild()
                else:
                    _notify_error(response.error or "delete failed")

            with ui.row():
                ui.button("Delete", color="negative", on_click=_do_delete)
                ui.button("Cancel", on_click=dialog.close)

    def _access_dialog(user: dict):
        """Edit one user's reach (server checkboxes), per-server function
        masks (comma-separated deny-lists, E1 semantics scoped to the user),
        and the E3.5 data-plane grants (db presets + rag collections)."""
        username = user.get("username", "?")
        current_servers = user.get("servers") or []
        current_masks = user.get("masked_functions") or {}
        known_servers = sorted(set(PORTS) | set(current_servers))
        db_presets_current = list(user.get("db_presets") or [])
        rags_current = list(user.get("rag_collections") or [])

        with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
            ui.label(f"Access for {username}").classes("text-subtitle1")
            checks: dict = {}
            mask_inputs: dict = {}
            with ui.column().classes("w-full gap-2"):
                for srv in known_servers:
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        checks[srv] = ui.checkbox(srv, value=srv in current_servers)
                        mask_inputs[srv] = ui.input(
                            "masked (comma-sep)",
                            value=", ".join(current_masks.get(srv, [])),
                        ).classes("w-64").tooltip("Functions this user must NOT see on this server")

            ui.separator()
            ui.label("Data-plane grants (E3.5)").classes("text-subtitle2")

            preset_checks: dict = {}
            rags_checks: dict = {}
            presets_col = ui.column().classes("w-full")
            with presets_col:
                ui.label("databasemcp presets:").classes("text-caption text-grey")
            rags_col = ui.column().classes("w-full")
            with rags_col:
                ui.label("ragmcp collections:").classes("text-caption text-grey")

            async def _load_grant_options():
                import json as _json

                from fastmcp import Client
                from fastmcp.client.auth import BearerAuth

                async def _mcp_list(tool: str, mcp_tool: str, args: dict) -> str:
                    cfg = json.loads((Path(__file__).resolve().parents[2]
                                      / "tools" / tool / "config.json").read_text())
                    url = f"http://127.0.0.1:{PORTS[tool]}/mcp"
                    async with Client(url, auth=BearerAuth(cfg["auth"]["api_key"])) as c:
                        r = await c.call_tool(mcp_tool, args)
                        return r.content[0].text if getattr(r, "content", None) else ""

                try:
                    raw = await _mcp_list("databasemcp", "list_presets", {})
                    with presets_col:
                        for line in raw.splitlines():
                            if line.startswith("- "):
                                num = line[2:].split(" ")[0]
                                # seed current grants — an unseeded checkbox
                                # renders unchecked and Save would WIPE the
                                # user's presets (UI test 2026-09-09)
                                preset_checks[num] = ui.checkbox(
                                    num, value=num in db_presets_current)
                except Exception as e:
                    with presets_col:
                        ui.label(f"presets unavailable: {e}").classes("text-negative text-caption")

                try:
                    raw = await _mcp_list("ragmcp", "list_collections", {})
                    data = _json.loads(raw) if raw.strip().startswith(("[", "{")) else []
                    if isinstance(data, dict):
                        data = data.get("collections", [])
                    with rags_col:
                        for c in data:
                            name = c if isinstance(c, str) else c.get("name", "?")
                            rags_checks[name] = ui.checkbox(
                                name, value=name in rags_current)
                except Exception as e:
                    with rags_col:
                        ui.label(f"collections unavailable: {e}").classes("text-negative text-caption")

            asyncio.create_task(_load_grant_options())

            async def _save():
                servers = [s for s, cb in checks.items() if cb.value]
                masked = {
                    s: [f.strip() for f in (mi.value or "").split(",") if f.strip()]
                    for s, mi in mask_inputs.items()
                    if (mi.value or "").strip()
                }
                ok1 = await get_api_client().set_user_servers(username, servers)
                ok2 = await get_api_client().set_user_masked_functions(username, masked)
                presets = [p for p, cb in preset_checks.items() if cb.value]
                collections = [c for c, cb in rags_checks.items() if cb.value]
                ok3 = await get_api_client().set_user_db_presets(username, presets)
                ok4 = await get_api_client().set_user_rag_collections(username, collections)
                dialog.close()
                if all(o.success for o in (ok1, ok2, ok3, ok4)):
                    ui.notify("Access updated.", type="positive")
                    _rebuild()
                else:
                    _notify_error("update failed — see notifications")

            ui.button("Save", on_click=_save).props("outline dense")
        dialog.open()

    def _add_user_dialog():
        # PORTS (ports.json mcp assignments) — NOT get_state().tools, which is
        # only populated by the Overview page's loader and is EMPTY when the
        # user lands directly on /users (zero checkboxes, UI test 2026-09-09).
        known_servers = sorted(PORTS)
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
                with ui.dialog() as key_dialog, ui.card():
                    ui.label(f"MCP key for {username.value}").classes("text-subtitle1")
                    ui.label(key or "?").classes("font-mono text-sm whitespace-pre-wrap")
                    ui.label("Copy it now — it will not be shown again.")\
                        .classes("text-caption text-negative")

                    def _close_and_refresh():
                        key_dialog.close()
                        _rebuild()

                    ui.button("Close", on_click=_close_and_refresh)
                # NOTE: _rebuild must run ONLY after the key dialog closes —
                # container.clear() deletes an open dialog before it renders
                # (once-only key was invisible, UI test 2026-09-09).
                key_dialog.open()

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
