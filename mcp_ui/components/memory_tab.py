"""E2 — Memory Explorer tab (memorymcp only).

Search / browse / detail / delete through memorymcp's MCP surface via
MemoryMcpClient — the UI never touches the backend directly.

NiceGUI-trap compliance (mcp_ui audit): no password_toggle_button, no
@ui.refreshable (explicit container.clear() rebuilds), no absolute-center
positioning. State is a plain dict rebuilt per render.
"""

from nicegui import ui
from typing import Any

import asyncio

from ..memory_client import get_memory_client, MemoryMcpError

PAGE_SIZE = 20


async def render_memory_tab(container) -> None:
    """Entry point: (re)build the whole tab into `container`."""
    client = get_memory_client()
    state: dict[str, Any] = {
        "mode": "browse",       # browse | search | detail
        "offset": 0,
        "tag": "",
        "detail_id": None,
        "last_query": "",
    }

    def _go(mode: str, **kw):
        state.update(kw)
        state["mode"] = mode
        _rebuild()

    def _rebuild():
        container.clear()
        with container:
            _build()

    def _build():
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.label("Memory Explorer").classes("text-h6")
            ui.space()
            ui.button("Browse", icon="list",
                      on_click=lambda: _go("browse")).props('flat dense'
                                                            + (' color=primary' if state["mode"] == "browse" else ''))
            ui.button("Search", icon="search",
                      on_click=lambda: _go("search")).props('flat dense'
                                                            + (' color=primary' if state["mode"] == "search" else ''))

        body = ui.column().classes("w-full")
        with body:
            if state["mode"] == "browse":
                _build_browse(body)
            elif state["mode"] == "detail":
                _build_detail(body)
            else:
                _build_search(body)

    def _build_browse(body):
        with ui.row().classes("w-full items-center gap-2 mb-2"):
            tag_input = ui.input("Tag filter", value=state["tag"]).classes("w-48")

            def _apply_tag():
                state["tag"] = (tag_input.value or "").strip()
                state["offset"] = 0
                _rebuild()

            ui.button("Filter", icon="filter_alt", on_click=_apply_tag).props("flat dense")
            ui.button("Refresh", icon="refresh", on_click=lambda: _rebuild()).props("flat dense")

        loading = ui.label("Loading memories…").classes("text-grey")
        rows_box = ui.column().classes("w-full")

        async def _load():
            try:
                data = await client.list_memories(
                    limit=PAGE_SIZE, offset=state["offset"],
                    tag=state["tag"] or None,
                )
            except MemoryMcpError as e:
                loading.set_text(str(e))
                loading.classes("text-negative")
                return
            loading.set_text("")
            rows_box.clear()
            memories = data.get("memories", [])
            if isinstance(data.get("error"), str):
                with rows_box:
                    ui.label(f"Error: {data['error']}").classes("text-negative")
                return
            if not memories:
                with rows_box:
                    ui.label("No memories found.").classes("text-grey")
                return
            with rows_box:
                for m in memories:
                    _memory_row(m, body)

            with rows_box:
                with ui.row().classes("items-center gap-2 mt-2"):
                    prev_disabled = state["offset"] <= 0
                    next_disabled = data.get("next_offset") is None
                    ui.button("Prev", icon="chevron_left", on_click=_prev_page)\
                        .props("flat dense" + (" disable" if prev_disabled else ""))
                    ui.button("Next", icon="chevron_right", on_click=_next_page)\
                        .props("flat dense" + (" disable" if next_disabled else ""))
                    ui.label(f"{data.get('offset', 0) + 1}–"
                             f"{data.get('offset', 0) + len(memories)} of {data.get('total', '?')}"
                             ).classes("text-caption text-grey")

        def _prev_page():
            state["offset"] = max(0, state["offset"] - PAGE_SIZE)
            _rebuild()

        def _next_page():
            state["offset"] = state["offset"] + PAGE_SIZE
            _rebuild()

        asyncio.create_task(_load())

    def _memory_row(m: dict, body):
        with ui.card().classes("w-full cursor-pointer").on("click",
                lambda _e, mid=m["id"]: _go("detail", detail_id=mid)):
            with ui.row().classes("w-full justify-between items-start gap-2 flex-wrap"):
                with ui.column().classes("gap-1"):
                    preview = (m.get("preview") or "")[:160]
                    ui.label(preview + ("…" if len(m.get("preview") or "") > 160 else ""))\
                        .classes("text-sm")
                    with ui.row().classes("items-center gap-1"):
                        ui.badge(m.get("memory_type") or "?", color="indigo-7")
                        for t in (m.get("tags") or [])[:4]:
                            ui.badge(str(t), color="grey-8")
                        if m.get("sensitivity") and m.get("sensitivity") != "low":
                            ui.badge(m["sensitivity"], color="red-7")
                with ui.column().classes("items-end gap-1"):
                    ui.label(f"{m.get('usage_count', 0)}×").classes("text-caption text-grey")
                    ui.label(str(m.get("created_at", ""))[:10]).classes("text-caption text-grey")

    def _build_detail(body):
        mid = state.get("detail_id") or ""

        def _back():
            _go("browse")

        with ui.row().classes("w-full items-center gap-2 mb-2"):
            ui.button("Back", icon="arrow_back", on_click=_back).props("flat dense")
            ui.label(f"Memory {mid[:8]}…").classes("text-subtitle1")

        status = ui.label("").classes("text-grey")

        detail_box = ui.column().classes("w-full")

        async def _load():
            try:
                full = await client.get(mid)
                audit = await client.audit(mid, limit=20)
            except MemoryMcpError as e:
                status.set_text(str(e))
                status.classes("text-negative")
                return
            status.set_text("")
            detail_box.clear()
            with detail_box:
                with ui.card().classes("w-full"):
                    ui.label("Content").classes("text-subtitle2")
                    ui.label(full if isinstance(full, str) else str(full))\
                        .classes("text-sm whitespace-pre-wrap")
                with ui.card().classes("w-full"):
                    ui.label("Audit trail").classes("text-subtitle2")
                    ui.label(audit if isinstance(audit, str) else str(audit))\
                        .classes("text-xs whitespace-pre-wrap text-grey")

                def _confirm_delete():
                    with ui.dialog() as dialog, ui.card():
                        ui.label(f"Delete memory {mid[:8]}…? This cannot be undone.")
                        with ui.row():
                            ui.button("Delete", color="negative",
                                      on_click=lambda: dialog.submit("yes"))
                            ui.button("Cancel", on_click=lambda: dialog.submit("no"))

                    async def _wait():
                        outcome = await dialog
                        if outcome == "yes":
                            try:
                                await client.delete(mid)
                                ui.notify("Memory deleted.", type="positive")
                                _back()
                            except MemoryMcpError as e:
                                ui.notify(f"Delete failed: {e}", type="negative")

                    asyncio.create_task(_wait())

                ui.button("Delete", icon="delete", color="negative",
                          on_click=_confirm_delete).props("outline dense")

        asyncio.create_task(_load())

    def _build_search(body):
        state.setdefault("last_query", "")
        query_input = ui.input("Semantic search (queryMemory)").classes("w-full")
        result_box = ui.column().classes("w-full")
        loading = ui.label("").classes("text-grey")

        async def _run_search():
            q = (query_input.value or "").strip()
            if not q:
                return
            state["last_query"] = q
            loading.set_text("Searching…")
            result_box.clear()
            try:
                text = await client.query(q, k=10)
            except MemoryMcpError as e:
                loading.set_text(str(e))
                loading.classes("text-negative")
                return
            loading.set_text("")
            with result_box:
                ui.label(text if isinstance(text, str) else str(text))\
                    .classes("text-sm whitespace-pre-wrap")

        with ui.row():
            ui.button("Search", icon="search", on_click=_run_search).props("outline dense")

        if state["last_query"]:
            query_input.value = state["last_query"]
        ui.label("Results appear as queryMemory formats them (scored list).")\
            .classes("text-caption text-grey")

    _build()
