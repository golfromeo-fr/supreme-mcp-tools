"""Container log viewer (`/logs`): same control set as the mcp_ui Logs tab
(tail/font/cols/rows/Max out from the shared MCP_UI_LOG_* env defaults), but
polls `podman logs --tail N <container>` for any container in any env."""

from __future__ import annotations

import os

from nicegui import ui

from .. import auth, state


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return max(lo, min(hi, int(raw))) if raw else default
    except ValueError:
        return default


async def page() -> None:
    if (body := auth.frame("Logs")) is None:
        return

    view = {"font": _env_int("MCP_UI_LOG_FONT", 12, 8, 24),
            "cols": _env_int("MCP_UI_LOG_COLS", 160, 40, 1000),
            "rows": _env_int("MCP_UI_LOG_ROWS", 30, 10, 400),
            "max": os.environ.get("MCP_UI_LOG_MAX", "").strip().lower() in ("1", "true", "yes", "on")}
    tail_default = _env_int("MCP_UI_LOG_TAIL", 500, 50, 20000)
    tail_size = {"value": tail_default}
    nums: dict = {}
    sel = None

    names = state.known_log_containers()
    if not names:
        ui.label("no containers found (nothing created yet)").classes("opacity-60")
        return

    with body:
        with ui.row().classes("w-full items-center gap-2"):
            sel = ui.select({n: n for n in names}, value=names[0], label="container",
                            on_change=lambda e: refresh(),
                            ).classes("w-72").props("dense outlined")
            tail_options = sorted({200, 500, 1000, 2000, 5000, 10000, tail_default})
            ui.select({v: str(v) for v in tail_options}, value=tail_default, label="tail",
                      on_change=lambda e: tail_size.update(value=e.value),
                      ).classes("w-28").props("dense outlined")
            auto = ui.switch("Auto", value=True)
            ui.space()
            ui.switch("Max out", value=view["max"], on_change=lambda e: set_max(e.value))
            for label, key in (("font", "font"), ("cols", "cols"), ("rows", "rows")):
                ui.label(label).classes("text-caption opacity-60")
                ui.button(icon="remove", on_click=lambda k=key: bump(k, -1)) \
                    .props("flat dense size=sm")
                num = ui.number(value=view[key], format="%.0f",
                                on_change=lambda e, k=key: set_value(k, e.value))
                num.classes("w-24").props("dense outlined hide-bottom-space")
                nums[key] = num
                ui.button(icon="add", on_click=lambda k=key: bump(k, +1)) \
                    .props("flat dense size=sm")
        with ui.row().classes("w-full items-center gap-2"):
            grep_input = ui.input(placeholder="filter (shown lines, substring)")
            grep_input.classes("w-96").props("dense outlined clearable")
            ui.button("Refresh", icon="refresh", on_click=lambda: refresh())
        meta = ui.label("").classes("text-caption opacity-60")
        log_view = ui.log(max_lines=20000).classes("w-full font-mono")

    def apply_view() -> None:
        if view["max"]:
            css = "width:100%; height:calc(100vh - 250px);"
        else:
            height = int(view["rows"] * view["font"] * 1.35)
            css = f"width:{view['cols']}ch; height:{height}px;"
        css += f"font-size:{view['font']}px; line-height:1.35;"
        log_view.style(replace=css)

    def set_value(key: str, value) -> None:
        if value is None:
            return
        view[key] = int(value)
        apply_view()

    def bump(key: str, delta: int) -> None:
        bounds = {"font": (8, 24), "cols": (40, 1000), "rows": (10, 400)}[key]
        view[key] = max(bounds[0], min(bounds[1], view[key] + delta))
        nums[key].value = view[key]

    def set_max(enabled: bool) -> None:
        view["max"] = bool(enabled)
        apply_view()

    async def refresh() -> None:
        container = sel.value
        text = state.container_log(container, tail_size["value"])
        needle = (grep_input.value or "").strip().lower()
        log_view.clear()
        shown = 0
        for line in text.splitlines():
            if needle and needle not in line.lower():
                continue
            log_view.push(line)
            shown += 1
        meta.set_text(f"{container} · tail {tail_size['value']} · {shown} shown"
                      + (f" of {len(text.splitlines())}" if needle else ""))

    async def auto_tick() -> None:
        if auto.value:
            await refresh()

    apply_view()
    await refresh()
    ui.timer(5.0, auto_tick)
