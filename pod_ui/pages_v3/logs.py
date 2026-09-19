"""v3 logs (`/v3/logs`): cluster-GROUPED container picker, defaults to a
RUNNING container (the v1 failure mode: it defaulted to the alphabetically
first container, usually an exited one → an empty pane that looked dead),
honors ?container= from the node drawer, and says so when the selected
container is not running."""

from __future__ import annotations

import os

from nicegui import ui

from .. import state
from .chrome import frame_v3


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return max(lo, min(hi, int(raw))) if raw else default
    except ValueError:
        return default


def page() -> None:
    if (body := frame_v3("Logs")) is None:
        return

    grouped_rows = state.containers_by_env()
    # FLAT options (the nested grouped form raises 'Invalid value' on
    # NiceGUI 3.16): the label carries the cluster prefix instead
    options: dict = {}
    for env_key in ("work", "test", "other"):
        for row in grouped_rows.get(env_key, []):
            group = {"work": "BASIC", "test": "TEST"}.get(env_key, "other")
            options[row.name] = f"{group} · {row.name}"
    if not options:
        with body:
            ui.label("no containers found (nothing created yet)").classes("opacity-60")
        return

    flat_rows = {r.name: r for rows in grouped_rows.values() for r in rows}
    requested = ui.query("container") or None
    default = None
    if requested and requested in flat_rows:
        default = requested
    else:
        running = [r.name for r in flat_rows.values() if r.running]
        default = running[0] if running else next(iter(flat_rows))

    view = {"font": _env_int("MCP_UI_LOG_FONT", 12, 8, 24),
            "cols": _env_int("MCP_UI_LOG_COLS", 160, 40, 1000),
            "rows": _env_int("MCP_UI_LOG_ROWS", 26, 10, 400),
            "max": os.environ.get("MCP_UI_LOG_MAX", "").strip().lower()
            in ("1", "true", "yes", "on")}
    tail_default = _env_int("MCP_UI_LOG_TAIL", 500, 50, 20000)
    tail = {"value": tail_default}
    nums: dict = {}

    with body:
        with ui.row().classes("w-full items-center gap-2"):
            sel = ui.select(options, value=default, label="container",
                            on_change=lambda e: refresh()).classes("w-80") \
                .props("dense outlined")
            grep_input = ui.input(placeholder="filter (shown lines)")
            grep_input.classes("w-80").props("dense outlined clearable")
            auto = ui.switch("Auto", value=True)
            ui.space()
            gear = ui.button(icon="tune").props("flat dense").tooltip("view controls")
        advanced = ui.row().classes("w-full items-center gap-2")
        with advanced:
            tail_options = sorted({200, 500, 1000, 2000, 5000, 10000, tail_default})
            ui.select({v: str(v) for v in tail_options}, value=tail_default, label="tail",
                      on_change=lambda e: tail.update(value=e.value)) \
                .classes("w-28").props("dense outlined")
            ui.switch("Max out", value=view["max"], on_change=lambda e: set_max(e.value))
            for key in ("font", "cols", "rows"):
                ui.label(key).classes("text-caption opacity-60")
                ui.button(icon="remove", on_click=lambda k=key: bump(k, -1)) \
                    .props("flat dense size=sm")
                num = ui.number(value=view[key], format="%.0f",
                                on_change=lambda e, k=key: set_value(k, e.value))
                num.classes("w-24").props("dense outlined hide-bottom-space")
                nums[key] = num
                ui.button(icon="add", on_click=lambda k=key: bump(k, +1)) \
                    .props("flat dense size=sm")
        advanced.set_visibility(False)
        gear.on_click(lambda: advanced.toggle())

        state_label = ui.label("").classes("text-caption")
        meta = ui.label("").classes("text-caption opacity-60")
        log_view = ui.log(max_lines=20000).classes("w-full font-mono")

    def apply_view() -> None:
        if view["max"]:
            css = "width:100%; height:calc(100vh - 300px);"
        else:
            css = f"width:{view['cols']}ch; height:{int(view['rows']*view['font']*1.35)}px;"
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
        row = flat_rows.get(sel.value)
        if row is None:
            state_label.set_text("")
        elif row.running:
            state_label.set_text("").classes("text-caption")
        else:
            state_label.set_text(f"⚠ {sel.value} is not running — output may be stale/empty") \
                .classes("text-caption text-orange-8")
        text = state.container_log(sel.value, tail["value"])
        needle = (grep_input.value or "").strip().lower()
        log_view.clear()
        shown = 0
        for line in text.splitlines():
            if needle and needle not in line.lower():
                continue
            log_view.push(line)
            shown += 1
        meta.set_text(f"tail {tail['value']} · {shown} shown")

    async def tick() -> None:
        if auto.value:
            await refresh()

    apply_view()
    import asyncio
    asyncio.ensure_future(refresh())
    ui.timer(5.0, tick)
