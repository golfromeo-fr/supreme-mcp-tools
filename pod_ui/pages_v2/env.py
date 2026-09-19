"""v2 env page (`/v2/env/{work|test}`): everything about ONE environment —
actions grouped by intent (Lifecycle / Data / Danger), its own log viewer
(advanced controls behind a gear), its backups with restore, and the
pod-owned data export/import. Destructive actions keep typed confirmations."""

from __future__ import annotations

import os

from nicegui import ui

from .. import ops, state
from ..confirm import confirm_or_run
from .chrome import fire, frame_v2


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return max(lo, min(hi, int(raw))) if raw else default
    except ValueError:
        return default


def _logs_section(env_prefix: str) -> None:
    with ui.card().classes("w-full gap-2"):
        ui.label("logs").classes("text-h6")
        names = [c.name for c in state.containers_by_env().get(
            "work" if env_prefix == "mcp-work_" else "test", [])]
        if not names:
            ui.label("no containers yet").classes("opacity-60")
            return
        view = {"font": _env_int("MCP_UI_LOG_FONT", 12, 8, 24),
                "cols": _env_int("MCP_UI_LOG_COLS", 160, 40, 1000),
                "rows": _env_int("MCP_UI_LOG_ROWS", 24, 10, 400),
                "max": os.environ.get("MCP_UI_LOG_MAX", "").strip().lower()
                in ("1", "true", "yes", "on")}
        tail_default = _env_int("MCP_UI_LOG_TAIL", 500, 50, 20000)
        tail = {"value": tail_default}
        nums: dict = {}

        with ui.row().classes("w-full items-center gap-2"):
            sel = ui.select({n: n for n in names}, value=names[0],
                            on_change=lambda e: refresh()).classes("w-72") \
                .props("dense outlined")
            grep_input = ui.input(placeholder="filter (shown lines)")
            grep_input.classes("w-80").props("dense outlined clearable")
            auto = ui.switch("Auto", value=True)
            ui.space()
            gear = ui.button(icon="tune").props("flat dense").tooltip("view controls")
        advanced = ui.row().classes("w-full items-center gap-2")
        with advanced:
            tail_options = sorted({200, 500, 1000, 2000, 5000, 10000, tail_default})
            ui.select({v: str(v) for v in tail_options}, value=tail_default,
                      label="tail", on_change=lambda e: tail.update(value=e.value)) \
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
        advanced.set_visibility(False)  # advanced controls hidden until the gear

        log_view = ui.log(max_lines=20000).classes("w-full font-mono")
        meta = ui.label("").classes("text-caption opacity-60")

        def apply_view() -> None:
            if view["max"]:
                css = "width:100%; height:calc(100vh - 420px);"
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

        gear.on_click(lambda: advanced.toggle())

        async def refresh() -> None:
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


def _backups_section(env_key: str) -> None:
    with ui.card().classes("w-full gap-2"):
        ui.label("backups & pod-owned data").classes("text-h6")
        with ui.row().classes("items-center gap-2"):
            ui.button("Backup now", icon="backup",
                      on_click=lambda: fire(env_key, "backup")).props("dense")

        dumps = [b for b in state.list_backups()
                 if b.kind == "sql" and (b.env or "work") == env_key]
        if dumps:
            for b in dumps[:8]:
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(b.path.name).classes("text-caption font-mono")
                    if b.size == 0:
                        ui.badge("empty", color="warning")
                    ui.space()
                    ui.label(f"{b.size:,} B").classes("text-caption opacity-60")

                    async def restore(i=b) -> None:
                        ui.notify(f"restoring from {i.path.name} — watch the console",
                                  type="info")
                        await ops.runner.start(env_key, "restore", str(i.path))
                        ui.navigate.to(f"/v2/env/{env_key}")

                    ui.button("Restore", icon="restore",
                              on_click=lambda i=b: confirm_or_run(
                                  lambda: restore(i),
                                  f"restore {env_key} from {i.path.name} "
                                  "(wipes its state volume)", word="wipe")) \
                        .props("dense flat color=negative")
        else:
            ui.label("no dumps for this env yet").classes("opacity-60")

        ui.separator()
        with ui.row().classes("items-center gap-2"):
            ui.button("Export data", icon="download",
                      on_click=lambda: fire(env_key, "export-data")).props("dense") \
                .tooltip("pod-owned stores only — the host memory planes are never touched")

        if env_key == "work":
            tars = sorted(state.BACKUPS_DIR.glob("work-artifacts-*.tar.gz")) \
                if state.BACKUPS_DIR.is_dir() else []
            if tars:
                tsel = ui.select({str(p): p.name for p in tars}, value=str(tars[-1])) \
                    .classes("w-96").props("dense outlined label=import from")
                ui.button("Import data", icon="upload",
                          on_click=lambda: fire(env_key, "import-data", tsel.value)) \
                    .props("dense")
        else:
            dirs = [p for p in sorted(state.BACKUPS_DIR.glob("test-minio-*"))
                    if p.is_dir()] if state.BACKUPS_DIR.is_dir() else []
            if dirs:
                msel = ui.select({str(p): p.name for p in dirs}, value=str(dirs[-1])) \
                    .classes("w-96").props("dense outlined label=import from")
                ui.button("Import data", icon="upload",
                          on_click=lambda: fire(env_key, "import-data", msel.value)) \
                    .props("dense")


def page(env_key: str) -> None:
    env = state.ENV_BY_KEY.get(env_key)
    if env is None:
        ui.navigate.to("/v2")
        return
    if (body := frame_v2(env.title)) is None:
        return

    live = next((e for e in state.env_statuses() if e.key == env_key), env)
    rebuild_verb = "up" if env_key == "work" else "pg"

    with body:
        with ui.card().classes("w-full gap-2"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label(env.title).classes("text-h5")
                ui.badge(live.state, color="positive" if live.state == "running" else "grey")
                for label, value in live.health.items():
                    ui.badge(f"{label}: {value}",
                             color="positive" if value == "ok" else "negative")
                ui.space()
                ui.button("Start", icon="play_arrow",
                          on_click=lambda: fire(env_key, "start")).props("dense")
                ui.button("Stop", icon="stop",
                          on_click=lambda: fire(env_key, "stop")).props("dense")
                ui.button("Rebuild", icon="refresh",
                          on_click=lambda: fire(env_key, rebuild_verb)).props("dense") \
                    .tooltip("stop + rebuild the image + start")
        with ui.card().classes("w-full gap-2"):
            ui.label("data").classes("text-h6")
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.button("Scrape → bundle", icon="download",
                          on_click=lambda: fire(env_key, "scrape")).props("dense") \
                    .tooltip("promote this pod's config into its bundle")
                ui.button("Backup", icon="backup",
                          on_click=lambda: fire(env_key, "backup")).props("dense")
                ui.button("Export data", icon="file_download",
                          on_click=lambda: fire(env_key, "export-data")).props("dense")
        with ui.card().classes("w-full gap-2"):
            ui.label("danger zone").classes("text-h6 text-red")
            ui.label("clean removes containers (work: AND the state volume — "
                     "users/masks — after a typed confirmation)").classes("text-caption opacity-70")
            ui.button("Clean", icon="delete_forever",
                      on_click=lambda: confirm_launch_clean(env_key)) \
                .props("color=negative dense")

        _logs_section(env.prefix)
        _backups_section(env_key)


def confirm_launch_clean(env_key: str) -> None:
    if env_key == "test":
        # test clean does NOT prompt in startcluster (volumes kept) — still
        # confirm in-UI because it removes containers
        word = "clean"
    else:
        word = "wipe"  # the script's own confirmation word (stdin pipe)

    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(f"clean {env_key} — type '{word}' to confirm")
        typed = ui.input(f"type '{word}'")
        with ui.row():
            run_btn = ui.button("Run", color="negative")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        def do_run() -> None:
            dialog.close()
            fire(env_key, "clean")  # the runner pipes 'wipe' for the work env

        run_btn.on_click(do_run)
        run_btn.set_enabled(False)
        typed.on_value_change(
            lambda e: run_btn.set_enabled((e.value or "").strip() == word))
    dialog.open()
