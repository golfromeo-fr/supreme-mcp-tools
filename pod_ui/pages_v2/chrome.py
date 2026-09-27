"""v2 chrome: nav frame + the GLOBAL CONSOLE — a fixed bottom panel mounted
on every v2 page. Any op started from any v2 page streams here; the panel
auto-opens when a new op starts. One NiceGUI timer per mounted console
polls the shared runner (same process for v1 and v2)."""

from __future__ import annotations

import asyncio

from nicegui import ui

from .. import auth, ops

NAV = [
    ("/v2", "Home", "dashboard"),
    ("/v2/env/work", "work", "work"),
    ("/v2/env/test", "test", "science"),
    ("/v2/bundles", "Bundles", "inventory_2"),
]


def fire(env_key: str | None, action: str, *args) -> None:
    """Non-blocking op launch for v2 buttons: refuses while one runs, else
    starts it in the background — the console shows the stream."""
    if ops.runner.running():
        ui.notify("an operation is already running — watch the console", type="warning")
        return
    asyncio.ensure_future(ops.runner.start(env_key, action, *args))


def frame_v2(title: str):
    """Same auth gate as v1; v2 nav + a link back to the current layout.
    Returns the padded content container, or None when redirected."""
    if not auth.require():
        return None
    ui.dark_mode(True)
    with ui.header().classes("w-full p-3 items-center"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label("Pod Manager — v2").classes("text-h6")
            ui.label(title).classes("text-caption opacity-60")
            ui.button("back to current layout", icon="swap_horiz",
                      on_click=lambda: ui.navigate.to("/")).props("flat dense")
    # value=True: without it NiceGUI probes drawer state via JS on every
    # client connect (1s timeout) — a reconnecting tab after a server
    # restart spams a TimeoutError traceback into the console
    with ui.drawer(side="left", value=True).props("bordered behavior=desktop"):
        with ui.column().classes("w-full p-3 gap-1"):
            for path, label, icon in NAV:
                ui.button(label, icon=icon, on_click=lambda p=path: ui.navigate.to(p)) \
                    .classes("w-full").props("flat align=left")
            ui.separator()
            ui.button("Logout", icon="logout", on_click=auth.logout).classes("w-full").props("flat")
    body = ui.column().classes("w-full p-4 gap-3")
    mount_console()
    return body


def mount_console() -> None:
    """Fixed bottom console: toggle button + panel (status, live output,
    kill, history). Auto-opens on a NEW op (identity = started timestamp)."""
    runner = ops.runner

    with ui.element("div").classes("fixed bottom-2 right-2 z-50"):
        toggle = ui.button(icon="terminal").props("round dense color=primary") \
            .tooltip("console — operation output")

    with ui.element("div").classes(
            "fixed bottom-0 left-0 right-0 z-40 q-pa-md hidden"
    ).style("background: rgba(20,20,20,.97); border-top: 1px solid #444;") as panel:
        with ui.row().classes("w-full items-center gap-2"):
            status = ui.label("idle").classes("text-caption")
            ui.space()
            ui.button("Kill", icon="cancel", on_click=lambda: runner.kill()) \
                .props("flat dense color=negative")
            close_btn = ui.button(icon="close").props("flat dense")
        out_log = ui.log(max_lines=4000).classes("w-full h-52 font-mono text-xs")
        with ui.expansion("history", caption="recent ops").classes("w-full"):
            history_slot = ui.column().classes("w-full")

    def render_history() -> None:
        history_slot.clear()
        with history_slot:
            for h in reversed(runner.history[-8:]):
                mark = "✓" if h.get("rc") == 0 else "✗"
                ui.label(f"{h.get('started', '')}  {mark} rc={h.get('rc')}  "
                         f"{h.get('label')}").classes("text-caption font-mono")

    def set_open(open_: bool) -> None:
        panel.classes(remove="hidden" if open_ else None,
                      add=None if open_ else "hidden")

    def flip() -> None:
        set_open("hidden" in (panel.classes or ""))

    toggle.on_click(flip)
    close_btn.on_click(lambda: set_open(False))

    last_id = {"v": None}

    def poll() -> None:
        op = runner.current
        op_id = (op.started, op.label) if op else None
        if op_id and op_id != last_id["v"]:
            set_open(True)          # a new op started anywhere → show the console
        last_id["v"] = op_id
        if op is None:
            status.set_text("idle")
            return
        status.set_text(("running: " if not op.done else "finished: ") + op.label)
        seen = getattr(poll, "_seen", 0)
        if seen > len(op.lines):
            out_log.clear()
            seen = 0
        for line in op.lines[seen:]:
            out_log.push(line)
        poll._seen = len(op.lines)  # type: ignore[attr-defined]
        if op.done:
            render_history()

    ui.timer(1.0, poll)
