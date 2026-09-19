"""v3 chrome: nav frame + the global console, now WITHOUT the v2 defect
(the fixed panel covered page content): the body's bottom padding grows
when the console opens, so nothing is ever cut off."""

from __future__ import annotations

from nicegui import ui

from .. import auth, ops

NAV = [
    ("/v3", "Clusters", "hub"),
    ("/v3/logs", "Logs", "article"),
    ("/v3/bundles", "Bundles & backups", "inventory_2"),
]


def frame_v3(title: str):
    if not auth.require():
        return None
    ui.dark_mode(True)
    with ui.header().classes("w-full p-3 items-center"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label("Pod Manager — v3").classes("text-h6")
            ui.label(title).classes("text-caption opacity-60")
            ui.button("back to current layout", icon="swap_horiz",
                      on_click=lambda: ui.navigate.to("/")).props("flat dense")
    with ui.drawer(side="left").props("bordered behavior=desktop"):
        with ui.column().classes("w-full p-3 gap-1"):
            for path, label, icon in NAV:
                ui.button(label, icon=icon, on_click=lambda p=path: ui.navigate.to(p)) \
                    .classes("w-full").props("flat align=left")
            ui.separator()
            ui.button("Logout", icon="logout", on_click=auth.logout).classes("w-full").props("flat")
    body = ui.column().classes("w-full p-4 gap-3")
    mount_console(body)
    return body


def mount_console(body) -> None:
    runner = ops.runner

    with ui.element("div").classes("fixed bottom-2 right-2 z-50"):
        ui.button(icon="terminal").props("round dense color=primary") \
            .on_click(lambda: set_open(not open_state["v"])) \
            .tooltip("console — operation output")

    with ui.element("div").classes(
            "fixed bottom-0 left-0 right-0 z-40 q-pa-md hidden"
    ).style("background: rgba(20,20,20,.97); border-top: 1px solid #444;") as panel:
        with ui.row().classes("w-full items-center gap-2"):
            status = ui.label("idle").classes("text-caption")
            ui.space()
            ui.button("Kill", icon="cancel", on_click=lambda: runner.kill()) \
                .props("flat dense color=negative")
            ui.button(icon="close").props("flat dense").on_click(lambda: set_open(False))
        out_log = ui.log(max_lines=4000).classes("w-full h-52 font-mono text-xs")
        with ui.expansion("history", caption="recent ops").classes("w-full"):
            history_slot = ui.column().classes("w-full")

    open_state = {"v": False}

    def set_open(open_: bool) -> None:
        open_state["v"] = open_
        panel.classes(remove="hidden" if open_ else None,
                      add=None if open_ else "hidden")
        # the v2 defect fix: reserve space so the panel never covers content
        body.style(f"padding-bottom: {'16rem' if open_ else '3.5rem'}")

    def render_history() -> None:
        history_slot.clear()
        with history_slot:
            for h in reversed(runner.history[-8:]):
                mark = "✓" if h.get("rc") == 0 else "✗"
                ui.label(f"{h.get('started', '')}  {mark} rc={h.get('rc')}  "
                         f"{h.get('label')}").classes("text-caption font-mono")

    last_id = {"v": None}

    def poll() -> None:
        op = runner.current
        op_id = (op.started, op.label) if op else None
        if op_id and op_id != last_id["v"]:
            set_open(True)
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

    body.style("padding-bottom: 3.5rem")
    ui.timer(1.0, poll)
