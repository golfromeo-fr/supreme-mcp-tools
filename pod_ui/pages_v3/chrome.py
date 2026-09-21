"""v3 chrome: nav frame + the global console as an IN-FLOW page footer.

v2/v3.0 used a position:fixed overlay with manual padding juggling — the
panel covered page content and got cut at the viewport edge (user-reported).
An in-flow footer can neither cover nor be cut: it simply lives at the end
of each page's content, with a floating pill that scrolls to it (and the
page auto-scrolls there when a new op starts)."""

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
    """In-flow console at the end of the page + floating jump pill."""
    runner = ops.runner

    console_holder = ui.column().classes("w-full gap-2")
    console_holder.set_visibility(False)  # hidden until first use / pill click

    body_state = {"expanded": False}

    with console_holder:
        with ui.card().classes("w-full gap-2").style(
                "border: 1px solid #444; background: rgba(20,20,20,.6);"):
            with ui.row().classes("w-full items-center gap-2"):
                console_title = ui.label("console").classes("text-h6")
                status = ui.label("idle").classes("text-caption")
                ui.space()
                ui.button("Kill", icon="cancel", on_click=lambda: runner.kill()) \
                    .props("flat dense color=negative")
                collapse_btn = ui.button(icon="expand_less").props("flat dense") \
                    .tooltip("collapse")
            body_col = ui.column().classes("w-full gap-2")
            with body_col:
                out_log = ui.log(max_lines=4000).classes("w-full h-52 font-mono text-xs")
                result = ui.label("").classes("text-caption")
                with ui.expansion("history", caption="recent ops").classes("w-full"):
                    history_slot = ui.column().classes("w-full")

        def set_expanded(expanded: bool) -> None:
            body_state["expanded"] = expanded
            body_col.set_visibility(expanded)
            collapse_btn.props('icon="expand_less"' if expanded else 'icon="expand_more"')

        def flip() -> None:
            set_expanded(not body_state["expanded"])

        collapse_btn.on_click(flip)
        set_expanded(False)  # start collapsed to one title strip

    # floating pill: scrolls the page to the in-flow console
    pill = ui.button(icon="terminal").classes("fixed bottom-2 right-2 z-50") \
        .props("round dense color=primary").tooltip("jump to console")

    def scroll_to_console() -> None:
        console_holder.set_visibility(True)
        ui.run_javascript(
            "document.querySelector('.podui-console')?.scrollIntoView({behavior:'smooth', block:'start'})")

    pill.on_click(scroll_to_console)
    console_holder.classes("podui-console")

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
            set_expanded(True)   # a new op started anywhere → expand + jump
            scroll_to_console()
        last_id["v"] = op_id
        if op is None:
            status.set_text("idle")
            return
        console_title.set_text("console — " + op.label)
        status.set_text("running" if not op.done else "finished")
        seen = getattr(poll, "_seen", 0)
        if seen > len(op.lines):
            out_log.clear()
            seen = 0
        for line in op.lines[seen:]:
            out_log.push(line)
        poll._seen = len(op.lines)  # type: ignore[attr-defined]
        result.set_text(op.result_line())
        if op.done:
            render_history()

    ui.timer(1.0, poll)
