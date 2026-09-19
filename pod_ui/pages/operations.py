"""Operations (`/operations`): lifecycle buttons per environment with streamed
output, typed wipe confirmations, kill, and op history. File-parameterized
verbs (restore / import-data) live on the Backups page where the file is
picked. The guided migration walkthrough is at the bottom: one confirmed
step at a time, never an automatic sequence."""

from __future__ import annotations

from nicegui import ui

from .. import auth, ops

WIPE_WORD = "wipe"

ACTIONS = {
    "work": [
        ("up (rebuild)", "up", False),
        ("start", "start", False),
        ("stop", "stop", False),
        ("clean", "clean", True),
        ("backup", "backup", False),
        ("scrape → bundle", "scrape", False),
        ("export-data", "export-data", False),
    ],
    "test": [
        ("up pg (rebuild)", "pg", False),
        ("start", "start", False),
        ("stop", "stop", False),
        ("clean", "clean", False),
        ("backup", "backup", False),
        ("scrape → bundle", "scrape", False),
        ("export-data", "export-data", False),
    ],
}

MIGRATION_STEPS = [
    ("scrape the OLD pod into the bundle", "scrape"),
    ("backup the OLD pod state plane", "backup"),
    ("export the OLD pod-owned data", "export-data"),
]


def page() -> None:
    if (body := auth.frame("Operations")) is None:
        return
    runner = ops.runner

    async def launch(env_key: str, action: str, *extra) -> None:
        op = await runner.start(env_key, action, *extra)
        if op is None:
            ui.notify("an operation is already running", type="warning")
            return
        poll._seen = 0  # type: ignore[attr-defined]  # new op: reset the pane
        poll._hist_len = -1  # type: ignore[attr-defined]  # force history redraw

    def confirm_launch(env_key: str, action: str) -> None:
        with ui.dialog() as dialog, ui.card().classes("gap-2"):
            ui.label(f"this runs '{env_key} {action}' and pipes '{WIPE_WORD}' to its confirmation")
            typed = ui.input(f"type '{WIPE_WORD}' to enable")
            with ui.row():
                run_btn = ui.button("Run", color="negative")
                ui.button("Cancel", on_click=dialog.close).props("flat")

            async def do_run() -> None:
                dialog.close()
                await launch(env_key, action)

            run_btn.on_click(do_run)
            run_btn.set_enabled(False)
            typed.on_value_change(
                lambda e: run_btn.set_enabled((e.value or "").strip() == WIPE_WORD))
        dialog.open()

    with body:
        with ui.row().classes("w-full flex-wrap gap-3"):
            for env_key, actions in ACTIONS.items():
                with ui.card().classes("gap-1 min-w-xs"):
                    ui.label(env_key).classes("text-h6")
                    for label, action, wipe in actions:
                        if wipe:
                            ui.button(label, on_click=lambda e=env_key, a=action:
                                      confirm_launch(e, a)) \
                                .classes("w-full").props("color=negative")
                        else:
                            ui.button(label, on_click=lambda e=env_key, a=action:
                                      launch(e, a)) \
                                .classes("w-full")

        ui.separator()
        status = ui.label("idle").classes("text-caption")
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("output").classes("text-caption opacity-60")
            ui.space()
            ui.button("Kill", icon="cancel", on_click=lambda: runner.kill()) \
                .props("flat dense color=negative")
        out_log = ui.log(max_lines=5000).classes("w-full h-80 font-mono text-xs")
        result = ui.label("").classes("text-caption")
        history_box = ui.column().classes("w-full")

        def render_history() -> None:
            history_box.clear()
            with history_box:
                with ui.expansion("history", caption=f"{len(runner.history)} ops").classes("w-full"):
                    for h in reversed(runner.history[-10:]):
                        mark = "✓" if h.get("rc") == 0 else "✗"
                        ui.label(f"{h.get('started', '')}  {mark} rc={h.get('rc')}  "
                                 f"{h.get('label')}").classes("text-caption font-mono")

        def poll() -> None:
            op = runner.current
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
            result.set_text(op.result_line())
            # redraw history only when it actually changed — a redraw every
            # tick would collapse the user's open expansion
            hist_len = len(runner.history)
            if hist_len != getattr(poll, "_hist_len", -1) or op.done:
                render_history()
                poll._hist_len = hist_len  # type: ignore[attr-defined]

        ui.separator()
        with ui.card().classes("w-full gap-2"):
            ui.label("guided migration (old pod → bundle → new pod)").classes("text-h6")
            ui.label("one step at a time — each runs the matching startcluster verb; "
                     "after the exports: stop/clean the old env here, boot the new one "
                     "from the bundle (Bundles page), then restore + import-data "
                     "(Backups page).").classes("text-caption opacity-70")
            ui.label("1. pick the source environment, then run the steps top to bottom:")
            src = ui.select({e: e for e in ("work", "test")}, value="work",
                            label="source env").classes("w-40").props("dense outlined")
            for i, (label, verb) in enumerate(MIGRATION_STEPS, 1):
                ui.button(f"{i}. {label}",
                          on_click=lambda v=verb: launch(src.value, v)) \
                    .classes("w-full").props("flat align=left")
            ui.label("2. stop/clean the old env (buttons above) → 3. boot the new pod from "
                     "the refreshed bundle (Bundles page) → 4. restore + import-data "
                     "(Backups page).").classes("text-caption opacity-70")

        poll()
        ui.timer(1.0, poll)
