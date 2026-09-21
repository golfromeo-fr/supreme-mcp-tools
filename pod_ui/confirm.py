"""Shared confirmation helpers for pod_ui actions.

Policy (three tiers):
  immediate            light, non-destructive (backup, scrape, export, start)
  confirm_action       disruptive but reversible (Stop all, Rebuild, Boot over
                       a running env) — plain Run/Cancel dialog
  confirm_or_run       destroys data or removes containers/volumes (clean,
                       restore, delete bundle) — TYPED confirmation required

Factories may be sync or async; both are supported."""

from __future__ import annotations

import asyncio

from nicegui import ui


async def _call(f):
    result = f()
    if asyncio.iscoroutine(result):
        await result


def confirm_or_run(coro_factory, title: str, word: str | None = None) -> None:
    """coro_factory: zero-arg callable returning a coroutine (or None).
    word=None → fire immediately (task + 'started' notify).
    word set  → dialog whose Run button enables only when the typed word matches."""
    if word is None:
        asyncio.ensure_future(_call(coro_factory))
        ui.notify("started — watch the console", type="positive")
        return
    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(f"{title} — type '{word}' to confirm")
        typed = ui.input(f"type '{word}'")
        with ui.row():
            run_btn = ui.button("Run", color="negative")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        async def do_run() -> None:
            dialog.close()
            await _call(coro_factory)

        run_btn.on_click(do_run)
        run_btn.set_enabled(False)
        typed.on_value_change(
            lambda e: run_btn.set_enabled((e.value or "").strip() == word))
    dialog.open()


def confirm_action(coro_factory, title: str, description: str = "",
                   button: str = "Run") -> None:
    """Disruptive-but-reversible action: plain Run/Cancel dialog (no typing)."""
    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(title).classes("text-h6")
        if description:
            ui.label(description).classes("text-caption opacity-70")
        with ui.row():
            run_btn = ui.button(button, color="primary")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        async def do_run() -> None:
            dialog.close()
            await _call(coro_factory)

        run_btn.on_click(do_run)
    dialog.open()
