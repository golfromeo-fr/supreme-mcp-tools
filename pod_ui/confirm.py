"""Shared typed-confirmation runner for one-click (no confirm) and
destructive (type-the-word) actions. Used by the bundles and backups pages."""

from __future__ import annotations

import asyncio

from nicegui import ui


def confirm_or_run(coro_factory, title: str, word: str | None = None) -> None:
    """coro_factory: zero-arg callable returning a coroutine.
    word=None → fire immediately (task + 'started' notify).
    word set  → dialog whose Run button enables only when the typed word matches."""
    if word is None:
        asyncio.ensure_future(coro_factory())
        ui.notify("started — watch Operations for output", type="positive")
        return
    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(f"{title} — type '{word}' to confirm")
        typed = ui.input(f"type '{word}'")
        with ui.row():
            run_btn = ui.button("Run", color="negative")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        async def do_run() -> None:
            dialog.close()
            await coro_factory()

        run_btn.on_click(do_run)
        run_btn.set_enabled(False)
        typed.on_value_change(
            lambda e: run_btn.set_enabled((e.value or "").strip() == word))
    dialog.open()
