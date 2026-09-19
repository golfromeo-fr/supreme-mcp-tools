"""Backups & data (`/backups`): state-plane dumps (restore with typed wipe)
and the pod-owned data stores (export/import-data). File-parameterized verbs
live here because the file is picked from a list (never free text, E17)."""

from __future__ import annotations

from nicegui import ui

from .. import auth, ops, state


def page() -> None:
    if (body := auth.frame("Backups & data")) is None:
        return
    holder = ui.column().classes("w-full gap-3")

    def confirm_run(coro_factory, title: str, word: str | None = None) -> None:
        if word is None:
            import asyncio
            asyncio.create_task(coro_factory())
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
            typed.on_value_change(lambda e: run_btn.set_enabled((e.value or "").strip() == word))
        dialog.open()

    async def restore(info) -> None:
        env = info.env or "work"
        await ops.runner.start(env, "restore", str(info.path))
        ui.notify(f"restoring {env} from {info.path.name} — watch Operations", type="info")
        refresh()

    async def backup(env_key: str) -> None:
        await ops.runner.start(env_key, "backup")
        ui.notify(f"{env_key} backup started — watch Operations", type="info")
        refresh()

    async def export_data(env_key: str) -> None:
        await ops.runner.start(env_key, "export-data")
        ui.notify(f"{env_key} export started — watch Operations", type="info")
        refresh()

    async def import_data(env_key: str, path) -> None:
        await ops.runner.start(env_key, "import-data", str(path))
        ui.notify(f"{env_key} import started — watch Operations", type="info")
        refresh()

    def build() -> None:
        # ---- state-plane dumps ----
        ui.label("state-plane dumps (pg_dump)").classes("text-h6")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("work backup", icon="backup",
                      on_click=lambda: confirm_run(lambda: backup("work"))).props("dense")
            ui.button("test backup", icon="backup",
                      on_click=lambda: confirm_run(lambda: backup("test"))).props("dense")
        backups = state.list_backups()
        sqls = [b for b in backups if b.kind == "sql"]
        if not sqls:
            ui.label("no dumps yet in ~/supreme-mcp-tools-backups/").classes("opacity-60")
        for b in sqls:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(b.path.name).classes("text-caption font-mono")
                if b.size == 0:
                    ui.badge("0 B — empty (env down when taken?)", color="warning")
                ui.space()
                ui.label(f"{b.size:,} B · env {b.env or '?'}").classes("text-caption opacity-60")
                ui.button("Restore", icon="restore",
                          on_click=lambda i=b: confirm_run(
                              lambda: restore(i),
                              f"restore {i.env or 'work'} from {i.path.name} "
                              "(wipes its state volume)",
                              word="wipe")).props("dense flat color=negative")

        ui.separator()
        # ---- pod-owned data stores ----
        ui.label("pod-owned data stores (export / import)").classes("text-h6")
        with ui.card().classes("w-full gap-1"):
            ui.label("work — artifacts volume (large-memory blobs)").classes("text-caption")
            with ui.row().classes("items-center gap-2"):
                ui.button("Export", icon="download",
                          on_click=lambda: confirm_run(lambda: export_data("work"))).props("dense")
            ui.label("the memory data planes are HOST resources (turso dir / LAN PG) "
                     "— never exported.").classes("text-caption opacity-60")
        with ui.card().classes("w-full gap-1"):
            ui.label("test — MinIO shared artifacts").classes("text-caption")
            with ui.row().classes("items-center gap-2"):
                ui.button("Export", icon="download",
                          on_click=lambda: confirm_run(lambda: export_data("test"))).props("dense")
            exports = sorted(state.BACKUPS_DIR.glob("test-minio-*")) if state.BACKUPS_DIR.is_dir() else []
            export_dirs = [p for p in exports if p.is_dir()]
            if export_dirs:
                sel = ui.select({str(p): p.name for p in export_dirs},
                                value=str(export_dirs[0]), label="import from",
                                ).classes("w-80").props("dense outlined")
                ui.button("Import", icon="upload",
                          on_click=lambda: confirm_run(lambda: import_data("test", sel.value))) \
                    .props("dense")
            else:
                ui.label("no test-minio-* export dirs yet — export first").classes("text-caption opacity-60")

    def refresh() -> None:
        holder.clear()
        with holder:
            build()

    refresh()
