"""Backups & data (`/backups`): state-plane dumps (restore with typed wipe)
and the pod-owned data stores (export/import-data). File-parameterized verbs
live here because the file is picked from a list (never free text, E17)."""

from __future__ import annotations

from nicegui import ui

from .. import auth, ops, state
from ..confirm import confirm_or_run


def page() -> None:
    if (body := auth.frame("Backups & data")) is None:
        return

    async def restore(info) -> None:
        env = info.env or "work"
        ui.notify(f"restoring {env} from {info.path.name} — watch Operations", type="info")
        op = await ops.runner.start(env, "restore", str(info.path))
        if op is None:
            ui.notify("an operation is already running", type="warning")
            return
        refresh()

    async def backup(env_key: str) -> None:
        ui.notify(f"{env_key} backup started — watch Operations", type="info")
        op = await ops.runner.start(env_key, "backup")
        if op is None:
            ui.notify("an operation is already running", type="warning")
            return
        refresh()

    async def export_data(env_key: str) -> None:
        ui.notify(f"{env_key} export started — watch Operations", type="info")
        op = await ops.runner.start(env_key, "export-data")
        if op is None:
            ui.notify("an operation is already running", type="warning")
            return
        refresh()

    async def import_data(env_key: str, path) -> None:
        ui.notify(f"{env_key} import started — watch Operations", type="info")
        op = await ops.runner.start(env_key, "import-data", str(path))
        if op is None:
            ui.notify("an operation is already running", type="warning")
            return
        refresh()

    with body:
        holder = ui.column().classes("w-full gap-3")

        def build() -> None:
            # ---- state-plane dumps ----
            ui.label("state-plane dumps (pg_dump)").classes("text-h6")
            with ui.row().classes("w-full items-center gap-2"):
                ui.button("work backup", icon="backup",
                          on_click=lambda: confirm_or_run(lambda: backup("work"))).props("dense")
                ui.button("test backup", icon="backup",
                          on_click=lambda: confirm_or_run(lambda: backup("test"))).props("dense")
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
                              on_click=lambda i=b: confirm_or_run(
                                  lambda: restore(i),
                                  f"restore {i.env or 'work'} from {i.path.name} "
                                  "(wipes its state volume)",
                                  word="wipe")).props("dense flat color=negative")

            ui.separator()
            # ---- pod-owned data stores ----
            ui.label("pod-owned data stores (export / import)").classes("text-h6")

            work_tars = ([p for p in state.BACKUPS_DIR.glob("work-artifacts-*.tar.gz")]
                         if state.BACKUPS_DIR.is_dir() else [])
            with ui.card().classes("w-full gap-1"):
                ui.label("work — artifacts volume (large-memory blobs)").classes("text-caption")
                with ui.row().classes("items-center gap-2"):
                    ui.button("Export", icon="download",
                              on_click=lambda: confirm_or_run(lambda: export_data("work"))) \
                        .props("dense")
                    if work_tars:
                        tsel = ui.select({str(p): p.name for p in sorted(work_tars)},
                                         value=str(sorted(work_tars)[-1]),
                                         label="import from").classes("w-96").props("dense outlined")
                        ui.button("Import", icon="upload",
                                  on_click=lambda: confirm_or_run(
                                      lambda: import_data("work", tsel.value))) \
                            .props("dense")
                    else:
                        ui.label("no work-artifacts-*.tar.gz yet — export first") \
                            .classes("text-caption opacity-60")
                ui.label("the memory data planes are HOST resources (turso dir / LAN PG) "
                         "— never exported.").classes("text-caption opacity-60")

            test_dirs = ([p for p in state.BACKUPS_DIR.glob("test-minio-*") if p.is_dir()]
                         if state.BACKUPS_DIR.is_dir() else [])
            with ui.card().classes("w-full gap-1"):
                ui.label("test — MinIO shared artifacts").classes("text-caption")
                with ui.row().classes("items-center gap-2"):
                    ui.button("Export", icon="download",
                              on_click=lambda: confirm_or_run(lambda: export_data("test"))) \
                        .props("dense")
                    if test_dirs:
                        msel = ui.select({str(p): p.name for p in sorted(test_dirs)},
                                         value=str(sorted(test_dirs)[-1]),
                                         label="import from").classes("w-96").props("dense outlined")
                        ui.button("Import", icon="upload",
                                  on_click=lambda: confirm_or_run(
                                      lambda: import_data("test", msel.value))) \
                            .props("dense")
                    else:
                        ui.label("no test-minio-* export dirs yet — export first") \
                            .classes("text-caption opacity-60")

        def refresh() -> None:
            holder.clear()
            with holder:
                build()

        refresh()
