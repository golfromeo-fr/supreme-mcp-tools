"""Backups & data (`/v3/backups`): state-plane dumps (pg_dump + typed-wipe
restore) and the POD-OWNED data stores (export/import). File-parameterized
verbs pick from lists, never free text."""

from __future__ import annotations

from nicegui import ui

from .. import auth, ops, state
from ..ops import fire


def page() -> None:
    if (body := auth.frame("Backups & data")) is None:
        return

    def build() -> None:
        # ---- state-plane dumps ----
        ui.label("state-plane dumps (pg_dump)").classes("text-h6")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("BASIC backup", icon="backup",
                      on_click=lambda: fire("work", "backup")).props("dense")
            ui.button("TEST backup", icon="backup",
                      on_click=lambda: fire("test", "backup")).props("dense")
        dumps = [b for b in state.list_backups() if b.kind == "sql"]
        if not dumps:
            ui.label("no dumps yet in ~/supreme-mcp-tools-backups/").classes("opacity-60")
        for b in dumps[:10]:

            async def restore(i=b) -> None:
                ui.notify(f"restoring {i.env or 'work'} — watch the console", type="info")
                await ops.runner.start(i.env or "work", "restore", str(i.path))

            with ui.row().classes("w-full items-center gap-2"):
                ui.label(b.path.name).classes("text-caption font-mono")
                if b.size == 0:
                    ui.badge("empty (env down when taken?)", color="warning")
                ui.space()
                ui.label(f"{b.size:,} B · {b.env or '?'}").classes("text-caption opacity-60")
                ui.button("Restore", icon="restore",
                          on_click=lambda i=b: confirm_or_run(
                              lambda: restore(i),
                              f"restore {i.env or 'work'} from {i.path.name} "
                              "(wipes its state volume)", word="wipe")) \
                    .props("dense flat color=negative")

        ui.separator()
        # ---- pod-owned data stores ----
        ui.label("pod-owned data stores (export / import)").classes("text-h6")
        for env_key, pattern in (("work", "work-artifacts-*.tar.gz"),
                                 ("test", "test-minio-*")):
            with ui.card().classes("w-full gap-1"):
                ui.label(env_key).classes("text-caption w-12 inline-block")
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.button("Export", icon="download",
                              on_click=lambda k=env_key: fire(k, "export-data")).props("dense")
                    cands = ([p for p in sorted(state.BACKUPS_DIR.glob(pattern))
                              if p.is_dir() or p.suffix == ".gz"]
                             if state.BACKUPS_DIR.is_dir() else [])
                    if cands:
                        s = ui.select({str(p): p.name for p in cands}, value=str(cands[-1])) \
                            .classes("w-96").props("dense outlined label=import from")
                        ui.button("Import", icon="upload",
                                  on_click=lambda k=env_key, sel=s: fire(k, "import-data", sel.value)) \
                            .props("dense")
                    else:
                        ui.label(f"no {pattern} yet — export first").classes("text-caption opacity-60")
                if env_key == "work":
                    ui.label("the memory data planes are HOST resources (turso dir / "
                             "LAN PG) — never exported.").classes("text-caption opacity-60")

    def confirm_or_run(coro_factory, title: str, word: str) -> None:
        from ..confirm import confirm_or_run as _c
        _c(coro_factory, title, word=word)

    with body:
        build()
