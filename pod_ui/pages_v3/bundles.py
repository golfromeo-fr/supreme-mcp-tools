"""v3 bundles & backups (`/v3/bundles`): one page for the artifact flows —
harvest on top, bundle cards with a single Boot… menu, then the dumps table
(restore with typed wipe) and pod-owned data export/import. The guided
migration walkthrough stays collapsed at the bottom."""

from __future__ import annotations

import json
import re

from nicegui import ui

from .. import ops, state
from ..confirm import confirm_or_run
from ..ops import fire
from .chrome import frame_v3

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

MIGRATION_STEPS = [
    ("scrape the OLD cluster into the bundle", "scrape"),
    ("backup the OLD cluster state plane", "backup"),
    ("export the OLD cluster-owned data", "export-data"),
]


async def _boot(env_key: str, info) -> None:
    await ops.runner.start(env_key, "up" if env_key == "work" else "pg", str(info.path))
    ui.navigate.to("/v3")


async def _scrape_into(env_key: str, info) -> None:
    await ops.runner.start(env_key, "scrape", str(info.path))


def page() -> None:
    if (body := frame_v3("Bundles & backups")) is None:
        return

    def show_manifest(info) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-full"):
            ui.label(f"manifest — {info.name}").classes("text-h6")
            text = json.dumps(info.manifest, indent=2, ensure_ascii=False) if info.manifest else "{}"
            ui.code(text).classes("w-full max-h-[36rem]")
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    def _delete_bundle(i) -> None:
        import shutil

        async def do_delete() -> None:
            shutil.rmtree(i.path, ignore_errors=True)
            ui.notify(f"bundle {i.name} deleted", type="positive")
            refresh()
        confirm_or_run(do_delete, f"delete bundle {i.name} (rm -rf)", word=i.name)

    def bundle_card(info) -> None:
        m = info.manifest or {}
        with ui.card().classes("w-full gap-1"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label(info.name).classes("text-h6")
                git = (m.get("git") or {}).get("describe", "")
                ui.badge(f"git: {git}" if git else "no manifest",
                         color="grey" if git else "negative")
                if m.get("redacted"):
                    ui.badge("REDACTED", color="warning")
                if m.get("scraped"):
                    ui.badge("scraped from pod", color="info")
                ui.label(f"created {m.get('created_utc', '?')} · "
                         f"{len(m.get('files', []))} files").classes("text-caption opacity-60")
                ui.space()
                with ui.button("Boot…", icon="rocket_launch").props("dense color=primary"):
                    with ui.menu().props("no-wrap"):
                        ui.menu_item("Boot BASIC from this",
                                     on_click=lambda i=info: confirm_or_run(
                                         lambda: _boot("work", i),
                                         "boot the BASIC cluster from this bundle (stops the "
                                         "running one; canonical ports must be free)"))
                        ui.menu_item("Boot TEST (pg) from this",
                                     on_click=lambda i=info: confirm_or_run(
                                         lambda: _boot("test", i),
                                         "boot the TEST cluster from this bundle"))
                        ui.separator()
                        ui.menu_item("Scrape BASIC into",
                                     on_click=lambda i=info: confirm_or_run(
                                         lambda: _scrape_into("work", i)))
                        ui.menu_item("Scrape TEST into",
                                     on_click=lambda i=info: confirm_or_run(
                                         lambda: _scrape_into("test", i)))
                with ui.button(icon="more_vert").props("flat dense").tooltip("more"):
                    with ui.menu().props("no-wrap"):
                        ui.menu_item("Manifest", on_click=lambda i=info: show_manifest(i))
                        ui.separator()
                        ui.menu_item("Delete", on_click=lambda i=info: _delete_bundle(i))

    def backups_section() -> None:
        ui.separator()
        ui.label("state-plane dumps").classes("text-h6")
        with ui.row().classes("items-center gap-2"):
            ui.button("BASIC backup", icon="backup",
                      on_click=lambda: fire("work", "backup")).props("dense")
            ui.button("TEST backup", icon="backup",
                      on_click=lambda: fire("test", "backup")).props("dense")
        dumps = [b for b in state.list_backups() if b.kind == "sql"]
        if not dumps:
            ui.label("no dumps yet in ~/supreme-mcp-tools-backups/").classes("opacity-60")
        for b in dumps[:10]:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(b.path.name).classes("text-caption font-mono")
                if b.size == 0:
                    ui.badge("empty", color="warning")
                ui.space()
                ui.label(f"{b.size:,} B · {b.env or '?'}").classes("text-caption opacity-60")

                async def restore(i=b) -> None:
                    ui.notify(f"restoring {i.env or 'work'} — watch the console", type="info")
                    await ops.runner.start(i.env or "work", "restore", str(i.path))

                ui.button("Restore", icon="restore",
                          on_click=lambda i=b: confirm_or_run(
                              lambda: restore(i),
                              f"restore {i.env or 'work'} from {i.path.name} "
                              "(wipes its state volume)", word="wipe")) \
                    .props("dense flat color=negative")

        ui.label("pod-owned data").classes("text-h6")
        for env_key, pattern, pick_label in (("work", "work-artifacts-*.tar.gz", "tar"),
                                             ("test", "test-minio-*", "dir")):
            with ui.row().classes("items-center gap-2"):
                ui.label(env_key).classes("text-caption w-12")
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
                    ui.label(f"no {pattern} yet").classes("text-caption opacity-60")

    with body:
        with ui.card().classes("w-full gap-2"):
            ui.label("harvest the working tree into a new bundle").classes("text-caption opacity-70")
            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                name = ui.input("bundle name (letters/digits/._-)") \
                    .classes("w-72").props("dense outlined")
                zip_c = ui.checkbox("zip")
                redact_c = ui.checkbox("redact")
                db_c = ui.checkbox("from work DB")
                ui.button("Harvest", icon="save", on_click=lambda: do_harvest()).props("dense")
            harvest_msg = ui.label("").classes("text-caption")

        async def do_harvest() -> None:
            if not NAME_RE.match(name.value or ""):
                harvest_msg.set_text("invalid name (letters/digits/._- only)")
                return
            flags = []
            if zip_c.value:
                flags.append("--zip")
            if redact_c.value:
                flags.append("--redact")
            if db_c.value:
                flags.append("--from-work-db")
            ui.notify("harvest started — watch the console", type="positive")
            await ops.runner.start(None, "harvest", name.value, *flags)
            harvest_msg.set_text("harvest finished")
            refresh()

        holder = ui.column().classes("w-full gap-3")

        def refresh() -> None:
            holder.clear()
            with holder:
                bundles = state.list_bundles()
                if not bundles:
                    ui.label("no bundles yet — harvest one above").classes("opacity-60")
                for info in bundles:
                    bundle_card(info)
                backups_section()

        with ui.card().classes("w-full"):
            with ui.expansion("guided migration (old cluster → bundle → new cluster)") \
                    .classes("w-full"):
                ui.label("one step at a time; after the exports: Stop all on the old "
                         "cluster (Clusters page), Boot… the new cluster from the "
                         "refreshed bundle, then Restore + Import above.") \
                    .classes("text-caption opacity-70")
                src = ui.select({e: e for e in ("work", "test")}, value="work",
                                label="source cluster").classes("w-40").props("dense outlined")
                for i, (label, verb) in enumerate(MIGRATION_STEPS, 1):
                    ui.button(f"{i}. {label}",
                              on_click=lambda v=verb, s=src: fire(s.value, v)) \
                        .classes("w-full").props("flat align=left")

        refresh()
