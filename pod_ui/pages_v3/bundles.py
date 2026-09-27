"""v3 bundles & backups (`/v3/bundles`): one page for the artifact flows —
harvest on top, bundle cards with a single Boot… menu, then the dumps table
(restore with typed wipe) and pod-owned data export/import. The guided
migration walkthrough stays collapsed at the bottom."""

from __future__ import annotations

import json
import re

from nicegui import ui

from .. import ops, state
from ..confirm import confirm_action, confirm_or_run
from ..ops import fire
from .chrome import frame_v3

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

MIGRATION_STEPS = [
    ("scrape the OLD cluster into the bundle", "scrape"),
    ("backup the OLD cluster state plane", "backup"),
    ("export the OLD cluster-owned data", "export-data"),
]


async def _boot(env_key: str, info) -> None:
    op = await ops.runner.start(env_key, ops.boot_action(env_key),
                                "-b", str(info.path))
    if op is None:
        ui.notify("an operation is already running — watch the console", type="warning")
        return
    ui.navigate.to("/v3")


def _scrape_into(env_key: str, info) -> None:
    fire(env_key, "scrape", str(info.path))


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
                                     on_click=lambda i=info: confirm_action(
                                         lambda: _boot("work", i),
                                         "Boot the BASIC cluster from this bundle?",
                                         "Stops the currently running environment "
                                         "(canonical ports must be free). Rebuild "
                                         "takes minutes.", button="Boot"))
                        ui.menu_item("Boot TEST (pg) from this",
                                     on_click=lambda i=info: confirm_action(
                                         lambda: _boot("test", i),
                                         "Boot the TEST cluster from this bundle?",
                                         "Builds/starts node1, node2, db, lb, minio. "
                                         "Takes minutes.", button="Boot"))
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
            op = await ops.runner.start(None, "harvest", name.value, *flags)
            if op is None:
                ui.notify("an operation is already running — watch the console", type="warning")
                return
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
