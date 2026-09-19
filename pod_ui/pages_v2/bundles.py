"""v2 bundles (`/v2/bundles`): harvest on top; per-bundle ONE primary action
(Boot…) plus an overflow menu — scrape-into / manifest / delete live behind
⋮ instead of six same-weight buttons. The guided migration walkthrough is a
collapsed section here (it is a bundle story)."""

from __future__ import annotations

import json
import re

from nicegui import ui

from .. import ops, state
from ..confirm import confirm_or_run
from .chrome import fire, frame_v2

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

MIGRATION_STEPS = [
    ("scrape the OLD pod into the bundle", "scrape"),
    ("backup the OLD pod state plane", "backup"),
    ("export the OLD pod-owned data", "export-data"),
]


async def _boot(env_key: str, info) -> None:
    await ops.runner.start(env_key, "up" if env_key == "work" else "pg", str(info.path))
    ui.navigate.to(f"/v2/env/{env_key}")


async def _scrape_into(env_key: str, info) -> None:
    await ops.runner.start(env_key, "scrape", str(info.path))


def page() -> None:
    if (body := frame_v2("Bundles")) is None:
        return

    def show_manifest(info) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-full"):
            ui.label(f"manifest — {info.name}").classes("text-h6")
            text = json.dumps(info.manifest, indent=2, ensure_ascii=False) if info.manifest else "{}"
            ui.code(text).classes("w-full max-h-[36rem]")
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    with body:
        # ---- harvest ----
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

        def build_bundle_card(info) -> None:
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
                        with ui.menu().props("no-wrap") as menu:
                            ui.menu_item("Boot work from this",
                                         on_click=lambda i=info: confirm_or_run(
                                             lambda: _boot("work", i),
                                             "boot the work env from this bundle "
                                             "(stops the running one; canonical ports "
                                             "must be free)"))
                            ui.menu_item("Boot test (pg) from this",
                                         on_click=lambda i=info: confirm_or_run(
                                             lambda: _boot("test", i),
                                             "boot the test cluster from this bundle"))
                            ui.separator()
                            ui.menu_item("Scrape work into",
                                         on_click=lambda i=info: confirm_or_run(
                                             lambda: _scrape_into("work", i)))
                            ui.menu_item("Scrape test into",
                                         on_click=lambda i=info: confirm_or_run(
                                             lambda: _scrape_into("test", i)))
                    with ui.button(icon="more_vert").props("flat dense").tooltip("more"):
                        with ui.menu().props("no-wrap") as menu2:
                            ui.menu_item("Manifest",
                                         on_click=lambda i=info: show_manifest(i))
                            ui.separator()

                            def _delete(i) -> None:
                                import shutil

                                async def do_delete() -> None:
                                    shutil.rmtree(i.path, ignore_errors=True)
                                    ui.notify(f"bundle {i.name} deleted", type="positive")
                                    refresh()
                                confirm_or_run(do_delete,
                                               f"delete bundle {i.name} (rm -rf)",
                                               word=i.name)

                            ui.menu_item("Delete", on_click=lambda i=info: _delete(i))

        def refresh() -> None:
            holder.clear()
            with holder:
                bundles = state.list_bundles()
                if not bundles:
                    ui.label("no bundles yet — harvest one above").classes("opacity-60")
                    return
                for info in bundles:
                    build_bundle_card(info)

        # ---- guided migration (collapsed by default) ----
        with ui.card().classes("w-full"):
            with ui.expansion("guided migration (old pod → bundle → new pod)") \
                    .classes("w-full"):
                ui.label("one step at a time; after the exports: stop/clean the old "
                         "env (its page), boot the new pod from the refreshed bundle "
                         "(Boot… above), then restore + import data on the new env's "
                         "page.").classes("text-caption opacity-70")
                src = ui.select({e: e for e in ("work", "test")}, value="work",
                                label="source env").classes("w-40").props("dense outlined")
                for i, (label, verb) in enumerate(MIGRATION_STEPS, 1):
                    ui.button(f"{i}. {label}",
                              on_click=lambda v=verb, s=src: fire(s.value, v)) \
                        .classes("w-full").props("flat align=left")

        refresh()
