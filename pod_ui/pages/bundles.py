"""Bundles (`/bundles`): harvest a new bundle, then list my-bundles/ — inspect
the manifest, boot an environment from a bundle, scrape a pod into it, delete."""

from __future__ import annotations

import json
import re

from nicegui import ui

from .. import auth, ops, state
from ..confirm import confirm_or_run

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def page() -> None:
    if (body := auth.frame("Bundles")) is None:
        return

    def show_manifest(info) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-full"):
            ui.label(f"manifest — {info.name}").classes("text-h6")
            text = json.dumps(info.manifest, indent=2, ensure_ascii=False) if info.manifest else "{}"
            ui.code(text).classes("w-full max-h-[36rem]")
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    async def boot(env_key: str, info) -> None:
        action = "up" if env_key == "work" else "pg"
        await ops.runner.start(env_key, action, str(info.path))
        ui.notify(f"{env_key} boot finished — see Operations for the log", type="info")
        refresh()

    async def scrape_into(env_key: str, info) -> None:
        await ops.runner.start(env_key, "scrape", str(info.path))
        ui.notify(f"scrape finished — see Operations for the log", type="info")
        refresh()

    def delete_bundle(info) -> None:
        import shutil

        async def do_delete() -> None:
            shutil.rmtree(info.path, ignore_errors=True)
            ui.notify(f"bundle {info.name} deleted", type="positive")
            refresh()
        confirm_or_run(do_delete, f"delete bundle {info.name} (rm -rf)", word=info.name)

    def build() -> None:
        bundles = state.list_bundles()
        if not bundles:
            ui.label("no bundles yet — harvest one below, or ./startcluster harvest") \
                .classes("opacity-60")
            return
        for info in bundles:
            m = info.manifest or {}
            with ui.card().classes("w-full gap-1"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.label(info.name).classes("text-h6")
                    git = (m.get("git") or {}).get("describe", "")
                    if git:
                        ui.badge(f"git: {git}", color="grey")
                    else:
                        ui.badge("no manifest", color="negative")
                    if m.get("redacted"):
                        ui.badge("REDACTED", color="warning")
                    if m.get("scraped"):
                        ui.badge("scraped from pod", color="info")
                    ui.label(f"created {m.get('created_utc', '?')} · "
                             f"{len(m.get('files', []))} files").classes("text-caption opacity-60")
                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                    ui.button("Boot work from this", icon="rocket_launch",
                              on_click=lambda i=info: confirm_or_run(
                                  lambda: boot("work", i),
                                  "boot the work env from this bundle (stops the running one; "
                                  "canonical ports must be free)")).props("dense")
                    ui.button("Boot test pg from this", icon="science",
                              on_click=lambda i=info: confirm_or_run(
                                  lambda: boot("test", i),
                                  "boot the test cluster (pg) from this bundle")).props("dense")
                    ui.button("Scrape work into", icon="download",
                              on_click=lambda i=info: confirm_or_run(
                                  lambda: scrape_into("work", i))).props("dense flat")
                    ui.button("Scrape test into", icon="download",
                              on_click=lambda i=info: confirm_or_run(
                                  lambda: scrape_into("test", i))).props("dense flat")
                    ui.button("Manifest", icon="description",
                              on_click=lambda i=info: show_manifest(i)).props("flat dense")
                    ui.button("Delete", icon="delete",
                              on_click=lambda i=info: delete_bundle(i)) \
                        .props("flat dense color=negative")

    with body:
        # harvest form FIRST — the list below refers to it
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
            ui.notify("harvest started — watch Operations for output", type="positive")
            await ops.runner.start(None, "harvest", name.value, *flags)
            harvest_msg.set_text("harvest finished — see Operations")
            refresh()

        holder = ui.column().classes("w-full gap-3")

        def refresh() -> None:
            holder.clear()
            with holder:
                build()

        refresh()
