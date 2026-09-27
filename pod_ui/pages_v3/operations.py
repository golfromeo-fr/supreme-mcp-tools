"""Operations (`/v3/operations`): the FULL verb set per cluster, ported from
v1 onto the v3 chrome. Cluster cards stay lean (Start/Stop all/Boot/Rebuild);
this page carries everything incl. the destructive Clean (typed 'wipe').
All ops fire through ops.fire() -> shared runner -> the global console
auto-opens and streams; no output pane here. Guided migration lives here."""

from __future__ import annotations

from nicegui import ui

from .. import auth, ops
from ..confirm import confirm_action, confirm_or_run
from ..ops import fire

WIPE_WORD = "wipe"

ACTIONS = {
    "work": [
        ("up (rebuild)", "up", "light"),
        ("start", "start", "light"),
        ("stop", "stop", "light"),
        ("clean", "clean", "danger"),
        ("backup", "backup", "light"),
        ("scrape → bundle", "scrape", "light"),
        ("export-data", "export-data", "light"),
    ],
    "test": [
        ("up pg (rebuild)", "pg", "light"),
        ("start", "start", "light"),
        ("stop", "stop", "light"),
        ("clean", "clean", "danger"),
        ("backup", "backup", "light"),
        ("scrape → bundle", "scrape", "light"),
        ("export-data", "export-data", "light"),
    ],
}

MIGRATION_STEPS = [
    ("scrape the OLD cluster into the bundle", "scrape"),
    ("backup the OLD cluster state plane", "backup"),
    ("export the OLD cluster-owned data", "export-data"),
]


def _launch(env_key: str, action: str, *extra) -> None:
    if ops.runner.running():
        ui.notify("an operation is already running — watch the console", type="warning")
        return
    fire(env_key, action, *extra)


def _confirm_clean(env_key: str) -> None:
    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(f"clean {env_key} — this removes containers "
                 f"{'AND the state volume (users/masks)' if env_key == 'work' else '(volumes kept)'}")
        typed = ui.input(f"type '{WIPE_WORD}' to enable")
        with ui.row():
            run_btn = ui.button("Run clean", color="negative")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        def do_run() -> None:
            dialog.close()
            _launch(env_key, "clean")  # the runner pipes 'wipe' for the work env

        run_btn.on_click(do_run)
        run_btn.set_enabled(False)
        typed.on_value_change(
            lambda e: run_btn.set_enabled((e.value or "").strip() == WIPE_WORD))
    dialog.open()


def page() -> None:
    if (body := auth.frame("Operations")) is None:
        return

    def action_button(env_key: str, label: str, verb: str, tier: str) -> None:
        if tier == "danger":
            ui.button(label, on_click=lambda: _confirm_clean(env_key)) \
                .classes("w-full").props("color=negative")
        else:
            ui.button(label, on_click=lambda: _launch(env_key, verb)) \
                .classes("w-full")

    with body:
        with ui.row().classes("w-full flex-wrap gap-3 items-start"):
            for env_key, actions in ACTIONS.items():
                with ui.card().classes("gap-1 min-w-xs"):
                    ui.label(env_key).classes("text-h6")
                    for label, verb, tier in actions:
                        action_button(env_key, label, verb, tier)

        ui.separator()
        ui.label("Every action streams into the console (bottom / terminal pill) — "
                 "it opens automatically when one starts.").classes("text-caption opacity-60")

        with ui.card().classes("w-full"):
            with ui.expansion("guided migration (old cluster → bundle → new cluster)") \
                    .classes("w-full"):
                ui.label("one step at a time; after the exports: Stop all the old "
                         "cluster (Clusters page), boot the new one from the refreshed "
                         "bundle (Bundles page), then Restore + Import data (Backups "
                         "page).").classes("text-caption opacity-70")
                src = ui.select({e: e for e in ("work", "test")}, value="work",
                                label="source cluster").classes("w-40").props("dense outlined")
                for i, (label, verb) in enumerate(MIGRATION_STEPS, 1):
                    ui.button(f"{i}. {label}",
                              on_click=lambda v=verb, s=src: _launch(s.value, v)) \
                        .classes("w-full").props("flat align=left")
