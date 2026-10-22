"""Lifecycle (`/v3/operations`): the per-cluster lifecycle verbs —
Rebuild (topology-aware for TEST), Start, Stop, Clean (typed 'wipe').
Data verbs live on Backups & data; scrape/boot live on Bundles.
All ops fire through ops.fire() -> shared runner -> the global console
auto-opens and streams; no output pane here."""

from __future__ import annotations

from nicegui import ui

from .. import ops, state
from ..ops import fire
from .chrome import frame_v3

WIPE_WORD = "wipe"


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


def _topo_select() -> ui.select:
    """TEST topology select (per the unified plan §2.2): bundle mode offers
    all four (node.env regenerates per topology); host mode pins to the
    node.env-matching topology — the others would boot a broken cluster."""
    choices = state.topology_choices()
    topo_val = state.UI_PREFS.get("test_topology", choices["current"])
    if topo_val not in choices["options"]:
        topo_val = choices["current"]
    sel = ui.select(choices["options"], value=topo_val, label="topology",
                    on_change=lambda e: state.UI_PREFS.update({"test_topology": e.value}),
                    ).classes("w-56").props("dense outlined") \
        .tooltip("Rebuild/Boot topology"
                 + ("" if choices["mode"] == "bundle"
                    else " — host mode: fixed by node.env (boot from a bundle to switch)"))
    return sel


def page() -> None:
    if (body := frame_v3("Lifecycle")) is None:
        return

    def _rebuild(env_key: str) -> None:
        topo = state.UI_PREFS.get("test_topology", "pg") if env_key == "test" else "up"
        with ui.dialog() as dialog, ui.card().classes("gap-2"):
            ui.label(f"Rebuild cluster {env_key} ({topo} topology)?")
            ui.label("Stops the cluster, rebuilds the node image, restarts it. "
                     "Takes minutes — the console shows progress.")
            with ui.row():
                ui.button("Rebuild", color="primary",
                          on_click=lambda: (dialog.close(), _launch(env_key, topo)))
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()

    with body:
        for env_key, title in (("work", "BASIC"), ("test", "TEST")):
            with ui.card().classes("gap-1 min-w-xs"):
                ui.label(title).classes("text-h6")
                ui.button("Rebuild", icon="refresh",
                          on_click=lambda e=env_key: _rebuild(e)) \
                    .classes("w-full") \
                    .tooltip("stop + rebuild the image + start (startcluster up)")
                ui.button("Start", icon="play_arrow",
                          on_click=lambda e=env_key: _launch(e, "start")) \
                    .classes("w-full")
                ui.button("Stop", icon="stop",
                          on_click=lambda e=env_key: _launch(e, "stop")) \
                    .classes("w-full")
                ui.button("Clean", icon="delete_forever",
                          on_click=lambda e=env_key: _confirm_clean(e)) \
                    .classes("w-full").props("color=negative") \
                    .tooltip("removes containers"
                             + (" AND the state volume (users/masks)" if env_key == "work"
                                else " (volumes kept)"))

        ui.separator()
        ui.label("Every action streams into the console (bottom / terminal pill) — "
                 "it opens automatically when one starts.").classes("text-caption opacity-60")
