"""v2 home (`/v2`): the control room — one card per environment with its
PRIMARY action inline (Start when down, Stop when up), health at a glance,
container detail collapsed by default. No read-only dead ends."""

from __future__ import annotations

from nicegui import ui

from .. import state
from .chrome import fire, frame_v2

STATE_COLORS = {"running": "positive", "partial": "warning",
                "stopped": "grey-7", "absent": "grey-6"}


def _primary(env) -> tuple[str, str]:
    """(label, verb) for the env's main action."""
    if env.state == "running" or env.state == "partial":
        return ("Stop", "stop")
    if env.state == "absent":
        return ("Boot (build)", "up" if env.key == "work" else "pg")
    return ("Start", "start")


def _env_card(env, sizes: dict[str, str]):
    with ui.card().classes("w-full gap-2"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(env.title).classes("text-h6")
            ui.badge(env.state, color=STATE_COLORS.get(env.state, "grey"))
            for label, value in env.health.items():
                ui.badge(f"{label}: {value}",
                         color="positive" if value == "ok" else "negative")
        with ui.row().classes("items-center gap-2"):
            label, verb = _primary(env)
            ui.button(label, icon="power_settings_new",
                      on_click=lambda v=verb: fire(env.key, v)).props("color=primary")
            ui.button("Manage", icon="tune",
                      on_click=lambda k=env.key: ui.navigate.to(f"/v2/env/{k}")).props("flat")
            rebuild = "up" if env.key == "work" else "pg"
            ui.button("Rebuild", icon="refresh",
                      on_click=lambda r=rebuild: fire(env.key, r)).props("flat") \
                .tooltip("stop + rebuild the image + start (startcluster up)")
        if env.state == "absent":
            ui.label("never booted — Boot builds the image and starts it").classes("opacity-60")
        else:
            with ui.expansion("containers",
                              caption=f"{len(env.containers)} · volumes "
                                      + " ".join(
                                          f"{k.replace(env.prefix, '')}={v}"
                                          for k, v in sorted(sizes.items())
                                          if k.startswith(env.prefix))).classes("w-full"):
                ui.table(
                    columns=[{"name": c, "label": c, "field": c} for c in
                             ("name", "state", "status", "size", "ports")],
                    rows=[{"name": c.name, "state": c.state, "status": c.status,
                           "size": c.size, "ports": c.ports} for c in env.containers],
                    row_key="name").props("dense flat")


def page() -> None:
    if (body := frame_v2("Home")) is None:
        return
    with body:
        holder = ui.column().classes("w-full gap-3")

        def rebuild() -> None:
            holder.clear()
            with holder:
                statuses = state.env_statuses()
                sizes = state.volume_sizes()
                with ui.row().classes("w-full flex-wrap gap-3 items-start"):
                    for env in statuses:
                        _env_card(env, sizes)
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label(f"node image: {state.image_version()}").classes("text-caption")
                        ui.space()
                        ui.button("harvest a bundle", icon="save",
                                  on_click=lambda: ui.navigate.to("/v2/bundles")).props("flat dense")

        rebuild()
        ui.timer(5.0, rebuild)
