"""Dashboard (`/`): one card per environment — state chip, health badges,
container table, volume sizes — plus the podman df summary and image version.
Auto-refresh 5s via container.clear() rebuild (never @ui.refreshable)."""

from __future__ import annotations

from nicegui import ui

from .. import auth, state

STATE_COLORS = {"running": "positive", "partial": "warning",
                "stopped": "grey-7", "absent": "grey-6"}


def _env_card(env, sizes: dict[str, str]):
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(env.title).classes("text-h6")
            ui.badge(env.state, color=STATE_COLORS.get(env.state, "grey"))
            ui.space()
            for label, value in env.health.items():
                color = "positive" if value == "ok" else "negative"
                ui.badge(f"{label}: {value}", color=color)
        if env.containers:
            with ui.table(
                columns=[
                    {"name": "name", "label": "container", "field": "name"},
                    {"name": "state", "label": "state", "field": "state"},
                    {"name": "status", "label": "status", "field": "status"},
                    {"name": "size", "label": "size", "field": "size"},
                    {"name": "ports", "label": "ports", "field": "ports"},
                ],
                rows=[{"name": c.name, "state": c.state, "status": c.status,
                       "size": c.size, "ports": c.ports} for c in env.containers],
                row_key="name",
            ).classes("w-full") as t:
                t.props("dense flat")
        else:
            ui.label("no containers (never created, or removed with clean)").classes("opacity-60")
        env_vols = {k: v for k, v in sizes.items() if k.startswith(env.prefix)}
        if env_vols:
            ui.label("volumes: " + " · ".join(f"{k} = {v}" for k, v in sorted(env_vols.items()))) \
                .classes("text-caption opacity-60")
        ui.space()
        ui.button("Operations", icon="play_circle",
                  on_click=lambda e=env.key: ui.navigate.to(f"/operations?env={e.key}")) \
            .props("flat dense")


def render() -> None:
    body = ui.column().classes("w-full gap-3")
    with body:
        statuses = state.env_statuses()
        sizes = state.volume_sizes()
        with ui.row().classes("w-full flex-wrap gap-3"):
            for env in statuses:
                _env_card(env, sizes)
        with ui.card().classes("w-full"):
            ui.label("node image").classes("text-caption opacity-60")
            ui.label(state.image_version()).classes("text-mono")
        with ui.card().classes("w-full"):
            ui.label("podman disk usage").classes("text-caption opacity-60")
            rows = state.podman_df_summary()
            if rows:
                ui.table(
                    columns=[{"name": c, "label": c, "field": c} for c in
                             ("Type", "Total", "Active", "Size", "Reclaimable")],
                    rows=rows, row_key="Type").props("dense flat")
            else:
                ui.label("podman system df --format json unavailable").classes("opacity-60")
        other = state.containers_by_env().get("other", [])
        if other:
            with ui.card().classes("w-full"):
                ui.label("other containers (read-only)").classes("text-caption opacity-60")
                ui.label(", ".join(c.name for c in other)).classes("text-caption")
    # note: state chips rebuild via the page timer below


def page() -> None:
    if (body := auth.frame("Dashboard")) is None:
        return
    holder = ui.column().classes("w-full")

    def rebuild() -> None:
        holder.clear()
        with holder:
            render()

    rebuild()

    async def tick() -> None:
        rebuild()

    ui.timer(5.0, tick)
