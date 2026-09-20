"""v3 home (`/v3`): the cluster map. One card per cluster: header with state
+ inline cluster ops, the topology graph (nodes = live containers colored by
state, edges = functional links) with a per-card GRAPH <-> TABLE toggle
(the user's choice survives the 5s refresh), and the data row. Node click
opens the container drawer: safe actions visible; restart/stop-single are
behind a typed confirmation naming the container (no casual disruption)."""

from __future__ import annotations

import subprocess

from nicegui import ui

from .. import state
from ..ops import fire
from .chrome import frame_v3

STATE_COLORS = {"running": "positive", "partial": "warning",
                "stopped": "grey-7", "absent": "grey-6"}

NODE_COLORS = {"Up": "#38b45c", "Exited": "#e53935", "Created": "#888", "absent": "#555"}

# Static cluster topology (from the compose files — the functional links the
# deploy docs draw). env keys stay 'work'/'test' (machine names); TITLES are
# what the user asked for: basic = the single-node full stack.
TOPOLOGY = {
    "work": {
        "title": "Cluster 1 · BASIC",
        "nodes": [("ui", 60, 90), ("work", 220, 90), ("db", 380, 90)],
        "edges": [("ui", "work", "manages"), ("work", "db", "state plane")],
        "container_of": {"ui": "mcp-work_ui_1", "work": "mcp-work_work_1",
                         "db": "mcp-work_db_1"},
    },
    "test": {
        "title": "Cluster 2 · TEST",
        "nodes": [("lb", 50, 90), ("node1", 210, 40), ("node2", 210, 140),
                  ("db", 380, 40), ("minio", 380, 165)],
        "edges": [("lb", "node1", "balances"), ("lb", "node2", "balances"),
                  ("node1", "db", "state"), ("node2", "db", "state"),
                  ("node1", "minio", "artifacts"), ("node2", "minio", "artifacts")],
        "container_of": {"lb": "mcp-multihost_lb_1", "node1": "mcp-multihost_node1_1",
                         "node2": "mcp-multihost_node2_1", "db": "mcp-multihost_db_1",
                         "minio": "mcp-multihost_minio_1"},
    },
}


def _graph_options(env_key: str, rows_by_name: dict) -> dict:
    topo = TOPOLOGY[env_key]
    nodes = []
    for name, x, y in topo["nodes"]:
        row = rows_by_name.get(topo["container_of"][name])
        state_word = row.state if row else "absent"
        nodes.append({
            "name": name, "x": x, "y": y,
            "symbolSize": 46,
            "itemStyle": {"color": NODE_COLORS.get(state_word, "#888"),
                          **({"opacity": 0.35} if not row else {})},
            "label": {"show": True, "color": "#eee"},
            "value": f"{state_word}" + (f" · {row.status}" if row else " · not created"),
        })
    edges = [{"source": s, "target": t, "label": {"show": True, "formatter": lab,
                                                  "color": "#999", "fontSize": 10}}
             for s, t, lab in topo["edges"]]
    return {
        "animation": False,
        "tooltip": {},
        "series": [{
            "type": "graph", "layout": "none", "roam": False,
            "data": nodes, "links": edges,
            "edgeSymbol": ["none", "arrow"], "edgeSymbolSize": 8,
            "lineStyle": {"color": "#666", "width": 2, "curveness": 0},
            "emphasis": {"focus": "adjacency"},
        }],
    }


def _power_confirm(full: str, short: str, action: str) -> None:
    """Disruptive single-container op behind a typed confirm naming the
    container itself — cannot be mis-clicked (user decision: no casual restart)."""
    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(f"{action} the container '{short}' — type '{short}' to confirm") \
            .classes("text-caption")
        typed = ui.input(f"type '{short}'")
        with ui.row():
            run_btn = ui.button(action.capitalize(), color="negative")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        def do_run() -> None:
            dialog.close()
            subprocess.Popen(["podman", action, full],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ui.notify(f"{action} sent to {full}", type="info")

        run_btn.on_click(do_run)
        run_btn.set_enabled(False)
        typed.on_value_change(
            lambda e: run_btn.set_enabled((e.value or "").strip() == short))
    dialog.open()


def _node_drawer(env_key: str, short: str, rows_by_name: dict) -> None:
    topo = TOPOLOGY[env_key]
    full = topo["container_of"].get(short)
    row = rows_by_name.get(full)
    with ui.dialog() as dialog, ui.card().classes("gap-2 w-96"):
        ui.label(short).classes("text-h6")
        if row is None:
            ui.label("container not created — boot the cluster first").classes("opacity-70")
        else:
            with ui.column().classes("gap-0 w-full"):
                ui.label(f"state: {row.state}").classes("text-caption")
                ui.label(f"status: {row.status}").classes("text-caption opacity-70")
                if row.ports:
                    ui.label(f"ports: {row.ports}").classes("text-caption opacity-70")
                if row.size:
                    ui.label(f"size: {row.size}").classes("text-caption opacity-70")
            with ui.row().classes("gap-2"):
                # safe action: straight into this container's logs
                ui.button("Logs", icon="article",
                          on_click=lambda: ui.navigate.to(f"/v3/logs?container={full}")) \
                    .props("dense")
                # disruptive actions: typed confirm naming the container
                ui.button("Restart", icon="restart_alt",
                          on_click=lambda: _power_confirm(full, short, "restart")) \
                    .props("dense flat color=negative")
                ui.button("Stop", icon="stop",
                          on_click=lambda: _power_confirm(full, short, "stop")) \
                    .props("dense flat color=negative")
        ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


def _cluster_card(env, sizes: dict[str, str], view_modes: dict) -> None:
    topo = TOPOLOGY[env.key]
    rows_by_name = {c.name: c for c in env.containers}
    with ui.card().classes("w-full gap-2"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(topo["title"]).classes("text-h6")
            ui.badge(env.state, color=STATE_COLORS.get(env.state, "grey"))
            for label, value in env.health.items():
                ui.badge(f"{label}: {value}",
                         color="positive" if value == "ok" else "negative")
            ui.space()
            if env.state in ("running", "partial"):
                ui.button("Stop all", icon="stop", on_click=lambda k=env.key: fire(k, "stop")) \
                    .props("dense")
            elif env.state == "absent":
                boot = "up" if env.key == "work" else "pg"
                ui.button("Boot cluster", icon="rocket_launch",
                          on_click=lambda v=boot: fire(env.key, v)).props("dense color=primary")
            else:
                ui.button("Start all", icon="play_arrow",
                          on_click=lambda k=env.key: fire(k, "start")).props("dense color=primary")
            rebuild = "up" if env.key == "work" else "pg"
            ui.button("Rebuild", icon="refresh",
                      on_click=lambda v=rebuild: fire(env.key, v)).props("dense flat") \
                .tooltip("stop + rebuild the image + start (startcluster up)")
            # graph <-> table toggle (per card; choice survives the 5s refresh)
            ui.button(icon="table_chart",
                      on_click=lambda k=env.key: _flip(k)).props("flat dense") \
                .tooltip("switch graph / table view")

        graph_box = ui.column().classes("w-full max-w-4xl mx-auto")
        table_box = ui.column().classes("w-full")

        with graph_box:
            chart = ui.echart(_graph_options(env.key, rows_by_name)) \
                .classes("w-full h-64").style("background: rgba(255,255,255,.02)")

            def on_node(e) -> None:
                args = getattr(e, "args", None) or {}
                if isinstance(args, dict):
                    data = args.get("data") or {}
                    short = data.get("name") if isinstance(data, dict) else None
                else:
                    short = None
                if not short:
                    return
                rows = {c.name: c for c in state.containers_by_env().get(env.key, [])}
                _node_drawer(env.key, short, rows)

            # NiceGUI's echart JS emits clicks as 'componentClick' and
            # 'chart:click' (see elements/echart/echart.js) — a plain
            # .on("click") subscribes to an event that never fires
            chart.on("componentClick", on_node)
            chart.on("chart:click", on_node)

        with graph_box:
            # deterministic click path (and legend): one chip per node,
            # colored by state — same drawer as a graph-node click
            with ui.row().classes("gap-1"):
                for short, _x, _y in topo["nodes"]:
                    row = rows_by_name.get(topo["container_of"][short])
                    state_word = row.state if row else "absent"
                    color = {"Up": "positive", "absent": "grey-6"}.get(
                        state_word, "negative")
                    ui.chip(short, color=color, icon="circle",
                            on_click=lambda s=short: _node_drawer(
                                env.key, s, rows_by_name)) \
                        .props("outline dense clickable")

        with table_box:
            if env.containers:
                ui.table(
                    columns=[{"name": c, "label": c, "field": c} for c in
                             ("name", "state", "status", "size", "ports")],
                    rows=[{"name": c.name, "state": c.state, "status": c.status,
                           "size": c.size, "ports": c.ports} for c in env.containers],
                    row_key="name").props("dense flat")
            else:
                ui.label("no containers (never created)").classes("opacity-60")

        def apply_mode() -> None:
            graph_box.set_visibility(view_modes.get(env.key, "graph") == "graph")
            table_box.set_visibility(view_modes.get(env.key, "graph") != "graph")

        def _flip(key: str) -> None:
            view_modes[key] = "table" if view_modes.get(key, "graph") == "graph" else "graph"
            apply_mode()

        apply_mode()

        vols = " · ".join(f"{k.replace(env.prefix, '').replace('_', '')}={v}"
                          for k, v in sorted(sizes.items()) if k.startswith(env.prefix))
        ui.label(f"volumes: {vols or '—'}").classes("text-caption opacity-60")


def page() -> None:
    if (body := frame_v3("Clusters")) is None:
        return
    with body:
        holder = ui.column().classes("w-full gap-3")
        view_modes: dict = {}  # page-level: survives the 5s content rebuilds

        def rebuild() -> None:
            holder.clear()
            with holder:
                statuses = state.env_statuses()
                sizes = state.volume_sizes()
                with ui.row().classes("w-full flex-wrap gap-3 items-start"):
                    for env in statuses:
                        _cluster_card(env, sizes, view_modes)
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label(f"node image: {state.image_version()}").classes("text-caption")
                        df = state.podman_df_summary()
                        if df:
                            ui.space()
                            ui.label(" · ".join(
                                f"{r.get('Type')}: {r.get('Size')} (reclaim {r.get('Reclaimable')})"
                                for r in df)).classes("text-caption opacity-60")

        rebuild()
        ui.timer(5.0, rebuild)
