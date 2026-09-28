"""v3 home (`/v3`): the cluster map. One card per cluster: header with state
+ inline cluster ops, the topology graph (nodes = live containers colored by
state, edges = functional links) with a per-card GRAPH <-> TABLE toggle
(the user's choice survives the 5s refresh), and the data row. Node click
opens the container drawer: safe actions visible; restart/stop-single are
behind a typed confirmation naming the container (no casual disruption)."""

from __future__ import annotations

import asyncio
import json

from nicegui import context, ui


from .. import state
from ..confirm import confirm_action
from ..ops import boot_action, fire, runner
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


def _power_confirm(env_key: str, full: str, short: str, action: str) -> None:
    """Disruptive single-container op behind a typed confirm naming the
    container itself — cannot be mis-clicked (user decision: no casual restart).
    Shells out through the runner: `startcluster <env> service <action> <name>`
    (F9: UI MUST shell out, never reimplement) — lock, streamed console history
    and process-group kill all apply."""
    with ui.dialog() as dialog, ui.card().classes("gap-2"):
        ui.label(f"{action} the container '{short}' — type '{short}' to confirm") \
            .classes("text-caption")
        typed = ui.input(f"type '{short}'")
        with ui.row():
            run_btn = ui.button(action.capitalize(), color="negative")
            ui.button("Cancel", on_click=dialog.close).props("flat")

        def do_run() -> None:
            dialog.close()
            if runner.running():
                ui.notify("an operation is already running — watch the console",
                          type="warning")
                return
            asyncio.ensure_future(runner.start(env_key, "service", action, full))
            ui.notify(f"{action} sent to {full}", type="info")

        run_btn.on_click(do_run)
        run_btn.set_enabled(False)
        typed.on_value_change(
            lambda e: run_btn.set_enabled((e.value or "").strip() == short))
    dialog.open()


async def _inspect_dialog(full: str) -> None:
    """Full `podman inspect` for one container: curated summary (secrets
    masked on screen) + collapsible raw JSON. The fetch runs in a thread;
    the dialog is built back in the handler task, which KEEPS its slot
    across awaits (creating UI from ensure_future tasks does not)."""
    data = await asyncio.to_thread(state.container_inspect, full)
    with ui.dialog() as dialog, ui.card().classes("gap-2 w-[44rem]"):
        ui.label(full).classes("text-h6")
        if not data:
            ui.label("podman inspect returned nothing (container gone?)") \
                .classes("opacity-70")
        else:
            cfg = data.get("Config") or {}
            st = data.get("State") or {}
            with ui.column().classes("gap-0 w-full"):
                ui.label(f"image: {cfg.get('Image')}").classes("text-caption font-mono")
                ui.label(f"created: {data.get('Created')}").classes("text-caption")
                ui.label(f"started: {st.get('StartedAt', '')[:19]}").classes("text-caption")
                ui.label(f"restarts: {st.get('RestartCount')}").classes("text-caption")
                ui.label(f"status: {st.get('Status')}").classes("text-caption")
            health = ((st.get("Health") or {}).get("Log")) or []
            if health:
                with ui.expansion("health probes").classes("w-full"):
                    for h in reversed(health[-4:]):
                        ui.label(f"exit={h.get('ExitCode')} · {h.get('Output', '')[:120]}") \
                            .classes("text-caption font-mono")
            mounts = data.get("Mounts") or []
            if mounts:
                with ui.expansion("mounts").classes("w-full"):
                    ui.table(
                        columns=[{"name": c, "label": c, "field": c} for c in
                                 ("Source", "Destination")],
                        rows=[{"Source": m.get("Source", "?"),
                               "Destination": m.get("Destination", "?")}
                              for m in mounts],
                        row_key="Source").props("dense flat")
            env_lines = state.mask_secrets((cfg.get("Env") or []))
            with ui.expansion("environment (secrets masked)").classes("w-full"):
                for line in env_lines:
                    ui.label(line).classes("text-caption font-mono")
            with ui.expansion("raw JSON").classes("w-full"):
                ui.code(json.dumps(data, indent=2)[:20000]) \
                    .classes("w-full max-h-[36rem]")
        ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


async def _await_inspect(full: str) -> None:
    await _inspect_dialog(full)


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
                ui.button("Inspect", icon="plagiarism",
                          on_click=lambda: _await_inspect(full)).props("dense")
                # disruptive actions: typed confirm naming the container
                ui.button("Restart", icon="restart_alt",
                          on_click=lambda: _power_confirm(env_key, full, short, "restart")) \
                    .props("dense flat color=negative")
                ui.button("Stop", icon="stop",
                          on_click=lambda: _power_confirm(env_key, full, short, "stop")) \
                    .props("dense flat color=negative")
        ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


MODE_LABEL = {"graph": "GRAPH", "list": "LIST", "both": "BOTH", "side": "SIDE"}
MODE_ICON = {"graph": "insights", "list": "list",
             "both": "view_agenda", "side": "splitscreen"}


def _cluster_card(env, sizes: dict[str, str], view_modes: dict, open_state: dict,
                  foreign: list[int] | None = None, minio_ok: bool = True):
    topo = TOPOLOGY[env.key]
    rows_by_name = {c.name: c for c in env.containers}
    foreign = foreign or []

    # running clusters start EXPANDED (user: show the useful info at startup);
    # stopped/absent start folded. open_state overrides per user clicks.
    if env.key not in open_state:
        open_state[env.key] = env.state in ("running", "partial")
    open_now = open_state[env.key]
    chevron_icon = "expand_more" if open_now else "chevron_right"

    with ui.card().classes("w-full gap-0").style("padding: 4px 16px"):
        # SINGLE header row: chevron + title + state/health + cluster ops.
        # No separate expansion row — the ^ lives here (user request).
        with ui.row().classes("w-full items-center gap-2 py-1"):
            chevron = ui.button(icon=chevron_icon,
                                on_click=lambda: flip_open()).props("flat dense") \
                .tooltip("show/hide cluster detail")
            ui.label(topo["title"]).classes("text-h6 cursor-pointer") \
                .on("click", lambda: flip_open())
            ui.badge(env.state, color=STATE_COLORS.get(env.state, "grey"))
            for label, value in env.health.items():
                ui.badge(f"{label}: {value}",
                         color="positive" if value == "ok" else "negative")
            ui.space()
            mode = view_modes.get(env.key, "graph")
            mode_btn = ui.button(MODE_LABEL[mode], icon=MODE_ICON[mode]) \
                .props("dense flat") \
                .tooltip("view: graph / list / both — click to cycle")
            if foreign:
                ui.badge("host stack holds " + ",".join(str(p) for p in foreign)
                         + " — boot will fail", color="negative")
            if env.key == "test" and not minio_ok:
                ui.badge("MINIO_ROOT_PASSWORD missing — boot will fail", color="warning")
            if env.key == "test":
                choices = state.topology_choices()
                topo_val = state.UI_PREFS.get("test_topology", choices["current"])
                if topo_val not in choices["options"]:
                    topo_val = choices["current"]
                ui.select(choices["options"], value=topo_val, label="topology",
                          on_change=lambda e: state.UI_PREFS.update({"test_topology": e.value}),
                          ).classes("w-52").props("dense outlined") \
                    .tooltip("boot/rebuild topology"
                             + ("" if choices["mode"] == "bundle"
                                else " — host mode: fixed by node.env (bundle boot to switch)"))
            if env.state in ("running", "partial"):
                ui.button("Stop all", icon="stop",
                          on_click=lambda k=env.key: confirm_action(
                              lambda: fire(k, "stop"),
                              f"Stop cluster {k}?",
                              f"All {sum(1 for c in env.containers if c.running)} running "
                              "containers stop. 'Start all' brings them back.",
                              button="Stop all")).props("dense color=negative")
            elif env.state == "absent":
                boot = boot_action(env.key)
                if env.key == "test":
                    boot = state.UI_PREFS.get("test_topology", boot)
                ui.button("Boot cluster", icon="rocket_launch",
                          on_click=lambda v=boot: fire(env.key, v)).props("dense color=primary")
            else:
                ui.button("Start all", icon="play_arrow",
                          on_click=lambda k=env.key: fire(k, "start")).props("dense color=primary")
            rebuild = boot_action(env.key)
            if env.key == "test":
                rebuild = state.UI_PREFS.get("test_topology", rebuild)
            ui.button("Rebuild", icon="refresh",
                      on_click=lambda k=env.key, v=rebuild: confirm_action(
                          lambda: fire(k, v),
                          f"Rebuild cluster {k} ({v} topology)?",
                          "Stops the cluster, rebuilds the node image, restarts it. "
                          "Takes minutes — the console shows progress.",
                          button="Rebuild")).props("dense flat") \
                .tooltip("stop + rebuild the image + start (startcluster up)")
            if env.key == "work" and env.containers:
                ui.button("MCP UI", icon="web",
                          on_click=lambda: fire("work", "ui")).props("dense flat") \
                    .tooltip("start the in-cluster management UI on :8400 — "
                             "stop the host startui first; useful only while the cluster is up")

        # BODY: hidden while collapsed; visibility re-applied on rebuilds
        body = ui.column().classes("w-full gap-2")
        with body:
            ui.label("(click a node — or its chip — for actions)").classes(
                "text-caption opacity-50")
            # one container, three arrangements:
            #   graph -> centered column (graph + chips)
            #   list  -> centered table
            #   both  -> grid: graph | table side by side
            views = ui.element("div").classes("w-full")

            with views:
                graph_box = ui.column().classes("w-full")
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
                        # rows_by_name is the card's own snapshot — do NOT re-query
                        # podman here (a blocking call in a handler stalls the loop,
                        # the very bug this page was rewritten to avoid)
                        rows = rows_by_name
                        _node_drawer(env.key, short, rows)

                    chart.on("componentClick", on_node)
                    chart.on("chart:click", on_node)

                    with ui.row().classes("gap-1 pt-1"):
                        for short, _x, _y in topo["nodes"]:
                            row = rows_by_name.get(topo["container_of"][short])
                            state_word = row.state if row else "absent"
                            color = {"Up": "positive", "absent": "grey-6"}.get(
                                state_word, "negative")
                            ui.chip(short, color=color, icon="circle",
                                    on_click=lambda s=short: _node_drawer(
                                        env.key, s, rows_by_name)) \
                                .props("outline dense clickable")

                table_box = ui.column().classes("w-full")
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

            def apply_view() -> None:
                m = view_modes.get(env.key, "graph")
                graph_box.set_visibility(m in ("graph", "both", "side"))
                table_box.set_visibility(m in ("list", "both", "side"))
                if m == "side":
                    views.style("display:grid; grid-template-columns:1fr 1fr; "
                                "gap:1.5rem; align-items:start; width:100%; "
                                "max-width:150rem; margin:0 auto")
                    graph_box.style("width:100%; max-width:none")
                    table_box.style("width:100%; max-width:none")
                else:
                    views.style("display:flex; flex-direction:column; "
                                "align-items:center; width:100%")
                    if m == "list":
                        table_box.style("width:100%; max-width:56rem")
                    if m == "both":
                        graph_box.style("width:100%; max-width:56rem")
                        table_box.style("width:100%; max-width:56rem")
                    if m == "graph":
                        graph_box.style("width:100%; max-width:56rem")

            def _flip(key: str, modes: dict) -> None:
                order = ["graph", "list", "both", "side"]
                nxt = order[(order.index(modes.get(key, "graph")) + 1) % 4]
                modes[key] = nxt
                apply_view()
                mode_btn.props(f"icon={MODE_ICON[nxt]} label='{MODE_LABEL[nxt]}'")

            vols = {k: v for k, v in sorted(sizes.items()) if k.startswith(env.prefix)}
            vols_txt = " · ".join(f"{k.replace(env.prefix, '')}={v}" for k, v in vols.items())
            disk = state.cluster_disk(env.containers, vols)
            ui.label(f"cluster disk: {disk} · volumes: {vols_txt or '—'}") \
                .classes("text-caption opacity-60")

        def flip_open() -> None:
            open_state[env.key] = not open_state.get(env.key, False)
            chevron.props(f'icon="{ "expand_more" if open_state[env.key] else "chevron_right" }"')
            body.set_visibility(open_state[env.key])

        # the header button is created before _flip exists — wire it here,
        # at card scope (NOT inside _flip: chicken-and-egg dead button)
        mode_btn.on_click(lambda: _flip(env.key, view_modes))

        # apply the stored arrangement ONCE at build — without this the page
        # opened raw/un-centered until the first mode click (user report)
        apply_view()

        body.set_visibility(open_now)


def page() -> None:
    if (body := frame_v3("Clusters")) is None:
        return
    with body:
        # capture the client HERE (inside the slot): context.client cannot be
        # read from a background task — the slot stack is empty there
        client = context.client
        last_sig = {"v": None}
        view_modes: dict = {e.key: "both" for e in state.ENVIRONMENTS}  # default BOTH
        open_state: dict = {}  # per-cluster body open/closed; running clusters default open

        def _gather_cheap() -> dict:
            """Cheap part (ONE podman ps + one inspect) — enough to decide
            whether a rebuild is needed at all. Threaded: blocking the loop
            here stalled the socket heartbeat (Connection-lost cycle)."""
            grouped = state.containers_by_env()
            return {"statuses": state.env_statuses(grouped),
                    "image": state.image_version(),
                    "other": grouped.get("other", [])}

        def _gather_full() -> dict:
            """Expensive part (df -v + df --format json) — fetched only when the
            signature changed, so the steady-state tick stays cheap."""
            return {"sizes": state.volume_sizes(),
                    "df": state.podman_df_summary()}

        def _render(data: dict) -> None:
            statuses = data["statuses"]
            visible = [e for e in statuses
                       if state.UI_PREFS["show_stopped"]
                       or e.state in ("running", "partial")]
            holder.clear()
            with holder:
                if not visible:
                    ui.label("all clusters stopped — flip 'show stopped' in the side bar to review them") \
                        .classes("opacity-60")
                with ui.row().classes("w-full flex-wrap gap-3 items-start"):
                    for env in visible:
                        _cluster_card(env, data["sizes"], view_modes, open_state)
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.label(f"node image: {data['image']}").classes("text-caption")
                        df = data.get("df")
                        if df:
                            ui.space()
                            ui.label(" · ".join(
                                f"{r.get('Type')}: {r.get('Size')} "
                                f"(reclaim {r.get('Reclaimable')})" for r in df)) \
                                .classes("text-caption opacity-60")
                    rows = df
                    if rows:
                        ui.table(
                            columns=[{"name": c, "label": c, "field": c} for c in
                                     ("Type", "Total", "Active", "Size", "Reclaimable")],
                            rows=rows, row_key="Type").props("dense flat")
                    else:
                        ui.label("podman system df --format json unavailable") \
                            .classes("opacity-60")
                if data["other"]:
                    with ui.card().classes("w-full"):
                        ui.label("other containers (read-only)").classes(
                            "text-caption opacity-60")
                        ui.label(", ".join(c.name for c in data["other"])).classes(
                            "text-caption")

        # serialize rebuilds: the initial task, the 5s tick and the toggle can
        # otherwise overlap, and two coroutines doing holder.clear()/with holder
        # interleave DOM creation (duplicated / misplaced elements)
        _rebuild_lock = asyncio.Lock()

        async def rebuild(force: bool = False) -> None:
            # a closed browser tab leaves this page's timer alive: without this
            # guard every tab ever opened keeps polling podman every 5s forever
            # (and renders into a deleted client — "Client has been deleted")
            if client.is_deleted or not client.has_socket_connection:
                return
            async with _rebuild_lock:
                if client.is_deleted or not client.has_socket_connection:
                    return
                data = await asyncio.to_thread(_gather_cheap)
                # change-gate BEFORE the expensive gather: unchanged state =>
                # untouched DOM *and* no wasted df/system-df subprocesses
                sig = (state.env_signature(data["statuses"]) + f"#{data['image']}"
                       f"#{state.UI_PREFS['show_stopped']}"
                       f"#{','.join(c.name for c in data['other'])}")
                if sig == last_sig["v"] and not force:
                    return
                last_sig["v"] = sig
                data.update(await asyncio.to_thread(_gather_full))
                _render(data)

        holder = ui.column().classes("w-full gap-3")

        async def tick() -> None:
            await rebuild()

        # initial build: thread-gathered like every tick (never block the loop)
        asyncio.ensure_future(rebuild())
        ui.timer(5.0, tick)
