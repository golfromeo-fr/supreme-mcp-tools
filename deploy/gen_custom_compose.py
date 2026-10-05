#!/usr/bin/env python3
"""gen_custom_compose — registry entry -> compose YAML for a custom cluster.

Pure function of the registry JSON (unit-pinned in tests). The caller writes
the output to deploy/compose-<name>.run.yml and runs
`podman-compose -p <name> -f compose-<name>.run.yml up -d`.

usage: gen_custom_compose.py <clusters.d/<name>.json>
"""

import json
import sys

import yaml

# tool -> container port (canonical assignments, config/ports.json)
PORT_CONTAINER = {"databasemcp": 8000, "webmcp": 8001, "simplemcp": 8002,
                  "convertermcp": 8003, "ragmcp": 8004, "memorymcp": 8005}


def db_service(name: str, topology: str) -> dict:
    if topology == "turso":
        return {
            "image": "ghcr.io/tursodatabase/libsql-server:latest",
            "env_file": [f"./{name}.env"],
            "volumes": [f"{name}_sqldata:/var/lib/sqld"],
            # no curl/wget in the libsql-server image — bash /dev/tcp check
            # (found live via gwork; see db_service_work)
            "healthcheck": {
                "test": ["CMD-SHELL",
                         "bash -c 'exec 3<>/dev/tcp/127.0.0.1/8080' || exit 1"],
                "interval": "5s", "timeout": "3s", "retries": 20,
            },
        }
    return {
        "image": "docker.io/library/postgres:16-alpine",
        "env_file": [f"./{name}.env"],
        "volumes": [f"{name}_pgdata:/var/lib/postgresql/data"],
        "healthcheck": {
            "test": ["CMD-SHELL", "pg_isready -h 127.0.0.1"],
            "interval": "5s", "timeout": "3s", "retries": 10,
        },
    }


def node_service(name: str, node: str, central_host: int, tools: list,
                 tool_ports: dict, depends: dict) -> dict:
    svc = {
        "build": {"context": "..", "dockerfile": "deploy/Containerfile"},
        "env_file": [f"./{name}.env"],
        "environment": {
            "TOOLS_LIST": " ".join(tools),
            "MCP_NODE_NAME": node,
            "MCP_NODE_CENTRAL_URL": f"http://{node}:8200",
        },
        # plan F4/C2: bundle envs set EMBEDDING_PROVIDER=local — without the
        # host HF-cache mount the node cannot load BGE-M3. The compose
        # ${HOST_HF_CACHE} interpolation resolves at `up` time (custom_up
        # exports it); podman-compose appends override -f volumes (F17).
        "volumes": ["${HOST_HF_CACHE}:/root/.cache/huggingface"],
        "ports": [f"{central_host}:8200"]
        + [f"{tool_ports[t]}:{PORT_CONTAINER[t]}" for t in tools],
        "healthcheck": {
            "test": ["CMD", "python3", "/app/healthcheck.py"],
            "interval": "15s", "timeout": "5s", "retries": 5,
            "start_period": "60s",
        },
        "restart": "unless-stopped",
    }
    if depends:
        svc["depends_on"] = depends
    return svc


# ── work layout (grounds plan D2/D3: gwork grounds) ──────────────────────
# The work pod's shape: host networking on the CANONICAL ports (clients keep
# working unchanged), identity db published to host loopback (the node
# shares the HOST netns and cannot resolve podman DNS names — the publish is
# load-bearing, see plan §4 D8), pgvector for pg parity, work-shaped volumes.
# Data planes stay EPHEMERAL: the bundle override adds a container-local
# turso file, never a real-data mount.

WORK_DB_PORT = {"pg": 5433, "turso": 9080}


def db_service_work(name: str, topology: str) -> dict:
    if topology == "turso":
        return {
            "image": "ghcr.io/tursodatabase/libsql-server:latest",
            "env_file": [f"./{name}.env"],
            "ports": [f"127.0.0.1:{WORK_DB_PORT['turso']}:8080"],
            "volumes": [f"{name}_sqldata:/var/lib/sqld"],
            # the libsql-server image ships NO curl/wget — a curl healthcheck
            # can never pass (found live: compose up hung on service_healthy).
            # bash IS present, so use a /dev/tcp connect check.
            "healthcheck": {
                "test": ["CMD-SHELL",
                         "bash -c 'exec 3<>/dev/tcp/127.0.0.1/8080' || exit 1"],
                "interval": "5s", "timeout": "3s", "retries": 20,
            },
        }
    return {
        "image": "docker.io/pgvector/pgvector:pg16",
        "env_file": [f"./{name}.env"],
        "ports": [f"127.0.0.1:{WORK_DB_PORT['pg']}:5432"],
        "volumes": [f"{name}_pgdata:/var/lib/postgresql/data"],
        "healthcheck": {
            "test": ["CMD-SHELL", "pg_isready -h 127.0.0.1"],
            "interval": "5s", "timeout": "3s", "retries": 10,
        },
    }


def node_service_work(name: str, tools: list) -> dict:
    return {
        "image": "mcp-node:latest",
        "build": {"context": "..", "dockerfile": "deploy/Containerfile",
                  "args": {"IMAGE_VERSION": "${IMAGE_VERSION:-dev}",
                           "GIT_SHA": "${GIT_SHA:-unknown}"}},
        "network_mode": "host",
        "env_file": [f"./{name}.env"],
        "environment": {
            "TOOLS_LIST": " ".join(tools),
            "MCP_NODE_NAME": name,
            "MCP_NODE_CENTRAL_URL": "http://127.0.0.1:8200",
        },
        "volumes": [
            "${HOST_HF_CACHE}:/root/.cache/huggingface",
            f"{name}_artifacts:/root/.memorymcp/artifacts",
            f"{name}_logs:/app/logs",
        ],
        "healthcheck": {
            "test": ["CMD", "python3", "/app/healthcheck.py"],
            "interval": "15s", "timeout": "5s", "retries": 5,
            "start_period": "60s",
        },
        "restart": "unless-stopped",
        "depends_on": {"db": {"condition": "service_healthy"}},
    }


def generate(entry: dict) -> str:
    """Registry entry dict -> compose YAML string."""
    name = entry["name"]
    tools = entry.get("tools") or []
    ports = entry.get("ports") or {}
    central = ports.get("central")
    if not name or central is None:
        raise ValueError("registry entry needs name + ports.central")
    topology = entry.get("topology", "pg")

    if entry.get("layout") == "work":
        # work-shaped ground: host networking + canonical ports + published
        # identity db. Registry ports exist for attribution/probes only.
        services = {"db": db_service_work(name, topology),
                    "node1": node_service_work(name, tools)}
        volumes = {}
        if topology == "pg":
            volumes[f"{name}_pgdata"] = None
        if topology == "turso":
            volumes[f"{name}_sqldata"] = None
        # work-shaped node mounts declare named volumes for artifacts/logs
        # (podman-compose requires every named volume in the top-level dict)
        volumes[f"{name}_artifacts"] = None
        volumes[f"{name}_logs"] = None
        doc = {"name": name, "services": services, "volumes": volumes}
        return yaml.dump(doc, sort_keys=False, default_flow_style=False)

    tool_ports = {t: ports.get("tools", {}).get(t) for t in tools}
    missing = [t for t, p in tool_ports.items() if p is None]
    if missing:
        raise ValueError(f"ports.tools missing entries for: {', '.join(missing)}")

    services = {"db": db_service(name, topology)}
    # podman-compose expects the top-level volumes as a DICT (name -> spec);
    # a list makes fix_mount_dict crash ('list' object has no attribute 'get')
    volumes = {}
    if topology == "pg":
        volumes[f"{name}_pgdata"] = None
    if topology == "turso":
        volumes[f"{name}_sqldata"] = None

    services["node1"] = node_service(
        name, "node1", central, tools, tool_ports,
        {"db": {"condition": "service_started"}})
    if entry.get("nodes", 1) == 2:
        central2 = ports.get("central2")
        if central2 is None:
            raise ValueError("nodes=2 needs ports.central2")
        services["node2"] = node_service(
            name, "node2", central2, tools, tool_ports,
            {"node1": {"condition": "service_started"}})

    doc = {"name": name, "services": services, "volumes": volumes}
    return yaml.dump(doc, sort_keys=False, default_flow_style=False)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: gen_custom_compose.py <registry.json>", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        entry = json.load(f)
    try:
        sys.stdout.write(generate(entry))
    except (KeyError, ValueError) as e:
        print(f"gen_custom_compose: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
