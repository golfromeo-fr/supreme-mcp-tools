"""Unit pin for deploy/gen_custom_compose.py — Phase 1.1 of
plans/testing-grounds-and-work-topology-2026-10-05.md (C5: the custom-cluster
path had ZERO coverage before the first ground boots on it).

The generator is a pure function of the registry JSON (deploy/clusters.d/
<name>.json) -> compose YAML string. Pins here cover the shapes startgrounds
relies on; port VALUES come from the registry (created by startcluster
create, pinned in test_startcluster_create.py).
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deploy"))

import gen_custom_compose as gcc  # noqa: E402


def make_entry(**over):
    """A registry entry as startcluster create writes it (post +2 offset fix):
    central=base, tools at base+2..base+7 by canonical container port."""
    e = {
        "name": "g1-pg",
        "nodes": 1,
        "topology": "pg",
        "tools": ["databasemcp", "webmcp", "simplemcp", "convertermcp",
                  "ragmcp", "memorymcp"],
        "port_base": 20000,
        "ports": {
            "central": 20000,
            "central2": None,
            "tools": {"databasemcp": 20002, "webmcp": 20003,
                      "simplemcp": 20004, "convertermcp": 20005,
                      "ragmcp": 20006, "memorymcp": 20007},
        },
    }
    e.update(over)
    return e


def gen(entry_dict):
    return yaml.safe_load(gcc.generate(entry_dict))


# ── db service shapes ────────────────────────────────────────────────────

def _hc_cmd(db):
    return " ".join(db["healthcheck"]["test"])


def test_pg_db_shape():
    doc = gen(make_entry())
    db = doc["services"]["db"]
    assert db["image"] == "docker.io/library/postgres:16-alpine"
    assert db["env_file"] == ["./g1-pg.env"]
    assert db["volumes"] == ["g1-pg_pgdata:/var/lib/postgresql/data"]
    assert "pg_isready" in _hc_cmd(db)


def test_turso_db_shape():
    doc = gen(make_entry(name="g1-turso", topology="turso"))
    db = doc["services"]["db"]
    assert db["image"] == "ghcr.io/tursodatabase/libsql-server:latest"
    assert db["env_file"] == ["./g1-turso.env"]
    assert db["volumes"] == ["g1-turso_sqldata:/var/lib/sqld"]
    # the libsql image has no curl/wget — the healthcheck must be the bash
    # /dev/tcp connect check (found live via gwork, Phase 2a)
    assert "/dev/tcp/127.0.0.1/8080" in _hc_cmd(db)
    assert "curl" not in _hc_cmd(db)


def test_volumes_are_dict_not_list():
    """podman-compose's fix_mount_dict crashes on a top-level volume LIST
    ('list' object has no attribute 'get') — the generator must emit a dict."""
    raw = yaml.safe_load(gcc.generate(make_entry()))
    assert isinstance(raw["volumes"], dict)
    assert raw["volumes"]["g1-pg_pgdata"] is None
    raw_turso = yaml.safe_load(gcc.generate(make_entry(topology="turso")))
    assert isinstance(raw_turso["volumes"], dict)
    assert "g1-pg_pgdata" not in raw_turso["volumes"]


# ── node service ─────────────────────────────────────────────────────────

def test_node_service_shape():
    doc = gen(make_entry())
    n1 = doc["services"]["node1"]
    assert n1["environment"]["TOOLS_LIST"] == \
        "databasemcp webmcp simplemcp convertermcp ragmcp memorymcp"
    assert n1["environment"]["MCP_NODE_NAME"] == "node1"
    assert n1["environment"]["MCP_NODE_CENTRAL_URL"] == "http://node1:8200"
    assert n1["env_file"] == ["./g1-pg.env"]


def test_node_publishes_central_plus_each_tool_on_canonical_container_port():
    doc = gen(make_entry())
    ports = doc["services"]["node1"]["ports"]
    # central on the registry's central, each tool on its canonical in-container port
    assert "20000:8200" in ports
    assert "20002:8000" in ports   # databasemcp
    assert "20003:8001" in ports   # webmcp
    assert "20004:8002" in ports   # simplemcp
    assert "20005:8003" in ports   # convertermcp
    assert "20006:8004" in ports   # ragmcp
    assert "20007:8005" in ports   # memorymcp


def test_node_carries_hf_cache_mount():
    """F4/C2: bundle envs set EMBEDDING_PROVIDER=local — without the host
    HF-cache mount the node cannot load BGE-M3 (the probe exists to catch
    exactly this, but the shape must be right first)."""
    doc = gen(make_entry())
    assert "${HOST_HF_CACHE}:/root/.cache/huggingface" \
        in doc["services"]["node1"]["volumes"]


def test_single_node_has_no_node2_and_depends_on_db_started():
    doc = gen(make_entry())
    assert "node2" not in doc["services"]
    assert doc["services"]["node1"]["depends_on"] == \
        {"db": {"condition": "service_started"}}


def test_two_nodes_shape():
    e = make_entry(nodes=2,
                   ports={"central": 20000, "central2": 21000,
                          "tools": {"databasemcp": 20002, "webmcp": 20003,
                                    "simplemcp": 20004, "convertermcp": 20005,
                                    "ragmcp": 20006, "memorymcp": 20007}})
    doc = gen(e)
    n2 = doc["services"]["node2"]
    assert n2["environment"]["MCP_NODE_NAME"] == "node2"
    assert n2["environment"]["MCP_NODE_CENTRAL_URL"] == "http://node2:8200"
    assert "21000:8200" in n2["ports"]
    assert n2["depends_on"] == {"node1": {"condition": "service_started"}}


# ── error cases ──────────────────────────────────────────────────────────

def test_missing_central_raises():
    e = make_entry()
    del e["ports"]["central"]
    with pytest.raises(ValueError, match="central"):
        gcc.generate(e)


def test_missing_tool_port_raises_naming_the_tool():
    e = make_entry()
    del e["ports"]["tools"]["ragmcp"]
    with pytest.raises(ValueError, match="ragmcp"):
        gcc.generate(e)


def test_missing_ports_block_raises():
    with pytest.raises(ValueError):
        gcc.generate({"name": "x", "ports": {}})


def test_generated_yaml_round_trips_through_compose_shape():
    """Smoke: the raw string is valid compose-ish YAML with name+services."""
    raw = gcc.generate(make_entry())
    doc = yaml.safe_load(raw)
    assert doc["name"] == "g1-pg"
    assert set(doc["services"]) == {"db", "node1"}


# ── work layout (grounds plan §4 D7/D8 — gwork grounds) ──────────────────

def make_work_entry(**over):
    e = make_entry(name="gwork-pg", layout="work", nodes=1,
                   ports={"central": 8200, "central2": None, "db": 5433,
                          "tools": {"databasemcp": 8000, "webmcp": 8001,
                                    "simplemcp": 8002, "convertermcp": 8003,
                                    "ragmcp": 8004, "memorymcp": 8005}})
    e.update(over)
    return e


def test_work_layout_pg_shape():
    doc = gen(make_work_entry())
    n1 = doc["services"]["node1"]
    assert n1["network_mode"] == "host"
    assert "ports" not in n1  # host networking publishes nothing
    assert n1["environment"]["MCP_NODE_CENTRAL_URL"] == "http://127.0.0.1:8200"
    assert n1["depends_on"] == {"db": {"condition": "service_healthy"}}
    vols = " ".join(n1["volumes"])
    assert "huggingface" in vols and "artifacts" in vols and "/app/logs" in vols
    db = doc["services"]["db"]
    assert db["image"] == "docker.io/pgvector/pgvector:pg16"  # work parity
    assert db["ports"] == ["127.0.0.1:5433:5432"]  # the publish is LOAD-BEARING
    assert doc["volumes"]["gwork-pg_pgdata"] is None


def test_work_layout_turso_shape():
    doc = gen(make_work_entry(name="gwork-turso", topology="turso"))
    db = doc["services"]["db"]
    assert db["image"] == "ghcr.io/tursodatabase/libsql-server:latest"
    assert db["ports"] == ["127.0.0.1:9080:8080"]
    assert doc["volumes"]["gwork-turso_sqldata"] is None


def test_custom_layout_unchanged_by_work_branch():
    """The default custom shape must be untouched by the layout branch."""
    doc = gen(make_entry())
    assert "network_mode" not in doc["services"]["node1"]
    assert doc["services"]["node1"]["ports"][0] == "20000:8200"


def test_two_nodes_node2_ports_offset_by_1000():
    """Regression (Phase 3): node2 used to republish node1's tool ports —
    both nodes could never coexist. node2 = node1 + 1000 (mirrors central2)."""
    doc = gen(make_entry(nodes=2, tools=["databasemcp", "simplemcp"],
                         ports={"central": 20000, "central2": 21000,
                                "tools": {"databasemcp": 20002,
                                          "simplemcp": 20004}}))
    n1p = doc["services"]["node1"]["ports"]
    n2p = doc["services"]["node2"]["ports"]
    assert "20002:8000" in n1p and "20004:8002" in n1p
    assert "21002:8000" in n2p and "21004:8002" in n2p
    assert not (set(n1p) & set(n2p)), "node1/node2 publish the same host port"
