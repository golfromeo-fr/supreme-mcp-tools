"""Tier-2 real-pod smoke probes — plans/testing-grounds-and-work-topology-2026-10-05.md §3.5.

Runs ONLY against a testing ground (a disposable real pod booted from a
pinned bundle via ./startgrounds). Opt-in:

    POD_GROUNDS=g1-pg python -m pytest tests/test_pod_smoke.py -v
    # or: ./startgrounds probe g1-pg

Unset POD_GROUNDS -> everything skips, so the default suite stays hermetic.
These are the checks fixtures CANNOT make (tier-1 has no access to
container-boundary facts): image contents, mounts, published ports, engine
reachability, identity import, embedding-model availability.

Probe facts below were verified live on the first g1-pg boot (2026-10-05):
- tool servers speak JSON-RPC on /mcp-stateless, X-API-Key auth, 401 unauth
- the memory roundtrip needs the ephemeral data plane (file: turso inside
  the container) — with no vector backend the tools degrade to silent no-ops
  (NullVectorStore), which is why the plane is provisioned by the override
- masks from the bundle (webmcp: brave_search_web, simplemcp: get_secret)
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import requests

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not os.environ.get("POD_GROUNDS", "").strip(),
    reason="tier-2 ground not requested (set POD_GROUNDS=<name>)",
)

# canonical function per server — the floor oracle (R3): counts rot, one
# anchor function name per server is the stable minimum
CANONICAL = {
    "simplemcp": "double",
    "webmcp": "fetch_url",
    "convertermcp": "convert_docx_to_text",
    "databasemcp": "list_presets",
    "ragmcp": "list_collections",
    "memorymcp": "listMemoryTypes",
}
FLOOR = {"simplemcp": 3, "webmcp": 3, "convertermcp": 1,
         "databasemcp": 10, "ragmcp": 5, "memorymcp": 20}
# per-server floors: convertermcp genuinely ships ONE function; databasemcp
# (18) / memorymcp (26) / ragmcp (12) have large stable surfaces — observed
# live on g1-pg 2026-10-05

# one SAFE call per server (no external writes; webmcp/convertermcp calls
# need network/doc fixtures — covered by tools/list, documented here)
SAFE_CALLS = {
    "simplemcp": ("double", {"value": 21}, "42"),
    "databasemcp": ("list_presets", {}, None),
    "ragmcp": ("list_collections", {}, None),
    "memorymcp": ("listMemoryTypes", {}, None),
}


def _rpc(tool, key, method, params=None, port=None, timeout=60):
    """One JSON-RPC exchange against the ground's stateless endpoint.
    Returns (http_status, parsed-json-or-None)."""
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    r = requests.post(
        f"http://127.0.0.1:{port}/mcp-stateless",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream",
                 "X-API-Key": key},
        json=body, timeout=timeout)
    data = None
    for line in r.text.splitlines():
        if line.startswith("data: "):
            try:
                msg = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if msg.get("id") == 1:
                data = msg
    return r.status_code, data


def _registry():
    name = os.environ["POD_GROUNDS"].strip()
    return json.loads(
        (REPO / "deploy" / "clusters.d" / f"{name}.json").read_text())


def _tool_key(tool):
    bundle = Path(_registry().get("bundle", ""))
    return json.loads(
        (bundle / "tools" / tool / "config.json").read_text())["auth"]["api_key"]


def _result_payload(d):
    """unwrap fastmcp's structuredContent.result when present."""
    res = (d or {}).get("result", {})
    sc = res.get("structuredContent", {})
    if isinstance(sc, dict) and "result" in sc:
        return sc["result"]
    return res


# ── probe 1: every ground container Up and healthy ───────────────────────

def test_probe_01_all_containers_healthy(pod_ground):
    ps = subprocess.run(
        ["podman", "ps", "--filter", f"name={pod_ground.name}_",
         "--format", "{{.Names}}|{{.Status}}"],
        capture_output=True, text=True)
    rows = dict(l.split("|", 1) for l in ps.stdout.splitlines() if "|" in l)
    assert rows, "no containers running for the ground"
    for name, status in rows.items():
        assert status.startswith("Up"), f"{name}: {status}"
        assert "healthy" in status, f"{name}: {status} (healthcheck pending?)"


# ── probe 2: auth posture — unauth rejected, keyed accepted ─────────────

def test_probe_02_auth_gates(pod_ground):
    for tool, port in pod_ground.tool_ports.items():
        key = _tool_key(tool)
        st_unauth, _ = _rpc(tool, "wrong-key", "tools/list", {}, port=port)
        assert st_unauth == 401, f"{tool}: unauthenticated request was not rejected"
        st_keyed, d = _rpc(tool, key, "tools/list", {}, port=port)
        assert st_keyed == 200, f"{tool}: keyed request failed: {st_keyed}"
        assert "result" in (d or {}), f"{tool}: no JSON-RPC result"


def test_probe_02b_central_health(pod_ground):
    r = requests.get(f"http://127.0.0.1:{pod_ground.central}/health", timeout=10)
    assert r.status_code == 200


# ── probe 3: tools/list floor + canonical function (R3 oracle) ───────────

def test_probe_03_tools_list_floor_and_canonical(pod_ground):
    for tool, port in pod_ground.tool_ports.items():
        key = _tool_key(tool)
        st, d = _rpc(tool, key, "tools/list", {}, port=port)
        assert st == 200, f"{tool}: tools/list failed"
        names = [t["name"] for t in d["result"]["tools"]]
        assert len(names) >= FLOOR[tool], \
            f"{tool}: only {len(names)} functions: {names}"
        anchor = CANONICAL[tool]
        assert anchor in names, \
            f"{tool}: canonical function '{anchor}' missing from {names}"


# ── probe 4: one safe call per server that has one ───────────────────────

def test_probe_04_safe_calls(pod_ground):
    for tool, (fn, args, expect) in SAFE_CALLS.items():
        key = _tool_key(tool)
        st, d = _rpc(tool, key, "tools/call",
                     {"name": fn, "arguments": args},
                     port=pod_ground.tool_ports[tool])
        assert st == 200, f"{tool}.{fn}: HTTP {st}"
        res = d.get("result", {})
        assert not res.get("isError"), f"{tool}.{fn}: tool error: {res}"
        if expect is not None:
            assert expect in json.dumps(_result_payload(d)), \
                f"{tool}.{fn}: expected '{expect}' in result"


# ── probe 5: the embedding model loads INSIDE the container (F4 catcher) ─

def test_probe_05_embedding_model_loads_in_container(pod_ground):
    node = f"{pod_ground.name}_node1_1"
    r = subprocess.run(
        ["podman", "exec", node, "python3", "-c",
         "from sentence_transformers import SentenceTransformer;\n"
         "m = SentenceTransformer('BAAI/bge-m3');\n"
         "print('DIM', m.encode(['ground smoke probe']).shape)"],
        capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, \
        f"embedding load failed in-container: {r.stderr[-500:]}"
    assert "DIM (1, 1024)" in r.stdout, r.stdout[-200:]


# ── probe 6: the state plane is real SQL and reachable ───────────────────

def _db_scalar(pod_ground, sql):
    """Run a scalar SQL on the ground's state plane — pg via psql in the db
    container; turso via libsql from node1 (the db is sqld, published on the
    host loopback the host-networked node shares)."""
    if pod_ground.topology == "pg":
        r = subprocess.run(
            ["podman", "exec", f"{pod_ground.name}_db_1",
             "psql", "-U", "mcp", "-d", "mcp", "-tAc", sql],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()
    node = f"{pod_ground.name}_node1_1"
    py = ("import libsql_experimental as l;\n"
          f"c = l.connect('http://127.0.0.1:{pod_ground.ports_db}');\n"
          f"print(c.execute({sql!r}).fetchone()[0])")
    r = subprocess.run(["podman", "exec", node, "python3", "-c", py],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"sqld query failed: {r.stderr[-400:]}"
    return r.stdout.strip()


def test_probe_06_state_plane_tables(pod_ground):
    if pod_ground.topology == "pg":
        out = _db_scalar(pod_ground,
                         "SELECT to_regclass('mcp_state_docs') IS NOT NULL AND "
                         "to_regclass('mcp_users_store') IS NOT NULL")
        assert out == "t", "identity/state tables missing from the ground db"
    else:
        out = _db_scalar(pod_ground,
                         "SELECT count(*) FROM sqlite_master WHERE name IN "
                         "('mcp_state_docs','mcp_users_store')")
        assert int(out) == 2,             f"identity/state tables missing from sqld (count={out})"


# ── probe 7: identity import actually took the db path (R9 oracle) ───────

def test_probe_07_identity_import_in_db(pod_ground):
    # users.json was mounted RO into node1 and imported ONCE at first boot —
    # assert the DATA is in SQL and the node never fell back to JSON
    n = _db_scalar(pod_ground, "SELECT count(*) FROM mcp_users_store")
    assert int(n or "0") >= 1, \
        "mcp_users_store empty — identity import did not run"
    logs = subprocess.run(
        ["podman", "logs", f"{pod_ground.name}_node1_1"],
        capture_output=True, text=True)
    combined = logs.stdout + logs.stderr
    assert "staying on users.json" not in combined, \
        "node fell back to the JSON users store — the db path silently failed"


# ── probe 8: function masks from the bundle are enforced ─────────────────

def test_probe_08_masks_enforced(pod_ground):
    masks = {"webmcp": "brave_search_web", "simplemcp": "get_secret"}
    for tool, masked in masks.items():
        key = _tool_key(tool)
        st, d = _rpc(tool, key, "tools/list", {},
                     port=pod_ground.tool_ports[tool])
        assert st == 200, f"{tool}: tools/list failed"
        names = [t["name"] for t in d["result"]["tools"]]
        assert masked not in names, \
            f"{tool}: masked function '{masked}' is visible in tools/list"


# ── probe 9: memory data-plane roundtrip (upsert → get → query → delete) ─

def test_probe_09_memory_roundtrip(pod_ground):
    key = _tool_key("memorymcp")
    port = pod_ground.tool_ports["memorymcp"]

    def call(fn, args):
        st, d = _rpc("memorymcp", key, "tools/call",
                     {"name": fn, "arguments": args}, port=port)
        assert st == 200, f"{fn}: HTTP {st}"
        return _result_payload(d)

    mid = call("upsertMemory", {
        "text": "tier2 ground smoke probe memory",
        "memory_type": "fact",
        "tags": ["ground-smoke"],
        "source": "ground-probe"})
    assert mid and "error" not in str(mid).lower(), \
        f"upsert failed: {mid} — data plane degraded?"

    got = call("getMemory", {"memory_id": mid})
    assert "not found" not in str(got).lower(), \
        f"upserted memory invisible to getMemory — split data plane: {got}"

    hits = call("queryMemory", {"query": "smoke probe", "k": 3})
    assert "No memories found" not in str(hits), \
        f"semantic query found nothing — vector plane degraded: {hits}"

    deleted = call("deleteMemory", {"memory_id": mid})
    assert "Deleted" in str(deleted), f"delete failed: {deleted}"
    gone = call("getMemory", {"memory_id": mid})
    assert "not found" in str(gone).lower(), "memory survived delete"


# ── probe 10: attribution — we are testing THE ground, nothing else ──────

def test_probe_10_attribution(pod_ground):
    reg = _registry()
    assert reg["ports"]["tools"] == pod_ground.tool_ports, \
        "probed tool ports drifted from the registry — wrong target?"
    ps = subprocess.run(
        ["podman", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    mine = [n for n in ps.stdout.splitlines()
            if n.startswith(f"{pod_ground.name}_")]
    assert mine, "no containers with the ground's project prefix are running"
    assert f"{pod_ground.name}_node1_1" in mine


# ── probe 11: two-node grounds share the identity plane ──────────────────

def test_probe_11_cross_node_identity(pod_ground):
    """The multihost invariant at ground scale: both nodes resolve identity
    against the ONE shared state plane, so the same admin key authenticates
    on node1 AND node2 (g2 grounds only — single-node grounds skip)."""
    if not pod_ground.node2_ports:
        pytest.skip("single-node ground — cross-node probes need --nodes 2")
    for tool, port in pod_ground.node2_ports.items():
        key = _tool_key(tool)
        st, d = _rpc(tool, key, "tools/list", {}, port=port)
        assert st == 200, f"node2 {tool}: keyed call failed (HTTP {st})"
        assert "result" in (d or {}), f"node2 {tool}: no JSON-RPC result"
    # and the two nodes are genuinely different servers
    n1 = pod_ground.tool_ports["simplemcp"]
    n2 = pod_ground.node2_ports["simplemcp"]
    assert n1 != n2
