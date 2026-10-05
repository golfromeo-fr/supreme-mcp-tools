"""Automated identity-plane switch round trip on WORK-LAYOUT grounds —
the Phase-2b switch, re-proven by the suite on every run instead of by a
manual rehearsal (the user does not work on the pods; see TODO.md).

Opt-in like the smoke probes:
    POD_GROUNDS=gwork-pg python -m pytest tests/test_pod_switch.py -v
Skips on anything that is not a work-layout ground. Leaves the ground on
its ORIGINAL topology (self-cleaning): pg → turso → pg.
"""

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


def _startgrounds(*args, check=True):
    r = subprocess.run([str(REPO / "startgrounds"), *args],
                       capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stderr + r.stdout
    return r


def _status(name):
    import json
    return json.loads(_startgrounds("status", name).stdout)["grounds"][0]


def test_switch_round_trip(pod_ground):
    if pod_ground.topology != "pg":
        pytest.skip("round trip starts from pg")
    entry = __import__("json").loads(
        (REPO / "deploy" / "clusters.d" / f"{pod_ground.name}.json").read_text())
    if entry.get("layout") != "work":
        pytest.skip("switch is a work-layout ground operation")

    envf = REPO / "deploy" / f"{pod_ground.name}.env"
    pg_pw = [l.split("=", 1)[1] for l in envf.read_text().splitlines()
             if l.startswith("POSTGRES_PASSWORD=")][0]

    # pg → turso: sqld serves identity, users re-import from the bundle
    _startgrounds("switch", pod_ground.name, "turso")
    health = _status(pod_ground.name)
    assert health["topology"] == "turso" and health["running"]
    assert health["ports"]["db"] == 9080

    # turso → pg: the DORMANT block revives — same password, same users
    _startgrounds("switch", pod_ground.name, "pg")
    health = _status(pod_ground.name)
    assert health["topology"] == "pg" and health["running"]
    assert health["ports"]["db"] == 5433
    pg_pw_after = [l.split("=", 1)[1] for l in envf.read_text().splitlines()
                   if l.startswith("POSTGRES_PASSWORD=")][0]
    assert pg_pw_after == pg_pw, \
        "pg password changed across the round trip — pgdata would be bricked"

    # the revived identity plane actually authenticates (central health gate
    # is passed by startgrounds switch; the tools answer with the bundle key)
    import json as _json
    key = _json.loads(Path(pod_ground.bundle, "tools", "simplemcp",
                           "config.json").read_text())["auth"]["api_key"]
    r = requests.post("http://127.0.0.1:8002/mcp-stateless",
                      headers={"Content-Type": "application/json",
                               "Accept": "application/json, text/event-stream",
                               "X-API-Key": key},
                      json={"jsonrpc": "2.0", "id": 1,
                            "method": "tools/list"}, timeout=30)
    assert r.status_code == 200, f"keyed call failed after round trip: {r.status_code}"
