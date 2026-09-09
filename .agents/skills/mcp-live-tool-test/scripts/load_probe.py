#!/usr/bin/env python3
"""E3/load — concurrent-agent load probe for the SAFE MCP servers.

Mimics N concurrent agents (one session each, sequential calls) against
simplemcp, databasemcp, memorymcp and reports throughput + latency
percentiles, with a 1-agent serial baseline for the scaling factor.

- simplemcp : greet/double/square (trivial CPU)
- databasemcp: throwaway libSQL file DB — INSERT+SELECT churn through the
  full MCP surface (exercises executor threads, per-entry lock, to_thread)
- memorymcp : listMemories paging on the live store (READ-ONLY by design;
  queryMemory would inflate usage_count on every hit — not used)

webmcp/ragmcp deliberately excluded: external quotas / heavy local
embeddings (user decision 2026-09-08).

Usage:
  python load_probe.py --tools simplemcp,databasemcp,memorymcp \
                       --agents 5 --calls 10
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastmcp import Client  # noqa: E402
from fastmcp.client.auth import BearerAuth  # noqa: E402

PORTS = json.loads((ROOT / "config" / "ports.json").read_text())["assignments"]["mcp"]


def _key(tool: str) -> str:
    return json.loads(
        (ROOT / "tools" / tool / "config.json").read_text()
    )["auth"]["api_key"]


def _pct(sorted_lat: list[float], p: float) -> float:
    if not sorted_lat:
        return 0.0
    i = min(int(len(sorted_lat) * p), len(sorted_lat) - 1)
    return sorted_lat[i] * 1000


class _Probe:
    """Per-tool setup/call/teardown through the MCP surface itself."""

    def __init__(self, tool: str):
        self.tool = tool
        self.url = f"http://127.0.0.1:{PORTS[tool]}/mcp"
        self.key = _key(tool)

    async def setup(self) -> dict:
        """Tool-specific setup; returns the scenario context."""
        ctx: dict = {}
        if self.tool == "databasemcp":
            async with Client(self.url, auth=BearerAuth(self.key)) as c:
                await c.call_tool("connect_database", {
                    "name": "loadprobe", "db_type": "libsql",
                    "params": {"url": "file:/tmp/databasemcp_loadprobe.db"},
                })
                await c.call_tool("execute_sql", {
                    "sql": "CREATE TABLE IF NOT EXISTS load_t "
                           "(id INTEGER PRIMARY KEY, seq INTEGER)",
                    "connection": "loadprobe",
                })
            ctx["conn"] = "loadprobe"
        return ctx

    async def call(self, ctx: dict, i: int, client: Client,
                   conn_holder: dict | None = None) -> None:
        """One call over the agent's PERSISTENT session (real agents hold
        one session — per-call sessions would dominate the numbers).
        conn_holder overrides the connection name (per-agent connections
        variant: the documented multi-agent mitigation for databasemcp)."""
        conn = (conn_holder or {}).get("conn") or ctx.get("conn")
        if self.tool == "simplemcp":
            tool = ["double", "square", "greet"][i % 3]
            args = {"value": i} if tool != "greet" else {"name": f"load{i}"}
            await client.call_tool(tool, args)
        elif self.tool == "databasemcp":
            await client.call_tool("execute_sql", {
                "sql": f"INSERT INTO load_t VALUES ({i}, 'probe')",
                "connection": conn,
            })
            await client.call_tool("query", {
                "sql": "SELECT count(*) AS n FROM load_t",
                "connection": conn,
            })
        elif self.tool == "memorymcp":
            offset = (i * 20) % 120
            await client.call_tool("listMemories",
                                   {"limit": 20, "offset": offset})

    async def teardown(self, ctx: dict) -> None:
        if self.tool == "databasemcp":
            try:
                async with Client(self.url, auth=BearerAuth(self.key)) as c:
                    await c.call_tool("execute_sql", {
                        "sql": "DROP TABLE IF EXISTS load_t",
                        "connection": "loadprobe",
                    })
                    await c.call_tool("disconnect_database", {"name": "loadprobe"})
            except Exception as e:
                print(f"  teardown note: {type(e).__name__}: {e}")


async def _worker(url: str, key: str, probe: _Probe, ctx: dict,
                  calls: int, lat: list, err: list, offset: int = 0):
    async with Client(url, auth=BearerAuth(key)) as client:
        for i in range(calls):
            t0 = time.perf_counter()
            try:
                await probe.call(ctx, i + offset, client)
            except Exception as e:
                err.append(f"{type(e).__name__}: {str(e)[:80]}")
            lat.append(time.perf_counter() - t0)


async def _run_load(url: str, key: str, probe: _Probe, ctx: dict,
                    agents: int, calls: int) -> dict:
    lat: list[float] = []
    err: list[str] = []
    t0 = time.perf_counter()
    await asyncio.gather(*[
        _worker(url, key, probe, ctx, calls, lat, err, offset=k * calls)
        for k in range(agents)
    ])
    wall = time.perf_counter() - t0
    total = agents * calls
    s = sorted(lat)
    return {
        "total": total,
        "wall_s": round(wall, 2),
        "cps": round(total / wall, 1),
        "med_ms": round(median(lat) * 1000, 1) if lat else 0,
        "p95_ms": round(_pct(s, 0.95), 1),
        "max_ms": round(s[-1] * 1000, 1) if s else 0,
        "errors": len(err),
        "err_sample": err[:2],
    }


async def main(tools: list[str], agents: int, calls: int) -> int:
    failures = 0
    for tool in tools:
        probe = _Probe(tool)
        ctx = await probe.setup()
        try:
            if tool == "databasemcp" and "conn" not in ctx:
                print(f"{tool:<12} setup failed — skipping")
                failures += 1
                continue
            # serial baseline: 1 agent, same total call count
            base = await _run_load(probe.url, probe.key, probe, ctx, 1,
                                   min(agents * calls, 20))
            conc = await _run_load(probe.url, probe.key, probe, ctx,
                                   agents, calls)
            scaling = round(conc["cps"] / base["cps"], 2) if base["cps"] else 0
            print(f"{tool:<12} serial {base['cps']:>6} calls/s (med "
                  f"{base['med_ms']}ms) | {agents} agents: {conc['cps']:>6} "
                  f"calls/s, med {conc['med_ms']}ms p95 {conc['p95_ms']}ms "
                  f"max {conc['max_ms']}ms | scaling x{scaling} | "
                  f"errors {conc['errors']}")
            if tool == "databasemcp":
                # the documented mitigation: per-agent connections instead
                # of one shared entry (which serializes on the entry lock)
                async def _connect_named(name: str, db_file: str):
                    async with Client(probe.url, auth=BearerAuth(probe.key)) as c:
                        await c.call_tool("connect_database", {
                            "name": name, "db_type": "libsql",
                            "params": {"url": f"file:{db_file}"}})

                conns = []
                for k in range(agents):
                    name = f"loadp{k}"
                    db_file = f"/tmp/databasemcp_loadprobe_{k}.db"
                    await _connect_named(name, db_file)
                    conns.append((name, db_file))
                conc2 = await _run_load(probe.url, probe.key, probe, ctx,
                                        agents, calls)
                print(f"{'':12} per-agent connections: {conc2['cps']:>6} "
                      f"calls/s (was {conc['cps']} shared) — scaling x"
                      f"{round(conc2['cps'] / base['cps'], 2)}")
                for _n, f in conns:
                    Path(f).unlink(missing_ok=True)
            if conc["errors"]:
                for e in conc["err_sample"]:
                    print(f"    err: {e}")
                failures += 1
        finally:
            await probe.teardown(ctx)
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", default="simplemcp,databasemcp,memorymcp")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--calls", type=int, default=10,
                        help="calls per agent (concurrent run)")
    args = parser.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    allowed = {"simplemcp", "databasemcp", "memorymcp"}
    bad = [t for t in tools if t not in allowed]
    if bad:
        print(f"refusing: {bad} — the probe covers {sorted(allowed)} only "
              "(webmcp=quotas, ragmcp=heavy)")
        sys.exit(2)
    sys.exit(asyncio.run(main(tools, args.agents, args.calls)))
