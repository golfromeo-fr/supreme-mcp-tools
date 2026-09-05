"""C7 — bounded child shutdown (restart race, third variant).

A dying launcher used to be able to hold tool ports past the next launcher's
20s port-retry window: ``stop_all_servers`` awaited each uvicorn shutdown with
no deadline, so one stuck server (e.g. pinned by a live SSE stream) stalled
the whole teardown. These tests pin the new contract:

1. A server whose ``shutdown()`` never returns cannot stall
   ``stop_all_servers`` beyond the per-server deadline.
2. After real uvicorn servers stop, their ports are rebindable immediately —
   exactly what the next launcher's port probe checks.
3. The deadlines are actually wired: ``timeout_graceful_shutdown`` reaches the
   uvicorn config and the launchmcp watchdog keeps its forcible exit.
"""

import asyncio
import socket
import sys
import time
from pathlib import Path
from unittest import mock

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher.server_manager import (  # noqa: E402
    GRACEFUL_SHUTDOWN_TIMEOUT,
    TASK_CANCEL_TIMEOUT,
    ServerInstance,
    ServerManager,
)


def _run(coro):
    # Local event loop, never asyncio.run(): asyncio.run() leaves
    # set_event_loop(None) behind, which breaks legacy get_event_loop()
    # helpers in later test files (prior art: tests/test_era_negotiation.py).
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _trivial_app(scope, receive, send):
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _real_instance(name: str) -> ServerInstance:
    port = _free_port()
    config = uvicorn.Config(
        app=_trivial_app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    return ServerInstance(
        tool_name=name,
        port=port,
        app=_trivial_app,
        server_config=config,
        server=server,
    )


def _hung_instance(name: str) -> ServerInstance:
    """Instance whose uvicorn shutdown never completes (stuck connection)."""
    instance = _real_instance(name)

    async def hung_shutdown():
        await asyncio.sleep(3600)

    instance.server = mock.Mock(started=True, shutdown=hung_shutdown)
    return instance


class TestC7HungShutdownBounded:
    """One stuck server must not stall the whole teardown."""

    def test_stuck_server_cannot_stall_stop_all_servers(self, monkeypatch):
        from launcher import server_manager as sm

        monkeypatch.setattr(sm, "GRACEFUL_SHUTDOWN_TIMEOUT", 0.5)
        monkeypatch.setattr(sm, "TASK_CANCEL_TIMEOUT", 0.5)

        manager = ServerManager()
        hung = _hung_instance("hung")
        manager.servers["hung"] = hung

        started = time.monotonic()
        _run(manager.stop_all_servers())
        elapsed = time.monotonic() - started

        assert elapsed < sm.GRACEFUL_SHUTDOWN_TIMEOUT * 2 + 1, (
            f"stop_all_servers took {elapsed:.1f}s with a hung server — "
            "deadlines not enforced"
        )
        assert manager.servers["hung"].status == "stopped"


class TestC7RealPortRelease:
    """After stop_all_servers, every tool port is rebindable right away."""

    def test_ports_rebindable_immediately_after_stop(self):
        async def scenario():
            manager = ServerManager()
            instances = {}
            for name in ("tool_a", "tool_b"):
                inst = _real_instance(name)
                manager.servers[name] = inst
                manager.tasks[name] = asyncio.create_task(manager._run_server(inst))
                instances[name] = inst
            for inst in instances.values():
                for _ in range(100):
                    if inst.server.started:
                        break
                    await asyncio.sleep(0.05)
                assert inst.server.started, f"{inst.tool_name} never started"
            await manager.stop_all_servers()
            return instances

        instances = _run(scenario())

        # The next launcher's probe: bind the same port again, immediately.
        for inst in instances.values():
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", inst.port))


class TestC7DeadlinesWired:
    """The deadlines are actually plumbed into uvicorn and the entry point."""

    def test_uvicorn_config_sets_graceful_timeout(self):
        source = (PROJECT_ROOT / "launcher/server_manager.py").read_text()
        assert "timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT" in source

    def test_launchmcp_has_forcible_exit_watchdog(self):
        source = (PROJECT_ROOT / "launchmcp.py").read_text()
        assert "os._exit(1)" in source
        assert "stop_all_servers" in source

    def test_deadlines_are_ordered_inside_the_port_retry_window(self):
        # stop_all_servers' worst case (3s cancel + 5s drain, concurrent) must
        # stay under launchmcp's 15s watchdog and the next launcher's 20s
        # port-retry window.
        assert TASK_CANCEL_TIMEOUT + GRACEFUL_SHUTDOWN_TIMEOUT < 15
