"""Pinned-mgmt-port fallback (2026-09-07 databasemcp first-run failure).

databasemcp is the only tool with a pinned mgmt port (8110). When a stale
holder outlives the busy-retry window, the tool used to FAIL for the whole
launcher run ("first run databasemcp doesn't show up, second run is ok").
A pinned mgmt port is only a preference — its URL reaches consumers via the
service registry — so a timed-out pin now degrades to corridor allocation.
Pinned MCP ports still fail loudly (clients pin those by URL).
"""

import socket
import unittest

from launcher.errors import PortConflictError
from launcher.port_manager import PortManager
import launcher.port_manager as pm_mod


class _PortHolder:
    """Hold a port the same way uvicorn does (bind on DEFAULT_HOST)."""

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.listen(1)

    def stop(self):
        self.sock.close()


class TestPinnedMgmtFallback(unittest.TestCase):
    def _config(self):
        # synthetic ranges so tests never touch the real 8xxx space
        return {
            "ranges": {"mcp": [9000, 9099], "mgmt": [9100, 9199], "system": [9200, 9299]},
            "reserved": {"central_management": 9200},
            "assignments": {"mcp": {"probe": 9010}, "mgmt": {"probe": 9110}},
        }

    def setUp(self):
        # keep the busy-retry window short in tests
        self._orig_secs = pm_mod.PORT_BUSY_RETRY_SECS
        pm_mod.PORT_BUSY_RETRY_SECS = 1.5

    def tearDown(self):
        pm_mod.PORT_BUSY_RETRY_SECS = self._orig_secs

    def test_pinned_mgmt_busy_degrades_to_corridor(self):
        holder = _PortHolder(9110)
        try:
            pm = PortManager(ports_config=self._config(), mode="manual")
            port = pm.allocate_port("probe", "mgmt")  # must NOT raise
        finally:
            holder.stop()
        self.assertEqual(pm.tool_ports["probe"], port)
        self.assertGreaterEqual(port, 9100)
        self.assertLessEqual(port, 9199)
        self.assertNotEqual(port, 9110)  # the busy pinned port was skipped

    def test_pinned_mgmt_free_gets_the_pin(self):
        pm = PortManager(ports_config=self._config(), mode="manual")
        port = pm.allocate_port("probe", "mgmt")
        self.assertEqual(port, 9110)

    def test_pinned_mcp_busy_still_raises_loudly(self):
        """MCP ports are pinned in client configs — never silently degraded."""
        holder = _PortHolder(9010)
        try:
            pm = PortManager(ports_config=self._config(), mode="manual")
            with self.assertRaises(PortConflictError):
                pm.allocate_port("probe", "mcp")
        finally:
            holder.stop()


if __name__ == "__main__":
    unittest.main()
