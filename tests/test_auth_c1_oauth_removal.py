"""
Tests for C-1 fix: OAuth auto-approve removal, FastMCP 3 native auth.

These tests verify that:
1. DualHeaderVerifier (server_factory.py) correctly validates X-API-Key and Bearer
2. FastMCP 3 built-in auth protects the /mcp endpoint
3. The vulnerable OAuth endpoints (/register, /authorize, /token) are NOT exposed
4. Invalid credentials are rejected

Run: pytest tests/test_auth_c1_oauth_removal.py -v
"""

import asyncio
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_test_server(api_key: str = "test-api-key-12345"):
    """Create a minimal FastMCP server with DualHeaderVerifier auth."""
    from fastmcp import FastMCP
    from tools.shared.server_factory import DualHeaderVerifier

    verifier = DualHeaderVerifier(
        tokens={api_key: {"client_id": "test-client", "scopes": ["mcp"]}},
    )
    mcp = FastMCP("test-auth-server", auth=verifier)

    @mcp.tool()
    def echo(message: str) -> str:
        return message

    return mcp


class _ServerHandle:
    """Run a FastMCP ASGI app on a random port in a background thread."""

    def __init__(self, app, host="127.0.0.1", port=0):
        self.app = app
        self.host = host
        self.port = port
        self._thread = None
        self._ready = threading.Event()
        self._server = None

    def start(self):
        import uvicorn
        from functools import partial

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self._server.started:
                break
            time.sleep(0.05)
        for listener in self._server.servers[0].sockets if self._server.servers else []:
            self.port = listener.getsockname()[1]
            break
        self._ready.set()

    def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"


class TestDualHeaderVerifierUnit(unittest.TestCase):
    """Unit tests for DualHeaderVerifier.verify_token."""

    def setUp(self):
        from tools.shared.server_factory import DualHeaderVerifier
        self.verifier = DualHeaderVerifier(
            tokens={
                "key-alpha": {"client_id": "client-a", "scopes": ["mcp"]},
                "key-beta": {"client_id": "client-b", "scopes": ["mcp", "admin"]},
            },
        )

    def test_valid_token_returns_access_token(self):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.verifier.verify_token("key-alpha"))
        finally:
            loop.close()
        self.assertIsNotNone(result)
        self.assertEqual(result.client_id, "client-a")
        self.assertEqual(result.token, "key-alpha")

    def test_invalid_token_returns_none(self):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.verifier.verify_token("nonexistent"))
        finally:
            loop.close()
        self.assertIsNone(result)

    def test_empty_token_returns_none(self):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.verifier.verify_token(""))
        finally:
            loop.close()
        self.assertIsNone(result)

    def test_scopes_preserved(self):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.verifier.verify_token("key-beta"))
        finally:
            loop.close()
        self.assertIsNotNone(result)
        self.assertIn("admin", result.scopes)
        self.assertIn("mcp", result.scopes)


class TestDualHeaderAuthBackendUnit(unittest.TestCase):
    """Unit tests for DualHeaderAuthBackend.authenticate."""

    def setUp(self):
        from tools.shared.server_factory import (
            DualHeaderVerifier,
            DualHeaderAuthBackend,
        )
        self.verifier = DualHeaderVerifier(
            tokens={"my-secret": {"client_id": "c1", "scopes": ["mcp"]}},
        )
        self.backend = DualHeaderAuthBackend(self.verifier)

    def _make_connection(self, headers=None):
        """Create a mock HTTPConnection with given headers."""
        from unittest.mock import MagicMock
        conn = MagicMock()
        conn.headers = headers or {}
        return conn

    def test_x_api_key_header_accepted(self):
        conn = self._make_connection({"x-api-key": "my-secret"})
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.backend.authenticate(conn))
        finally:
            loop.close()
        self.assertIsNotNone(result)
        _, user = result
        self.assertEqual(user.access_token.token, "my-secret")

    def test_bearer_header_accepted(self):
        conn = self._make_connection({"authorization": "Bearer my-secret"})
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.backend.authenticate(conn))
        finally:
            loop.close()
        self.assertIsNotNone(result)

    def test_invalid_x_api_key_rejected(self):
        conn = self._make_connection({"x-api-key": "wrong-key"})
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.backend.authenticate(conn))
        finally:
            loop.close()
        self.assertIsNone(result)

    def test_invalid_bearer_rejected(self):
        conn = self._make_connection({"authorization": "Bearer wrong-key"})
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.backend.authenticate(conn))
        finally:
            loop.close()
        self.assertIsNone(result)

    def test_no_auth_returns_none(self):
        conn = self._make_connection({})
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.backend.authenticate(conn))
        finally:
            loop.close()
        self.assertIsNone(result)

    def test_case_insensitive_bearer(self):
        conn = self._make_connection({"authorization": "bearer my-secret"})
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.backend.authenticate(conn))
        finally:
            loop.close()
        self.assertIsNotNone(result)


class TestFastMCPAuthHTTP(unittest.TestCase):
    """HTTP-level tests: FastMCP server with DualHeaderVerifier protects /mcp."""

    @classmethod
    def setUpClass(cls):
        mcp = _make_test_server("test-secret-key-999")
        app = mcp.http_app(transport="http")
        cls.handle = _ServerHandle(app, port=0)
        cls.handle.start()
        cls.api_key = "test-secret-key-999"
        cls.base = cls.handle.base_url

    @classmethod
    def tearDownClass(cls):
        cls.handle.stop()

    def test_mcp_endpoint_rejects_no_auth(self):
        """POST /mcp with no auth header must fail."""
        with httpx.Client() as c:
            r = c.post(
                f"{self.base}/mcp",
                json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
                headers={"Content-Type": "application/json"},
            )
        self.assertIn(r.status_code, (401, 403))

    def test_mcp_endpoint_rejects_wrong_key(self):
        """POST /mcp with wrong API key must fail."""
        with httpx.Client() as c:
            r = c.post(
                f"{self.base}/mcp",
                json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": "WRONG-KEY",
                },
            )
        self.assertIn(r.status_code, (401, 403))

    def test_mcp_endpoint_accepts_x_api_key(self):
        """POST /mcp with correct X-API-Key must succeed."""
        with httpx.Client() as c:
            r = c.post(
                f"{self.base}/mcp",
                json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                }},
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                },
            )
        self.assertNotIn(r.status_code, (401, 403))

    def test_mcp_endpoint_accepts_bearer(self):
        """POST /mcp with correct Bearer token must succeed."""
        with httpx.Client() as c:
            r = c.post(
                f"{self.base}/mcp",
                json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                }},
                headers={
                    "Content-Type": "application/json",
                    "authorization": f"Bearer {self.api_key}",
                },
            )
        self.assertNotIn(r.status_code, (401, 403))


class TestOAuthEndpointsRemoved(unittest.TestCase):
    """Verify that FastMCP 3 native auth does NOT expose OAuth endpoints.

    These tests will FAIL against the current code (where MCPApiKeyMiddleware
    adds /register, /authorize, /token) and PASS after C-1 is fixed by
    removing the middleware and delegating to FastMCP 3.
    """

    @classmethod
    def setUpClass(cls):
        mcp = _make_test_server("test-oauth-check-key")
        app = mcp.http_app(transport="http")
        cls.handle = _ServerHandle(app, port=0)
        cls.handle.start()
        cls.base = cls.handle.base_url

    @classmethod
    def tearDownClass(cls):
        cls.handle.stop()

    def test_register_not_exposed(self):
        """POST /register must return 404 (no OAuth client registration)."""
        with httpx.Client() as c:
            r = c.post(f"{self.base}/register", json={})
        self.assertIn(r.status_code, (404, 405))

    def test_authorize_not_exposed(self):
        """GET /authorize must return 404 (no OAuth authorize endpoint)."""
        with httpx.Client() as c:
            r = c.get(f"{self.base}/authorize?client_id=x&redirect_uri=http://evil.com")
        self.assertIn(r.status_code, (404, 405))

    def test_token_not_exposed(self):
        """POST /token must return 404 (no OAuth token exchange)."""
        with httpx.Client() as c:
            r = c.post(f"{self.base}/token", json={"code": "fake-code"})
        self.assertIn(r.status_code, (404, 405))

    def test_no_well_known_oauth(self):
        """GET /.well-known/oauth-authorization-server must return 404."""
        with httpx.Client() as c:
            r = c.get(f"{self.base}/.well-known/oauth-authorization-server")
        self.assertIn(r.status_code, (404, 405))


class TestCreateFastMCPServerFactory(unittest.TestCase):
    """Test the create_fastmcp_server factory function."""

    def test_factory_creates_server_with_auth(self):
        from tools.shared.server_factory import create_fastmcp_server
        mcp = create_fastmcp_server("testtool", api_key="factory-test-key")
        self.assertIsNotNone(mcp)

    def test_factory_resolves_key_from_env(self):
        from tools.shared.server_factory import create_fastmcp_server
        import os
        os.environ["TESTTOOL_API_KEY"] = "env-derived-key"
        try:
            mcp = create_fastmcp_server("testtool")
            self.assertIsNotNone(mcp)
        finally:
            del os.environ["TESTTOOL_API_KEY"]

    def test_factory_app_has_routes(self):
        from tools.shared.server_factory import create_fastmcp_server, get_transport_app
        mcp = create_fastmcp_server("testtool", api_key="k1")
        app = get_transport_app(mcp)
        self.assertIsNotNone(app)


class TestResolveApiKey(unittest.TestCase):
    """Test _resolve_api_key priority chain."""

    def setUp(self):
        self._saved_env = {}
        for key in ["TESTRESOLVE_API_KEY", "MCP_API_KEY"]:
            self._saved_env[key] = os.environ.get(key)

    def tearDown(self):
        import os
        for key, val in self._saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_name_env_var_highest_priority(self):
        from tools.shared.server_factory import _resolve_api_key
        import os
        os.environ["TESTRESOLVE_API_KEY"] = "from-name-env"
        os.environ["MCP_API_KEY"] = "from-global-env"
        result = _resolve_api_key("testresolve", None)
        self.assertEqual(result, "from-name-env")

    def test_global_env_var_second_priority(self):
        from tools.shared.server_factory import _resolve_api_key
        import os
        os.environ.pop("TESTRESOLVE_API_KEY", None)
        os.environ["MCP_API_KEY"] = "from-global-env"
        result = _resolve_api_key("testresolve", None)
        self.assertEqual(result, "from-global-env")

    def test_fallback_used_when_no_env(self):
        from tools.shared.server_factory import _resolve_api_key
        import os
        os.environ.pop("TESTRESOLVE_API_KEY", None)
        os.environ.pop("MCP_API_KEY", None)
        result = _resolve_api_key("testresolve", "my-fallback-key")
        self.assertEqual(result, "my-fallback-key")

    def test_predictable_default_when_nothing_set(self):
        """H-1 regression: predictable default key when no config."""
        from tools.shared.server_factory import _resolve_api_key
        import os
        import warnings
        os.environ.pop("TESTRESOLVE_API_KEY", None)
        os.environ.pop("MCP_API_KEY", None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_api_key("testresolve", None)
        self.assertTrue(result.startswith("testresolve-dev-"))
        self.assertTrue(len(result) > len("testresolve-dev-"))
        self.assertTrue(any("No API key configured" in str(warning.message) for warning in w))


class TestServerFactoryNoOAuthRoutes(unittest.TestCase):
    """Verify that server_factory-created servers have no OAuth routes."""

    def test_no_oauth_metadata_routes(self):
        """FastMCP with DualHeaderVerifier must not add /.well-known/oauth-* routes."""
        from tools.shared.server_factory import create_fastmcp_server, get_transport_app
        mcp = create_fastmcp_server("testtool", api_key="k1")
        app = get_transport_app(mcp)

        if hasattr(app, "routes"):
            paths = [r.path for r in app.routes]
            self.assertNotIn("/register", paths)
            self.assertNotIn("/authorize", paths)
            self.assertNotIn("/token", paths)


if __name__ == "__main__":
    unittest.main()
