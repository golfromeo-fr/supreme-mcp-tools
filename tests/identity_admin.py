"""Environment-agnostic identity management for integration tests.

Manages test users through the CENTRAL API (/api/users...) of whatever
environment holds the canonical ports — host launcher, work pod, or any
node — instead of writing the host users.json directly. This makes the
e35 suite able to test ANY environment without touching host state.

Auto-detection of the identity plane (host-shared vs isolated):
after creating a user via the central API, we check whether the HOST
users.json sees it. If yes, the target shares the host identity plane
(host launcher) and file-level test sections may run; if no, the target
has its own plane (work pod) and file sections must be skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

HOST_STORE = (
    Path(os.environ.get("MCP_USERS_STORE"))
    if os.environ.get("MCP_USERS_STORE")
    else Path.home() / ".config" / "supreme-mcp-tools" / "users.json"
)


class IdentityAdmin:
    """Central-API user lifecycle for integration tests."""

    def __init__(self, base_url: str = "http://127.0.0.1:8200",
                 api_key: str | None = None, timeout: float = 15.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("MCP_MANAGEMENT_API_KEY")
        if not self.api_key:
            # test processes run outside the launcher — read the root .env
            try:
                from dotenv import dotenv_values
                self.api_key = dotenv_values(
                    Path(__file__).resolve().parents[1] / ".env"
                ).get("MCP_MANAGEMENT_API_KEY")
            except Exception:
                pass
        self.timeout = timeout
        self._created: list[str] = []
        self.shares_host_identity: bool | None = None  # detected on create

    # -- low level -------------------------------------------------------
    def _req(self, method: str, path: str, json_body: dict | None = None) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = httpx.request(method, f"{self.base}{path}", json=json_body,
                          headers=headers, timeout=self.timeout)
        if r.status_code >= 500:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:200]}")
        try:
            return {"status": r.status_code, "body": r.json()}
        except Exception:
            return {"status": r.status_code, "body": {}}

    # -- lifecycle -------------------------------------------------------
    def create(self, username: str, password: str, role: str = "user",
               servers: list[str] | None = None, **grants) -> dict:
        """Create via central API; returns the record incl. the once-only
        mcp_key. 400 'already exists' -> adopt silently (leftover from a
        crashed run): rotate to regain a known key."""
        body = {"username": username, "password": password, "role": role,
                "servers": servers or []}
        r = self._req("POST", "/api/users", body)
        if r["status"] == 400 and "already exists" in str(r["body"]):
            self.rotate(username)
            for setter, arg in (
                ("/servers", servers or []),
                ("/masked-functions", grants.get("masked_functions", {})),
                ("/db-presets", grants.get("db_presets", [])),
                ("/rag-collections", {"collections": grants.get("rag_collections", [])}),
                ("/role", role),
            ):
                self._req("PUT", f"/api/users/{username}{setter}", arg)
            rec = self.get(username)
        elif r["status"] != 200:
            raise RuntimeError(f"create {username} failed: {r['body']}")
        else:
            rec = r["body"]
            if rec.get("mcp_key"):
                self._keys[username] = rec["mcp_key"]
            # fresh create: apply the grants the create endpoint doesn't take
            if servers:
                self.set_servers(username, servers)
            if grants.get("db_presets"):
                self.set_db_presets(username, grants["db_presets"])
            if grants.get("masked_functions"):
                self.set_masked(username, grants["masked_functions"])
        self._created.append(username)
        if self.shares_host_identity is None:
            self.shares_host_identity = self._host_file_sees(username)
        return rec

    def get(self, username: str) -> dict:
        users = self.list()
        for u in users:
            if u.get("username") == username:
                rec = dict(u)
                # list_users masks mcp_key; recover from create-time cache
                if "mcp_key" not in rec or not rec.get("mcp_key"):
                    rec["mcp_key"] = self._keys.get(username)
                return rec
        raise RuntimeError(f"user '{username}' not visible via central API")

    _keys: dict[str, str] = {}

    def list(self) -> list[dict]:
        r = self._req("GET", "/api/users")
        return r["body"].get("users", []) if r["status"] == 200 else []

    def delete(self, username: str) -> None:
        self._req("DELETE", f"/api/users/{username}")
        self._keys.pop(username, None)

    def rotate(self, username: str) -> dict:
        r = self._req("POST", f"/api/users/{username}/rotate-key")
        key = r["body"].get("mcp_key") if r["status"] == 200 else None
        if key:
            self._keys[username] = key
        return {"mcp_key": key}

    def set_role(self, username: str, role: str) -> None:
        self._req("PUT", f"/api/users/{username}/role", {"role": role})

    def set_enabled(self, username: str, enabled: bool) -> None:
        self._req("POST", f"/api/users/{username}/enabled",
                  {"enabled": enabled})

    def set_servers(self, username: str, servers: list[str]) -> None:
        self._req("PUT", f"/api/users/{username}/servers", {"servers": servers})

    def set_rag_collections(self, username: str, collections: list[str]) -> None:
        self._req("PUT", f"/api/users/{username}/rag-collections",
                  {"collections": collections})

    def set_masked(self, username: str, masked: dict) -> None:
        self._req("PUT", f"/api/users/{username}/masked-functions",
                  {"masked_functions": masked})

    def set_db_presets(self, username: str, presets: list[str]) -> None:
        self._req("PUT", f"/api/users/{username}/db-presets",
                  {"presets": presets})

    # -- host-plane detection --------------------------------------------
    def _host_file_sees(self, username: str) -> bool:
        try:
            users = json.loads(HOST_STORE.read_text())
            users = users.get("users", users)
            return username in users
        except Exception:
            return False

    def cleanup(self) -> None:
        for u in self._created:
            try:
                self.delete(u)
            except Exception:
                pass
        self._created.clear()

    def key(self, record_or_name) -> str:
        if isinstance(record_or_name, dict):
            return record_or_name.get("mcp_key", "")
        return self.get(record_or_name).get("mcp_key", "")
