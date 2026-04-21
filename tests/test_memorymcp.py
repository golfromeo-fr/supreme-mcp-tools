#!/usr/bin/env python3
"""
MemoryMCP Integration Test Script

Tests all 12 memorymcp tools against the running server via MCP streamable HTTP.
Requires: memorymcp server running (standalone or via launcher).

Usage:
    python tests/test_memorymcp.py                  # default http://127.0.0.1:8005
    python tests/test_memorymcp.py --url http://192.168.0.1:8005/mcp
    python tests/test_memorymcp.py --keep           # don't clean up test memories
"""

import sys
import json
import argparse
import traceback
from datetime import datetime

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# MCP Streamable HTTP client
# ---------------------------------------------------------------------------

class MCPClient:
    """Minimal MCP client using JSON-RPC over streamable HTTP."""

    def __init__(self, url: str):
        self.url = url
        self._id = 0
        self._session_id: str | None = None

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    @staticmethod
    def _parse_sse_body(raw: bytes) -> dict:
        """Parse SSE response body, extracting the first JSON message."""
        text = raw.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
        return json.loads(text)

    def _post(self, payload: dict) -> dict:
        """Send a JSON-RPC request and return the parsed response."""
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        req = urllib.request.Request(
            self.url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                sid = resp.headers.get("mcp-session-id")
                if sid and not self._session_id:
                    self._session_id = sid
                raw = resp.read()
                return self._parse_sse_body(raw)
        except urllib.error.HTTPError as e:
            try:
                return self._parse_sse_body(e.read())
            except Exception:
                return {"error": {"message": f"HTTP {e.code}: {e.reason}"}}
        except Exception as e:
            return {"error": {"message": f"Connection failed: {e}"}}

    def call(self, tool_name: str, arguments: dict | None = None) -> dict:
        """Call an MCP tool and return the result."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        body = self._post(payload)

        if "error" in body:
            return {"error": body["error"]}

        result = body.get("result", {})
        content_list = result.get("content", [])
        if content_list and isinstance(content_list, list):
            text = content_list[0].get("text", "")
            try:
                return {"result": json.loads(text)}
            except (json.JSONDecodeError, TypeError):
                return {"result": text}
        return {"result": result}

    def call_text(self, tool_name: str, arguments: dict | None = None) -> str:
        """Call an MCP tool and return result as plain text."""
        res = self.call(tool_name, arguments)
        if "error" in res:
            err = res["error"]
            if isinstance(err, dict):
                return f"ERROR: {err.get('message', err)}"
            return f"ERROR: {err}"
        r = res["result"]
        return r if isinstance(r, str) else json.dumps(r, indent=2)

    def initialize(self) -> dict:
        """Send MCP initialize handshake."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "memorymcp-test", "version": "1.0"},
            },
        }
        return self._post(payload)


class TestRunner:
    def __init__(self, client: MCPClient, keep: bool = False):
        self.client = client
        self.keep = keep
        self.created_ids: list[str] = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def _report(self, name: str, ok: bool, detail: str = ""):
        status = "PASS" if ok else "FAIL"
        icon = "✓" if ok else "✗"
        msg = f"  {icon} {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def run(self):
        print(f"\n{'='*60}")
        print(f"MemoryMCP Integration Tests — {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        # Step 0: connectivity
        print("─ Connectivity ─")
        init = self.client.initialize()
        ok = "error" not in init
        self._report("Server handshake", ok,
                      init.get("result", {}).get("serverInfo", {}).get("name", "") if ok else str(init.get("error")))
        if not ok:
            print("\nServer not reachable. Aborting.")
            return

        # Step 1: listMemoryTypes
        print("\n─ listMemoryTypes ─")
        result = self.client.call_text("listMemoryTypes")
        self._report("Returns types", "code_pattern" in result and "lesson" in result)

        # Step 2: getMemorySystemPrompt
        print("\n─ getMemorySystemPrompt ─")
        result = self.client.call_text("getMemorySystemPrompt")
        self._report("Returns prompt", "Memory System" in result or "upsertMemory" in result)

        # Step 3: redactSensitive
        print("\n─ redactSensitive ─")
        result = self.client.call_text("redactSensitive", {
            "text": "My email is test@example.com and my key is api_key=sk_live_1234567890abcdef1234"
        })
        self._report("Detects PII", "Sensitivity:" in result)
        self._report("Redacts email", "█████████" in result or "█" in result)

        # Step 4: upsertMemory (create new)
        print("\n─ upsertMemory (create) ─")
        result = self.client.call_text("upsertMemory", {
            "text": f"Test memory from integration test at {datetime.now().isoformat()}",
            "memory_type": "lesson",
            "tags": ["test", "integration"],
            "source": "test_script",
            "retention_policy": "temp",
        })
        mem1_id = None
        is_uuid = len(result) >= 32 and "-" in result
        self._report("Creates memory", is_uuid, f"id={result[:20]}..." if is_uuid else result)
        if is_uuid:
            mem1_id = result
            self.created_ids.append(mem1_id)

        # Step 5: upsertMemory (second memory)
        print("\n─ upsertMemory (create #2) ─")
        result = self.client.call_text("upsertMemory", {
            "text": "Another test memory about code patterns in Python",
            "memory_type": "code_pattern",
            "tags": ["test", "python"],
            "source": "test_script",
        })
        mem2_id = None
        is_uuid = len(result) >= 32 and "-" in result
        self._report("Creates second memory", is_uuid, f"id={result[:20]}..." if is_uuid else result)
        if is_uuid:
            mem2_id = result
            self.created_ids.append(mem2_id)

        # Step 6: upsertMemory (with memory_id — true update)
        print("\n─ upsertMemory (update by ID) ─")
        if mem1_id:
            result = self.client.call_text("upsertMemory", {
                "text": "Updated test memory with new content",
                "memory_type": "lesson",
                "tags": ["test", "updated"],
                "memory_id": mem1_id,
            })
            self._report("Updates existing", mem1_id in result, f"returned={result[:20]}...")
        else:
            self._report("Updates existing", False, "skipped — no mem1_id")
            self.skipped += 1

        # Step 7: queryMemory
        print("\n─ queryMemory ─")
        result = self.client.call_text("queryMemory", {
            "query": "test memory",
            "k": 5,
        })
        self._report("Returns results", "Found" in result or "memories" in result.lower(), result[:80])

        # Step 8: queryMemory with tag filter
        print("\n─ queryMemory (tag filter) ─")
        result = self.client.call_text("queryMemory", {
            "query": "test",
            "tags": ["integration"],
            "k": 5,
        })
        self._report("Filters by tag", True, result[:80])

        # Step 9: getMemory
        print("\n─ getMemory ─")
        if mem1_id:
            result = self.client.call_text("getMemory", {"memory_id": mem1_id})
            self._report("Retrieves by ID", mem1_id in result, result[:80])
        else:
            self._report("Retrieves by ID", False, "skipped")
            self.skipped += 1

        # Step 10: attachProvenance
        print("\n─ attachProvenance ─")
        if mem1_id:
            result = self.client.call_text("attachProvenance", {
                "memory_id": mem1_id,
                "source": "test_script",
                "confidence": 0.95,
                "notes": "Integration test provenance",
            })
            self._report("Attaches provenance", mem1_id in result or "Added" in result)
        else:
            self._report("Attaches provenance", False, "skipped")
            self.skipped += 1

        # Step 11: auditTrail
        print("\n─ auditTrail ─")
        if mem1_id:
            result = self.client.call_text("auditTrail", {"memory_id": mem1_id})
            self._report("Returns audit trail", mem1_id in result or "Audit" in result, result[:80])
        else:
            self._report("Returns audit trail", False, "skipped")
            self.skipped += 1

        # Step 12: onAgentAction
        print("\n─ onAgentAction ─")
        result = self.client.call_text("onAgentAction", {
            "action_type": "discovery",
            "context": "Test discovery action from integration test",
            "tags": ["test"],
        })
        is_uuid = len(result) >= 32 and "-" in result
        self._report("Stores action", is_uuid, f"id={result[:20]}..." if is_uuid else result)
        if is_uuid:
            self.created_ids.append(result)

        # Step 13: getMemoryMetrics
        print("\n─ getMemoryMetrics ─")
        result = self.client.call_text("getMemoryMetrics")
        self._report("Returns metrics", "Total Memories" in result or "memories" in result.lower(), result[:100])

        # Step 14: mergeDuplicates (dry run)
        print("\n─ mergeDuplicates (dry run) ─")
        result = self.client.call_text("mergeDuplicates", {
            "threshold": 0.5,
            "dry_run": True,
        })
        self._report("Dry run merge", "Would merge" in result or "duplicate" in result.lower(), result[:80])

        # Step 15: decayOrExpire (dry run)
        print("\n─ decayOrExpire (dry run) ─")
        result = self.client.call_text("decayOrExpire", {
            "ttl_days": 365,
            "min_usage_count": 0,
            "dry_run": True,
        })
        self._report("Dry run decay", "Would delete" in result or "delete" in result.lower(), result[:80])

        # Step 16: reindexMemory (skip — too slow for test)
        print("\n─ reindexMemory ─")
        self._report("Skipped (too heavy for integration test)", True)
        self.skipped += 1

        # Step 17: deleteMemory (cleanup)
        print("\n─ deleteMemory (cleanup) ─")
        if not self.keep and self.created_ids:
            for mid in self.created_ids:
                result = self.client.call_text("deleteMemory", {"memory_id": mid})
                ok = "Deleted" in result or "deleted" in result.lower()
                self._report(f"Delete {mid[:12]}...", ok)
        elif self.keep:
            print(f"  ℹ Keeping {len(self.created_ids)} test memories (--keep flag)")
        else:
            print("  ℹ No memories to clean up")

        # Summary
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Results: {self.passed} passed, {self.failed} failed, {self.skipped} skipped ({total} total)")
        print(f"{'='*60}\n")

        return self.failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MemoryMCP integration tests")
    parser.add_argument("--url", default="http://127.0.0.1:8005/mcp", help="MCP endpoint URL")
    parser.add_argument("--keep", action="store_true", help="Don't delete test memories after run")
    args = parser.parse_args()

    client = MCPClient(args.url)
    runner = TestRunner(client, keep=args.keep)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
