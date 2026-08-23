---
name: mcp-live-tool-test
description: Live-test the MCP tool servers by having the agent itself call every exposed function through its native mcp__<server>__<function> tools — the same path a real harness client uses. Use whenever the user asks to "test the MCP tools", "do tool calls", verify tools after a launcher restart or upgrade, check that servers/functions work, or says "are the tools up". The agent must make the calls itself, as many functions as possible — not one lazy call per server, and not a Python script doing the calls.
---

# Live MCP tool testing (agent-driven)

The agent makes real MCP tool calls through its native `mcp__<server>__<function>`
tools. That exercises the full client path (harness MCP client → auth → session →
server → tool), which is the thing the user actually wants validated. Scripts and
raw HTTP are only for checks a native call cannot express (see "Complementary
HTTP checks") — never a substitute for native calls.

## Procedure

1. **Enumerate what is exposed.** Scan the session's available tools for
   `mcp__<server>__*`. Cross-check the expected function set against the repo:
   `tools/<name>/config.json` → `"tools"` section lists each tool's name and
   input schema. A server may be running but not wired into this session —
   say so explicitly if a configured server has no native tools.

2. **Call every function** — the user's expectation is a full sweep, not one
   call per server. Safe argument choices:
   - deterministic math/string tools → assert the exact expected value
     (`double(21)` must return 42)
   - fetch/search tools → cheap, harmless targets (`https://example.com`,
     query "smoke test")
   - exercise optional parameters on at least one call (e.g. a `greeting`
     override, a pagination `start_index`) — optional paths break too
   - **skip** mutation-sounding functions (`delete*`, `upsert*`, `create*`,
     `execute*`, `write*`, …) unless the user explicitly asks for them

3. **Verify each result**: a response came back, no error flag, plausible
   content. For deterministic tools, compare the exact value.

4. **Report a table**: function | arguments used | result | PASS/FAIL/SKIP,
   then a one-line summary (X passed / Y failed / Z skipped). Flag anything
   that is an upstream error rather than a server defect.

## Known failure mode: `Session not found` (code -32600)

The harness client holds a **stale `Mcp-Session-Id`** — typically after a
launcher restart or someone calling `POST /admin/flush-sessions` (it terminates
ALL live sessions, including the harness's own). The server answers 404 +
`{"error":{"code":-32600,"message":"Session not found"}}` and ZCode does not
auto-reinitialize mid-turn (see repo root `zcode-mcp-session-churn-bug.md`:
ZCode reconnects at task boundaries, ~4 parallel initializers per server).

Recovery — the CLIENT must re-initialize, nothing server-side fixes it:
1. Usually the **next user turn** (task boundary) re-establishes sessions.
2. Otherwise the user restarts the ZCode session / reconnects MCP servers.
3. Do not retry-loop the same call. Report the cause, ask for a new turn or
   reconnect, then verify recovery with one cheap deterministic call first
   (e.g. `square`) before resuming the sweep.

When testing flush-sessions recovery deliberately: do it LAST, after all
native sweeps are done, and warn that it kills the harness's own sessions.

## Expected arguments (current repo tools)

| Function | Safe args | Deterministic? |
|---|---|---|
| `mcp__simplemcp__double` | `{"value": 21}` | yes → 42 |
| `mcp__simplemcp__square` | `{"value": 7}` | yes → 49 |
| `mcp__simplemcp__greet` | `{"name": "smoke", "greeting": "Howdy"}` | yes → "Howdy, smoke!" |
| `mcp__simplemcp__get_secret` | `{}` | returns the SIMPLEMCP_SECRET fixture |
| webmcp `brave_search_api` | `{"query": "...", "count": 3}` | 429 = upstream rate limit, not a defect |
| webmcp `google_search_api` | `{"query": "..."}` | no |
| webmcp `post_url` | `https://httpbin.org/post` + small JSON body | echoed payload |
| webmcp `fetch_url` | `{"url": "https://example.com"}` | assert: status 200 AND body contains "Example Domain" AND no U+FFFD replacement chars |

For functions not listed: read the input schema from the session's tool
definition or the tool's `config.json`, pick minimal safe args, and skip with
a note if none are safe.

### Known webmcp defect (as of 2026-08-23)

`webmcp` stores origin response bodies verbatim without honoring
`Content-Encoding`, so gzip-compressed answers surface as mojibake/binary:

- **Exclude `brave_search_web` from sweeps** — its scraped HTML comes back
  compressed garbage intermittently (reproduced on FastMCP 3.4.2 and
  4.0.0b3). Re-add after the decode fix lands.
- The `fetch_url` assertions above are the regression probe for the same
  root cause: they currently FAIL (example.com arrives as ~318 raw bytes).
  A failure there means the known bug still exists — do not report it as a
  FastMCP-version or transport regression.

Fixing the decode handling in `webmcp_fastmcp.py` flips both: re-add
`brave_search_web` and expect `fetch_url` green.

## Companion scripts (run these instead of retyping probes)

Both run against a launcher you started, using the fastmcp >=4 client in the
active interpreter; ports/keys resolve from repo config automatically.

- **`scripts/sweep_all.py`** — full safe-function sweep of every running
  server (`--only simplemcp,webmcp` to narrow). Encodes the expectations
  table above: brave_search_web excluded, fetch_url asserted and reported as
  KNOWN-DEFECT while the gzip bug exists. Exit code = unexpected failures,
  so it is scriptable.
- **`scripts/probe_eras.py [server] [--wire]`** — simulates a NEWEST-era MCP
  client (fastmcp default mode) plus a legacy control on one server;
  `--wire` additionally shows raw HTTP behavior (Mcp-Session-Id issuance,
  stateless vs stateful transport).

```bash
python .agents/skills/mcp-live-tool-test/scripts/sweep_all.py
python .agents/skills/mcp-live-tool-test/scripts/probe_eras.py simplemcp --wire
```

## Complementary HTTP checks (only when explicitly wanted)

Native calls cannot express these; use a short inline `python` + `httpx`
snippet (repo has httpx):

- **Auth enforced**: POST `/mcp` `initialize` without auth header → expect
  HTTP 401. A 200 here is an auth hole — report immediately.
- **Flush recovery**: initialize (get session id) → `POST /admin/flush-sessions`
  with `X-API-Key` → old session id must now get HTTP 404 → fresh initialize
  yields a new id. Remember: this terminates all live sessions of that tool.

Ports come from `config/ports.json` (`assignments.mcp`), keys from
`tools/<name>/config.json` (`auth.api_key`).
