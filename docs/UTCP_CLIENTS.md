# UTCP clients — consuming the tools over the fourth dialect

Every tool serves UTCP alongside MCP (`GET /utcp` manual, `POST /utcp/call/<tool>`,
same port, same auth). One generated config serves every client shape:

```bash
./scripts/setup_utcp_client.sh          # writes (outside the repo, no gitignore needed):
#   ~/.config/supreme-mcp-tools/utcp-client.json   — secret-free client config
#   ~/.config/supreme-mcp-tools/utcp-client.env   — the API keys (dotenv-loaded)
```

The config lists all manuals (`http://127.0.0.1:<port>/utcp`, `X-API-Key` auth
template, placeholder `${UTCP_<NAME>_API_KEY}`) plus the `tool_repository` /
`tool_search_strategy` the reference clients expect. Keys live in the sibling
.env (`load_variables_from` dotenv loader) under BOTH the plain variable name
and the provider-namespaced name (`simplemcp_UTCP_...`) — reference clients
resolve manual placeholders under a `<manual>_` scope (live-proven on the
Python client).

## 1. MCP-only harness → `@utcp/mcp-bridge`

For harnesses that can only register MCP servers (Kilo Code, Claude Desktop,
most GUI clients): register the bridge as ONE stdio MCP server and the whole
tool set arrives through it — testing the non-MCP path from an MCP-only
client, zero per-tool server entries. Verified contract (bridge README):
exposes `register_manual`, `deregister_manual`, `call_tool`, `search_tools`,
`list_tools`, `get_required_keys_for_tool`, `tool_info`.

```json
{
  "mcpServers": {
    "utcp": {
      "command": "npx",
      "args": ["-y", "@utcp/mcp-bridge"],
      "env": { "UTCP_CONFIG_FILE": "<absolute path — ~ is NOT expanded in env values>" }
    }
  }
}
```

Notes:
- The harness sees the 7 bridge tools (call_tool/search_tools/...), NOT 64
  individual functions — the agent then calls e.g.
  `call_tool(tool_name="simplemcp.square", arguments={"body": {"value": 5}})`
  (arg is `tool_name`; verified live 2026-11-04 over stdio MCP: all 7 bridge
  tools listed, square(6) → 36 through the bridge). That indirection is the
  code-mode trade-off: fewer context tokens, one extra hop per call.
- Use an ABSOLUTE path for UTCP_CONFIG_FILE (env values don't expand `~`).
- Traffic really flows over UTCP: the server-side `mcp.access` log shows
  `GET /utcp` + `POST /utcp/call/<tool>` lines (with E3 `user=` attribution),
  never `POST /mcp`.

## 2. Shell-capable agents → `@utcp/code-mode-cli`

```bash
cd <dir with .utcp_config.json>   # copy or symlink the generated config there
npx -y @utcp/code-mode-cli prompt  # built-in agent guide
npx -y @utcp/code-mode-cli search "memory"          # discover + TS interfaces
npx -y @utcp/code-mode-cli run <<'EOF'
const r = await simplemcp.square({ body: { value: 5 } });
const m = await memorymcp.getMemoryMetrics({});
return { square: r, memory: typeof m };
EOF
```

(CLI is v0.1.1 — subcommands verified against its README, but untested against
our servers; treat as experimental.)

## 3. Python clients

```python
from utcp.utcp_client import UtcpClient
client = await UtcpClient.create(config="/home/<you>/.config/supreme-mcp-tools/utcp-client.json")
tools = await client.search_tools("square")          # word-match strategy — use real terms
result = await client.call_tool("simplemcp.square", {"body": {"value": 5}})
# tool names are "<manual>.<tool>"; args go under "body" (HttpCallTemplate.body_field)
```

Live-verified 2026-11-04 against the running launcher: 6 manuals registered
(64 tools), search + `simplemcp.square` → `25.0`, `databasemcp.query` honest
tool-error surfaced.

Note: the config pins the tools' MCP ports from `config/ports.json`. Those
ports are pinned and fail loudly if taken (unlike mgmt ports, which
auto-allocate) — but if a port ever shifts, re-run the generator and restart
the client; same staleness property as any hardcoded MCP client entry.

## 4. Testing your UTCP client setup

**Automated (suite):** `python -m pytest tests/test_utcp_client_integration.py -v`
boots an ephemeral multi-transport app on a real port and drives it with the
official Python UTCP client — manual registration from `GET /utcp` (auth
template + `${VAR}` placeholder resolution), discovery + word-match search,
`{"body": ...}` calls, honest 400 surfacing, and bad-key rejection. No
launcher needed; needs the `utcp` pins from requirements.txt (client-only).

**Quick live probes** (launcher running; key = `tools/<name>/config.json`
`auth.api_key`):

```bash
# manual (discoverability)
curl -s -H "X-API-Key: $KEY" http://127.0.0.1:8002/utcp | python3 -m json.tool
# expect 401 without a key
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8002/utcp
```

```bash
# call route — args go under "body"
curl -s -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"body":{"value":5}}' http://127.0.0.1:8002/utcp/call/square
```

**Python client against the generated config:**

```python
import asyncio
from utcp.utcp_client import UtcpClient

async def main():
    client = await UtcpClient.create(
        config="/home/<you>/.config/supreme-mcp-tools/utcp-client.json")
    print(await client.search_tools("square", 10))
    print(await client.call_tool("simplemcp.square", {"body": {"value": 6}}))
    await client.close()

asyncio.run(main())
```

Gotchas (2026-11-17 live + suite-proven):
- `call_tool` returns a plain dict (`{"result": "36.0"}` — string numbers);
  there is no `list_tools` — use `search_tools("", N)` to list everything.
- Missing/invalid args → HTTP 400 surfaces as a client exception carrying the
  pydantic detail; masked tools never register → client-side `Tool not found`.
- Manual (tool) names with underscores: the reference client doubles
  underscores when deriving variable scopes (every `_` doubles:
  `foo_bar` → `foo__bar`). Keep tool names underscore-free (all six are); the
  generator (setup_utcp_client.sh) handles the doubling since 2026-11-17.
- The server-side proof that traffic rode UTCP: `mcp.access` log shows
  `GET /utcp` + `POST /utcp/call/<tool>` lines — never `POST /mcp`.
