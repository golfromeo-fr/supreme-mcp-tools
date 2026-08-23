# MCP `2026-07-28` Stateless Protocol — FastMCP 4 Upgrade Reference & Plan

*Created 2026-08-11. Reviewed 2026-08-11 against current `pip index versions` and the actual
codebase. Re-reviewed 2026-08-14: status refreshed (beta now 4.0.0b3), blast radius corrected,
and Phase -1 (legacy SSE removal) added — see Appendix B. Part 1 = status/reference (re-check
weekly); Part 2 = the upgrade plan; Appendix A = the 2026-08-11 audit; Appendix B = the
2026-08-14 re-review audit.*

---

## PART 1 — STATUS & REFERENCE

### TL;DR / Decision

- **FastMCP 4.0 is in beta now** — latest pre-release `4.0.0b3` (2026-08-14; b1 Jul 28, b2 Aug 7, i.e.
  ~weekly cadence); stable `4.0.0` is likely **weeks away**. `pip index versions fastmcp` shows only
  *stable* releases, which is why it lists 3.4.7 as latest (still the latest **stable** as of
  2026-08-14 — trigger condition #1 remains unmet).
- **The lower-level MCP Python SDK is already GA at 2.0.0** (verified 2026-08-11: `pip index versions mcp`
  → `LATEST: 2.0.0`; still latest on 2026-08-14 — v1.x ended at 1.29.0, critical fixes only). So the
  wire-protocol side of this upgrade is unblocked today; only FastMCP's higher-level wrapper is still
  in beta. Two facts from the official 4.0 upgrade guide (checked 2026-08-14): `mcp.types` moved to a
  standalone `mcp_types` package **but is re-exported as `mcp.types`** — existing imports keep
  working; and the legacy-era client knob is `Client(..., mode="legacy")`, not `protocol_version=`.
- **FastMCP 4 serves BOTH protocol eras from one server** (auto-negotiates per client): the sessionless
  `2026-07-28` era for new clients AND the old `initialize` handshake for legacy clients. So an upgrade
  does **not** break ZCode/opencode/Claude Code — they keep working on the handshake era. This resolves
  the “upgrading is pointless until clients catch up” worry: one server handles everyone.
- **Our environment is already 4.0-ready** (pydantic 2.13.4 ≥ 2.12, fastapi 0.135.1 ≥ 0.133,
  starlette ≥ 1.0.1). No floor bumps needed.
- **Our blast radius is tiny but not literally one file** (see Appendix A). The active hot file is
  `tools/shared/server_factory.py`. A second file, `tools/shared/utils.py`, also references FastMCP
  internals (`from fastmcp import FastMCP`, `FastMCP(name)`, `self.mcp_instance.http_app()`) but
  that code is currently dead — *nothing* in the repo imports `FastMCPBase`/`create_mcp_instance`/
  `setup_common_logging`/`get_tool_logger`. Only `is_internal_url` is used. Plan recommends deleting
  the dead FastMCP surface in the same change. The 2026-08-14 re-review found three more `mcp`-SDK
  importers (`launcher/tool_discovery.py` SSE tier, dead `launcher/streamable_http/` package, dead
  `tools/convertermcp/convertermcp.py`) — all eliminated by the new **Phase -1** cleanup (Appendix B),
  which restores the port's blast radius to `server_factory.py` + `utils.py`.

### FastMCP vs `mcp` — the relationship

```
your code  →  fastmcp  (high-level framework, jlowin/Prefect, Apache-2.0)
                  ↓ depends on / wraps
             mcp     (official MCP Python SDK, modelcontextprotocol/python-sdk)
                  ↓
             wire protocol (2026-07-28 / 2025-11-25)
```

- **`mcp`** = official Anthropic SDK. Originally *derived from* FastMCP 1.0; FastMCP then continued as a
  richer wrapper **on top** of it. Confirmed in-repo: `fastmcp` imports `from mcp` and pulls the SDK in
  via its server extra.
- **`fastmcp`** = ergonomic layer (`@mcp.tool`, `Client`, `Context`, auth, proxy). FastMCP **3.x → mcp 1.x**
  (us today); FastMCP **4.0 → mcp 2.0** (SDK v2).
- ⚠️ Don’t confuse with the **TypeScript** `fastmcp` npm package (punkpeye) — different project, no 2026-07-28.

### Is the code readable / mature?

Yes. Apache-2.0, readable Python on GitHub (**PrefectHQ/fastmcp**), ~1M+ daily downloads, stewarded by
Prefect (maintainer = Prefect CEO), full docs + executable upgrade tests, ~3–4 week minor cadence. 4.0
beta shipped alongside the spec.

### Project migration risk (verified by code scan)

| Migration item (from the 4.0 guide) | Found here? | Impact |
|---|---|---|
| Env floors: pydantic≥2.12, fastapi≥0.133, starlette≥1.0.1 | ✅ already met | None |
| `ctx.elicit()` / `ctx.sample()` / `ctx.list_roots()` (era-gated/removed) | ❌ none | None |
| `task=True` / `TaskConfig` (now a separate extension) | ❌ none | None |
| Removed server methods (`import_server`, `as_proxy`, `mount(prefix=)`, tool transforms) | ❌ none | None |
| FastMCP `Client` / client-side usage (needs httpx2) | ❌ **none — server-only project** | None |
| `httpx` → `httpx2` | ⚠️ many files use `httpx` | **Non-issue** — all uses are tool-internal HTTP fetching (webmcp/convertermcp/ragmcp/memory_text), not around FastMCP client calls. “HTTP inside your own tools is yours” — unaffected. |
| camelCase field reads (`inputSchema`, …) | ⚠️ `tools/convertermcp/convertermcp.py` — **moot: file deleted in Phase -1** (only `inputSchema` at :434 ever existed; `isError` appears nowhere in the repo) | — |
| **Auth internals** (`fastmcp.server.auth` + `mcp.server.auth.middleware.*`) | ⚠️ **`server_factory.py`** | **Main item** — see plan |
| **Session internals** (`_server_instances`, `StreamableHTTPASGIApp`, lifespan) | ⚠️ **`server_factory.py`** | **Secondary item** — see plan |

### What the `2026-07-28` spec changes (summary)

Stateless core (no `initialize`/`Mcp-Session-Id`; `_meta` carries identity); header-based routing
(`Mcp-Method`/`Mcp-Name`/`Mcp-Param-*`); MRTR (replaces held-open server→client streams for
elicitation/sampling/roots); cacheable list results; Tasks extension; auth hardening (RFC 9207 `iss`,
CIMD over DCR); 12-month deprecations (Roots, Sampling, Logging, legacy HTTP+SSE).

Naming trap: `stateless_http=True` in FastMCP 3.x is transport-level statelessness on the *old*
protocol — NOT the 2026-07-28 no-handshake core.

### Client / harness support

| Client | 2026-07-28 status |
|---|---|
| Claude / Claude Code | 🟡 “rolling out soon” |
| Cursor / Copilot / Gemini / VS Code | 🟡 unconfirmed GA |
| opencode | 🔴 likely not yet |
| ZCode (this env) | 🔴 no statement |
| Harnesses on official SDK v2 clients | ✅ free once they upgrade |

Now less of a gate, because FastMCP 4 serves both eras. Backward-compat both ways during the 12-month window.

---

## PART 2 — UPGRADE PLAN

### Blast radius (the good news)

```
tools/shared/server_factory.py   ← ACTIVE hot file: FastMCP/SDK internals (auth + sessions + transport)
                                   (Phase -1 first strips its SSE branch + legacy fallback)
tools/shared/utils.py            ← DEAD code that also imports FastMCP (FastMCPBase, create_mcp_instance,
                                   http_app()). Nothing calls it. Delete as part of the upgrade.
tools/<name>/*_fastmcp.py        ← effectively unchanged: `mcp = create_fastmcp_server(...)` + `app = get_transport_app(mcp)`
                                   (Phase -1 only drops log-only `if transport == "sse"` lines in __main__)
launcher/tool_discovery.py       ← Phase -1 deletes its SSE-validation tier (the launcher's ONLY `mcp` imports);
                                   afterwards the launcher just imports `app` and checks isinstance(module.app, Starlette)
launcher/streamable_http/        ← DEAD package (imported only by its own tests) — deleted whole in Phase -1
tools/convertermcp/convertermcp.py ← DEAD legacy server (zero importers) — deleted in Phase -1
tools/<name>/* tool defs         ← unchanged: plain `@mcp.tool()` servers
tools/shared/impls/* (stores)    ← unchanged: transport-agnostic
```

So the upgrade is essentially: **run Phase -1 (delete SSE + dead transports on 3.4.x), then rework
`server_factory.py` for FastMCP 4, delete dead FastMCP code in `utils.py`, bump deps, fix the 3–4
tests that assert on internals.** Nothing else in the codebase changes.

### Phase -1 — Legacy SSE & dead-transport removal (on FastMCP 3.4.x, before the port)

*Added 2026-08-14.* A repo audit (Appendix B) found the legacy SSE path is entirely vestigial: no
`*_sse.py` files exist (deleted 2026-05-06), nothing anywhere sets `MCP_TRANSPORT=sse` (repo `.env`,
`config/*.json`, scripts all clean; live config has no `transport` key, so the code default
`streamable-http` always wins), the management UI has no transport selector, and the
`REQUIRED_EXPORTS_SSE` validation tier is unreachable for current `_fastmcp.py` tools. Removing SSE
now — while still on FastMCP 3.4.x, where both transports work — shrinks the 4.0 blast radius back to
`server_factory.py` + `utils.py`, eliminates every in-repo importer of `mcp.server.sse` /
`mcp.server.lowlevel` (their SDK-v2 fate stops mattering — no extra spike question needed), and gets
ahead of the spec's 12-month SSE offramp. **Run as its own branch/commit (`chore/remove-sse`),
independently verifiable from the FastMCP 4 port.**

Checklist:

- `tools/shared/server_factory.py` — collapse `get_transport_app` (~:213-228) to streamable-http-only:
  raise a clear `ValueError` when `MCP_TRANSPORT=sse` (loud, not a silent switch), drop the `"sse"`
  mapping and the legacy `hasattr(mcp, "streamable_http_app")` / `sse_app()` fallback (that
  fallback's removal was already planned in Phase 4 — doing it here simplifies the port).
- `launcher/tool_discovery.py` — delete `REQUIRED_EXPORTS_SSE` (~:39), the SSE validation tier
  (~:284-325) including its `mcp.server.lowlevel` / `mcp.server.sse` imports (an ImportError there
  becomes a hard `ValidationError`, which is exactly the kind of SDK-v2 break we want to pre-empt),
  the SSE metadata extraction in `_extract_metadata` (~:359), and the now-vacuous `_sse` exclusion
  pattern (~:46). Verify no `mcp` imports remain in the file.
- Delete the dead `launcher/streamable_http/` package wholesale — only its own tests/examples import
  it (`launcher/streamable_http/tests/` incl. `example_client.py` / `example_server.py` and
  `config.example.json` go with it). The launcher runtime never imports it.
- Delete `tools/convertermcp/convertermcp.py` — legacy hand-rolled server, zero importers, skipped by
  discovery. This also moots the camelCase `inputSchema` migration item.
- Remove `"sse"` from the `--transport` CLI choices: `launcher/__main__.py:165-171` and
  `launchmcp.py:204-210`.
- Remove the log-only `if transport == "sse"` branches in the six `*_fastmcp.py` `__main__` blocks:
  webmcp:1097, simplemcp:405, ragmcp:2047, memorymcp:88, oraclemcp:1012, convertermcp:440.
- Stale refs: `tools/simplemcp/config.json:8` (`"file": "simplemcp_sse.py"`) and
  `tools/oraclemcp/config.json:8` (`"script": "oraclemcp_sse.py"`) — machine-written files, preserve
  the JSON structure; `tests/fef_v3/README.md:31-33`; stale `*_sse*.pyc` under `tools/*/__pycache__/`.
- Docs: drop SSE rows from the AGENTS.md transport-switching table (~:122-126), tools/AGENTS.md
  legacy layout (~:10, 26, 31), and README.md's legacy SSE contract examples (~:252-276, 426-430,
  470-475 — light touch; README is already known stale).
- Tests: update `tests/test_review_fixes.py:72-131` (sse_app / transport-flag string asserts);
  update or remove the LOW-16 string-read of the deleted `convertermcp.py` in
  `tests/test_bug_fixes_low.py:190`.

**Gate:** `python -m pytest` (full suite) green on FastMCP 3.4.x with SSE gone. Phase -1 is NOT
gated on stable FastMCP 4 — merge to `main` as soon as it's green.

### Phase 0 — Beta spike (throwaway venv; independent of Phase -1)

Goal: answer the open questions before committing to a plan. FastMCP 4 auth is explicitly unstable
(“exempt from the stability policy, expected to break”), so we must empirically confirm the auth hooks
survive before writing real code.

```bash
python -m venv /tmp/fmcp4 && . /tmp/fmcp4/bin/activate
pip install --pre "fastmcp>=4.0.0b3"    # LATEST BETA (b3 as of 2026-08-14); pulls mcp 2.0 + starlette 1.x + pydantic 2.12+
pip install uvicorn
```

Then a ~30-line script that imports and inspects (do NOT run a server yet — just resolve the API):

```python
# Spike: confirm the FastMCP 4 auth + transport surface our code depends on.
import inspect
from fastmcp import FastMCP
from fastmcp.server import auth as fmcp_auth

# Q1: TokenVerifier / AccessToken still present? get_middleware() still overridable?
print("TokenVerifier:", hasattr(fmcp_auth, "TokenVerifier"))
print("AccessToken:  ", hasattr(fmcp_auth, "AccessToken"))
TV = getattr(fmcp_auth, "TokenVerifier", None)
if TV: print("get_middleware sig:", inspect.signature(TV.get_middleware))

# Q2: do the official-SDK auth middleware modules still import at these paths?
for m in ("mcp.server.auth.middleware.auth_context.AuthContextMiddleware",
          "mcp.server.auth.middleware.bearer_auth.AuthenticatedUser"):
    mod, _, name = m.rpartition(".")
    try:
        print(m, "->", getattr(__import__(mod, fromlist=[name]), name))
    except Exception as e:
        print(m, "-> MISSING:", e)

# Q3: http_app signature + transport options
print("http_app:", inspect.signature(FastMCP.http_app))
```

Record the answers; they determine Phase 2 below. If `TokenVerifier.get_middleware()` / the custom
`AuthenticationBackend` hook is gone in 4.0, the dual-header auth needs a different mechanism (see Phase 2).

Also run the official guide’s checklist prompt (linked in Sources) against `server_factory.py` for an
automated second opinion.

### Phase 1 — Dependencies

Pins live in **three** files. Update all three.

**`requirements.txt` (top-level, hand-curated):**

```diff
- fastmcp>=3.4.2
- mcp>=1.28.0
+ fastmcp>=4.0.0          # pin exact when stable: fastmcp==4.0.0
+ mcp>=2.0.0              # floor; pin to the version 4.0 was tested against — `pip show fastmcp` lists its Requires:
  starlette>=1.0.1        # unchanged (floor already correct)
- fastapi>=0.104.0
+ fastapi>=0.133.0        # 4.0 floors Starlette ≥1.0.1; FastAPI <0.133 caps Starlette <1.0
- pydantic>=2.5.0
+ pydantic>=2.12.0        # 4.0 floor
  httpx>=0.27.0           # unchanged (tool-internal only; not FastMCP client code)
```

**`tools/memorymcp/requirements.txt` (only tool with a separate pin):**

```diff
- fastmcp>=3.3.1
+ fastmcp>=4.0.0
- pydantic>=2.0.0
+ pydantic>=2.12.0
```

**`requirements.merged.txt` — DON'T hand-edit.** It's regenerated by `python requirements_manager.py
merge`. Run that after the two edits above and re-grep to confirm the fastmcp/mcp pins propagated.

While on the beta, pin the prerelease explicitly (latest — `fastmcp==4.0.0b3` as of 2026-08-14) and
constrain the transitive
`fastmcp-slim` (see the upgrade guide's uv note). **Don't ship the upgrade against a beta** — wait for
`fastmcp==4.0.0` stable before merging (per the Rollout section).

### Phase 2 — `server_factory.py`: auth (the crux)

**Requirement to preserve:** accept **both** `X-API-Key` (VS Code Copilot) and `Authorization: Bearer`
(other clients) against a static token map; no OAuth endpoints advertised; no `WWW-Authenticate` 401.

Current implementation depends on three unstable surfaces:
- `from fastmcp.server.auth import TokenVerifier, AccessToken`
- `from mcp.server.auth.middleware.auth_context import AuthContextMiddleware` (official SDK)
- `from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser` (official SDK)
- `DualHeaderVerifier.get_middleware()` returns a Starlette `AuthenticationMiddleware` + `AuthContextMiddleware`.

**Plan, branched on Phase 0 findings (in preference order):**

1. **Mechanical port (preferred).** Keep `DualHeaderVerifier` / `DualHeaderAuthBackend` / `_AuthenticatedUser`;
   fix any moved imports (likely the `mcp.server.auth.middleware.*` paths). Lowest effort, preserves the
   exact OAuth-removal guarantee that `test_auth_c1_oauth_removal.py` enforces.
   *(Verdict 2026-08-21 simplification: drop `DualHeaderAuthBackend`/`_AuthenticatedUser` — the default
   `BearerAuthBackend` + `verify_token` + an outermost X-API-Key→Authorization normalizer delivers
   dual-header; 4.0 gates on Authorization-header presence. And re-verify TokenVerifier's OAuth route
   advertising (`auth.get_routes()`) stays empty — `test_auth_c1_oauth_removal.py` is the guard.)*
2. **Port onto whatever new hook the verdict surfaces** (e.g. a `http_app(middleware=[...])` kwarg,
   a 4.0 auth-provider class, or a lifespan-wrap pattern). The dual-header logic itself (read both
   headers, `hmac.compare_digest` against the token map, return `AccessToken`) is
   transport-independent and ports as-is — only the wiring changes.
3. **Fall back: stay on FastMCP 3.4.x** if 4.0 auth is too unstable in beta. This is a legitimate option
   since 3.x keeps working with all clients. Document the deferral in AGENTS.md so the question isn't
   re-raised next review.

The verdict in `plans/mcp-2026-07-28-phase0-verdict.md` picks the branch.

Add a runtime guard so a future auth-API break surfaces loudly instead of silently accepting all requests:
after building the app, assert the `/mcp` route is actually protected (the existing
`test_auth_c1_oauth_removal.py` HTTP tests do this — keep them green). Run those tests for **both
eras** if Q9 indicates a cross-era auth gap.

### Execution order (the actionable sequence)

Run these in this exact order. Each step's exit criteria gate the next.

| Step | Output | Gate before next |
|---|---|---|
| -1. **Phase -1 SSE cleanup** on `chore/remove-sse` — can start immediately, independent of the spike. | SSE + dead transports gone on 3.4.x. | Full `python -m pytest` green on FastMCP 3.4.x. |
| 0. **Phase 0 spike** in `/tmp/fmcp4` — **✅ DONE 2026-08-21: all Q1–Q9 answered YES (Q9 with the normalizer pattern); see `plans/mcp-2026-07-28-phase0-verdict.md`.** | `plans/mcp-2026-07-28-phase0-verdict.md` | Verdict file exists and answers Q1–Q9. |
| 1. **Draft `tests/test_era_negotiation.py`** using the verdict's concrete symbols (no best-guess code; legacy client knob per Q8). — **✅ DONE 2026-08-23: every symbol live-probed on 4.0.0b3 before writing (`Client(mode=)`, `client.protocol_version` → modern `2026-07-28` vs legacy `2025-11-25`). On b3: 3 passed + 1 strict-xfailed (Q9 gate, flips when Phase 2 normalizer lands). On the 3.4.2 interpreter the module skips cleanly, so both suites stay green.** | New test file. | Draft compiles against real 4.0 symbols only. |
| 2. **Apply Phase 1 deps** to `requirements.txt` and `tools/memorymcp/requirements.txt`, then run `python requirements_manager.py merge`. — **✅ DONE 2026-08-22: fresh-venv install clean; pydantic resolves to 2.13.4 stable (no double-beta); 29/30 auth+flush tests pass on 4.0.0b3 with UNPORTED code — the 1 failure is the spike-predicted X-API-Key gate (Phase 2's normalizer fixes it).** Re-check 2026-08-23 (b3 venv): 36 passed / 1 failed / 1 xfailed across auth+flush+era — the sole failure is still exactly that gate (`test_mcp_endpoint_accepts_x_api_key`). | Updated pins. | `pip install -r requirements.txt` succeeds in a fresh venv. |
| 3. **Port `server_factory.py`** per Phase 2 + Phase 3 (REWORK) + Phase 4 using the spike verdict to pick the wiring. — **✅ DONE 2026-08-23: Phase 2 = `ApiKeyFallbackMiddleware` (pure-ASGI, registered via `add_middleware`, runs outside the router ahead of per-route `RequireAuthMiddleware`; app stays a real Starlette so tool discovery's `isinstance(app, Starlette)` check holds); `DualHeaderAuthBackend`/`_AuthenticatedUser`/`get_middleware()` deleted per verdict. Phase 3 = native `http_app(session_idle_timeout=...)` on FastMCP ≥4 (signature-detected), 3.x lifespan wrap kept as temporary fallback until the runtime env upgrades. Phase 4 = `transport="http"` confirmed valid on 4.0b3 (verdict Q7).** | File compiles; import succeeds. | — |
| 4. **Rewrite `tests/test_session_flush.py`** for the 4.0 session API. — **✅ NO REWRITE NEEDED (2026-08-23): all four cases pass UNCHANGED on 4.0.0b3 — verdict Q6 held (`_server_instances` runtime attr + `terminate()` survived).** | Test passes alone (`pytest tests/test_session_flush.py -v`). | All four cases (no-auth, bad-auth, auth/0, auth/N) green. |
| 5. **Rewrite `tests/test_integration_flush.py`** to exercise the end-to-end restart-recovery flow. — **✅ NO REWRITE NEEDED (2026-08-23): passes unchanged on 4.0.0b3.** | Test passes. | E2E restart-recovery green. |
| 6. **Update `tests/test_auth_c1_oauth_removal.py`** for any 4.0 shape changes (call `mcp.http_app(...)` etc). Add a legacy-era variant if Q9 says cross-era auth is split. — **✅ DONE 2026-08-23: HTTP tests now build via `get_transport_app` (production wiring incl. normalizer); obsolete `TestDualHeaderAuthBackendUnit` removed (backend deleted per verdict). No legacy-era variant needed — Q9 says auth stack is era-agnostic; `test_era_negotiation.py` guards legacy anyway.** | C1 tests green for the modern era; legacy-era variant green if added. | C1 acceptance criteria met both ways. |
| 7. **Finalize `tests/test_era_negotiation.py`** with concrete 4.0 symbols from the verdict; run. — **✅ DONE 2026-08-23: strict-xfail on the Q9 X-API-Key gate flipped to a real pass once the normalizer landed; fixture builds via `get_transport_app`. On b3: modern AND legacy clients authenticate dual-header and negotiate different protocolVersions.** | Era-negotiation test green. | Modern AND legacy client both authenticate. |
| 8. **Full suite** + **live harness smoke** per Validation strategy #6 and #7. — **✅ DONE 2026-08-23 (with one caveat): full suite green in a complete fastmcp-4 env (529 passed; required fixing a raw-`Path` sys.path entry in ragmcp_fastmcp.py that crashed hypothesis imports, and converting two legacy `get_event_loop()` test helpers to local loops) and on the 3.4.2 interpreter (525 passed / 4 skipped, era module skips). Live smoke: launcher started under the b3 venv, all six tools up on 8000–8005; auth matrix on simplemcp:8002 = no-auth 401 / Bearer 200 / X-API-Key-only 200 (Q9 fix verified live) / wrong-key 401; legacy-handshake client negotiated 2025-03-26. Caveat: the native ZCode-client tool call could not be captured this turn due to an agent-side tool-emission fault — re-verify with one cheap native call (square) on the next turn.** | All green. | — |
| 9. **Roll out** per the Rollout section (merge to `main` once stable FastMCP 4 ships). | Released. | — |

**Phase 0 verdict template** (`plans/mcp-2026-07-28-phase0-verdict.md`):

```markdown
# FastMCP 4 Phase 0 Spike Verdict

**Date:** YYYY-MM-DD
**fastmcp version tested:** fastmcp==X.Y.Z (e.g. 4.0.0b3)
**mcp version installed transitively:** mcp==X.Y.Z

## Questions (mirror Open questions Q1–Q9)
| # | Question | Answer | Evidence (path/symbol/snippet) |
|---|---|---|---|
| 1 | TokenVerifier.get_middleware() still overridable? | YES/NO | … |
| 2 | mcp.server.auth.middleware.auth_context/bearer_auth still import? | YES/NO | … |
| 3 | FastMCP(name, auth=<TokenVerifier>) still constructs? | YES/NO | … |
| 4 | http_app() kwarg for native session_idle_timeout? | YES/NO; kwarg name = … | … |
| 5 | Where is StreamableHTTPSessionManager reachable? | … | … |
| 6 | Per-session transport.terminate() / close_all() API? | … | … |
| 7 | http_app(transport=…) accepted string values? | … | … |
| 8 | Client(protocol_version=…) knob exists? | YES/NO | … |
| 9 | Cross-era auth: does DualHeaderVerifier run for legacy clients? | YES/NO | … |

## Phase 2 path chosen
**Mechanical port / 4.0 hook / Stay on 3.4.x** (pick one)

## Notes for Phase 3 REWORK
Idle-timeout kwarg name (if any): …
Session-manager reach path: …
Per-session termination API: …
```

The verdict file is the only artifact required to proceed past Phase 0.

### Phase 3 — `server_factory.py`: session machinery (**REWORK** — preserve flush for handshake-era clients)

**Decision (per user, 2026-08-11):** REWORK. Defense-in-depth — preserve the operator recovery hatch
(`/admin/flush-sessions`, `MCP_SESSION_IDLE_TIMEOUT`, `MCP_DISABLE_FLUSH_ENDPOINT`) because at least
some handshake-era clients in our environment (ZCode/opencode) may cache stale `Mcp-Session-Id`
values and never reconnect on their own.

Current `_wire_session_management` / `_get_session_manager` / `_apply_session_idle_timeout_via_lifespan`
reach into `StreamableHTTPASGIApp`, `StreamableHTTPSessionManager._server_instances`, and
`router.lifespan_context`. SDK v2 *removed session-id access from its streamable HTTP transport* and
FastMCP “reconstructs it on the transport object” — so the exact import paths and class names will move.

**Phase 0 spike MUST answer before this code is touched (added to Open questions below):**
- Where does FastMCP 4 surface the `StreamableHTTPSessionManager` singleton from which
  `_server_instances` is reachable? (Possibilities: a) a public attribute on the transport ASGI app;
  b) a registry on the `FastMCP` instance itself; c) a new `SessionStore`/`SessionManager` API; d)
  fully internal — must be reached via lifespan walk like today.)
- Does FastMCP 4 expose `session_idle_timeout` natively on `http_app()` or on the session manager?
  If yes, retire the lifespan-wrap hack.
- Does the streamable transport's per-request `.terminate()` still exist?

**If the spike answers YES to a native idle-timeout kwarg:**
1. Remove `_apply_session_idle_timeout_via_lifespan`. Set `session_idle_timeout` (or whatever the
   4.0 kwarg name is) directly via `http_app(...)`.
2. Keep `_wire_session_management` only for the flush route — rewrite `_get_session_manager` and the
   `_server_instances` walk to use whatever 4.0 exposes (likely an attribute on the transport app).
3. `/admin/flush-sessions` stays in `get_transport_app`; the HTTP path, auth, and response shape are
   FastMCP-version-independent.

**If the spike answers NO (no native kwarg, internals opaque):**
1. Keep `_wire_session_management` and `_apply_session_idle_timeout_via_lifespan` largely intact.
2. Update only the moved imports (e.g. `fastmcp.server.http.StreamableHTTPASGIApp` → whatever 4.0
   exposes) and any renamed attribute names.
3. The lifespan-walk pattern may still work — it's Starlette, not FastMCP, so it's stable.

**If spike surfaces a third path** (e.g. FastMCP 4 replaces `_server_instances.clear()` with a
`SessionManager.close_all()` method), follow the new API.

**Fallback:** if REWORK proves impossible in 4.0b (auth still unstable AND session internals
inaccessible), defer the upgrade entirely — stay on FastMCP 3.4.x. Document the deferral in AGENTS.md.

**Out of scope:** the management UI (port 8200), FEF extensions, and operator scripts do **not** call
`/admin/flush-sessions` (verified by grep, 2026-08-11), so REWORK requires no downstream changes
outside `server_factory.py` and the flush tests.

### Phase 4 — `server_factory.py`: transport app call

Trivial. `get_transport_app` currently does `mcp.http_app(transport="http"|"sse")`. In 4.0 confirm
(via Phase 0 spike):
- `http_app(transport=...)` still accepted; **what are the accepted string values?** Current code uses
  `"http"` to mean "streamable HTTP". FastMCP 4's deployment doc uses `http_app(stateless_http=True)`,
  `http_app(middleware=...)`, `http_app(path=...)`. The exact kwarg values may have been renamed
  (`"streamable-http"`? `"http"`? a new enum?). Resolve before writing the call.
- New option to consider: `stateless_http=True` (per-request fresh context, old-protocol) — relevant only
  if you scale horizontally behind a LB. Not required for single-process.
- Legacy SSE transport: **removed by Phase -1** — `get_transport_app` is streamable-http-only by the
  time Phase 4 runs. (Supersedes the earlier "keep the `sse` branch for the offramp" decision; the
  2026-08-14 audit found zero SSE consumers — Appendix B.)

The `hasattr(mcp, "streamable_http_app")` legacy branch (`mcp.server.fastmcp`) is already removed by
Phase -1 — and under `mcp` 2.0 the class is `mcp.server.mcpserver.MCPServer` and that old method no
longer exists anyway.

### Phase 5 — Tests

| Test | Action |
|---|---|
| `tests/test_session_flush.py` | **REWRITE** against 4.0 session API. Asserts on `_server_instances`; the attribute location/name will move in 4.0 (spike resolves). Re-verify the 401 (no-auth), 200 (auth, 0 sessions), and 200 (auth, N sessions cleared) cases. |
| `tests/test_integration_flush.py` | **REWRITE** counterpart for the end-to-end path — restart one tool, call flush, confirm a stub handshake-era client can re-initialize. |
| `tests/test_auth_c1_oauth_removal.py` | **Keep — your safety net.** Update imports/constructs if `http_app(transport="http")` or auth shapes changed. This test is the acceptance criterion for dual-header auth. |
| `tests/test_review_fixes.py` | sse/transport string-asserts already updated by Phase -1; the rest likely still valid — adjust if the transport API is renamed. |
| `tests/test_fastmcp_critical_fixes.py` | **Keep as regression guard.** Doesn't touch internals, but is the cross-tool behavioral safety net. Run first after each tool starts. |
| `tests/test_regression.py`, `tests/test_bug_fixes_*` | String-greps over tool source. Re-run after edits. |
| Backend tests (`test_memory_text`, `test_qdrant_contracts`, `test_sql_store`, …), `tests/fef_v3/*`, `tests/test_oracle_thread_safety` | **Unaffected** (transport-agnostic). Must still be green. (`launcher/streamable_http/tests/*` is deleted in Phase -1 — it imported `mcp.types` directly and was never transport-agnostic.) |
| `tests/test_bug_fixes_low.py` | LOW-16 string-reads the `convertermcp.py` source — update or remove in Phase -1 when the file is deleted. |

Add one new test: **`tests/test_era_negotiation.py` — era negotiation smoke test.** Spin up the 4.0
server, connect a modern client (`fastmcp.Client` defaults to the latest era) AND a legacy client
(`fastmcp.Client(..., mode="legacy")` — the knob named in the 4.0 upgrade guide; the spike confirms
the exact kwarg, with `protocol_version=` as fallback — else use a raw JSON-RPC client via `httpx`
that omits `Mcp-Session-Id`), assert both authenticate via dual-header
(`X-API-Key` and `Authorization: Bearer`) and call a trivial tool, and that the negotiated
`protocolVersion` differs between them. This locks in the "one server, both eras" guarantee.

**Test-isolation note:** several tests (`test_session_flush.py`, `test_auth_c1_oauth_removal.py`,
`test_integration_flush.py`) insert `tools/` into `sys.path` at runtime so they can import
`tools.shared.server_factory` from a test that doesn't run inside the package. After the upgrade,
verify those `sys.path` shims still resolve the renamed symbols (the spike should confirm).

### Per-file change checklist

```
requirements.txt                 bump fastmcp>=4.0.0, mcp>=2.0.0, fastapi>=0.133.0, pydantic>=2.12.0
tools/memorymcp/requirements.txt bump fastmcp>=4.0.0, pydantic>=2.12.0
requirements.merged.txt          regenerated via `python requirements_manager.py merge`; do not hand-edit
— Phase -1 (SSE removal, own branch chore/remove-sse) —
tools/shared/server_factory.py   collapse get_transport_app to streamable-http; ValueError on MCP_TRANSPORT=sse
launcher/tool_discovery.py       drop SSE validation tier + REQUIRED_EXPORTS_SSE + _sse exclusion
launcher/streamable_http/        DELETE package (dead; own tests only) incl. tests/ + config.example.json
launcher/__main__.py, launchmcp.py  drop "sse" from --transport choices
6× tools/<name>/<name>_fastmcp.py  remove log-only `if transport == "sse"` branches
tools/simplemcp+oraclemcp/config.json  drop stale *_sse.py refs (preserve JSON structure)
tests/test_review_fixes.py, tests/test_bug_fixes_low.py  update sse/convertermcp string asserts
AGENTS.md / tools/AGENTS.md / README.md  drop SSE transport docs
— FastMCP 4 port —
tools/shared/server_factory.py   [auth port — Phase 2] [session REWORK — Phase 3] [transport — Phase 4]
tools/shared/utils.py            DELETE FastMCPBase, create_mcp_instance, get_app, run, setup_common_logging,
                                 get_tool_logger (lines ~10, 32-173, 176-187). Keep is_internal_url +
                                 _is_internal_ip + _check_ipv4_mapped (these have no FastMCP dependency
                                 and are imported by tools + tests).
tools/convertermcp/convertermcp.py  DELETE (Phase -1; zero importers — moots the camelCase item)
tests/test_session_flush.py      REWRITE for 4.0 session API (Phase 3/5)
tests/test_integration_flush.py  REWRITE for 4.0 session API (Phase 3/5)
tests/test_auth_c1_oauth_removal.py  update for 4.0 shapes (Phase 5)
tests/test_review_fixes.py       adjust string asserts if needed (Phase 5)
AGENTS.md                        keep "Session management" section; only update FastMCP-specific
                                 notes if any class names or env vars moved in 4.0 (Phase 3).
(NEW) tests/test_era_negotiation.py  both-era dual-header smoke test (Phase 5)
(NEW) plans/mcp-2026-07-28-phase0-verdict.md  spike output — Gate to Phase 2
```

### Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FastMCP 4 auth API (`TokenVerifier`/`get_middleware`) changed → dual-header breaks | Medium-High | High (auth) | Phase 0 spike first; fallback chain in Phase 2 (port → `middleware=` kwarg → lifespan-wrap → stay on 3.x) |
| Official `mcp.server.auth.middleware.*` paths move in SDK v2 | High (types moved to `mcp_types`) | Medium | Spike resolves; port imports |
| Session internals moved → flush feature breaks | **High** (we're REWORKing, so we depend on this) | **Medium** (operator hatch offline until rewrite lands; handshake-era clients may pile up stale sessions) | Phase 0 spike MUST answer Q4–Q6 before Phase 3 starts. If unmovable, fall through to DROP or stay on 3.x |
| Cross-era auth gap (Q9): legacy `initialize` clients use a different auth surface than modern | Low–Medium | High (silent auth regression on legacy clients) | Add a legacy-era variant to `test_auth_c1_oauth_removal.py`. C1 tests must run against both eras |
| Silent auth bypass if migration wrong | Low (but catastrophic) | Critical | Keep `test_auth_c1_oauth_removal.py` green; add startup assert that /mcp is protected |
| Beta API churn before stable 4.0 | Medium | Medium | Pin exact; re-run spike when 4.0.0 stable ships; don’t ship the upgrade until stable |
| External client pointed at a `/sse` URL breaks when Phase -1 lands | Low (audit found zero SSE consumers in-repo: no config/env/UI/scripts) | Medium for that client | `MCP_TRANSPORT=sse` fails fast with a clear error; release note tells operators to move any `/sse` client config to `/mcp` |

### Validation strategy

Run in this strict order — each step gates the next.

0. **Phase -1 SSE cleanup** (independent of the spike; can run first or in parallel): `git checkout -b
   chore/remove-sse`, apply the Phase -1 checklist. **GATE: full `python -m pytest` green on FastMCP
   3.4.x.** Merge to `main` when green — not gated on stable 4.0.
1. **Phase 0 spike** (`/tmp/fmcp4` venv) resolves Open Questions Q1–Q9. **No server is started in this
   phase.** Spike output is `plans/mcp-2026-07-28-phase0-verdict.md`. **GATE: this file must exist
   before Phase 2.**
2. **Branch + bump.** `git checkout -b feature/fastmcp-4` (from post-Phase--1 `main`), apply Phase 1 deps
   exactly as written.
3. **Port `server_factory.py` (Phases 2 + 3 + 4)** — auth, sessions (REWORK), transport. Stop and
   ask if the verdict says a branch was impossible.
4. **Run the *narrow* auth/session tests first:** `pytest tests/test_auth_c1_oauth_removal.py
   tests/test_session_flush.py tests/test_integration_flush.py -v`. All must pass. **GATE: dual-header
   auth + flush endpoint still work end-to-end.**
5. **Run the new smoke test** `pytest tests/test_era_negotiation.py -v`. Both modern AND legacy
   client must authenticate and call a trivial tool. **GATE: cross-era compatibility holds.**
6. **Full suite:** `python -m pytest tests/ -q`. Target: all green. Investigate any regression; the
   backend tests must still pass (they're transport-agnostic).
7. **Live smoke against a real handshake-era harness.** Point ZCode or opencode at the upgraded tool,
   restart the tool, then either (a) let the client auto-reconnect (it should — that's the whole point
   of preserving flush) or (b) call `/admin/flush-sessions` once and confirm the client recovers with
   a fresh session.
8. **Confirm advertised protocol.** A modern client logs `protocolVersion: 2026-07-28`; a legacy
   client `2025-11-25`. Both succeed.
9. **Stop the rollout** if any step fails — do not merge to main. Pin to 4.0b N>1 and re-evaluate.

### Rollout

- Do it on a branch (e.g. `feature/fastmcp-4`).
- **Phase -1 (`chore/remove-sse`) is the exception**: it merges to `main` as soon as the suite is green
  on 3.4.x — it does not wait for stable FastMCP 4.
- Merge only after **stable** `fastmcp==4.0.0` is out (not the beta) — unless you specifically want to
  harden against the beta early.
- Keep 3.4.x on `main` until the branch is validated end-to-end with a real harness.

### Operator comms (when the upgrade lands)

Nothing material changes for operators, but call these out in the release notes:

- **No new env vars**, no new ports, no new auth flow. The same `X-API-Key` and `Authorization: Bearer`
  headers keep working against the same tool API keys.
- **SSE transport removed (Phase -1).** `/sse` endpoints no longer exist; every tool serves
  streamable HTTP at `/mcp`. `MCP_TRANSPORT=sse` now fails fast with a clear error. Any external
  client config pointing at a `/sse` URL must switch to streamable HTTP (`/mcp`, same auth headers).
- **Operator recovery hatch preserved.** `POST /admin/flush-sessions`, `MCP_SESSION_IDLE_TIMEOUT`,
  `MCP_DISABLE_FLUSH_ENDPOINT` all still function. Document them in the release notes even though they
  live in AGENTS.md.
- **Restart picks up new deps.** Operators don't need to do anything; `python -m launcher --tools ...`
  re-resolves and starts on FastMCP 4.
- **Transient unavailability during restart.** The eight tool ports (~8008-8032) drop for ~toolstartup;
  handshake-era clients should call `/admin/flush-sessions` once after the restart clears their stale
  session IDs (or wait for `MCP_SESSION_IDLE_TIMEOUT`). Modern (2026-07-28) clients reconnect on next
  request automatically.
- **No data migration.** Memory/Qdrant/etc. are untouched.

### Out of scope (explicit)

- **Client-side adoption.** No client in this repo's environment supports 2026-07-28 today (per
  Part 1 client table). Plan does NOT add any client code.
- **Wire-protocol schema migration of stored memories.** The on-disk format is independent of the
  transport protocol; `tools/shared/store_models.py` is already transport-agnostic. No data
  migration is implied by the FastMCP 4 upgrade.
- **Migration tool (`migrate_store`)** is unaffected — it talks to the SQL/Vector ABCs, not FastMCP.
- **Deprecating the SSE transport — moved in-scope 2026-08-14:** removed by Phase -1 on 3.4.x (the
  repo audit found zero SSE consumers — Appendix B — and removal shrinks the 4.0 blast radius). The
  spec's 12-month SSE offramp no longer binds this repo because nothing here serves SSE after
  Phase -1.
- **Tool-set changes.** No new tools; no tool removals. The upgrade is invisible to MCP clients.
- **LGUI / management UI / FEF extensions.** None of them call the internals we're porting
  (validated 2026-08-11). They are unaffected.
- **Performance/caching work.** The era-negotiation smoke test does NOT benchmark. Cacheability of
  list results (per the 2026-07-28 spec) is a feature add, not a transport requirement.

---

### Open questions (resolve in Phase 0 spike)

> **✅ RESOLVED 2026-08-21 — see `plans/mcp-2026-07-28-phase0-verdict.md`.** Headlines: all YES; native
> `session_idle_timeout` kwarg; `_server_instances`/`terminate()` survive at runtime; legacy knob is
> `Client(..., mode="legacy")`; dual-header needs an X-API-Key→Authorization normalizer middleware
> (4.0 gates on Authorization-header presence) after which the custom auth backend is deletable;
> fastapi is no longer a FastMCP 4 dependency (httpx2 replaces httpx on the client side).

1. Does FastMCP 4 `TokenVerifier.get_middleware()` still allow a custom Starlette `AuthenticationBackend`?
2. Do `mcp.server.auth.middleware.auth_context.AuthContextMiddleware` and
   `...bearer_auth.AuthenticatedUser` import at the same paths under SDK v2?
3. Does `FastMCP(name, auth=<TokenVerifier>)` still construct the same way?
4. Does `http_app(transport="http"|"sse")` still work, and is there a **native** idle-session-TTL kwarg
   on `http_app()` or the session manager in 4.0? (If yes, retire the `_apply_session_idle_timeout_via_lifespan`
   hack.) **REQUIRED FOR PHASE 3 REWORK.**
5. **Where does FastMCP 4 expose `StreamableHTTPSessionManager._server_instances`** (or its replacement)?
   (Today it's reachable by walking the `/mcp` route's `.app` chain inward through
   `StreamableHTTPASGIApp`. 4.0 may move it.) **REQUIRED FOR PHASE 3 REWORK.**
6. **Does the per-session transport object in 4.0 still expose `terminate()`** for flush to call?
   Or does 4.0 prefer a `SessionManager.close_all()` / `disconnect_all()` API? **REQUIRED FOR PHASE 3 REWORK.**
7. **What are the accepted string values for `http_app(transport=...)` in 4.0?** Current code passes
   `"http"` to mean streamable-http. Confirm this still maps, or rename to `"streamable-http"` /
   whatever 4.0 expects. The Phase 4 implementation depends on this.
8. **What is the legacy-era client knob for the era-negotiation smoke test?** The 4.0 upgrade guide
   names `Client(..., mode="legacy")` (v4 defaults to `mode="auto"`). Confirm the exact kwarg (else
   `protocol_version=` as fallback), or drive the legacy client with raw JSON-RPC over `httpx` that
   omits `Mcp-Session-Id`.
9. When FastMCP 4 serves a legacy `initialize`-handshake client, **does the dual-header auth path still
   run** (i.e., is `DualHeaderVerifier.get_middleware()` part of the cross-era middleware stack, or is
   the legacy path on a different auth surface)? If different, the C1 acceptance test may need a
   legacy-era variant.

**Phase 0 spike deliverable:** a one-page verdict in `plans/mcp-2026-07-28-phase0-verdict.md`
answering each numbered question with YES/NO + the symbol path / kwarg name / snippet needed. The
verdict is a precondition for starting Phase 2.

---

## Trigger conditions (re-check weekly)

Act when **both** are true:
1. **FastMCP `4.0.0` (non-beta) is on PyPI** (`pip index versions fastmcp` shows a stable 4.x).
2. Phase 0 spike answers the open questions and the auth port is feasible (or you’ve decided to drop flush).

The MCP Python SDK v2 prerequisite (`mcp>=2.0.0`) is already satisfied as of 2026-08-11 (2.0.0 still
latest on 2026-08-14) — so trigger condition #1 is the only remaining gate **for the FastMCP 4 port**.
Phase -1 (SSE removal) needs no trigger at all and can run immediately.

Client adoption is **no longer a blocker** (4.0 serves both eras).

## Re-check commands

```bash
pip index versions fastmcp | head -2     # stable 4.x (betas thru 4.0.0b3 exist; add --pre to list them)
pip index versions mcp    | head -2      # 2.0.0 GA already out
python -c "import mcp.types as t; print('protocol:', t.LATEST_PROTOCOL_VERSION)"
pip install --pre "fastmcp>=4.0.0b3"     # try the latest beta (throwaway venv)
```

---

## Appendix A — Review audit (2026-08-11)

A code-grep audit was run during review to validate the plan's claims. Findings:

**Blast radius (validated):** `tools/shared/server_factory.py` is the only ACTIVE file referencing
FastMCP/SDK internals (`TokenVerifier`, `AccessToken`, `AuthContextMiddleware`, `AuthenticatedUser`,
`StreamableHTTPASGIApp`, `_server_instances`, `streamable_http_app`, `sse_app`). It is imported by
every `tools/<name>/<name>_fastmcp.py` via `create_fastmcp_server` + `get_transport_app`.

**Second touch point found:** `tools/shared/utils.py` also imports `from fastmcp import FastMCP` and
calls `FastMCP(name)` + `self.mcp_instance.http_app()` from a `FastMCPBase` helper class. **No code
in the repo imports `FastMCPBase`/`create_mcp_instance`/`setup_common_logging`/`get_tool_logger`** —
verified by grep across `tools/`, `launcher/`, and `tests/`. Only `is_internal_url` is used.
**Recommendation:** delete the dead FastMCP surface in the same change. Doing so also removes a
*latent* dual-header hole (the dead `create_mcp_instance` builds a `FastMCP(name)` with **no auth**
— if anyone resurrects it, the server would accept any caller).

**Tests touching internals (validated):** `tests/test_session_flush.py` (asserts `_server_instances`),
`tests/test_integration_flush.py` (end-to-end flush), `tests/test_auth_c1_oauth_removal.py` (the
acceptance test for dual-header — imports `DualHeaderVerifier`, calls `mcp.http_app(transport="http")`),
`tests/test_review_fixes.py` (string-greps for `get_transport_app`/`sse_app`/`streamable` in tool
sources). All other test files either exercise transports abstractly (memorymcp, ragmcp, oracle,
qdrant, sql_store) or string-grep tool source code without importing FastMCP.

**Flush-sessions downstream impact (validated):** `/admin/flush-sessions`, `MCP_SESSION_IDLE_TIMEOUT`,
and `MCP_DISABLE_FLUSH_ENDPOINT` are referenced ONLY in `server_factory.py`, `test_session_flush.py`,
`test_integration_flush.py`, and the root `AGENTS.md` docs table. The management UI (port 8200),
launcher (`launcher/`), and FEF extensions (`tools/fef_*`) do not call them. **No in-repo callers,
but the features must be preserved on the upgrade** (per Phase 3 user decision) — this audit
established that REWORK is bounded to `server_factory.py` + the two tests, with zero downstream
ripple.

**Current dependency state (verified 2026-08-11):**
- `pip index versions fastmcp` → `LATEST: 3.4.7`, `INSTALLED: 3.4.2` (stable channel; beta 4.0.0b1 not
  listed by `pip index`).
- `pip index versions mcp` → `LATEST: 2.0.0`, `INSTALLED: 1.28.1`. The SDK side of the upgrade is GA
  today; only FastMCP is still in beta.

**Naming caveat:** the filename `mcp-2026-07-28-stateless-upgrade.md` uses the **protocol** date
(2026-07-28 is when the spec was released). The **framework** being upgraded is FastMCP 4.0. Don't
confuse the two when discussing this plan.

## Appendix B — Re-review audit (2026-08-14)

Second review pass: fresh codebase sweep (blast radius, SSE consumers, dead code, tests) plus a
PyPI/docs status check. All findings below are folded into the plan above.

**Blast-radius correction** (refines Appendix A, which implied `launcher/*` was untouched): three
files beyond `server_factory.py`/`utils.py` import the `mcp` SDK. All are eliminated by Phase -1
rather than ported:

1. `launcher/tool_discovery.py:301-325` — the SSE validation tier imports
   `mcp.server.lowlevel.Server` + `mcp.server.sse.SseServerTransport`; an ImportError there becomes a
   hard `ValidationError`. Unreachable for current `_fastmcp.py` tools (they validate at the FastMCP
   tier), so deleting the tier is behavior-neutral today and pre-empts an SDK-v2 break.
2. `launcher/streamable_http/` — dead package: only its own tests/examples import it
   (`launcher/streamable_http/tests/*`, incl. `example_client.py`/`example_server.py`); the launcher
   runtime (`__main__.py`, `server_manager.py`, `management_server.py`) never touches it. Imports
   `mcp.types` JSONRPC classes behind a stub-on-ImportError guard.
3. `tools/convertermcp/convertermcp.py:20-22` — legacy hand-rolled server with zero importers (the
   live entry `convertermcp_fastmcp.py` never imports it; discovery skips it via the
   `_fastmcp`-priority rule).

**SSE-consumer audit (basis for Phase -1):** no `*_sse.py` files exist (deleted in commit `5584f2e`,
2026-05-06); nothing sets `MCP_TRANSPORT=sse` (repo `.env`, `tools/ragmcp/.env.example`,
`config/launcher_config.json`, all scripts clean; the live config has no `transport` key and the code
default is `streamable-http`); `mcp_ui/` has no transport selector (it only proxies management-API
start/stop/restart calls). Stale remnants found: `tools/simplemcp/config.json:8` and
`tools/oraclemcp/config.json:8` reference deleted `*_sse.py` files; `tests/fef_v3/README.md:31-33`
names deleted scripts; stale `*_sse*.pyc` bytecode sits in `tools/*/__pycache__/`.

**Claims corrected:**
- Migration-table row "camelCase reads (`inputSchema`, `isError`, …)": only `inputSchema` exists
  (`convertermcp.py:434`); `isError` appears nowhere in the repo; `content` is httpx-only. Moot once
  the file is deleted.
- Execution-order step 1 originally drafted tests "against the unresolved 4.0 API" *after* the spike
  had resolved it — fixed to use the verdict's symbols.
- Phase 5 listed `launcher/streamable_http/tests/*` as "Unaffected (transport-agnostic)" — it imports
  `mcp.types` directly and is deleted in Phase -1.

**Status refresh (PyPI, 2026-08-14):** fastmcp — latest stable 3.4.7 (Aug 10); pre-releases 4.0.0b3
(Aug 14), 4.0.0b2 (Aug 7), 4.0.0b1 (Jul 28); ~weekly beta cadence supports "stable in weeks";
trigger #1 still unmet. mcp — 2.0.0 (Jul 28) still latest; v1.x ended at 1.29.0 (critical fixes
only). From the official 4.0 upgrade guide: `mcp.types` → standalone `mcp_types` package but
re-exported as `mcp.types` (existing imports keep working); legacy-era client knob is
`Client(..., mode="legacy")`. The guide is silent on `TokenVerifier`/`get_middleware()` and session
internals — Q1–Q7/Q9 remain genuine spike questions.

**Dead-code claim re-verified:** `utils.py` delete ranges are exact — line 10 (`from fastmcp import
FastMCP`), 32-173 (`FastMCPBase`), 176-187 (logging helpers); keep `_is_internal_ip` /
`_check_ipv4_mapped` / `is_internal_url` (used by webmcp, convertermcp_fastmcp, and tests). The
latent no-auth hole is real: `utils.py:77-79` builds `FastMCP(name)` with no `auth=`.

## Sources

- Spec post: https://blog.modelcontextprotocol.io/posts/2026-07-28/
- Spec + changelog: https://modelcontextprotocol.io/specification/2026-07-28
- **FastMCP 4.0 upgrade guide (the authoritative migration checklist):** https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3
- FastMCP releases/cadence: https://gofastmcp.com/development/releases
- FastMCP HTTP deployment (routing headers, stateless, auth): https://gofastmcp.com/deployment/http
- FastMCP repo + `tests/test_upgrade_from_v3.py`: https://github.com/PrefectHQ/fastmcp
- Official Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Claude rollout: https://claude.com/blog/bringing-mcp-2026-07-28-to-claude
- Breaking-change roundup: https://stacktr.ee/blog/mcp-2026-spec-changes
