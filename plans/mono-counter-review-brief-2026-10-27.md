# Mono-mode safety — counter-review brief (2026-10-27)

*For an adversarial reviewer. Claims are numbered and verifiable; the admitted
limits section is where a good counter-review should start. Code state:
branch `feature/review-fixes-2026-10-24`, tip at the time of writing.*

## Context

Mono mode (`MCP_AUTH_MODE` unset) = one static system key per tool, no user
store, no identity gate. It is the operator's primary usage (fresh checkouts,
no `.env`). Two live mono bugs were found at a work deployment on 2026-10-26
and fixed in commit `20680b0`; a uniformity follow-up is in the tip commit.
This brief covers the claim "mono mode is now safe" — try to break it.

## The claims (each independently checkable)

**C1 — Every identity resolver treats an authenticated mono caller as
admin/unrestricted.** Sites and expected role-default semantics:
- `tools/ragmcp/ragmcp_fastmcp.py` `_caller_rag_access` (~line 128): mono →
  `("ragmcp", None)` = unrestricted.
- `tools/memorymcp/memory_tools.py` `_memory_caller` (~line 104): mono →
  admin, unscoped.
- `tools/databasemcp/connections.py` `_current_caller_grants` (~line 438):
  mono → `("system"/client_id, "admin", [])`.
- `tools/databasemcp/db_tools.py` `_assert_preset_grant` (~line 588) and
  `_refuse_non_admin`: mono → allowed.
- `tools/shared/server_factory.py` flush gate (~line 643): mono → allowed.

Convention everywhere: `entry.get("role", "user" if user_store else "admin")`.

**C2 — The multi direction stays fail-closed.** A role-less entry under a
user-store verifier defaults to user (scoped/refused), never admin. Pinned by
`test_ragmcp_multi_roleless_still_scoped` in `tests/test_review_fixes_2026_10_18.py`.

**C3 — Mono auth over real transport works.** `tests/test_auth_c1_oauth_removal.py`
`TestFastMCPAuthHTTP`: production `get_transport_app` + DualHeaderVerifier, real
HTTP `/mcp` calls with the system key.

**C4 — Mono anomaly paths stay fail-open (no HTTP context / no token /
unknown token under the MONO verifier).** Each resolver returns the
unrestricted tuple there. NOTE: under a user-store verifier those same paths
fail CLOSED by design (audit P2-3) — mono and multi differ here deliberately.

**C5 — No mono surface silently requires a user store.** The identity gate is
installed only when `MCP_AUTH_MODE=multi` (`server_factory.create_fastmcp_server`);
E1 masks and tx owner-attribution are mode-independent; mcp_ui/pod_ui logins
are env-pair based.

**C6 — Deliberately left WITHOUT the admin default (adjudicated, verify the
reasoning):** `tools/shared/identity.py:132` and
`launcher/tool_extensions/http_server.py:180` use `entry.get("role") == "admin"`
with NO default → a role-less entry is NOT admin there. Rationale: both code
paths only execute in multi (the gate is multi-only; `_is_admin`'s system-key
branch short-circuits mono first), and in multi a role-less entry SHOULD be
fail-closed. If the reviewer finds a mono path that reaches these lines, that
is a real finding.

## Admitted limits (start here)

1. **No full-deployment mono e2e test** — nothing boots a real launcher with
   no `.env` and calls every tool over HTTP. The work box is the de-facto test.
2. **The convention is duplicated at 6+ call sites, not centralized** — a new
   resolver can reintroduce the bug family. The unit pins only cover the
   three fixed sites + one multi direction.
3. **Fresh-checkout mono is keyless-central** — reads open, mutations 403
   (audit P2-2), admin/admin UI login refused on non-loopback (P1-1). All by
   design; all friction the operator will hit (checklist in the main report).
4. `seed_from_env` still runs the unlocked load→mutate→save (runs once per
   process; only reachable from multi/mcp_ui-multi paths).

## Verification recipe (no credentials needed; repo is public)

```bash
git fetch <remote> feature/review-fixes-2026-10-24 && git switch -c mono-review FETCH_HEAD
# Adversarial greps — any hit outside the adjudicated list is a finding:
grep -rn "get(\"role\"\|get('role'" tools/ launcher/ mcp_ui/ pod_ui/ --include='*.py' | grep -v tests
# For each hit: which verifier shapes reach it? what does the default do per mode?
python -m pytest tests/test_review_fixes_2026_10_18.py tests/test_auth_c1_oauth_removal.py -q
# Best mono reproduction: run a tool with NO .env (mono), call it with the
# system key over HTTP, exercise every gated tool (search/index, memory CRUD,
# db connect/query/execute).
```
