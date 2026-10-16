# Security Audit — 2026-10-01

**Scope:** full repo at `feature/pod-ui-gaps` (tip `193fca1`): `launcher/`, `tools/` (6 FastMCP servers), `mcp_ui/`, `pod_ui/`, `deploy/`, `config/`.
**Method:** three parallel read-only audit agents (auth/secrets/sessions · injection/data-handling · network-exposure/deploy) + `bandit` (28,786 LOC: 1 High / 53 Med / 99 Low) + `pip-audit`. Every P0/P1 claim below was independently re-verified in source by the main agent (marked ✅). Mimosa sealed-scan skill could NOT run: its `security_scan_start` MCP server is not bound in this session — enable the Mimosa MCP server in `~/.zcode/cli/config.json` + restart the session to get the sealed-artifact path too.

**Verdict discipline:** findings only; no "project is secure" claim is made or implied.

---

## P0 — exploitable now

### P0-1 · Six tool API keys are public in git history, and every MCP endpoint binds `0.0.0.0`
- `tools/{simplemcp,webmcp,memorymcp,ragmcp,databasemcp,convertermcp}/config.json` `auth.api_key` values are in the **public** repo's history on `main` (verified via `git show <sha>:...`); remote = `golfromeo-fr/supreme-mcp-tools`.
- Binds: `config/launcher_config.json:11` `"host": "0.0.0.0"`, `launcher/config_types.py:23` and `tools/shared/server_factory.py:44` `DEFAULT_HOST = "0.0.0.0"`; `deploy/compose-work.yml` uses `network_mode: host` so the pod is LAN-reachable exactly like the host.
- Impact: anyone on the network authenticates with public keys → full access to databasemcp SQL (incl. the live memory store via preset), memorymcp/ragmcp data planes, webmcp fetch, ragmcp indexing of arbitrary paths (P2-4).
- **Fix: this is the open TODO.md item — rotate all six keys** (configs + `~/.zcode/cli/config.json` + client envs), then default hosts to `127.0.0.1`. Everything else in this report compounds or is compounded by this.

## P1

### P1-1 · Default `admin/admin` is silently materialized into users.json ✅
`tools/shared/users_store.py:536-553` — `admin_password = os.environ.get("MCP_UI_PASSWORD") or "admin"`; `_ensure_seeded()` (line 598) runs on the first `central_tokens()` call, which the central API's `_verify_api_key` makes on **every** authenticated request (`launcher/management_server.py:357-360`). A role=admin PBKDF2("admin") record is auto-created even in mono deployments. UIs bind loopback by default (the only mitigation). **Fix:** refuse to seed / refuse login with the default password on any non-loopback bind.

### P1-2 · Plaintext `mcp_key` persisted to world-readable NiceGUI session files ✅
`mcp_ui/management_ui.py:455` stores the raw `mcp_key` via `nicegui_app.storage.user` → `.nicegui/storage-user-*.json`. Verified on disk: **5 files contain a real 43-char key, mode 664**; the prune docstring (`management_ui.py:1219-1221`) claims "nothing of value inside"; files outlive logout and key rotation. **Fix:** keep only username/role in `storage.user`, fetch key per request; `chmod 600` storage dir; purge on logout/rotate.

### P1-3 · `textToSmartGraph` accepts caller-supplied `base_url`/`api_key` → arbitrary-endpoint exfiltration ✅
`tools/memorymcp/memory_text.py:490-536` — the MCP tool takes `base_url` and POSTs the full `text` there. Any authenticated caller (in practice: an agent, possibly prompt-injected) can send memories/skill docs/source code to any URL. Contrast: `optimize_sql_with_ai` reads `AI_BASE_URL` from env only (correct). **Fix:** drop the params; env-only like optimize_sql_with_ai.

### P1-4 · `PostgresVectorStore` interpolates collection names into SQL with no sanitizer ✅
`tools/shared/impls/postgres_vector.py:105,123,133,139,143…` — `f"...vec_{collection}..."` everywhere; **no** `_s()` identifier check (only false-positive matches `_schema_sql`/`_serialize_vector`). The Turso twin has one (`turso_vector.py:74-86`, raises on non-`[A-Za-z0-9_]`). `collection_name` is a raw MCP-caller param in ragmcp (`ragmcp_fastmcp.py:512,872`). Paramless psycopg3 calls → multi-statement injection; parameterized calls still allow FROM-clause rewrite. **Latent** (PG+pgvector combo not active; live backend is Turso+Turso) but documented-supported. **Fix:** port `_s()` into postgres_vector.py.

### P1-5 · Test cluster publishes management APIs on all interfaces
`deploy/compose-common.yml` — `- "18200:8200"`, `- "19200:8200"`, `- "18002:8002"`, `- "19003:8003"`, `- "18400:8400"` with no `127.0.0.1:` prefix (podman publishes 0.0.0.0). Contradicts the project's own rule in `deploy/compose-lb.yml:2-4` (central mgmt "must not be reachable from other interfaces"); the LB itself is correctly pinned to loopback. **Fix:** prefix every node publish with `127.0.0.1:`.

## P2

| # | Finding | Evidence | Fix |
|---|---------|----------|-----|
| P2-1 | 81xx mgmt servers fail open without `auth.api_key`, and ignore the env key the MCP port honors → locked MCP port + **wide-open** 81xx (`/admin/function-masks`, extension query/mutate) | `launcher/tool_extensions/http_server.py:149-152,166-167`; fed only config.json by `launcher/server_manager.py:119-120` vs env-first resolution at `server_factory.py:690-708` | derive 81xx key from the same `_resolve_api_key()` chain; fail closed |
| P2-2 | Central API + 81xx fail OPEN when `MCP_MANAGEMENT_API_KEY` unset (loud warning, still serves; currently set everywhere → mitigated in practice) | `launcher/management_server.py:304-308,346-347`; `http_server.py:149-151` | refuse mutation routes without a key |
| P2-3 | E3.5 data-plane grants fail open on every anomaly path (no HTTP context, token not in map, preset lookup failure → allow) | `tools/databasemcp/db_tools.py:553-599`; same shape `connections.py:385-405`, `tools/ragmcp/ragmcp_fastmcp.py:85-106` | fail closed when a verifier exists but identity can't resolve |
| P2-4 | ragmcp indexing is unconfined: caller-supplied `workspace_root` — index `~/.ssh`, `/etc`, the repo itself; read back via `search` (argv-safe, no command injection) | `ragmcp_fastmcp.py:511,532`; `incremental_indexer.py:769+` | `allowed_roots` check on `workspace_root.resolve()` |
| P2-5 | ragmcp log-file path traversal via `collection_name` (`../../x` writes outside `tools/ragmcp/logs/`) | `ragmcp_fastmcp.py:542,1429` | validate collection name before any path use |
| P2-6 | WS auth token accepted from query string; uvicorn access_log default ON → `?token=<key>` lands in the log `/api/logs` serves | `management_server.py:573,972-973` | header-only WS auth or `access_log=False` on 8200 |
| P2-7 | Metrics server 8300 (legacy `launchmcp.py` only): unauthenticated `/metrics` on 0.0.0.0 exposing tool names/latencies; `fef_metrics.py:47` same class, currently dead code | `launchmcp.py:565,576`; `launcher/monitoring/fef_metrics.py:47`; `config/monitoring_config.json` `"enabled": true` | pass the API key or bind loopback; gate the dead path |
| P2-8 | Management UI binds 0.0.0.0 plain HTTP (`.env`/`work.env` override the safe 127.0.0.1 default; work UI uses `network_mode: host`) | `mcp_ui/management_ui.py:1247` vs root `.env` `MCP_UI_HOST=0.0.0.0` | loopback or front with the TLS'd nginx |
| P2-9 | World-readable DB backups with plaintext user keys: `pg_dump > out` no chmod; verified on disk `~/supreme-mcp-tools-backups/work-pg-20260919-063001.sql` mode 644 containing 3 `mcp_key`s + `password_hash`es | `deploy/startcluster.sh:156-160,219-223` | `umask 077` in backup/export paths; fix the existing file |
| P2-10 | Local secret files 664: `deploy/node.env`, `deploy/.env`, 7 `.env.bak-*` files with live third-party keys + `MCP_UI_SECRET` + DB/UI passwords (all gitignored — local exposure only) | verified `stat` | `chmod 600`; delete stale `.bak`s |
| P2-11 | Node containers run as root with sensitive host mounts (`users.json`, `tools_config.json`, turso data, HF cache) + `network_mode: host`; Containerfile has no `USER` | `deploy/Containerfile`, `deploy/compose-work.yml` | non-root USER; chown mounts |

## P3

- **Bandit B324 (its only High):** MD5 chunk point-IDs in `incremental_indexer.py:612` — non-crypto dedup ID, accepted as-is (optionally add `usedforsecurity=False`).
- **pip-audit:** `ecdsa 0.19.1` PYSEC-2026-1325 (Minerva timing on P-256), transitive via python-jose 3.5.0; **no fixed release exists yet**; unreachable in practice (no JWT/ECDSA usage anywhere — verified). Re-check when a fixed ecdsa ships.
- `/health` (8200 + 81xx) unauthenticated, returns tool list/topology (fingerprinting) — `management_server.py:396-404`.
- No login rate-limiting/lockout in either UI (`management_ui.py:448-464`, `pod_ui/auth.py:90-95`) — realistic only combined with P2-8/P1-1.
- flush-sessions admin gate `except Exception: pass` fail-open (auth still required; impact = session-kill DoS) — `server_factory.py:641-642`.
- Post-auth identity lookups use plain `dict.get(token)` instead of compare_digest (`identity.py:71`, `db_tools.py:587`, `ragmcp_fastmcp.py:102`, `server_factory.py:515`, `http_server.py:173-175`) — consistency gap, not practical.
- Failed central-API auth attempts not audit-logged; two polled reads log at DEBUG only — `management_server.py:349-375`.
- Default MinIO password `change-me-minio-secret` (loopback-pinned; rotate before any non-loopback bind); `MC_HOST` briefly ps-visible via `podman exec -e` — `deploy/.env:7`, `startcluster.sh:538,557,727`.
- `logs/` 664; legacy SSE session ids in URLs logged — `server_factory.py:205-208`.
- UI credential fallback admin/admin (`management_ui.py:243-244`, `pod_ui/auth.py:26-27`) — see P1-1.
- databasemcp presets point straight at the live memory store (dangerous-by-design; `execute_sql` is the product; no write allowlist by design; `_read_guard` self-describes as accident prevention) — `.env:110,114`.
- convertermcp `allowed_roots` = whole repo + /tmp (gate itself correctly implemented) — `tools/convertermcp/config.json`.
- webmcp/convertermcp SSRF checks run post-fetch (blind SSRF residue) + DNS-rebinding TOCTOU; `is_internal_url` itself is thorough — `webmcp_fastmcp.py:781-784`, `convertermcp_fastmcp.py:143-146`.
- `optimize_sql_with_ai` ships SQL + table schemas to the configured AI gateway (operator-controlled URL; informational data-sharing surface) — `db_tools.py:996-1043`.

## Verified clean (coverage record)

- **SQL (active backend, turso_sql/turso_vector):** values parameterized throughout; `_s()` identifier sanitizer enforced; FTS5 MATCH escaped; filter values bound; HIGH-14 trgm param-order fix intact.
- **Command injection:** zero `shell=True`/`os.system`/`os.popen` repo-wide; all 5 subprocess sites argv-list.
- **Deserialization:** no pickle/`yaml.load`/eval/exec/marshal anywhere.
- **CORS:** only two sites, both localhost origins, `allow_credentials=False`.
- **Key comparison:** `hmac.compare_digest` on every gate (central, per-tool, UIs, passwords PBKDF2-SHA256 390k with timing equalization).
- **No JWT usage** anywhere; no hardcoded dev key (missing key → random-generated + warning).
- **E3 reachability:** unauthorized-server keys rejected pre-`tools/call`; masks enforced for everyone incl. admins; last-admin demotion refused.
- **Secret echo:** `GET /api/tools/{t}/auth`, `/api/env`, `/api/users` all masked/public-view; create/rotate return key once by design.
- **Logs:** `mcp.access` lines carry session prefix/user/status only — no key material (P2-6 is the one concrete exception found).
- **XSS:** no `ui.html`/`ui.markdown`/innerHTML sinks in either UI; NiceGUI label escaping only.
- **XXE:** python-docx 1.2.0 disables entity resolution. **Zip:** no extractall/zip-slip surface. **TLS:** no `verify=False` anywhere.
- **harvest bundles:** dir 700 / secrets 600; `--redact` verified to blank keys/hashes/presets; `env_strip.py` mechanism matches docstring; images built secret-free ("image-sanitized" stage).
- **startcluster.sh (52KB):** no curl|bash, no sudo, no 777, no privileged/docker.sock/`/` mounts.
- **Loopback pins:** LB 18080, work PG 5433, MinIO 19000 — all correct; nginx/minio digest-pinned.
- **pod_ui:** auth gate on every page, stdin-confirmed wipe actions, single-op mutual exclusion.
- **Sessions:** idle TTL 1800s default; flush endpoint key-gated; open-redirect guard on post-login redirects.
- **Process user:** host launcher/UIs run as `gr`, never root.

## Suggested fix order

1. **Rotate the six tool API keys** (P0-1, user decision; unblocks everything).
2. Quick wins, one-liners: `127.0.0.1:` prefixes in `compose-common.yml` (P1-5); chmod 600 on the 664 secret/session/backup files (P1-2 partial, P2-9, P2-10); drop `base_url`/`api_key` params from `textToSmartGraph` (P1-3); header-only WS auth or access_log off (P2-6).
3. Small code fixes: `_s()` port to postgres_vector (P1-4); 81xx key resolution + fail-closed (P2-1, P2-2); E3.5 grant fail-closed (P2-3); ragmcp `allowed_roots` + collection-name validation (P2-4, P2-5); default-password seeding guard (P1-1); storage.user de-clerk (P1-2 remainder).
4. Decide-and-document: metrics auth (P2-7), UI bind (P2-8), container USER (P2-11), preset-vs-memory-store segmentation (P3), rate limiting (P3).
