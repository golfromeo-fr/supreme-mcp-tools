# M4 multi-host branch code review

**Date:** 2026-09-24  
**Branch:** `feature/m4-multihost`  
**Compared with:** `main...HEAD`  
**Merge base:** `3bcbb5eab0daf5a6f999b56c91b599086325891d`  
**Reviewed HEAD:** `835ded5`  
**Status:** Review only; no fixes were applied.

## Scope and verification

The branch contains the M4 multi-host runtime, shared state, config bundles, deployment tooling, and the v1/v2/v3 pod UI. The review covered the committed three-dot diff only. The pre-existing working-tree edits to `deploy/ARCHITECTURE.md`, `deploy/PODS-EXPLAINED.md`, and `deploy/README.md`, plus `gui-test-screenshots/`, were excluded.

Verification performed:

- `python -m pytest -q` — **981 passed**.
- Focused bundle, pod UI, cluster-state, deployment-config, and state-doc tests — **59 passed**.
- `bash -n` for all deployment shell scripts and root wrappers — **clean**.
- `git diff --check main...HEAD` — **clean**.
- A live temporary-file probe confirmed that two same-version `state_docs.save_doc_cas()` calls can both succeed because the version stamp has only one-second resolution.
- A shell probe confirmed that a failed left side of an `&&` chain under `set -euo pipefail` can still be followed by an unconditional successful exit.

The repository has no `docs/agents/issue-tracker.md`; issue-linked commit references could not be independently fetched. If that workflow is required, run `/setup-matt-pocock-skills`.

## Standards findings

### P0 — Tool credentials can be baked into published images

`deploy/Containerfile:35` copies the entire `tools/` tree, including all six tracked `tools/*/config.json` files. Those files contain non-empty, non-placeholder `auth.api_key` values. `deploy/publish-image.sh:22-35` can push the resulting image, while `.containerignore` does not exclude the tool configs. This contradicts the image's own comment that secrets are not baked into the image.

Sanitize or exclude runtime configs before publishing, and rotate any credentials that may already have been published.

### P1 — The documented primary launcher bypasses M4 bootstrap and configured management authentication

`launcher/__main__.py:33-36` only loads `.env`; it does not adopt shared env/auth overrides or register the node. At `launcher/__main__.py:273-279`, it passes only `args.api_key` to `ManagementServer`, rather than resolving `MCP_MANAGEMENT_API_KEY` from the environment.

The legacy path does this correctly at `launchmcp.py:28-39` and `launchmcp.py:635-647`. Therefore, `python -m launcher` can start with stale cluster configuration and an unauthenticated central API even when the key is present in `.env`.

### P1 — Shared-state write failures are reported as successful mutations

`tools/shared/cluster.py:131-146` returns a boolean from `mirror_env()` and `mirror_auth()`, but callers ignore it at `launcher/env_manager.py:369-375`, `launcher/env_manager.py:408-414`, and `launcher/management_server.py:913-920`.

A failed shared write can therefore return HTTP success, update only local state, and disappear on the next node boot—especially when the mounted `.env` is read-only.

### P1 — Environment deletion does not propagate to sibling nodes

`cluster.mirror_env(name, None)` removes the key from the shared document, but `adopt_env_overrides()` at `tools/shared/cluster.py:149-164` only applies keys that remain present. A sibling that already has the old value in its environment or mounted `.env` keeps the deleted variable.

This is especially serious for revoked credentials and configuration: central deletion is not enforcement.

### P1 — The state-document compare-and-swap lock has one-second resolution

`tools/shared/state_docs.py:93-94` stores `updated_at` with only second-level precision. `save_doc_cas()` at `tools/shared/state_docs.py:200-227` uses that value as the optimistic-lock version.

Two writers using the same expected version can both pass the update condition. A temporary-file probe confirmed that both calls returned `True` and the second payload silently replaced the first. The documented CAS behavior is therefore not safe for concurrent same-second cluster updates.

### P1 — The destructive pod UI defaults to all interfaces with `admin/admin`

`pod_ui/app.py:16` defaults to `0.0.0.0`, while `pod_ui/auth.py:24-29` falls back to `admin/admin`. Unlike the existing management UI, there is no loopback default or warning for this destructive host-side app.

Default the pod UI to `127.0.0.1` or require an explicit opt-in for remote binding, and warn clearly when default credentials are active.

### P2 — v3 single-container actions bypass the sole operation gateway

`pod_ui/ops.py:1-4` declares `OpRunner` as the only process spawner, but `pod_ui/pages_v3/home.py:91-95` directly calls `subprocess.Popen`. These actions bypass the global lock, streamed history, and process-group cancellation, allowing them to race full-cluster operations.

### P2 — Freshly generated secret-bearing env files are group/world-readable

`deploy/startcluster.sh:219` and `deploy/startcluster.sh:495` redirect secret-bearing output directly into `work.env` and `node.env` without `umask 077` or a final `chmod 600`. On a normal `0022` umask, a newly created file is mode `0644`.

### P2 — Work preflight omits port 8400 after already stopping the environment

`deploy/startcluster.sh:310-314` checks ports 8000-8005 and 8200, but `deploy/startcluster.sh:327-328` starts the UI profile on port 8400. A host management UI already listening on 8400 can therefore cause a partial startup after the old work environment has been torn down.

### P2 — The documented root `./startcluster` command is not tracked

`AGENTS.md:27-30` documents `./startcluster`, but `.gitignore:261` ignores the root wrapper and `git ls-files` contains no tracked `startcluster` file. A fresh clone can read the instructions but cannot execute the documented command unless the user creates the wrapper manually or invokes `deploy/startcluster.sh` directly.

### P2 — Test cleanup omits the LB and MinIO compose layers

`deploy/startcluster.sh:466-470` and `deploy/startcluster.sh:516-520` do not pass `compose-lb.yml` or `compose-artifacts.yml` and only explicitly remove `node1`, `node2`, and `db`. The clean/topology-transition paths can leave LB and MinIO containers running, causing partial clusters and occupied ports.

### Judgement call — repeated pod-UI action mappings

The same work/test action mapping is repeated in `pod_ui/pages/bundles.py:28-31`, `pod_ui/pages_v2/bundles.py:25-28`, and `pod_ui/pages_v3/bundles.py:26-28`. The v1/v2/v3 coexistence is intentional, but centralizing the mapping would have prevented the bundle-boot defect below.

## Specification findings

The specification sources reviewed were `plans/m4-multihost-2026-09-08.md`, `plans/m4-review-fixes-2026-09-14.md`, `plans/pod-config-bundle-2026-09-17.md`, `plans/pod-ui-plan-2026-09-19.md`, and `plans/post-m4-roadmap-2026-09-13.md`. Items explicitly marked resolved, deferred, or deliberately kept in the M4 review tracker were not re-proposed.

### P1 — Every pod-UI “Boot from bundle” action silently starts host mode

The v1, v2, and v3 bundle pages pass the bundle as a positional argument, for example `pod_ui/pages/bundles.py:28-31`. The script only consumes `--config-bundle`/`-b`, so the positional path is ignored. `tests/test_pod_ui_helpers.py:78-81` currently codifies the incorrect command shape.

### P1 — Bundle re-seeding passes a password value where the generator expects a filename

`deploy/startcluster.sh:215-220` and `deploy/startcluster.sh:491-496` extract `POSTGRES_PASSWORD` into `OLD_PW`, but `deploy/bundle_node_env.py:62-70` treats its third argument as an env-file path.

Existing PostgreSQL passwords are consequently discarded and replaced with a new random password, making the existing `pgdata` volume inaccessible on re-up.

### P1 — `external-pg` bundles do not emit PostgreSQL connection settings

`deploy/bundle_node_env.py:141-165` drops every `STATE_KEYS` entry before reaching the `external-pg` pass-through logic. Since external topologies start without an embedded database, the generated `node.env` has no usable external `POSTGRES_*` configuration.

### P1 — `external-turso` bundles discard the network authentication token

`deploy/bundle_node_env.py:145-160` preserves `TURSO_AUTH_TOKEN` for `external-pg` but drops it for `external-turso`. A network Turso database requiring token authentication will be unreachable.

### P1 — Export/import commands can return success after failure

The `tar`/`podman cp` operations at `deploy/startcluster.sh:191-199`, `deploy/startcluster.sh:201-209`, and `deploy/startcluster.sh:457-465` are followed by unconditional `exit 0`. Under `set -e`, a failed left side of an `&&` chain does not terminate the script.

Corrupt archives or failed MinIO operations can therefore be reported as successful, allowing a migration workflow to continue with incomplete data.

### P2 — `--config-bundle` is not ignored for non-up work actions

The dispatcher resolves the bundle before entering `work_env` at `deploy/startcluster.sh:624-628`, and `work_env` rewrites `work.env` before its action switch at `deploy/startcluster.sh:215-220`.

A missing bundle can make `stop`/`status` fail, while a valid bundle can unexpectedly mutate generated state during those operations. The bundle plan explicitly requires the flag to be ignored for non-up actions.

### P2 — Management UI inventory can be seeded from masked discovery

`mcp_ui/management_ui.py:535-552` performs discovery when a server's inventory is empty and then replaces the entry with the discovered list. That discovery can be filtered by enforced masks, so functions can be permanently omitted from the shared inventory. The union protection in `launcher/tools_config.py:358-368` does not cover this separate path.

### P2 — Bundle compose generation fails for optional and space-containing paths

The loops at `deploy/startcluster.sh:265-270` and `deploy/startcluster.sh:502-509` do not use `nullglob`; a missing optional config directory produces a literal `*.json` mount. Generated YAML does not quote paths, and `deploy/startcluster.sh:244` uses `xargs` for a potentially space-containing Turso path.

This violates the bundle plan's optional-file and path-with-spaces requirements.

### P2 — Harvest re-runs do not preserve manifest history

`deploy/harvest-config.sh:212-263` overwrites the bookkeeping and manifest without retaining a one-level `previous` summary, contrary to the bundle plan's E14 requirement.

### P2 — Redaction does not implement the explicit `POSTGRES_*` rule

The regex at `deploy/harvest-config.sh:163-173` only catches keys ending in `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, or `DSN`. `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, and `POSTGRES_DB` remain in a supposedly shareable redacted bundle.

### P2 — v3 direct `podman` execution is also a specification violation

The v3 route is live at `pod_ui/app.py:69-80`, but its destructive actions bypass the F9 requirement that every pod operation go through the locked, streamed `startcluster` runner.

## Merge recommendation

Do not treat the branch as merge-ready until the P0 credential-publication issue and the P1 startup, shared-state, bundle, and migration issues are fixed or explicitly vetoed. The green test suite is useful regression evidence, but it does not exercise the affected integration paths.
