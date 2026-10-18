# Pod-UI config writes — central persistence design (audit (d) / TODO 2026-09-25 (d))

**Status:** DESIGN ONLY — not approved, not scheduled. Build only if the
in-pod tools-editing flow is actually used.
**Scope:** the in-pod management UI (`work ui` → mcp_ui inside the work
container) writing tool configuration. Host-side `mcp_ui`/`pod_ui` are
unaffected (they already reach the central API or shell out to
`startcluster` on the host).

## The problem

`mcp_ui.management_ui._save_tools_config()` writes
`tools/<name>/config.json` **as a local file**. Inside the work container
that path is one of:

| deployment | `/app/tools/<n>/config.json` is | effect of a direct write |
|---|---|---|
| host dev (startlauncher) | the real tracked-then-untracked file | works (the only case it was designed for) |
| pod, no bundle mount | container-layer file | lost on every rebuild (`work up` re-creates the container) |
| pod, bundle mode | **`:ro` bind mount** of the bundle's file | write fails (read-only FS error surfaced in the UI) |

Either way the write never reaches the place the pod's next boot reads from
(the bundle), and the bundle is documented as the source of truth
("the BUNDLE is the source of truth — edit there and re-run startcluster").

## Target design — all config writes go through the central API

The central management API (8200, inside the pod) is the one component that
(a) is authenticated, (b) runs in the same mount namespace as the tools, and
(c) already owns the M4 shared-state mirror. It becomes the only writer.

### P1 — central endpoint

`PUT /api/tools/{tool}/config` (body: the full config JSON; the existing
`PUT /api/tools/{tool}/auth` stays the narrow, audited key-rotation route):

1. Validate against the tool's config schema (auth nesting, known keys).
2. Write the file **at the launcher's own config path** — the same path the
   launcher reads at boot — via `tools/shared/atomic_io` (flock + tmp +
   replace; never a partial file).
3. Mirror a redacted copy into `state_docs` (`tools_config` doc, M4 H2) so
   a second node adopts it and the write is reconstructible after a
   container rebuild even when the file write landed on a :ro mount.
4. Audit-log the write (`central.access`: who, tool, which top-level keys
   changed — never values).

Refuse (409) when the config path is a read-only mount AND the state_docs
mirror is unavailable — with a message telling the operator to edit the
bundle and re-up (the documented flow), instead of a silent FS error.

### P2 — mcp_ui switches to the client call

`_save_tools_config` is replaced by the existing central api_client call
(the UI already talks to 8200 for masks/users; same auth, same
attribution). The direct file write is deleted, not feature-flagged —
keeping both writers is how the drift happened.

### P3 — pod bundle flow (the real decision)

Two options, pick one when P1/P2 exist:

- **3a. rw config mounts** — drop `:ro` on the bundle config mounts and
  chown them to the node user. Writes then land in the bundle on the host
  (the bundle stays source of truth, edits survive, `work up` needs no
  rebuild). Cost: the container can now alter its own auth material —
  acceptable because every write is central-authenticated and audited.
- **3b. config-in-db** — state_docs becomes the source of truth for tool
  configs inside a pod; at boot the launcher materializes the doc over the
  image defaults; bundle configs act as first-boot seeds only. Cost: two
  source-of-truth regimes (host dev = file, pod = db) and a migration.

Recommendation: **3a** — it preserves the documented bundle philosophy and
is ~20 lines across the compose templates.

## Edge cases carried from the audit/bands

- :ro mounts (the 409 path above); concurrent UI writes (atomic_io flock);
  key echo (auth values masked on read-back, rotate returns once);
  E3 masks (a write must not be able to unmask the caller's own mask —
  the existing `disabled_tools` central routes already gate that);
  the `config.example.json` bootstrap (bf75e24) must not overwrite a
  central-written config (it doesn't — bootstrap only fires when the file
  is absent).

## Effort

P1 ~half a day (endpoint + tests), P2 ~1h, P3a ~20 lines + one `work up`
verification. Gate: do it only when the in-pod tools-editing flow matters
in practice; until then the documented flow is "edit the bundle, re-up".
