# Post-M4 roadmap (2026-09-13, feature/m4-multihost)

M4 is functionally complete: H1 (shared identity), H2 (shared masks +
inventory), H2b (env/auth mirror + mutation + boot adoption), H4 (runbook,
topologies, podman/compose), plus the cluster runtime pieces — node
registry, runtime mask fan-out, central env/auth mutation with boot
adoption, and artifact durability via declared volumes. All live-verified
on the two-node test cluster; a `work` pod environment runs as a daily
driver.

## Phase A — certification (before anything merges)

1. **Soak the work env.** Use the pod as the daily driver for a week:
   rebuilds (`startcluster work`), restarts, identity (keys/masks), data
   planes, UI (profile `ui`). Every anomaly is a branch-fix, not a
   workaround.
2. **Make the e35 suite environment-aware.** It currently manages users via
   the host `users.json`, so it can only test an identity plane that shares
   that file (host launcher). Switch its user lifecycle to the central
   API (`/api/users`) → the suite can then validate ANY environment (host,
   work pod, test cluster) without touching host state.
3. **Backup story for the work state plane**: nightly `pg_dump` of the
   work pg (users/masks/inventory) — one cron line, restore documented in
   the runbook.
4. **Merge decision** (user gate: "not until a long time"). Suggested
   trigger: work env soaked + one real second host deployed, or explicit
   user certification.

## Phase B — a real second host (M4 production truth)

The current cluster is single-machine. The genuine multi-host claims get
proven by:
1. Node 2 on a second machine/VM: identical image + `.env`, DSNs pointed at
   the network PG/Turso (never file:), central reachable cross-host.
2. LB (TLS, health-checked) over the node centrals; sticky sessions only
   for stateful `/mcp` clients; prefer `/mcp-stateless` everywhere.
3. Log aggregation: ship `mcp.access` (user= attribution included) to one
   place — the durable per-user audit trail.
4. Watch items from the scaling doc: per-node tool concurrency, the
   GIL-bound local embedding model (bge-m3) — consider remote embeddings
   when more than one busy node shares the quota.

## Phase C — postponed items, re-ranked by the pod reality

- **Packaging** (was postponed): the work env made it MORE relevant —
  the image IS the package now. Remaining: publish the image to a registry,
  version-pin releases, one-command host bootstrap.
- **Graphify** (was postponed): unchanged value; run it against the repo
  when planning resumes.
- **Artifact S3 backend**: ArtifactStore already supports S3 — configuring
  it removes the last node-local state, making nodes truly stateless
  (alternative to the artifacts volume for multi-host).

## Technical debt noticed during M4 (small, opportunistic)

- `tools_config.json` had three implementations; the UI now delegates —
  finish collapsing the remaining duplicate readers.
- `cluster_nodes` entries never expire (a dead node's entry lingers until
  manually removed); fan-out tolerates them, but a heartbeat/TTL would
  keep the registry honest.
- Adoption logging is WARNING-level by design (boot visibility); downgrade
  to INFO once the cluster is boring.
- The e35 suite's per-run debris (concurrent-test memories) could be
  swept by its cleanup like the other fixtures.

## Explicitly NOT planned (unless reality demands)

- Distributed transactions / consensus — last-writer-wins per document is
  the documented contract; the workloads are single-writer by nature.
- A dedicated auth/config micro-service (design Option C) — Option B's
  shared-store approach has not strained.
- Dynamic node discovery (network scanning) — explicit registration via
  env is simple and sufficient at this scale.
