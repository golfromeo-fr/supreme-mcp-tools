# Grounds — disposable real-pod testing environments (tier 2)

Design: `plans/testing-grounds-and-work-topology-2026-10-05.md` (§3).
Manager: `./startgrounds` (Phase 1 — does not exist yet; this directory is
the Phase-0 skeleton).

A **ground** is a throwaway pod booted from a pinned config bundle via the
existing `startcluster create` registry, used by the tier-2 smoke probes
(`tests/test_pod_smoke.py`, opt-in via `POD_GROUNDS=<name>`). Grounds are
recreatable at will; the REAL clusters (`work`, `test`) are never shaped by
what the tests need.

## Spec files (committed, key-free)

One `<name>.spec` per ground — `<key> <value>` lines, `#` comments:

```
# ground spec — key-free by policy (bundles carry the secrets, gitignored)
nodes 1
topology pg
tools databasemcp webmcp simplemcp convertermcp ragmcp memorymcp
bundle my-bundles/pre-pod-test        # relative to the repo root
sha256 <hash-of-bundle-contents>      # written/refreshed by startgrounds; drift -> ensure recreates
probes smoke-work                     # probe list (design §3.5)
```

Nothing in a spec may contain an API key, password, or token — bundles stay
in gitignored `my-bundles/` (F11). Planned grounds: `g1-pg` (Phase 1),
`g1-turso` (Phase 2), `g2-pg`/`g2-turso` (Phase 3), `gwork-*` (Phase 2,
work-layout via `--layout work`).

## Rules that matter (from the design)

- `ensure` recreates a drifted ground by default; `probe` never recreates
  (evidence preservation).
- Grounds are data-plane isolated by construction — they mount NO host data
  in v1; the Phase-2 `gwork-*` layout mounts COPIES of identity/data, never
  the real files.
- Probe failure keeps the ground running for inspection; exit code is the
  signal, not auto-repair.
