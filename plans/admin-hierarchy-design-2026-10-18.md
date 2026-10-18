# Admin hierarchy (master vs sub-admins) — DESIGN ONLY, 2026-10-18

Status: design for the last unstarted H-series item. NOT scheduled; build
only when a real second administrator exists. (The other H items are done
or vacuous: users-db = H1 shipped, state mirror = H2/H2b shipped, artifacts
= volume shipped, second-node runbook = H4 shipped, per-user central tokens
= shipped (6d66a25), prompts/resources identity parity = vacuous — no
prompts/resources registered anywhere; guard-note added to identity.py.)

## Current model

`role` is binary (admin|user). Admins are full peers: user/role/mask/grant
management, 81xx actions, central mutations; the system key is attributed
to the mono admin; last-admin demotion is refused; central admin tokens
exist (one per admin, veto-able one-key-per-admin).

## The ask (when it comes)

A MASTER admin (the owner) plus SUB-ADMINS with bounded powers — e.g. a
sub-admin who can onboard users and set masks but cannot rotate the master's
key, cannot edit databasemcp presets, cannot write tool configs.

## Design sketch

1. **Capabilities, not a third role.** Add `admin_capabilities` to the user
   record (default for role=admin: the full set; sub-admin = a subset).
   Capability keys mirror the central route families:
   `users_manage`, `masks_manage`, `grants_manage`, `env_manage`,
   `config_write`, `nodes_manage`.
2. **Enforcement points** (each already has an admin gate to extend):
   central routes (`_verify_api_key` -> capability check per route family),
   81xx actions (`_require_admin` -> the same check via the caller's token
   entry), UI (Users tab sections render per capabilities).
3. **Master invariants**: exactly one master (the seeded admin); masters
   are never capability-bounded; sub-admins cannot grant capabilities they
   do not hold (no escalation); the last-admin guard becomes the
   last-MASTER guard; sub-admin-created users are attributed (created_by)
   and reassignable.
4. **Token scoping**: central admin tokens inherit the admin's capability
   set; rotation unchanged.
5. **Migration**: existing admins become masters automatically; no store
   migration beyond the new optional field (absent = full set, so old
   stores behave identically — iso-features).

## Effort

~2-3 days: capability matrix + route-family checks (~1d), UI rendering
(~0.5d), tests for the escalation matrix (~1d). The escalation matrix
(sub-admin cannot self-promote/grant-unheld) is the test bulk.
