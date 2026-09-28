# Automation accessories (startcluster + ZCode)

Re-creatable definitions for the automation that supports the podman
environments. These are **not** installed by the repo — apply them on a new
system using the commands below.

## Related local scripts (not in git)

| Script | Role |
|---|---|
| `startlauncher` (repo root) | host working stack: 6 tools + central 8200 |
| `startui` (repo root) | management UI on 8400 |
| `startcluster` (repo root) | podman environments: `work` (daily driver, canonical ports) and `test` (M4 bench, ports 18xxx/19xxx) |

`startcluster` is generated/maintained on the machine that runs podman; a
copy of the currently-proven version lives at `deploy/startcluster.sh`
(sync it down if your local copy drifts).

## Nightly work-env backup (ZCode automation)

**What it does:** `pg_dump` of the work state plane (users, masks,
inventory) into `~/supreme-mcp-tools-backups/work-pg-<timestamp>.sql`,
then prunes to the 14 newest backups. Skips gracefully if the work cluster
is not running.

**Recreate on a new ZCode instance** (CronCreate, or ask the agent):
```
prompt: |
  Nightly backup of the supreme-mcp-tools work-env state plane. Run exactly:
  cd /home/gr/supreme-mcp-tools && ./startcluster work backup
  ...it pg_dumps the work Postgres (users, masks, inventory) to
  ~/supreme-mcp-tools-backups/. If it fails because the work cluster is not
  running, that is acceptable — reply with one line saying so. On success,
  prune old backups: keep only the 14 newest files matching
  ~/supreme-mcp-tools-backups/work-pg-*.sql (delete older ones). Reply in
  2-3 lines maximum: backup path + size, pruning result.
title: "Nightly work-env PG backup (06:30)"
cron: "30 6 * * *"
recurring: true
```

# Automation accessories (startcluster + ZCode)

Re-creatable definitions for the automation that supports the podman
environments. These are **not** installed by the repo — apply them on a new
system using the commands below.

## Related local scripts (not in git)

| Script | Role |
|---|---|
| `startlauncher` (repo root) | host working stack: 6 tools + central 8200 |
| `startui` (repo root) | management UI on 8400 |
| `startcluster` (repo root) | podman environments: `work` (daily driver, canonical ports) and `test` (M4 bench, ports 18xxx/19xxx) |

`startcluster` is generated/maintained on the machine that runs podman; a
copy of the currently-proven version lives at `deploy/startcluster.sh`
(sync it down if your local copy drifts).

## Nightly work-env backup (ZCode automation)

**What it does:** `pg_dump` of the work state plane (users, masks,
inventory) into `~/supreme-mcp-tools-backups/work-pg-<timestamp>.sql`,
then prunes to the 14 newest backups. Skips gracefully if the work cluster
is not running.

**Recreate on a new ZCode instance** (CronCreate, or ask the agent):
```
prompt: |
  Nightly backup of the supreme-mcp-tools work-env state plane. Run exactly:
  cd /home/gr/supreme-mcp-tools && ./startcluster work backup
  ...it pg_dumps the work Postgres (users, masks, inventory) to
  ~/supreme-mcp-tools-backups/. If it fails because the work cluster is not
  running, that is acceptable — reply with one line saying so. On success,
  prune old backups: keep only the 14 newest files matching
  ~/supreme-mcp-tools-backups/work-pg-*.sql (delete older ones). Reply in
  2-3 lines maximum: backup path + size, pruning result.
title: "Nightly work-env PG backup (06:30)"
cron: "30 6 * * *"
recurring: true
```

**Without ZCode** — plain crontab equivalent:
```
30 6 * * * cd /home/gr/supreme-mcp-tools && ./startcluster work backup && ls -1t "$HOME/supreme-mcp-tools-backups"/work-pg-*.sql | tail -n +15 | xargs -r rm
```

## Restore (manual, from any backup)

```bash
./startcluster work restore ~/supreme-mcp-tools-backups/work-pg-<ts>.sql
# asks for typed confirmation ('wipe'), recreates the pgdata volume,
# imports, waits for the node central on :8200
```

## Identity of the automation on this machine

- ZCode automation id: `automation-76ab0533-695c-4d87-947e-1291c0bfdc84`
  (list with CronList; delete with CronDelete before recreating).
- Backup directory: `~/supreme-mcp-tools-backups/` (also holds the
  pre-M4 full config snapshot `20260910-033925-pre-h2b-full/`).


## Restore (manual, from any backup)

```bash
./startcluster work restore ~/supreme-mcp-tools-backups/work-pg-<ts>.sql
# asks for typed confirmation ('wipe'), recreates the pgdata volume,
# imports, waits for the node central on :8200
```

## Identity of the automation on this machine

- ZCode automation id: `automation-76ab0533-695c-4d87-947e-1291c0bfdc84`
  (list with CronList; delete with CronDelete before recreating).
- Backup directory: `~/supreme-mcp-tools-backups/` (also holds the
  pre-M4 full config snapshot `20260910-033925-pre-h2b-full/`).


## Status on this machine (2026-09-16)

- The ZCode automation was PAUSED without ever firing (runCount 0) —
  nightly backups now run via the installed **user crontab** entry
  (verified with `crontab -l`), independent of ZCode.
- The paused automation (id `automation-76ab0533-…`) can be re-enabled in
  ZCode or deleted; the crontab line makes it redundant.
