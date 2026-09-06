"""
Configuration package for the launcher.

The FEF V3 persistence backends (JSON/SQLite ConfigPersistence, ConfigManager,
HA/distributed config) were deleted 2026-09-05 — they never ran in production
(design pass D1, plans/launcher-persistence-design-2026-09-05.md). The one
live mutation-log writer is `launcher.distributed_registry.ConfigPersistence`;
retention/secrets decisions for it are D3 (pending).
"""
