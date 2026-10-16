# ground gwork-pg — Phase 2a: WORK-LAYOUT ground (grounds plan §4/D7-D8)
# host networking + canonical ports (8000-8005/8200) + pgvector identity db
# published on 127.0.0.1:5433 — the same shape as the real BASIC cluster, so
# the topology switch (Phase 2b) is validated HERE before touching it.
# MUTUAL EXCLUSION: host launcher XOR mcp-work XOR gwork (canonical ports).
# Data plane is the ephemeral container-local turso file — never real data.
nodes 1
layout work
topology pg
tools databasemcp webmcp simplemcp convertermcp ragmcp memorymcp
bundle my-bundles/pre-pod-test
sha256 f5f996e98b3a162772eeb8a021a1e875fb7ace1929ddc2abd42c3cbf60bde6f3
probes smoke-work
