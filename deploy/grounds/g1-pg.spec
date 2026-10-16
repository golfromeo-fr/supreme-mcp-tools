# ground g1-pg — Phase 1 MVP ground (plans/testing-grounds-and-work-topology-2026-10-05.md §3)
# one node, all six tools, embedded-postgres state plane, booted from the
# live work bundle. Key-free: secrets live in the gitignored bundle.
nodes 1
topology pg
tools databasemcp webmcp simplemcp convertermcp ragmcp memorymcp
bundle my-bundles/pre-pod-test
sha256 f5f996e98b3a162772eeb8a021a1e875fb7ace1929ddc2abd42c3cbf60bde6f3
probes smoke-work
