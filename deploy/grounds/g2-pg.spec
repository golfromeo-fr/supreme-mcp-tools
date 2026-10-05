# ground g2-pg — Phase 3: two-node ground (grounds plan §3 D2)
# the multihost invariant at ground scale: two nodes, ONE shared state
# plane (the ground's postgres) — identity is shared, tool surfaces are
# per-node (node2 = node1 ports + 1000). Data planes stay per-node
# ephemeral turso files (node-local by design; the TEST bench's shared
# data plane is validated on the real TEST cluster).
nodes 2
topology pg
tools databasemcp webmcp simplemcp convertermcp ragmcp memorymcp
bundle my-bundles/pre-pod-test
sha256 f5f996e98b3a162772eeb8a021a1e875fb7ace1929ddc2abd42c3cbf60bde6f3
probes smoke-work
