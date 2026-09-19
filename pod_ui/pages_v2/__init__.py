"""v2 layout (candidate successor) — lives beside v1 under /v2/* routes so
both can run in one process for side-by-side comparison. Shared core
(state/ops/confirm) is untouched; delete this package + the v2 route block
in app.py to remove the loser after the user decides."""
