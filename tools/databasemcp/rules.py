"""User-local rules loader (P4).

The work-specific rule files (optimization.json, proc_rules.md) live OUTSIDE
the repo — in ~/.config/supreme-mcp-tools/databasemcp/ — so private content
never enters version control. Templates ship in tools/databasemcp/examples/.
"""
from pathlib import Path

RULES_DIR = Path.home() / ".config" / "supreme-mcp-tools" / "databasemcp"

TEMPLATES = {
    "optimization.json": "tools/databasemcp/examples/optimization.json",
    "proc_rules.md": "tools/databasemcp/examples/proc_rules.md",
}


def load(name: str) -> str:
    """Return the file's text; FileNotFoundError carries the fix hint."""
    path = RULES_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {path}. "
            f"Copy the template from {TEMPLATES.get(name, 'tools/databasemcp/examples/')} "
            "there and edit it with your own rules."
        ) from None
