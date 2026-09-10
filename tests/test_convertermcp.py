"""convertermcp — config-driven allowed roots + conversion core.

The tool hardcoded ALLOWED_ROOTS to a devcontainer path that doesn't exist
on real deployments, silently rejecting every local-path conversion
(2026-09-09). Roots now load from config.json; these tests lock that in.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_FILE = PROJECT_ROOT / "tools" / "convertermcp" / "convertermcp_fastmcp.py"


def _load(monkeypatch, allowed_roots, cfg_dir: Path):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text(json.dumps({"allowed_roots": allowed_roots}))
    spec = importlib.util.spec_from_file_location("cvt_test", TOOL_FILE)
    module = importlib.util.module_from_spec(spec)
    # point __file__-relative config lookup at the test's own config
    monkeypatch.setattr(spec.loader, "name", "cvt_test")
    spec.loader.exec_module(module)
    module.__file__ = str(TOOL_FILE)
    # re-resolve with the test config dir
    module.ALLOWED_ROOTS = module._load_allowed_roots.__wrapped__() \
        if hasattr(module._load_allowed_roots, "__wrapped__") else None
    return module


def test_roots_from_config(tmp_path, monkeypatch):
    """allowed_roots come from config.json, ~ expands, paths resolve."""
    import tools.convertermcp  # noqa: F401 — ensure package importable

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps(
        {"allowed_roots": [str(tmp_path), "~/docs", "/etc"]}))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    sig = "def _load_allowed_roots() -> list[Path]:"
    spec = importlib.util.spec_from_file_location("cvt_cfg", TOOL_FILE)
    source = TOOL_FILE.read_text()
    ns = {"__file__": str(cfg_dir / "convertermcp_fastmcp.py"),
          "json": json, "Path": Path}
    fn_src = source.split(sig)[1].split("ALLOWED_ROOTS =")[0]
    exec(sig + fn_src, ns)
    roots = ns["_load_allowed_roots"]()

    assert roots == [tmp_path.resolve(), (tmp_path / "home" / "docs").resolve(),
                     Path("/etc")]


def test_roots_fallback_without_config(tmp_path):
    """Missing/unreadable config → the historical /workspaces default."""
    source = TOOL_FILE.read_text()
    ns = {"__file__": str(tmp_path / "nope" / "convertermcp_fastmcp.py"),
          "json": json, "Path": Path}
    sig = "def _load_allowed_roots() -> list[Path]:"
    fn_src = source.split(sig)[1].split("ALLOWED_ROOTS =")[0]
    exec(sig + fn_src, ns)
    assert ns["_load_allowed_roots"]() == [Path("/workspaces")]


def test_is_under_allowed_roots():
    import tools.convertermcp
    from tools.convertermcp.convertermcp_fastmcp import (
        is_under_allowed_roots, ALLOWED_ROOTS)

    assert is_under_allowed_roots(
        ALLOWED_ROOTS[0] / "sub" / "file.docx", ALLOWED_ROOTS)
    assert not is_under_allowed_roots(Path("/etc/passwd"), ALLOWED_ROOTS)


def test_extract_docx_text_round_trip(tmp_path):
    """The conversion core extracts paragraph text from a real DOCX."""
    from docx import Document
    import tools.convertermcp
    from tools.convertermcp.convertermcp_fastmcp import extract_docx_text

    doc = Document()
    doc.add_paragraph("alpha beta")
    doc.add_paragraph("gamma delta")
    f = tmp_path / "sample.docx"
    doc.save(str(f))

    text = extract_docx_text(f)
    assert "alpha beta" in text and "gamma delta" in text


def test_local_size_cap_enforced(tmp_path, monkeypatch):
    """Local-path conversions enforce MAX_DOCX_SIZE_MB (URL downloads
    already did; local files previously bypassed it)."""
    import tools.convertermcp
    from tools.convertermcp.convertermcp_fastmcp import (
        convert_docx_to_text, ALLOWED_ROOTS)

    big = tmp_path / "big.docx"
    big.write_bytes(b"x" * (21 * 1024 * 1024))
    monkeypatch.setattr(sys.modules["tools.convertermcp.convertermcp_fastmcp"],
                        "MAX_DOCX_SIZE_MB", 20)

    coro = convert_docx_to_text(str(big))
    result = asyncio.run(coro)
    assert "exceeds limit" in result
