"""Entry-chain tests for scripts/startlauncher.sh.

The script ends in an ``exec``; these tests shim ``exec`` as a bash function
(which shadows the builtin) so the final command line is captured instead of
the launcher actually starting. No servers are touched.

- R6 (review 2026-11-11): the modern engine takes tools only via ``--tools``
  (comma-separated) — forwarding bare positional names used to die with
  argparse "unrecognized arguments".
- R10: ``--health-check-logs`` is only forwarded when explicitly given —
  the script's own default used to clobber the ``MCP_HEALTH_CHECK_LOGS``
  knob from .env (see test for the legacy default shape).
"""
import shlex
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _captured_exec(args: list[str]) -> str:
    """Run startlauncher.sh with ``exec`` captured; return the command line.

    ``exec`` is shadowed by an EXPORTED bash function — the script runs in its
    own nested bash, so a plain function definition would not survive, and the
    real launcher would boot (the timeout is a belt-and-braces guard against
    that ever happening again)."""
    shim = (
        "exec() { printf '%s\\n' \"$*\"; }\n"
        "export -f exec\n"
        f"bash '{PROJECT_ROOT}/scripts/startlauncher.sh' "
        + " ".join(shlex.quote(a) for a in args)
        + "\n"
    )
    result = subprocess.run(
        ["bash", "-c", shim],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_modern_tools_translated_to_flag():
    """R6: -t webmcp memorymcp under --engine modern becomes --tools a,b."""
    line = _captured_exec(["--engine", "modern", "-t", "webmcp memorymcp"])
    assert line.split() == ["python3", "-m", "launcher", "--tools", "webmcp,memorymcp"]


def test_modern_comma_form_normalized():
    """R6: the comma form is accepted and travels as a single --tools value."""
    line = _captured_exec(["--engine", "modern", "--tools=webmcp,memorymcp"])
    assert "--tools webmcp,memorymcp" in line


def test_modern_all_forwards_no_tools_filter():
    """--all under modern means "all discovered": no --tools on the line."""
    line = _captured_exec(["--engine", "modern", "--all"])
    assert line.split() == ["python3", "-m", "launcher"]


def test_modern_default_tools_reach_the_engine():
    """The default 6-tool list used to kill the modern engine on argv."""
    line = _captured_exec(["--engine", "modern"])
    assert line.endswith("--tools simplemcp,ragmcp,webmcp,memorymcp,databasemcp,convertermcp")


def test_legacy_keeps_positional_forwarding():
    """The legacy engine's historical positional shape is untouched — and the
    script's flags are PREPENDED: argparse can't parse positionals that follow
    optionals, so a forwarded bare word before --health-check-logs used to die
    with "unrecognized arguments"."""
    line = _captured_exec(["-t", "webmcp", "memorymcp"])
    parts = line.split()
    assert parts[0] == str(PROJECT_ROOT / "launchmcp.py")
    # flags come before the positionals, both tools survive
    assert "--health-check-logs" in parts and parts.index("--health-check-logs") < 3
    assert parts[-2:] == ["memorymcp", "webmcp"]