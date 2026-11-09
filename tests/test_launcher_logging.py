"""setup_logging() must still attach the rotating file handler (2026-11-09).

Regression found by an MCP live sweep: commit 06d205c shipped
`from launcher.config import Config` inside setup_logging(), but
launcher/config/__init__.py exports nothing (the persistence backends were
deleted 2026-09-05), so every `python -m launcher` boot swallowed an
ImportError and silently booted without its file handler. The correct home
is launcher.launcher_config.Config (same import the module header already
uses at the top of __main__.py).
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher.__main__ import setup_logging  # noqa: E402


def test_setup_logging_attaches_file_handler_without_warning():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    captured: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    # Own handler on the named logger: setup_logging() clears ROOT handlers,
    # which would silently detach pytest's caplog handler mid-test.
    catch = _Catch(level=logging.WARNING)
    lg = logging.getLogger("launcher.__main__")
    lg.addHandler(catch)
    try:
        setup_logging()
        assert not [m for m in captured if "file logging" in m], captured
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert file_handlers, (
            "setup_logging() attached no file handler — the launcher.config "
            "import regression is back, or file logging got disabled in "
            "config/launcher_config.json"
        )
    finally:
        lg.removeHandler(catch)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
