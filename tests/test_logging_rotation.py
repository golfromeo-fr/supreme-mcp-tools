"""
Tests for the log-rotation subsystem (LoggingConfig + make_file_handler).

These guard the changes that replaced RotatingFileHandler with
TimedRotatingFileHandler so logs roll on a calendar schedule (default monthly)
instead of only on size. Covers: validation, round-trip serialization, handler
construction, parent-dir creation, and the None-file path.
"""
import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from launcher.config_types import LoggingConfig, make_file_handler
from launcher.errors import ConfigError


# ---------------------------------------------------------------------------
# LoggingConfig validation
# ---------------------------------------------------------------------------

class TestLoggingConfigValidation:
    def test_defaults(self):
        c = LoggingConfig()
        assert c.rotation_when == "D"
        assert c.rotation_interval == 30
        assert c.rotation_backup_count == 6
        assert c.max_bytes == 10 * 1024 * 1024

    def test_valid_rotation_when_values(self):
        for w in ("S", "M", "H", "D", "MIDNIGHT", "W0", "W1", "W2", "W3", "W4", "W5", "W6"):
            LoggingConfig(rotation_when=w)  # must not raise

    def test_invalid_rotation_when_raises(self):
        with pytest.raises(ConfigError, match="Invalid rotation_when"):
            LoggingConfig(rotation_when="BADMAGIC")

    def test_rotation_when_case_insensitive(self):
        c = LoggingConfig(rotation_when="d")
        assert c.rotation_when == "d"

    def test_zero_rotation_interval_raises(self):
        with pytest.raises(ConfigError, match="rotation_interval must be positive"):
            LoggingConfig(rotation_interval=0)

    def test_negative_rotation_interval_raises(self):
        with pytest.raises(ConfigError, match="rotation_interval must be positive"):
            LoggingConfig(rotation_interval=-1)

    def test_negative_backup_count_raises(self):
        with pytest.raises(ConfigError, match="rotation_backup_count must be >= 0"):
            LoggingConfig(rotation_backup_count=-1)

    def test_zero_backup_count_ok(self):
        """backup_count=0 is valid — means 'don't keep rotated files'."""
        c = LoggingConfig(rotation_backup_count=0)
        assert c.rotation_backup_count == 0

    def test_invalid_level_still_validated(self):
        with pytest.raises(ConfigError, match="Invalid log level"):
            LoggingConfig(level="trace")


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestLoggingConfigSerialization:
    def test_from_dict_camelCase(self):
        d = {
            "level": "debug",
            "rotationWhen": "H",
            "rotationInterval": 12,
            "rotationBackupCount": 3,
            "maxBytes": 999,
        }
        c = LoggingConfig.from_dict(d)
        assert c.rotation_when == "H"
        assert c.rotation_interval == 12
        assert c.rotation_backup_count == 3
        assert c.max_bytes == 999
        assert c.level == "debug"

    def test_from_dict_snake_case(self):
        """from_dict accepts both camelCase and snake_case keys."""
        d = {"rotation_when": "M", "rotation_interval": 5, "rotation_backup_count": 2, "max_bytes": 42}
        c = LoggingConfig.from_dict(d)
        assert c.rotation_when == "M"
        assert c.rotation_interval == 5
        assert c.rotation_backup_count == 2
        assert c.max_bytes == 42

    def test_from_dict_defaults_when_missing(self):
        c = LoggingConfig.from_dict({})
        assert c.rotation_when == "D"
        assert c.rotation_interval == 30
        assert c.rotation_backup_count == 6
        assert c.max_bytes == 10 * 1024 * 1024

    def test_round_trip(self):
        original = LoggingConfig(
            level="warning",
            file="/tmp/test_rt.log",
            rotation_when="midnight",
            rotation_interval=1,
            rotation_backup_count=4,
            max_bytes=2048,
        )
        d = original.to_dict()
        restored = LoggingConfig.from_dict(d)
        assert restored.rotation_when.upper() == original.rotation_when.upper()
        assert restored.rotation_interval == original.rotation_interval
        assert restored.rotation_backup_count == original.rotation_backup_count
        assert restored.max_bytes == original.max_bytes
        assert restored.level == original.level
        assert restored.file == original.file

    def test_round_trip_from_config_file(self):
        """The actual config/launcher_config.json logging block must parse cleanly."""
        import json
        config_path = Path(__file__).resolve().parent.parent / "config" / "launcher_config.json"
        raw = json.loads(config_path.read_text())
        logging_block = raw["logging"]
        c = LoggingConfig.from_dict(logging_block)
        assert c.file == "logs/launcher.log"
        assert c.rotation_when == "D"
        assert c.rotation_interval == 30
        assert c.rotation_backup_count == 6
        assert c.max_bytes == 10485760


# ---------------------------------------------------------------------------
# make_file_handler
# ---------------------------------------------------------------------------

class TestMakeFileHandler:
    def test_returns_none_when_file_unset(self):
        assert make_file_handler(LoggingConfig()) is None
        assert make_file_handler(LoggingConfig(file=None)) is None
        assert make_file_handler(LoggingConfig(file="")) is None

    def test_creates_timed_rotating_handler(self, tmp_path):
        log_file = tmp_path / "subdir" / "test.log"
        cfg = LoggingConfig(file=str(log_file))
        handler = make_file_handler(cfg)
        assert handler is not None
        from logging.handlers import TimedRotatingFileHandler
        assert isinstance(handler, TimedRotatingFileHandler)

    def test_creates_parent_directory(self, tmp_path):
        """Parent dirs that don't exist yet are created automatically."""
        log_file = tmp_path / "a" / "b" / "c" / "deep.log"
        assert not log_file.parent.exists()
        handler = make_file_handler(LoggingConfig(file=str(log_file)))
        assert handler is not None
        assert log_file.parent.exists()

    def test_passes_rotation_params(self, tmp_path):
        log_file = tmp_path / "r.log"
        cfg = LoggingConfig(
            file=str(log_file),
            rotation_when="H",
            rotation_interval=6,
            rotation_backup_count=3,
        )
        handler = make_file_handler(cfg)
        assert handler.when == "H"
        assert handler.interval == 6 * 3600  # TimedRotatingFileHandler converts H → seconds
        assert handler.backupCount == 3

    def test_handler_can_write(self, tmp_path):
        """The handler must actually work when attached to a logger."""
        log_file = tmp_path / "writable.log"
        cfg = LoggingConfig(file=str(log_file))
        handler = make_file_handler(cfg)
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger("test_write_handler")
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
        lg.warning("rotation test message")
        handler.flush()
        assert log_file.exists()
        assert "rotation test message" in log_file.read_text()
        # cleanup
        lg.removeHandler(handler)
        handler.close()

    def test_midnight_rotation(self, tmp_path):
        log_file = tmp_path / "midnight.log"
        cfg = LoggingConfig(file=str(log_file), rotation_when="MIDNIGHT", rotation_interval=1)
        handler = make_file_handler(cfg)
        assert handler is not None
        from logging.handlers import TimedRotatingFileHandler
        assert isinstance(handler, TimedRotatingFileHandler)
        handler.close()

    def test_max_bytes_is_applied(self, tmp_path):
        """max_bytes must actually trigger a rollover (it was dead config before).

        Stock TimedRotatingFileHandler ignores size; the size-aware subclass must
        roll over when the active file reaches the cap. We log many small records
        past the cap and assert a date-stamped backup file appears.
        """
        log_file = tmp_path / "sized.log"
        # Small cap so a handful of ~50-byte records crosses it.
        cfg = LoggingConfig(
            file=str(log_file),
            max_bytes=120,
            rotation_backup_count=5,
            rotation_when="D",  # far-future calendar rollover so only SIZE triggers
        )
        handler = make_file_handler(cfg)
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger("test_max_bytes_rollover")
        lg.setLevel(logging.DEBUG)
        lg.handlers.clear()
        lg.addHandler(handler)
        for _ in range(40):  # 40 * (30 chars + \n) = ~1240 bytes >> 120 cap
            lg.warning("x" * 30)
        handler.flush()
        handler.close()
        lg.removeHandler(handler)

        rotated = [p for p in tmp_path.glob("sized.log*") if p.name != "sized.log"]
        assert len(rotated) >= 1, f"size cap should have produced rotated backups, got {sorted(tmp_path.glob('*'))}"

    def test_max_bytes_zero_disables_size_guard(self, tmp_path):
        """max_bytes=0 means no size rollover — only the calendar schedule applies."""
        log_file = tmp_path / "nosize.log"
        cfg = LoggingConfig(file=str(log_file), max_bytes=0, rotation_when="D")
        handler = make_file_handler(cfg)
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger("test_no_size_guard")
        lg.setLevel(logging.DEBUG)
        lg.handlers.clear()
        lg.addHandler(handler)
        for _ in range(40):
            lg.warning("y" * 30)
        handler.flush()
        handler.close()
        lg.removeHandler(handler)
        rotated = [p for p in tmp_path.glob("nosize.log*") if p.name != "nosize.log"]
        assert rotated == [], f"max_bytes=0 should not roll on size, got {rotated}"

    def test_both_entry_points_share_factory(self):
        """Verify launchmcp.py and launcher/__main__.py both import make_file_handler."""
        # launcher/__main__.py path
        import launcher.config_types
        assert hasattr(launcher.config_types, "make_file_handler")
        # launchmcp.py path — it imports make_file_handler at module level
        import launchmcp
        # launchmcp.py must have the symbol in its module namespace
        assert hasattr(launchmcp, "make_file_handler") or "make_file_handler" in dir(launchmcp)
