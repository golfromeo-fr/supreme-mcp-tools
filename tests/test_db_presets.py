"""P7 — connection presets: parser, masking, lazy bypass, autoconnect."""

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _identified_admin(monkeypatch):
    """Audit P2-3 (2026-10-17): unresolved identities fail closed in multi
    mode. These tests exercise registry/tx/bypass mechanics as an IDENTIFIED
    admin; per-test overrides of _current_caller_grants still win."""
    import connections
    monkeypatch.setattr(connections, "_current_caller_grants",
                        lambda: ("admin", "admin", []))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOL_DIR = PROJECT_ROOT / "tools" / "databasemcp"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from dialects._base import DbDialect  # noqa: E402
from presets import (  # noqa: E402
    apply_autoconnect,
    get_preset,
    load_presets,
    mask_url,
    parse_preset_url,
)


class FakeDialect(DbDialect):
    """Generic stand-in; set .explode = True on an instance to fail connect."""

    name = "fake"
    REQUIRED_PARAMS = ()
    explode = False

    def connect(self, params):
        if self.explode:
            raise RuntimeError("host unreachable")
        return object()

    def close(self, handle): pass
    def ping(self, handle): ...
    def run_select(self, handle, sql, max_rows): return [], False
    def execute(self, handle, sql): return -1
    def list_tables(self, handle): return []
    def describe_table(self, handle, table):
        return {"columns": [], "constraints": [], "foreign_keys": []}
    def explain(self, handle, sql): return ""
    def format_error(self, e):
        return {"error": "DB_ERROR", "code": None, "message": str(e), "offset": None}

    # --- tx stubs (E4 ABC) — fakes never enter transaction flows in these tests ---
    def open_tx(self, handle, params): raise NotImplementedError
    def select_tx(self, tx_handle, sql, max_rows): raise NotImplementedError
    def execute_tx(self, tx_handle, sql): raise NotImplementedError
    def commit_tx(self, tx_handle): raise NotImplementedError
    def rollback_tx(self, tx_handle): raise NotImplementedError
    def close_tx(self, handle, tx_handle): raise NotImplementedError


class TestParsePresetUrl:
    def test_oracle(self):
        dialect, params = parse_preset_url("oracle://scott:tiger@dbhost:1521/ORCLPDB1")
        assert dialect == "oracle"
        assert params == {"user": "scott", "password": "tiger", "host": "dbhost",
                          "port": "1521", "service_name": "ORCLPDB1"}

    def test_oracle_default_port_and_percent_decoding(self):
        _, params = parse_preset_url("oracle://u%40x:p%2Fss@dbhost/SVC")
        assert params["user"] == "u@x" and params["password"] == "p/ss" and params["port"] == "1521"

    def test_postgres(self):
        dialect, params = parse_preset_url("postgresql://gr:s3cret@127.0.0.1:5433/mydb")
        assert dialect == "postgres"
        assert params["dbname"] == "mydb" and params["port"] == "5433"

    def test_libsql_file_and_turso_token(self):
        dialect, params = parse_preset_url("file:/tmp/dev.db")
        assert dialect == "libsql" and params == {"url": "file:/tmp/dev.db"}
        dialect, params = parse_preset_url("libsql://db.turso.io?authToken=tok123")
        assert dialect == "libsql" and params["auth_token"] == "tok123"

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match="Unknown preset URL scheme"):
            parse_preset_url("mysql://u:p@h/db")

    def test_mask_url_hides_password(self):
        assert mask_url("oracle://scott:tiger@db:1521/S") == "oracle://scott:***@db:1521/S"


@pytest.fixture()
def env_presets(monkeypatch):
    # Hermetic: purge ambient DB_PRESET* first — importing the tool modules
    # load_dotenv()s the real .env, and any real _NAME/_DESC/AUTOCONNECT key
    # would otherwise merge into these fixtures (found live 2026-11-01: a
    # real DB_PRESET_03_NAME leaked into preset 03's fixture env).
    for _k in [k for k in os.environ if k.startswith("DB_PRESET")]:
        monkeypatch.delenv(_k, raising=False)
    monkeypatch.setenv("DB_PRESET_01", "oracle://scott:tiger@db:1521/ORCL")
    monkeypatch.setenv("DB_PRESET_01_DESC", "Work Oracle")
    monkeypatch.setenv("DB_PRESET_02", "file:/tmp/preset_two.db")
    monkeypatch.setenv("DB_PRESET_02_NAME", "pglocal")
    monkeypatch.setenv("DB_PRESET_03", "postgres://u:badpass@nowhere:5432/db")
    monkeypatch.setenv("DB_PRESET_03_DESC", "Never reachable")
    monkeypatch.setenv("DB_PRESET_AUTOCONNECT", "01,03")
    monkeypatch.delenv("USERID", raising=False)
    monkeypatch.delenv("DB_AUTOCONNECT", raising=False)


class TestLoadPresets:
    def test_sorted_by_number_with_metadata(self, env_presets):
        presets = load_presets()
        assert [p.number for p in presets] == ["01", "02", "03"]
        assert presets[0].dialect == "oracle" and presets[0].desc == "Work Oracle"
        assert presets[1].name == "pglocal"
        assert [p.autoconnect for p in presets] == [True, False, True]

    def test_get_preset_by_number_and_name(self, env_presets):
        assert get_preset("02").name == "pglocal"
        assert get_preset("pglocal").number == "02"      # NAME alias resolves
        assert get_preset("PGLOCAL").number == "02"      # case-insensitive alias

    def test_get_preset_unknown_lists_available(self, env_presets):
        with pytest.raises(LookupError, match="Unknown preset"):
            get_preset("nope")

    def test_urls_masked_in_listing_data(self, env_presets):
        # list_presets prints url_masked only — params stay internal
        flat = str([(p.url_masked, p.desc, p.number, p.name, p.dialect) for p in load_presets()])
        assert "tiger" not in flat and "badpass" not in flat
        assert "oracle://scott:***@db:1521/ORCL" in flat
        assert "postgres://u:***@nowhere:5432/db" in flat
        assert "Work Oracle" in flat and "Never reachable" in flat


class TestBypassAndAutoconnect:
    @pytest.fixture()
    def fake_dialects(self, monkeypatch, env_presets):
        import dialects

        fine = FakeDialect()
        exploding = FakeDialect()
        exploding.explode = True
        monkeypatch.setitem(dialects.DIALECTS, "oracle", fine)
        monkeypatch.setitem(dialects.DIALECTS, "postgres", exploding)
        monkeypatch.setitem(dialects.DIALECTS, "libsql", fine)
        return fine, exploding

    def test_preset_bypass_in_registry_get(self, env_presets, fake_dialects):
        from connections import ConnectionRegistry

        reg = ConnectionRegistry()
        # BYPASS: get() on an unconnected preset connects it on the fly
        entry = reg.get("02")
        assert entry.name == "pglocal" and entry.dialect == "libsql"
        # by NUMBER for a preset without an alias
        entry = reg.get("01")
        assert entry.name == "01" and entry.dialect == "oracle"
        # by NAME alias as the connection key
        assert reg.get("pglocal").name == "pglocal"

    def test_bypass_unknown_name_still_raises(self, env_presets, fake_dialects):
        from connections import ConnectionRegistry

        reg = ConnectionRegistry()
        with pytest.raises(LookupError, match="Unknown connection"):
            reg.get("not_a_preset")

    def test_autoconnect_tolerates_dead_db(self, env_presets, fake_dialects):
        from connections import ConnectionRegistry

        reg = ConnectionRegistry()
        connected, failed = apply_autoconnect(reg)  # 01 oracle OK, 03 postgres explodes
        assert (connected, failed) == (1, 1)
        assert reg.get("01").name == "01"           # the good one is up

    def test_autoconnect_connects_only_flagged_presets(self, env_presets, monkeypatch):
        import dialects
        from connections import ConnectionRegistry

        fine = FakeDialect()
        monkeypatch.setitem(dialects.DIALECTS, "oracle", fine)
        monkeypatch.setitem(dialects.DIALECTS, "postgres", fine)
        monkeypatch.setitem(dialects.DIALECTS, "libsql", fine)
        reg = ConnectionRegistry()
        connected, failed = apply_autoconnect(reg)
        assert (connected, failed) == (2, 0)         # 01 and 03 flagged, 02 not
        assert reg.get("01").name == "01"
        assert reg.get("03").name == "03"


class TestMaskUrlQueryTokens:
    """Credentials can ride in the QUERY STRING (Turso's ?authToken=…), not
    only the user:password segment — mask_url used to pass them through
    byte-for-byte into every listing (found via work-box analysis,
    2026-11). Sensitive keys are masked case-insensitively; non-sensitive
    params survive byte-for-byte (no re-encoding)."""

    def test_auth_token_masked(self):
        assert mask_url("libsql://db.turso.io?authToken=tok123") == \
            "libsql://db.turso.io?authToken=***"

    def test_sensitive_keys_masked_non_sensitive_preserved(self):
        out = mask_url("libsql://db.turso.io?authToken=T&region=fra&sslmode=require")
        assert out == "libsql://db.turso.io?authToken=***&region=fra&sslmode=require"
        assert "fra" in out and "require" in out

    def test_case_insensitive_including_underscore_form(self):
        # repo-canonical "auth_token" AND Turso's camelCase "authToken"
        out = mask_url("libsql://db?AUTH_TOKEN=x&keep=1")
        assert "AUTH_TOKEN=***" in out and "keep=1" in out

    def test_password_query_param_also_masked(self):
        out = mask_url("postgres://u:pw@h:5432/db?password=pp&opt=z")
        assert out == "postgres://u:***@h:5432/db?password=***&opt=z"

    def test_untouched_urls_pass_through(self):
        assert mask_url("file:/tmp/x.db") == "file:/tmp/x.db"
        assert mask_url("postgres://u:pw@h/db") == "postgres://u:***@h/db"

    def test_load_presets_listing_never_shows_query_token(self, env_presets, monkeypatch):
        # end-to-end: whatever mask_url does, the list_presets payload carries no token
        monkeypatch.setenv(
            "DB_PRESET_04", "libsql://secret-db.turso.io?authToken=sekrit-token")
        listing = [(p.url_masked, p.desc, p.number) for p in load_presets()]
        flat = str(listing)
        assert "sekrit-token" not in flat
        assert "authToken=***" in flat
