"""databasemcp connection presets (P7) — numbered .env connections.

Pattern (plans/databasemcp-overhaul-2026-09-06.md, P7):

    DB_PRESET_01=oracle://scott:tiger@dbhost:1521/ORCLPDB1
    DB_PRESET_01_DESC=Work Oracle          # optional display label
    DB_PRESET_02=postgres://gr:pw@127.0.0.1:5432/mydb
    DB_PRESET_02_NAME=pglocal              # optional connection alias
    DB_PRESET_03=file:/home/gr/databases/dev.db
    DB_PRESET_AUTOCONNECT=01,02            # eager connect at tool startup

The URL scheme selects the dialect; numbers give a stable user-chosen order
(os.environ is unordered). Secrets live only in .env and are masked in every
listing. .env is read at process start — preset changes need a restart.
"""
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, unquote

_PRESET_RE = re.compile(r"^DB_PRESET_(\d+)$")


@dataclass
class Preset:
    number: str                # as written, e.g. "01"
    dialect: str
    params: dict               # may hold secrets — never log/return unmasked
    url_masked: str            # URL with the password segment replaced
    desc: str = ""
    name: str | None = None    # optional connection alias (env-safe)
    autoconnect: bool = False

    @property
    def connection_name(self) -> str:
        return self.name or self.number


def mask_url(url: str) -> str:
    """Mask the password segment of a URL: scheme://user:***@host/..."""
    import re as _re

    return _re.sub(r"((?:[A-Za-z][A-Za-z0-9+.\-]*://)[^:/@\s]+:)([^@\s/]+)@", r"\1***@", url)


def parse_preset_url(url: str) -> tuple[str, dict]:
    """Parse a preset URL into (dialect, params). Raises ValueError on
    unknown schemes. Percent-decodes user/password/dbname."""
    split = urlsplit(url)
    scheme = (split.scheme or "").lower()

    if scheme == "oracle":
        if not split.hostname or not split.username:
            raise ValueError("oracle preset URL must be oracle://user:password@host:port/service")
        return "oracle", {
            "user": unquote(split.username),
            "password": unquote(split.password or ""),
            "host": split.hostname,
            "port": str(split.port or 1521),
            "service_name": unquote(split.path.lstrip("/")),
        }

    if scheme in ("postgres", "postgresql"):
        if not split.hostname or not split.username:
            raise ValueError("postgres preset URL must be postgres://user:password@host:port/dbname")
        return "postgres", {
            "user": unquote(split.username),
            "password": unquote(split.password or ""),
            "host": split.hostname,
            "port": str(split.port or 5432),
            "dbname": unquote(split.path.lstrip("/")),
        }

    if scheme in ("file", "libsql"):
        params = {"url": url}
        if split.scheme.lower() == "libsql" and split.query:
            from urllib.parse import parse_qs

            token = parse_qs(split.query).get("authToken", [None])[0]
            if token:
                params["auth_token"] = token
        return "libsql", params

    raise ValueError(
        f"Unknown preset URL scheme '{scheme or '(none)'}' — "
        "use oracle://, postgres:// (or postgresql://) or file:/libsql://"
    )


def load_presets() -> list[Preset]:
    """Scan the environment for DB_PRESET_<NN> keys; sorted by number."""
    autoconnect = {
        x.strip().lstrip("0") or "0"
        for x in os.environ.get("DB_PRESET_AUTOCONNECT", "").split(",")
        if x.strip()
    }
    presets = []
    for key, value in os.environ.items():
        match = _PRESET_RE.match(key)
        if not match or not value.strip():
            continue
        number = match.group(1)
        try:
            dialect, params = parse_preset_url(value.strip())
        except ValueError as e:
            from core import logger

            logger.warning(f"Preset {number} ignored: {e}")
            continue
        presets.append(Preset(
            number=number,
            dialect=dialect,
            params=params,
            url_masked=mask_url(value.strip()),
            desc=os.environ.get(f"DB_PRESET_{number}_DESC", ""),
            name=os.environ.get(f"DB_PRESET_{number}_NAME") or None,
            autoconnect=number.lstrip("0") in autoconnect or number in autoconnect,
        ))
    presets.sort(key=lambda p: int(p.number))
    return presets


def get_preset(key: str) -> Preset:
    """Resolve a preset by number ("01") or NAME alias ("pglocal"); case-
    insensitive on the alias. Raises LookupError listing what exists."""
    wanted = (key or "").strip()
    for preset in load_presets():
        if preset.number == wanted or (
            preset.name and preset.name.lower() == wanted.lower()
        ):
            return preset
    available = [
        f"{p.number}{'/' + p.name if p.name else ''} ({p.dialect})"
        for p in load_presets()
    ]
    raise LookupError(
        f"Unknown preset '{key}'. Available presets: {'; '.join(available) or 'none — add DB_PRESET_<NN> to .env'}"
    )


def apply_autoconnect(registry) -> tuple[int, int]:
    """Eagerly connect every AUTOCONNECT preset; tolerant failure.

    Returns (connected, failed). A dead DB logs a warning and is skipped —
    the launcher must never be blocked by an unreachable database.
    """
    connected = failed = 0
    for preset in load_presets():
        if not preset.autoconnect:
            continue
        try:
            registry.connect(preset.connection_name, preset.dialect, preset.params)
            connected += 1
        except Exception as e:
            failed += 1
            from core import logger

            logger.warning(
                f"Autoconnect preset {preset.number} ({preset.dialect}) failed: "
                f"{type(e).__name__} — it stays available via connect_preset."
            )
    return connected, failed
