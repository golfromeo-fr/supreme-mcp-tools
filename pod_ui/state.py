"""Pure data layer for the pod UI: podman/startcluster output parsing,
environment model, bundle and backup listings. NO NiceGUI imports — this
module is the test target (plans/pod-ui-plan-2026-09-19.md §5).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "startcluster.sh"
BUNDLES_DIR = REPO_ROOT / "my-bundles"
BACKUPS_DIR = Path.home() / "supreme-mcp-tools-backups"


@dataclass
class ContainerRow:
    name: str
    state: str  # first word of status: Up / Exited / Created / ...
    status: str  # raw status string (tooltip)
    running: bool
    size: str = ""
    ports: str = ""


@dataclass
class Environment:
    key: str
    title: str
    project: str
    prefix: str
    containers: list = field(default_factory=list)  # [ContainerRow]
    health: dict = field(default_factory=dict)  # {label: "ok"|"down"|"n/a"}

    @property
    def running(self) -> bool:
        return any(c.running for c in self.containers)

    @property
    def state(self) -> str:
        """absent (nothing in podman at all) / stopped / partial / running."""
        if not self.containers:
            return "absent"
        up = sum(1 for c in self.containers if c.running)
        if up == 0:
            return "stopped"
        if up == len(self.containers):
            return "running"
        return "partial"


# cross-page UI prefs (process-wide; single-admin tool). The drawer switch
# writes it, the dashboard's 5s tick applies it — no cross-component events.
UI_PREFS = {"show_stopped": True}


ENVIRONMENTS = [
    Environment("work", "work", "mcp-work", "mcp-work_"),
    Environment("test", "test", "mcp-multihost", "mcp-multihost_"),
]
ENV_BY_KEY = {e.key: e for e in ENVIRONMENTS}
HEALTH_URLS = {
    "work": {"central :8200": "http://127.0.0.1:8200/health"},
    "test": {"node1 :18200": "http://127.0.0.1:18200/health",
             "node2 :19200": "http://127.0.0.1:19200/health"},
}


def parse_ps_rows(output: str) -> list[ContainerRow]:
    """Parse `podman ps -a --format "{{.Names}}|{{.Status}}|{{.Size}}|{{.Ports}}"`.

    State comes from the FIRST WORD of Status — an exited row can still say
    '(healthy)' (probe fact F2), never trust the suffix."""
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if not parts or not parts[0].strip():
            continue
        name = parts[0].strip()
        status = parts[1].strip() if len(parts) > 1 else ""
        size = parts[2].strip() if len(parts) > 2 else ""
        ports = parts[3].strip() if len(parts) > 3 else ""
        state_word = status.split()[0] if status else "Unknown"
        rows.append(ContainerRow(name=name, state=state_word, status=status,
                                 running=state_word == "Up", size=size, ports=ports))
    return rows


def group_containers(rows: list[ContainerRow]) -> dict[str, list[ContainerRow]]:
    """Group into env keys by name prefix; anything else -> 'other' (read-only)."""
    grouped: dict[str, list[ContainerRow]] = {e.key: [] for e in ENVIRONMENTS}
    grouped["other"] = []
    for row in rows:
        for env in ENVIRONMENTS:
            if row.name.startswith(env.prefix):
                grouped[env.key].append(row)
                break
        else:
            grouped["other"].append(row)
    return grouped


def run(cmd: list[str], timeout: int = 20) -> str:
    """Thin capture wrapper; returns stdout, '' on any failure (callers render
    their own empty states — the dashboard must survive a broken podman)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def containers_by_env() -> dict[str, list[ContainerRow]]:
    # -s is REQUIRED for {{.Size}}: without it podman renders the size empty
    rows = parse_ps_rows(run(["podman", "ps", "-a", "-s", "--format",
                              "{{.Names}}|{{.Status}}|{{.Size}}|{{.Ports}}"]))
    return group_containers(rows)


_SIZE_UNITS = {"B": 1, "kB": 1000, "KB": 1024, "MB": 1000**2, "MiB": 1024**2,
               "GB": 1000**3, "GiB": 1024**3, "TB": 1000**4}


def parse_size(text: str) -> int:
    """'48.39MB' or '1.5MB (virtual 1.2GB)' -> bytes (0 on unparsable).
    Podman container sizes carry a '(virtual ...)' suffix — only the first
    token is the real value."""
    t = text.strip().split()[0] if text.strip() else ""
    for suffix, mult in sorted(_SIZE_UNITS.items(), key=lambda kv: -len(kv[0])):
        if t.endswith(suffix):
            try:
                return int(float(t[: -len(suffix)].strip()) * mult)
            except ValueError:
                return 0
    return 0


def human_size(n: int) -> str:
    """Bytes -> podman-style human string ('48.4MB')."""
    if n < 1000:
        return f"{n}B"
    for suffix, div in (("TB", 1000**4), ("GB", 1000**3), ("MB", 1000**2), ("kB", 1000)):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return f"{n}B"


def cluster_disk(containers: list[ContainerRow], volumes: dict[str, str]) -> str:
    """'containers X + volumes Y' human summary for one cluster card."""
    c_total = sum(parse_size(c.size) for c in containers)
    v_total = sum(parse_size(v) for k, v in volumes.items())
    return f"containers {human_size(c_total)} + volumes {human_size(v_total)}"


def probe_health(url: str, timeout: float = 3.0) -> str:
    """'ok' | 'down' ('n/a' is never returned from here)."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return "ok" if resp.status == 200 else "down"
    except Exception:
        return "down"


def env_statuses(grouped: dict | None = None) -> list[Environment]:
    """grouped: precomputed containers_by_env() — pass it to avoid a second
    podman ps when the caller already ran one (the dashboard cheap gather)."""

    if grouped is None:
        grouped = containers_by_env()
    result = []
    for env in ENVIRONMENTS:
        e = Environment(key=env.key, title=env.title, project=env.project,
                        prefix=env.prefix, containers=grouped.get(env.key, []))
        e.health = {label: probe_health(url) for label, url in HEALTH_URLS.get(env.key, {}).items()}
        result.append(e)
    return result


def env_signature(statuses: list[Environment]) -> str:
    """Change-gate for the dashboard refresh: a compact signature of the
    container states + health. Identical signature => skip the rebuild so
    the page never flickers or resets scroll/expansions while you read."""
    parts = []
    for env in statuses:
        parts.append(env.state)
        parts.append(",".join(f"{label}:{v}" for label, v in sorted(env.health.items())))
        parts.append(",".join(f"{c.name}:{c.state}" for c in env.containers))
    return "|".join(parts)


def image_version() -> str:
    """'version / sha' from the mcp-node image labels (may be 'dev / unknown'
    for compose-built images — shown honestly, see plan F3)."""
    out = run(["podman", "inspect", "localhost/mcp-node:latest", "--format",
               '{{index .Labels "org.opencontainers.image.version"}} / '
               '{{index .Labels "org.opencontainers.image.revision"}}'])
    return out.strip() or "image not found"


def volume_sizes() -> dict[str, str]:
    """{volume_name: size} for our volumes, parsed from `podman system df -v`
    (du under-reads root-owned volume dirs — probe fact F4/E19)."""
    text = run(["podman", "system", "df", "-v"], timeout=30)
    sizes: dict[str, str] = {}
    in_volumes = False
    for line in text.splitlines():
        if "Local Volumes space usage" in line:
            in_volumes = True
            continue
        if in_volumes:
            if not line.strip():
                if sizes:
                    break
                continue
            if line.strip().startswith("VOLUME NAME"):
                continue
            tokens = line.split()
            if len(tokens) >= 3:
                sizes[tokens[0]] = tokens[2]  # [name, links, size]
    return sizes


def podman_df_summary() -> list[dict]:
    """[{Type,Total,Active,Size,Reclaimable}] from `podman system df --format json`."""
    text = run(["podman", "system", "df", "--format", "json"], timeout=30)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except ValueError:
        return []


# ---------- node.env / topology / preflight (plans V3.9) ----------

NODE_ENV = REPO_ROOT / "deploy" / "node.env"

TOPOLOGY_LABELS = {
    "pg": "pg (postgres)",
    "turso": "turso (sqld) — ⚠ STREAM_EXPIRED on idle streams; pg recommended beyond demos",
    "external-pg": "external-pg (your postgres)",
    "external-turso": "external-turso (your libsql)",
}


def node_env_bundle() -> Path | None:
    """The bundle a reconnecting/last-boot node.env was generated from —
    the scrape verb uses the same header trick."""
    if not NODE_ENV.exists():
        return None
    m = re.search(r"^# \[bundle\] GENERATED by startcluster from (.+)$",
                  NODE_ENV.read_text(encoding="utf-8"), re.M)
    return Path(m.group(1)) if m else None


def test_mode() -> str:
    """'bundle' (node.env regenerated per topology on boot) or 'host'
    (node.env is hand-managed — topology switching is not offered)."""
    if node_env_bundle() is not None or (REPO_ROOT / "deploy" / "compose-test.bundle.yml").exists():
        return "bundle"
    return "host"


def _node_env_keys() -> dict[str, str]:
    """Uncommented KEY=VALUE pairs from node.env."""
    if not NODE_ENV.exists():
        return {}
    keys = {}
    for line in NODE_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        if k and v:
            keys[k.strip()] = v.strip()
    return keys


def node_env_topology() -> str:
    """Topology the CURRENT node.env points at (host mode: the only
    selectable one — the others would boot a broken cluster)."""
    keys = _node_env_keys()
    if keys.get("POSTGRES_HOST") == "db":
        return "pg"
    if keys.get("TURSO_DATABASE_URL", "").startswith("http://db:"):
        return "turso"
    if keys.get("POSTGRES_HOST"):
        return "external-pg"
    if keys.get("TURSO_DATABASE_URL"):
        return "external-turso"
    return "pg"


def topology_choices() -> dict:
    """{options: {topo: label}, current: topo, mode: bundle|host}.
    Host mode: only the node.env-matching topology is offered — the others
    would boot a broken cluster (nodes point at a state plane node.env
    does not describe)."""
    mode = test_mode()
    current = node_env_topology()
    if mode == "bundle":
        return {"options": dict(TOPOLOGY_LABELS), "current": current, "mode": mode}
    return {"options": {current: TOPOLOGY_LABELS[current]},
            "current": current, "mode": mode}


CANONICAL_PORTS = (8000, 8001, 8002, 8003, 8004, 8005, 8200)


def parse_ss_ports(text: str) -> set[int]:
    """Listening ports from `ss -ltn` output (local-address column)."""
    ports = set()
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 4 or not tokens[0].startswith("LISTEN"):
            continue
        local = tokens[3]
        if ":" in local:
            try:
                ports.add(int(local.rsplit(":", 1)[-1]))
            except ValueError:
                continue
    return ports


def listening_ports() -> set[int]:
    return parse_ss_ports(run(["ss", "-ltn"]))


def foreign_port_holders() -> list[int]:
    """Canonical ports held while the work cluster is DOWN — the CLI
    preflight would refuse a boot. [] when the work env runs or ports are free."""
    grouped = containers_by_env()
    if any(c.running for c in grouped.get("work", [])):
        return []
    return sorted(set(CANONICAL_PORTS) & listening_ports())


def minio_password_ok() -> bool:
    """deploy/.env carries a real MINIO_ROOT_PASSWORD (test boot fatals otherwise)."""
    env = REPO_ROOT / "deploy" / ".env"
    if not env.exists():
        return False
    return bool(re.search(r"^MINIO_ROOT_PASSWORD=..", env.read_text(encoding="utf-8"), re.M))


def container_inspect(name: str) -> dict | None:
    """Full `podman inspect` as dict (None when missing/broken)."""
    text = run(["podman", "inspect", name], timeout=20)
    if not text:
        return None
    try:
        data = json.loads(text)
        return data[0] if isinstance(data, list) and data else None
    except ValueError:
        return None


SECRET_NAME_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|DSN)$", re.I)


def mask_secrets(env_lines: list[str]) -> list[str]:
    """'POSTGRES_PASSWORD=x' -> 'POSTGRES_PASSWORD=•••' (KEY/SECRET/TOKEN/
    PASSWORD/DSN-named env values are masked on screen; raw JSON stays one
    click away for the admin)."""
    out = []
    for line in env_lines:
        k, _, v = line.partition("=")
        out.append(f"{k}=•••" if SECRET_NAME_RE.search(k) else line)
    return out


def harvest_preview() -> tuple[int, str]:
    """`harvest-config.sh --list` dry-run — (rc, output); writes nothing."""
    env = dict(os.environ, HARVEST_INVOCATION_PWD=str(REPO_ROOT))
    try:
        r = subprocess.run(["bash", str(REPO_ROOT / "deploy" / "harvest-config.sh"),
                            "--list"], capture_output=True, text=True,
                           timeout=30, cwd=REPO_ROOT, env=env)
        return r.returncode, (r.stdout + r.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"preview failed: {e}"


# ---------- bundles ----------


@dataclass
class BundleInfo:
    name: str
    path: Path
    manifest: dict | None


def list_bundles() -> list[BundleInfo]:
    if not BUNDLES_DIR.is_dir():
        return []
    out = []
    for mf in sorted(BUNDLES_DIR.glob("*/manifest.json")):
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        out.append(BundleInfo(name=mf.parent.name, path=mf.parent, manifest=manifest))
    return out


# ---------- backups / data exports ----------


@dataclass
class BackupInfo:
    path: Path
    kind: str          # "sql" | "tar.gz"
    size: int
    env: str | None    # work / test / None (inferred from the name prefix)


def backup_env_of(path: Path) -> str | None:
    name = path.name
    if name.startswith("work-"):
        return "work"
    if name.startswith("test-"):
        return "test"
    return None


def list_backups() -> list[BackupInfo]:
    if not BACKUPS_DIR.is_dir():
        return []
    out = []
    for p in BACKUPS_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix == ".sql":
            kind = "sql"
        elif p.name.endswith(".tar.gz"):
            kind = "tar.gz"
        else:
            continue
        out.append(BackupInfo(path=p, kind=kind, size=p.stat().st_size,
                              env=backup_env_of(p)))
    return sorted(out, key=lambda b: b.path.name, reverse=True)


def container_log(name: str, tail: int = 100) -> str:
    return run(["podman", "logs", "--tail", str(tail), name], timeout=20)


def known_log_containers() -> list[str]:
    """Container names worth offering in the log viewer (present in podman)."""
    grouped = containers_by_env()
    names = []
    for key in ("work", "test", "other"):
        for row in grouped.get(key, []):
            names.append(row.name)
    return sorted(names)
