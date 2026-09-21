"""Pure data layer for the pod UI: podman/startcluster output parsing,
environment model, bundle and backup listings. NO NiceGUI imports — this
module is the test target (plans/pod-ui-plan-2026-09-19.md §5).
"""

from __future__ import annotations

import json
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
    rows = parse_ps_rows(run(["podman", "ps", "-a", "--format",
                              "{{.Names}}|{{.Status}}|{{.Size}}|{{.Ports}}"]))
    return group_containers(rows)


def probe_health(url: str, timeout: float = 3.0) -> str:
    """'ok' | 'down' ('n/a' is never returned from here)."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return "ok" if resp.status == 200 else "down"
    except Exception:
        return "down"


def env_statuses() -> list[Environment]:
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
