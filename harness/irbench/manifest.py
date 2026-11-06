import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from irbench import __version__


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def git_state(root: Path) -> dict[str, Any]:
    """The commit and whether the tree had uncommitted changes (tracked or untracked files)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        )
        dirty: bool | None = bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {"git_sha": git_sha(root), "git_dirty": dirty}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    root = path
    while root != root.parent and not (root / ".git").exists():
        root = root.parent
    manifest = {
        "irbench_version": __version__,
        "git_sha": git_sha(root),
        "created_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
