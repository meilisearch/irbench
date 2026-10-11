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
