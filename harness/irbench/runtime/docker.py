import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
import yaml

from irbench.engines.base import poll
from irbench.paths import Workspace
from irbench.runtime.hardware import HardwareProfile

Runner = Callable[[list[str]], str]


class Runtime(Protocol):
    def up(self) -> None: ...
    def down(self) -> None: ...
    def image_info(self) -> dict[str, str | None]: ...
    def memory_peak_bytes(self) -> int | None: ...
    def disk_bytes(self, path: str) -> int | None: ...


def run_command(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found: is it installed and on PATH?") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"{' '.join(cmd)} failed ({e.returncode}): {e.stderr.strip()[-2000:]}"
        ) from e
    return out.stdout


class DockerRuntime:
    """One engine container, managed through its compose file."""

    def __init__(
        self,
        root: Path,
        engine: str,
        profile: HardwareProfile,
        runner: Runner = run_command,
        state_dir: Path | None = None,
    ):
        self.engine = engine
        self.profile = profile
        self.runner = runner
        self.compose_file = root / "engines" / engine / "docker" / "compose.yaml"
        self.project = f"irbench-{engine}"
        self.container = f"irbench-{engine}"
        self.override_file = (state_dir or Workspace(root).cache / "runtime") / (
            f"{engine}.override.yaml"
        )

    def _compose(self, *args: str) -> list[str]:
        cmd = ["docker", "compose", "-p", self.project, "-f", str(self.compose_file)]
        if self.override_file.exists():
            cmd += ["-f", str(self.override_file)]
        return cmd + list(args)

    def _write_override(self) -> None:
        limits = self.profile.engine
        service: dict[str, str] = {}
        if limits.cpuset:
            service["cpuset"] = limits.cpuset
        if limits.memory:
            service["mem_limit"] = limits.memory
            service["memswap_limit"] = limits.memory  # no swap: memory limit is a hard limit
        if service:
            self.override_file.parent.mkdir(parents=True, exist_ok=True)
            self.override_file.write_text(yaml.safe_dump({"services": {"engine": service}}))
        else:
            self.override_file.unlink(missing_ok=True)

    def up(self, timeout_s: int = 300) -> None:
        if not self.compose_file.exists():
            raise FileNotFoundError(f"{self.compose_file} missing: unknown engine {self.engine!r}")
        self._write_override()
        self.runner(self._compose("up", "-d", "--wait", "--wait-timeout", str(timeout_s)))

    def down(self) -> None:
        self.runner(self._compose("down", "-v", "--remove-orphans"))

    def image_info(self) -> dict[str, str | None]:
        try:
            image = self.runner(
                ["docker", "inspect", "--format", "{{.Config.Image}}", self.container]
            ).strip()
            digests = json.loads(
                self.runner(
                    ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image]
                )
            )
        except (RuntimeError, ValueError):
            return {"image": None, "digest": None}
        return {"image": image, "digest": digests[0] if digests else None}

    def memory_peak_bytes(self) -> int | None:
        # cgroup v2: peak memory of the container since it started.
        try:
            out = self.runner(
                ["docker", "exec", self.container, "cat", "/sys/fs/cgroup/memory.peak"]
            )
            return int(out.strip())
        except (RuntimeError, ValueError):
            return None

    def disk_bytes(self, path: str) -> int | None:
        try:
            out = self.runner(["docker", "exec", self.container, "du", "-sk", path])
            return int(out.split()[0]) * 1024
        except (RuntimeError, ValueError, IndexError):
            return None


def wait_healthy(
    client: httpx.Client, path: str, timeout_s: float, interval_s: float = 1.0
) -> None:
    def ok() -> bool:
        try:
            return client.get(path).status_code == 200
        except httpx.HTTPError:
            return False

    poll(ok, timeout_s, f"{client.base_url}{path} to answer 200", interval_s)
