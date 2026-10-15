"""Machine-checkable parity rules for engine configs (spec §7), run in CI by `irbench lint`."""

import functools
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import yaml

from irbench.engines.config import TRACKS, TrackConfig, load_engine_config
from irbench.io import read_jsonl
from irbench.models import BASE_MODES, Mode
from irbench.paths import Workspace

MODES = get_args(Mode)
REQUIRED_FILES = ("adapter.py", "README.md", "docker/compose.yaml")
# name:tag, optionally @sha256:<digest>; `latest` is rejected separately.
IMAGE_RE = re.compile(
    r"^[a-z0-9][a-z0-9._/-]*:(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]*)(@sha256:[0-9a-f]{64})?$"
)


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.rule}] {self.message}"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class QuerySet:
    texts: frozenset[str]
    ids: frozenset[str]

    @classmethod
    def of(cls, texts: list[str], ids: list[str]) -> "QuerySet":
        # Purely numeric ids ("0", "17") are skipped: they collide with ordinary numeric
        # settings. Variant and task ids ("0-typo", "0:keyword:keyword") are distinctive.
        return cls(
            frozenset(normalize(t) for t in texts), frozenset(i for i in ids if not i.isdigit())
        )

    @classmethod
    def load(cls, root: Path, scenario_id: str) -> "QuerySet":
        d = Workspace(root).scenario_dir(scenario_id)
        variants, tasks = d / "queries.variants.jsonl", d / "tasks.jsonl"
        if not variants.exists() or not tasks.exists():
            raise FileNotFoundError(
                f"{d} is not prepared: run `irbench prepare --scenario {scenario_id}` first"
            )
        rows = list(read_jsonl(variants))
        ids = [r["qid"] for r in rows] + [t["task_id"] for t in read_jsonl(tasks)]
        return cls.of([r["text"] for r in rows], ids)

    @functools.cached_property
    def multiword(self) -> list[tuple[str, re.Pattern[str]]]:
        # cached_property writes to __dict__ directly, so it works on a frozen dataclass.
        return [
            (t, re.compile(rf"(?<!\w){re.escape(t)}(?!\w)")) for t in sorted(self.texts) if " " in t
        ]

    def leak(self, value: str) -> str | None:
        norm = normalize(value)
        if norm in self.texts:
            return f"equals query text {norm!r}"
        if value in self.ids:
            return f"is a query or task id ({value!r})"
        for text, pattern in self.multiword:
            if pattern.search(norm):
                return f"contains query text {text!r}"
        return None


QueryLoader = Callable[[str], QuerySet]


def lint_repo(root: Path, queries: QueryLoader | None = None) -> tuple[int, list[Finding]]:
    """Returns (number of configs checked, findings)."""
    load = functools.cache(queries or (lambda scenario_id: QuerySet.load(root, scenario_id)))
    findings: list[Finding] = []
    n_configs = 0
    engine_dirs = (p for p in (root / "engines").iterdir() if p.is_dir() and p.name[0] not in "._")
    for engine_dir in sorted(engine_dirs):
        findings += _check_engine_files(root, engine_dir)
        for path in sorted((engine_dir / "configs").rglob("*.yaml")):
            n_configs += 1
            scenario_id = path.relative_to(engine_dir / "configs").with_suffix("").as_posix()
            findings += _check_config(root, engine_dir.name, scenario_id, path, load)
    return n_configs, findings


def _check_engine_files(root: Path, engine_dir: Path) -> list[Finding]:
    rel = engine_dir.relative_to(root).as_posix()
    out = [
        Finding(
            f"{rel}/{name}",
            "engine-files",
            "missing: every engine needs an adapter, "
            "a README with parity notes and a pinned compose file",
        )
        for name in REQUIRED_FILES
        if not (engine_dir / name).exists()
    ]
    compose = engine_dir / "docker" / "compose.yaml"
    if compose.exists():
        image = _compose_image(compose.read_text(encoding="utf-8"))
        match = IMAGE_RE.match(image or "")
        if not match or match["tag"] == "latest":
            out.append(
                Finding(
                    f"{rel}/docker/compose.yaml",
                    "image-pin",
                    f"services.engine.image must pin a version tag, got {image!r}",
                )
            )
    return out


def _compose_image(text: str) -> str | None:
    # A malformed compose file (bad YAML, or `services`/`services.engine` not a mapping)
    # has no image to pin; the caller turns that into an `image-pin` finding, not a crash.
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    services = data.get("services") if isinstance(data, dict) else None
    engine = services.get("engine") if isinstance(services, dict) else None
    image = engine.get("image") if isinstance(engine, dict) else None
    return image if isinstance(image, str) else None


def _check_config(
    root: Path, engine: str, scenario_id: str, path: Path, load: QueryLoader
) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    try:
        config = load_engine_config(root, engine, scenario_id)
    except (ValueError, yaml.YAMLError) as e:
        return [Finding(rel, "config-load", str(e))]

    tracks = [(name, config.track(name)) for name in TRACKS if name in config.raw]
    out: list[Finding] = []
    for name, track in tracks:
        for setting in _settings(track):
            if not _justified(setting, track.justification):
                out.append(
                    Finding(
                        rel,
                        "unjustified-setting",
                        f"[{name}] {setting} has no `justification:` entry; every "
                        "setting that departs from the engine as shipped is explained",
                    )
                )
        for key in track.unsupported:
            if not _valid_unsupported_key(key):
                out.append(
                    Finding(
                        rel,
                        "unsupported-key",
                        f"[{name}] {key!r} is not `<mode>`, `<mode>:<base>` or `*:<base>`",
                    )
                )
    try:
        query_set = load(scenario_id)
    except FileNotFoundError as e:
        return [*out, Finding(rel, "not-prepared", str(e))]
    for name, track in tracks:
        # Only index and search reach the engine; justification and N/A reasons are prose.
        for where, value in _strings({"index": track.index, "search": track.search}, ""):
            reason = query_set.leak(value)
            if reason:
                out.append(
                    Finding(
                        rel,
                        "query-leak",
                        f"[{name}] {where} {reason}: per-query tuning is not allowed",
                    )
                )
    return out


def _settings(track: TrackConfig) -> Iterator[str]:
    yield from _leaves(track.index, "index")
    yield from _leaves(track.search, "search")


def _leaves(data: dict[str, Any], prefix: str) -> Iterator[str]:
    for key, value in data.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            yield from _leaves(value, path)  # an empty dict changes nothing
        else:
            yield path


def _justified(setting: str, justification: dict[str, str]) -> bool:
    # A justification key names the setting or a parent, with or without the leading
    # "index." / "search." / "search.<base>.": "index.hnsw.ef" is covered by index,
    # index.hnsw, index.hnsw.ef, hnsw or hnsw.ef; "search.semantic.ef" also by ef.
    parts = setting.split(".")
    starts = (0, 1, 2) if parts[0] == "search" else (0, 1)
    candidates = {".".join(parts[s:e]) for s in starts for e in range(s + 1, len(parts) + 1)}
    return any(justification.get(c, "").strip() for c in candidates)


def _valid_unsupported_key(key: str) -> bool:
    mode, sep, base = key.partition(":")
    if not sep:
        return mode in MODES
    return (mode == "*" or mode in MODES) and base in BASE_MODES


def _strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield child, str(key)
            yield from _strings(item, child)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]")
    elif isinstance(value, str):
        yield path, value
