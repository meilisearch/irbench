import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from irbench.models import BASE_MODES, QueryTask

TRACKS = ("default", "tuned")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def config_diff(
    default: dict[str, Any], effective: dict[str, Any], prefix: str = ""
) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(set(default) | set(effective)):
        path = f"{prefix}{key}"
        a, b = default.get(key), effective.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diff.update(config_diff(a, b, path + "."))
        elif a != b:
            diff[path] = {"default": a, "effective": b}
    return diff


class TrackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: dict[str, Any] = {}
    search: dict[str, dict[str, Any]] = {}
    unsupported: dict[str, str] = {}
    justification: dict[str, str] = {}

    @field_validator("search")
    @classmethod
    def _search_keys_are_bases(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        unknown = sorted(set(value) - set(BASE_MODES))
        if unknown:
            raise ValueError(f"search keys must be retrieval bases {BASE_MODES}, got {unknown}")
        return value

    def unsupported_reason(self, task: QueryTask) -> str | None:
        for key in (f"{task.mode}:{task.base}", task.mode, f"*:{task.base}"):
            if key in self.unsupported:
                return self.unsupported[key]
        return None


class Credit(BaseModel):
    """Who contributed a tuned config; shown on the report's run page (spec §11)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    path: Path
    raw: dict[str, Any]

    def _data(self, name: str) -> dict[str, Any]:
        if name not in TRACKS:
            raise ValueError(f"unknown track {name!r}; known: {', '.join(TRACKS)}")
        default = self.raw.get("default") or {}
        if name == "default":
            return default
        if "tuned" not in self.raw:
            raise ValueError(f"{self.path}: no tuned track defined")
        return deep_merge(default, self.raw["tuned"] or {})

    def track(self, name: str) -> TrackConfig:
        data = self._data(name)
        try:
            return TrackConfig.model_validate(data)
        except ValueError as e:
            raise ValueError(f"{self.path} [{name}]: {e}") from e

    def diff(self, name: str) -> dict[str, dict[str, Any]]:
        return config_diff(self.track("default").model_dump(), self.track(name).model_dump())

    def hash(self, name: str) -> str:
        canonical = json.dumps(self.track(name).model_dump(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def credits(self) -> list[Credit]:
        items = self.raw.get("credits") or []
        if not isinstance(items, list):
            raise ValueError("credits must be a list of {name, url}")
        return [Credit.model_validate(c) for c in items]


def config_path(root: Path, engine: str, scenario_id: str) -> Path:
    return root / "engines" / engine / "configs" / f"{scenario_id}.yaml"


def load_engine_config(root: Path, engine: str, scenario_id: str) -> EngineConfig:
    path = config_path(root, engine, scenario_id)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: engine {engine!r} has no config for {scenario_id}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - set(TRACKS) - {"credits"})
    if unknown or "default" not in raw:
        raise ValueError(
            f"{path}: top-level keys must be 'default', optional 'tuned' and optional "
            f"'credits', got {sorted(raw)}"
        )
    config = EngineConfig(path, raw)
    for name in TRACKS:
        if name in raw:
            config.track(name)
    try:
        config.credits
    except ValueError as e:
        raise ValueError(f"{path}: credits: {e}") from e
    return config
