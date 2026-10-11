import importlib.util
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
import numpy as np

from irbench.engines.config import TrackConfig
from irbench.models import QueryTask
from irbench.scenario import Scenario


@dataclass(frozen=True)
class Request:
    """One HTTP request, exactly as written to a rendered-request file."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    body: Any = None
    content: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    protocol: Literal["http"] = "http"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "protocol": self.protocol,
            "method": self.method,
            "path": self.path,
            "headers": dict(self.headers),
        }
        if self.content is not None:
            out["content"] = self.content
        else:
            out["body"] = self.body
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Request":
        return cls(
            method=data["method"],
            path=data["path"],
            body=data.get("body"),
            content=data.get("content"),
            headers=dict(data.get("headers") or {}),
            protocol=data.get("protocol", "http"),
        )

    def send(self, client: httpx.Client, timeout: float | None = None) -> httpx.Response:
        kwargs: dict[str, Any] = {"headers": self.headers}
        if self.content is not None:
            kwargs["content"] = self.content
        elif self.body is not None:
            kwargs["json"] = self.body
        if timeout is not None:
            kwargs["timeout"] = timeout
        return client.request(self.method, self.path, **kwargs)


class EngineError(RuntimeError):
    pass


def call(client: httpx.Client, request: Request, timeout: float | None = None) -> Any:
    response = request.send(client, timeout)
    if response.status_code >= 400:
        raise EngineError(
            f"{request.method} {request.path} -> {response.status_code}: {response.text[:500]}"
        )
    return response.json() if response.content else None


@dataclass(frozen=True)
class SearchResult:
    ids: list[str]
    facets: dict[str, dict[str, int]] | None = None


@dataclass(frozen=True)
class IndexContext:
    scenario: Scenario
    config: TrackConfig
    index: str
    dim: int
    synonyms: list[list[str]]


class EngineAdapter(ABC):
    """Everything engine-specific: index setup, request rendering, response parsing."""

    name: ClassVar[str]
    host_port: ClassVar[int]
    health_path: ClassVar[str]
    data_dir: ClassVar[str]

    def __init__(self, ctx: IndexContext):
        self.ctx = ctx

    @abstractmethod
    def create_index(self, client: httpx.Client) -> None:
        """Drop any existing index with this name, then create it from the scenario schema."""

    @abstractmethod
    def load_documents(
        self, client: httpx.Client, docs: list[dict[str, Any]], vectors: np.ndarray
    ) -> None:
        """Send every document with its vector (row i of `vectors` belongs to docs[i])."""

    @abstractmethod
    def wait_until_indexed(self, client: httpx.Client, n_docs: int, timeout_s: float) -> None:
        """Return once all n_docs are searchable; raise TimeoutError or EngineError otherwise."""

    @abstractmethod
    def render(
        self, task: QueryTask, vector: list[float] | None, limit: int, facets: bool = True
    ) -> Request:
        """The search request for one task. `facets=False` renders the facet verification fetch."""

    @abstractmethod
    def parse(self, task: QueryTask, payload: Any) -> SearchResult:
        """Doc IDs in rank order (and facet counts when requested) from a response body."""

    def unsupported_reason(self, task: QueryTask) -> str | None:
        return self.ctx.config.unsupported_reason(task)

    def search_params(self, task: QueryTask) -> dict[str, Any]:
        return self.ctx.config.search.get(task.base, {})


def require_vector(task: QueryTask, vector: list[float] | None) -> list[float]:
    if vector is None:
        raise ValueError(f"task {task.task_id} needs a query vector")
    return vector


def index_name(scenario_id: str, model_id: str) -> str:
    return "irbench_" + re.sub(r"[^a-z0-9]+", "_", f"{scenario_id}__{model_id}".lower()).strip("_")


def poll(check: Callable[[], bool], timeout_s: float, what: str, interval_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if check():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout_s:g}s waiting for {what}")
        time.sleep(interval_s)


def load_adapter_class(root: Path, engine: str) -> type[EngineAdapter]:
    path = root / "engines" / engine / "adapter.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: unknown engine {engine!r}")
    spec = importlib.util.spec_from_file_location(f"irbench_engine_{engine}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Adapter", None)
    if not (isinstance(cls, type) and issubclass(cls, EngineAdapter)):
        raise TypeError(f"{path} must define `Adapter`, a subclass of EngineAdapter")
    if cls.name != engine:
        raise ValueError(f"{path}: Adapter.name is {cls.name!r}, expected {engine!r}")
    return cls
