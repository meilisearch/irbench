from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from irbench.embed.vectors import read_vectors
from irbench.engines.base import EngineAdapter, IndexContext, index_name, load_adapter_class
from irbench.engines.config import EngineConfig, TrackConfig, load_engine_config
from irbench.io import read_jsonl, read_qrels
from irbench.models import Qrels, QueryTask
from irbench.paths import Workspace
from irbench.runs import RunSpec
from irbench.scenario import Scenario, load_scenario
from irbench.terms import load_term_groups

ClientFactory = Callable[[str], httpx.Client]


def default_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=60.0)


class RunContext:
    """Inputs of one run, loaded on first use so early stages can create them."""

    def __init__(self, root: Path, spec: RunSpec, client_factory: ClientFactory | None = None):
        self.root = root
        self.spec = spec
        self.ws = Workspace(root)
        self.run_dir = self.ws.run_dir(spec.run_id)
        self.rendered_dir = self.run_dir / "rendered"
        self.client_factory = client_factory or default_client

    @cached_property
    def scenario(self) -> Scenario:
        return load_scenario(self.root, self.spec.scenario)

    @cached_property
    def config(self) -> EngineConfig:
        return load_engine_config(self.root, self.spec.engine, self.spec.scenario)

    @cached_property
    def track(self) -> TrackConfig:
        return self.config.track(self.spec.track)

    @cached_property
    def adapter_cls(self) -> type[EngineAdapter]:
        return load_adapter_class(self.root, self.spec.engine)

    @cached_property
    def docs(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.ws.dataset_dir(self.scenario.dataset) / "docs.jsonl"))

    @cached_property
    def docs_by_id(self) -> dict[str, dict[str, Any]]:
        return {d[self.scenario.id_field]: d for d in self.docs}

    def _embed_hint(self) -> str:
        return f"run `irbench embed --scenario {self.spec.scenario} --model {self.spec.model}`"

    @cached_property
    def doc_vectors(self) -> np.ndarray:
        path = self.ws.doc_vectors_path(self.scenario.dataset, self.spec.model)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: {self._embed_hint()}")
        ids, vectors = read_vectors(path)
        doc_ids = [d[self.scenario.id_field] for d in self.docs]
        if ids != doc_ids:
            raise ValueError(
                f"{path} does not match docs.jsonl ({len(ids)} vs {len(doc_ids)} ids): "
                f"{self._embed_hint()}"
            )
        return vectors

    @cached_property
    def query_vectors(self) -> dict[str, np.ndarray]:
        path = self.ws.query_vectors_path(self.spec.scenario, self.spec.model)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: {self._embed_hint()}")
        ids, vectors = read_vectors(path)
        return dict(zip(ids, vectors))

    @cached_property
    def tasks(self) -> list[QueryTask]:
        path = self.ws.scenario_dir(self.spec.scenario) / "tasks.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: run `irbench prepare --scenario {self.spec.scenario}`"
            )
        return [QueryTask.model_validate(r) for r in read_jsonl(path)]

    @cached_property
    def qrels(self) -> Qrels:
        return read_qrels(self.ws.scenario_dir(self.spec.scenario) / "qrels.tasks.tsv")

    @cached_property
    def adapter(self) -> EngineAdapter:
        return self.adapter_cls(
            IndexContext(
                scenario=self.scenario,
                config=self.track,
                index=index_name(self.spec.scenario, self.spec.model),
                dim=int(self.doc_vectors.shape[1]),
                synonyms=load_term_groups(self.scenario.synonyms_path),
                n_docs=len(self.docs),
            )
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.spec.host}:{self.adapter_cls.host_port}"

    def client(self) -> httpx.Client:
        return self.client_factory(self.base_url)

    def query_vector(self, task: QueryTask) -> list[float] | None:
        if task.base == "keyword":
            return None
        if task.qid not in self.query_vectors:
            raise ValueError(f"no query vector for {task.qid!r}: {self._embed_hint()}")
        return self.query_vectors[task.qid].tolist()

    def rendered_path(self, task: QueryTask, check: bool = False) -> Path:
        name = f"{task.qid}.check.json" if check else f"{task.qid}.json"
        return self.rendered_dir / task.mode / task.base / name
