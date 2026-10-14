from dataclasses import dataclass
from pathlib import Path

import httpx

from irbench.datasets import NORMALIZERS
from irbench.datasets.registry import fetch_dataset, load_dataset_spec
from irbench.io import read_jsonl, read_qrels, write_jsonl, write_qrels
from irbench.manifest import write_manifest
from irbench.models import Query
from irbench.paths import Workspace
from irbench.perturb.variants import make_variants
from irbench.scenario import load_scenario
from irbench.tasks import build_tasks
from irbench.terms import scenario_term_groups


@dataclass(frozen=True)
class PrepareResult:
    scenario_dir: Path
    n_docs: int
    n_queries: int
    n_variants: int
    n_tasks: int
    skipped_queries: list[str]


def prepare(root: Path, scenario_id: str, client: httpx.Client | None = None) -> PrepareResult:
    ws = Workspace(root)
    scenario = load_scenario(root, scenario_id)
    spec = load_dataset_spec(root, scenario.dataset)

    raw = fetch_dataset(spec, ws.raw_dir(spec.id), client)
    normalized = NORMALIZERS[spec.id](raw, ws.dataset_dir(spec.id))

    docs_by_id = {d[scenario.id_field]: d for d in read_jsonl(normalized.docs)}
    queries = [
        Query(
            qid=r["qid"],
            text=r["text"],
            variant_of=r["qid"],
            perturbation="original",
            group=r.get("group"),
        )
        for r in read_jsonl(normalized.queries)
    ]
    qrels = read_qrels(normalized.qrels)

    p = scenario.perturbations
    vocabulary = scenario_term_groups(scenario.vocabulary_path)
    variants = make_variants(queries, p.seed, list(p.kinds), vocabulary)
    task_set = build_tasks(scenario, variants, qrels, docs_by_id)

    out = ws.scenario_dir(scenario.id)
    write_jsonl(out / "queries.variants.jsonl", (v.model_dump(exclude_none=True) for v in variants))
    write_jsonl(
        out / "tasks.jsonl", (t.model_dump(mode="json", exclude_none=True) for t in task_set.tasks)
    )
    write_qrels(out / "qrels.tasks.tsv", task_set.qrels)

    result = PrepareResult(
        scenario_dir=out,
        n_docs=normalized.n_docs,
        n_queries=normalized.n_queries,
        n_variants=len(variants),
        n_tasks=len(task_set.tasks),
        skipped_queries=task_set.skipped_queries,
    )
    write_manifest(
        out / "prepare.manifest.json",
        {
            "stage": "prepare",
            "scenario": scenario.id,
            "dataset": {
                "id": spec.id,
                "version": spec.version,
                "files": {name: f.sha256 for name, f in spec.files.items()},
            },
            "seed": p.seed,
            "counts": {
                "docs": result.n_docs,
                "queries": result.n_queries,
                "variants": result.n_variants,
                "tasks": result.n_tasks,
                "skipped_queries": len(result.skipped_queries),
            },
            "skipped_queries": result.skipped_queries,
        },
    )
    return result
