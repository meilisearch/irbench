from collections import defaultdict

from pydantic import BaseModel
from ranx import Qrels as RanxQrels
from ranx import Run as RanxRun
from ranx import evaluate

from irbench.models import Qrels, QueryTask, RunResults

METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "hit_rate@1"]
REPORTED = {"hit_rate@1": "success@1"}


class GroupScore(BaseModel):
    mode: str
    base: str
    perturbation: str
    n: int
    metrics: dict[str, float]


def _ranked_scores(ids: list[str]) -> dict[str, float]:
    # A duplicate keeps its first (best) rank; later copies become non-relevant
    # placeholders so they still occupy their position and never pull docs up.
    scores: dict[str, float] = {}
    for i, doc_id in enumerate(ids):
        key = doc_id if doc_id not in scores else f"__duplicate_{i}"
        scores[key] = float(len(ids) - i)
    return scores


def _evaluate(task_ids: list[str], qrels: Qrels, run: RunResults) -> dict[str, float]:
    run_dict = {t: _ranked_scores(run[t]) for t in task_ids if run.get(t)}
    if not run_dict:
        return {REPORTED.get(m, m): 0.0 for m in METRICS}
    values = evaluate(
        RanxQrels.from_dict({t: qrels[t] for t in task_ids}),
        RanxRun.from_dict(run_dict),
        METRICS,
        make_comparable=True,
    )
    return {REPORTED.get(m, m): float(values[m]) for m in METRICS}


def score_relevance(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> list[GroupScore]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for task in tasks:
        if task.task_id not in qrels:
            raise ValueError(f"task {task.task_id} has no qrels")
        groups[(task.mode, task.base, "all")].append(task.task_id)
        groups[(task.mode, task.base, task.perturbation)].append(task.task_id)
    return [
        GroupScore(
            mode=mode, base=base, perturbation=p, n=len(ids), metrics=_evaluate(ids, qrels, run)
        )
        for (mode, base, p), ids in sorted(groups.items())
    ]
