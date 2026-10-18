import math
from collections import defaultdict

from pydantic import BaseModel
from ranx import Qrels as RanxQrels
from ranx import Run as RanxRun
from ranx import evaluate

from irbench.models import Qrels, QueryTask, RunResults

METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "hit_rate@1"]
REPORTED = {"hit_rate@1": "success@1"}
JUDGED_DEPTH = 10
Z_95 = 1.959964


class GroupScore(BaseModel):
    mode: str
    base: str
    perturbation: str
    n: int
    metrics: dict[str, float]
    group: str | None = None
    # 95% confidence interval of the mean nDCG@10 over the group's tasks (normal approximation).
    ndcg_ci95: tuple[float, float] | None = None


def _ranked_scores(ids: list[str]) -> dict[str, float]:
    # A duplicate keeps its first (best) rank; later copies become non-relevant
    # placeholders so they still occupy their position and never pull docs up.
    scores: dict[str, float] = {}
    for i, doc_id in enumerate(ids):
        key = doc_id if doc_id not in scores else f"__duplicate_{i}"
        scores[key] = float(len(ids) - i)
    return scores


def judged_at_k(task_ids: list[str], qrels: Qrels, run: RunResults, k: int = JUDGED_DEPTH) -> float:
    """Mean share of each task's top-k results that carry a judgment (any grade, 0 included).

    Unjudged documents count as non-relevant in every metric, so a low value means the
    other metrics underestimate this run. Tasks that returned nothing are left out."""
    shares = []
    for t in task_ids:
        top = list(dict.fromkeys(run.get(t, [])))[:k]
        if top:
            judged = qrels.get(t, {})
            shares.append(sum(d in judged for d in top) / len(top))
    return sum(shares) / len(shares) if shares else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, mean)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = Z_95 * sd / math.sqrt(n)
    return (mean - half, mean + half)


def _evaluate(
    task_ids: list[str], qrels: Qrels, run: RunResults
) -> tuple[dict[str, float], dict[str, float]]:
    """Group metrics, and nDCG@10 per task (0 for a task with no results)."""
    run_dict = {t: _ranked_scores(run[t]) for t in task_ids if run.get(t)}
    per_task = dict.fromkeys(task_ids, 0.0)
    if not run_dict:
        metrics = {REPORTED.get(m, m): 0.0 for m in METRICS}
    else:
        ranx_run = RanxRun.from_dict(run_dict)
        values = evaluate(
            RanxQrels.from_dict({t: qrels[t] for t in task_ids}),
            ranx_run,
            METRICS,
            make_comparable=True,
        )
        metrics = {REPORTED.get(m, m): float(values[m]) for m in METRICS}
        for t, v in ranx_run.scores["ndcg@10"].items():
            per_task[t] = float(v)
    metrics[f"judged@{JUDGED_DEPTH}"] = judged_at_k(task_ids, qrels, run)
    return metrics, per_task


def per_task_ndcg(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> dict[str, float]:
    """nDCG@10 of every task, for paired comparisons between runs of the same scenario."""
    return _evaluate([t.task_id for t in tasks], qrels, run)[1]


def score_relevance(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> list[GroupScore]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    by_query_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for task in tasks:
        if task.task_id not in qrels:
            raise ValueError(f"task {task.task_id} has no qrels")
        groups[(task.mode, task.base, "all")].append(task.task_id)
        groups[(task.mode, task.base, task.perturbation)].append(task.task_id)
        if task.group is not None:
            by_query_group[(task.mode, task.base, task.group)].append(task.task_id)
    rows = []
    for (mode, base, p), ids in sorted(groups.items()):
        metrics, per_task = _evaluate(ids, qrels, run)
        rows.append(
            GroupScore(
                mode=mode,
                base=base,
                perturbation=p,
                n=len(ids),
                metrics=metrics,
                ndcg_ci95=ci95(list(per_task.values())),
            )
        )
    rows += [
        GroupScore(
            mode=mode,
            base=base,
            perturbation="all",
            group=group,
            n=len(ids),
            metrics=_evaluate(ids, qrels, run)[0],
        )
        for (mode, base, group), ids in sorted(by_query_group.items())
    ]
    return rows
