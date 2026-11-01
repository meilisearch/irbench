import math
from collections import defaultdict

from pydantic import BaseModel
from ranx import Qrels as RanxQrels
from ranx import Run as RanxRun
from ranx import evaluate

from irbench.models import REFERENCE_PERTURBATIONS, Qrels, QueryTask, RunResults

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


class PairedDelta(BaseModel):
    """Mean nDCG@10 of a variant minus its original, over the queries that have both."""

    mode: str
    base: str
    perturbation: str
    group: str | None = None
    n: int
    delta: float
    ci95: tuple[float, float]


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
    """Rows per (mode, base, perturbation), plus one per query group.

    Reference kinds (a variant the dataset supplies, such as `corrected`) get their own
    perturbation row and are left out of every `all` row, so the headline keeps measuring
    what users type."""
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    by_query_group: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for task in tasks:
        if task.task_id not in qrels:
            raise ValueError(f"task {task.task_id} has no qrels")
        perturbations = [task.perturbation]
        if task.perturbation not in REFERENCE_PERTURBATIONS:
            perturbations.append("all")
        for p in perturbations:
            groups[(task.mode, task.base, p)].append(task.task_id)
            if task.group is not None:
                by_query_group[(task.mode, task.base, task.group, p)].append(task.task_id)
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
    # The `all` group rows come first, in the order they always had.
    group_keys = sorted(by_query_group, key=lambda k: (k[3] != "all", k[0], k[1], k[2], k[3]))
    rows += [
        GroupScore(
            mode=mode,
            base=base,
            perturbation=p,
            group=group,
            n=len(by_query_group[(mode, base, group, p)]),
            metrics=_evaluate(by_query_group[(mode, base, group, p)], qrels, run)[0],
        )
        for mode, base, group, p in group_keys
    ]
    return rows


def paired_deltas(tasks: list[QueryTask], per_task: dict[str, float]) -> list[PairedDelta]:
    """Variant minus original nDCG@10, per (mode, base, perturbation, group) and over all groups.

    A pair is a variant task and the original task of the same query, mode and base; both must
    have a score in `per_task` (which includes reference kinds)."""
    diffs: dict[tuple[str, str, str, str | None], list[float]] = defaultdict(list)
    for task in tasks:
        if task.perturbation == "original":
            continue
        original_id = f"{task.variant_of}:{task.mode}:{task.base}"
        if task.task_id not in per_task or original_id not in per_task:
            continue
        diff = per_task[task.task_id] - per_task[original_id]
        keys: list[str | None] = [None]
        if task.group is not None:
            keys.append(task.group)
        for group in keys:
            diffs[(task.mode, task.base, task.perturbation, group)].append(diff)
    return [
        PairedDelta(
            mode=mode,
            base=base,
            perturbation=p,
            group=group,
            n=len(values),
            delta=sum(values) / len(values),
            ci95=ci95(values),
        )
        for (mode, base, p, group), values in sorted(
            diffs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or "")
        )
    ]
