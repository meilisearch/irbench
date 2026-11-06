import math
from collections import defaultdict

from pydantic import BaseModel
from ranx import Qrels as RanxQrels
from ranx import Run as RanxRun
from ranx import evaluate

from irbench.models import REFERENCE_PERTURBATIONS, Qrels, QueryTask, RunResults

# Cutoffs reported per metric. clean@k and found@k come from the Meilisearch relevancy-bench
# protocol; nDCG@10 stays the headline score.
NDCG_CUTOFFS = (5, 10, 20, 100)
CLEAN_CUTOFFS = (5, 10, 20, 100)
PRECISION_CUTOFFS = (1, 5, 20, 100)
RECALL_CUTOFFS = (10, 20, 100)
METRICS = [
    *(f"ndcg@{k}" for k in NDCG_CUTOFFS),
    *(f"precision@{k}" for k in PRECISION_CUTOFFS),
    *(f"recall@{k}" for k in RECALL_CUTOFFS),
    "mrr@10",
    "hit_rate@1",
]
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
    # "tune" or "test" for a row scored on one half of the queries (irbench.score.halves);
    # None for the row over every query.
    half: str | None = None
    # 95% confidence interval of the mean nDCG@10 over the group's tasks (normal approximation).
    ndcg_ci95: tuple[float, float] | None = None


class PairedDelta(BaseModel):
    """Mean nDCG@10 of a variant minus its original, over the queries that have both."""

    mode: str
    base: str
    perturbation: str
    group: str | None = None
    # "tune" or "test" for a row over one half of the queries (irbench.score.halves); None for
    # the row over every query.
    half: str | None = None
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


def _ideal_gain(grades: list[int], n: int) -> float:
    """Sum of the `n` best relevant grades (fewer if there are fewer relevant documents)."""
    return float(sum(grades[:n]))


def clean_and_found_at_k(
    task_ids: list[str], qrels: Qrels, run: RunResults, k: int, with_clean: bool = True
) -> dict[str, float]:
    """clean@k and found@k, means over the tasks that have at least one relevant document.

    found@k: gain of the top k results over the best gain possible in k results
    (normalised cumulative gain, no rank discount; binary grades: relevant found / min(R, k)).

    clean@k: mean grade of the *judged* results among the top k, over the best mean grade
    possible for that many results. It drops when an engine pads its results with documents
    judged not relevant; unjudged documents are left out, and an engine that returns no judged
    result scores 1 (it showed no junk; found@k shows what it missed). Only meaningful when the
    qrels judge non-relevant documents, so the caller omits it otherwise."""
    clean: list[float] = []
    found: list[float] = []
    for t in task_ids:
        judged = qrels.get(t, {})
        grades = sorted((g for g in judged.values() if g > 0), reverse=True)
        if not grades:
            continue
        top = list(dict.fromkeys(run.get(t, [])))[:k]
        found.append(sum(judged.get(d, 0) for d in top) / _ideal_gain(grades, k))
        shown = [judged[d] for d in top if d in judged]
        if not shown:
            clean.append(1.0)
        else:
            # Best mean grade for len(shown) results: the top grades, then non-relevant ones.
            best = _ideal_gain(grades, len(shown)) / len(shown)
            clean.append(sum(shown) / len(shown) / best)
    out: dict[str, float] = {}
    if found:
        out[f"found@{k}"] = sum(found) / len(found)
        if with_clean:
            out[f"clean@{k}"] = sum(clean) / len(clean)
    return out


def has_negative_judgments(qrels: Qrels) -> bool:
    """True when the qrels mark some documents as judged not relevant (grade 0)."""
    return any(g <= 0 for judged in qrels.values() for g in judged.values())


def ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, mean)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = Z_95 * sd / math.sqrt(n)
    return (mean - half, mean + half)


def _evaluate(
    task_ids: list[str], qrels: Qrels, run: RunResults, with_clean: bool = True
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
    for k in CLEAN_CUTOFFS:
        metrics.update(clean_and_found_at_k(task_ids, qrels, run, k, with_clean))
    return metrics, per_task


def per_task_ndcg(tasks: list[QueryTask], qrels: Qrels, run: RunResults) -> dict[str, float]:
    """nDCG@10 of every task, for paired comparisons between runs of the same scenario."""
    return _evaluate([t.task_id for t in tasks], qrels, run)[1]


def score_relevance(
    tasks: list[QueryTask],
    qrels: Qrels,
    run: RunResults,
    halves: dict[str, str] | None = None,
) -> list[GroupScore]:
    """Rows over every task, then, when `halves` (task id -> half) is given, the same rows
    for each half of the queries."""
    rows = _score_rows(tasks, qrels, run, None)
    if halves:
        for half in sorted(set(halves.values())):
            subset = [t for t in tasks if halves.get(t.task_id) == half]
            if subset:
                rows += _score_rows(subset, qrels, run, half)
    return rows


def _score_rows(
    tasks: list[QueryTask], qrels: Qrels, run: RunResults, half: str | None
) -> list[GroupScore]:
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
    # clean@k needs judged-non-relevant documents; qrels listing only relevant ones (CQADupStack,
    # LoTTE) would score every run 1, so it is left out for them.
    with_clean = has_negative_judgments(
        {t: qrels[t] for task_ids in groups.values() for t in task_ids}
    )
    rows = []
    for (mode, base, p), ids in sorted(groups.items()):
        metrics, per_task = _evaluate(ids, qrels, run, with_clean)
        rows.append(
            GroupScore(
                mode=mode,
                base=base,
                perturbation=p,
                n=len(ids),
                metrics=metrics,
                half=half,
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
            half=half,
            n=len(by_query_group[(mode, base, group, p)]),
            metrics=_evaluate(by_query_group[(mode, base, group, p)], qrels, run, with_clean)[0],
        )
        for mode, base, group, p in group_keys
    ]
    return rows


def paired_deltas(
    tasks: list[QueryTask], per_task: dict[str, float], half: str | None = None
) -> list[PairedDelta]:
    """Variant minus original nDCG@10, per (mode, base, perturbation, group) and over all groups.

    A pair is a variant task and the original task of the same query, mode and base; both must
    have a score in `per_task` (which includes reference kinds). `half` labels rows computed
    over one half of the queries (the caller passes that half's tasks)."""
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
            half=half,
            n=len(values),
            delta=sum(values) / len(values),
            ci95=ci95(values),
        )
        for (mode, base, p, group), values in sorted(
            diffs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or "")
        )
    ]
