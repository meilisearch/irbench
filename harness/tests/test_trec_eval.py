"""Our metrics vs trec_eval (spec §10), through pytrec_eval's bindings to trec_eval's C code."""

import random

import pytest
import pytrec_eval

from irbench.models import QueryTask
from irbench.score.relevance import score_relevance

OURS_TO_TREC = {
    "ndcg@10": "ndcg_cut_10",
    "recall@10": "recall_10",
    "recall@100": "recall_100",
    "mrr@10": "recip_rank",  # computed on the top 10 only, below
    "success@1": "success_1",
}


def _fixture(seed: int = 7, n_tasks: int = 60, n_docs: int = 150):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(n_docs)]
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, list[str]] = {}
    for i in range(n_tasks):
        tid = f"t{i}"
        judged = rng.sample(docs, rng.randint(1, 20))
        grades = {d: rng.choice([0, 1, 2]) for d in judged}
        grades[judged[0]] = rng.choice([1, 2])  # prepare drops queries without a relevant doc
        qrels[tid] = grades
        if i % 10 != 9:  # every 10th task returned nothing: it must score 0
            run[tid] = rng.sample(docs, rng.randint(0, 120))
    return qrels, run


def _trec_eval(qrels, run, task_ids):
    # trec_eval orders by score (ties by docid), so give rank r a strictly decreasing score.
    full = {t: {d: float(1000 - r) for r, d in enumerate(run.get(t, []))} for t in task_ids}
    top10 = {t: {d: float(1000 - r) for r, d in enumerate(run.get(t, [])[:10])} for t in task_ids}
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.10", "recall.10", "recall.100", "recip_rank", "success.1"}
    )
    on_full, on_top10 = evaluator.evaluate(full), evaluator.evaluate(top10)
    out = {}
    for ours, trec in OURS_TO_TREC.items():
        source = on_top10 if trec == "recip_rank" else on_full
        # pytrec_eval omits queries with an empty run; they score 0, like ours.
        out[ours] = sum(source.get(t, {}).get(trec, 0.0) for t in task_ids) / len(task_ids)
    return out


def _task(tid: str) -> QueryTask:
    return QueryTask(
        task_id=tid,
        qid=tid,
        variant_of=tid,
        perturbation="original",
        text="x",
        mode="keyword",
        base="keyword",
    )


@pytest.mark.parametrize("seed", [7, 11, 2026])
def test_metrics_match_trec_eval(seed):
    qrels, run = _fixture(seed)
    tasks = [_task(t) for t in qrels]
    groups = score_relevance(tasks, qrels, run)
    ours = next(g for g in groups if g.perturbation == "all").metrics
    expected = _trec_eval(qrels, run, list(qrels))
    for name, value in expected.items():
        assert ours[name] == pytest.approx(value, abs=1e-6), name
