import pytest

from irbench.models import QueryTask
from irbench.score.relevance import (
    ci95,
    clean_and_found_at_k,
    judged_at_k,
    per_task_ndcg,
    score_relevance,
)


def _task(
    tid: str, perturbation: str = "original", mode: str = "keyword", group: str | None = None
) -> QueryTask:
    return QueryTask(
        task_id=tid,
        qid=tid,
        variant_of=tid,
        perturbation=perturbation,
        text="x",
        mode=mode,
        base="keyword" if mode != "semantic" else "semantic",
        group=group,
    )


QRELS = {"t1": {"d1": 2, "d2": 1, "d3": 0}, "t2": {"d5": 1}}


def _group(scores, perturbation="all", mode="keyword"):
    return next(s for s in scores if s.perturbation == perturbation and s.mode == mode)


def test_hand_computed_metrics():
    # trec_eval definitions: linear gain, log2(rank+1) discount.
    # DCG = 0/1 + 2/log2(3) = 1.26186; IDCG = 2/1 + 1/log2(3) = 2.63093.
    scores = score_relevance([_task("t1")], QRELS, {"t1": ["d3", "d1", "d9"]})
    m = _group(scores).metrics
    assert m["ndcg@10"] == pytest.approx(0.47962, abs=1e-4)
    assert m["recall@10"] == pytest.approx(0.5)
    assert m["mrr@10"] == pytest.approx(0.5)
    assert m["success@1"] == pytest.approx(0.0)
    # d1 at rank 2; d2 not returned. P@1 = 0, P@5 = 1/5, R@20 = 1/2; nDCG@k equal past rank 3.
    assert m["precision@1"] == pytest.approx(0.0)
    assert m["precision@5"] == pytest.approx(0.2)
    assert m["recall@20"] == pytest.approx(0.5)
    assert m["ndcg@5"] == m["ndcg@20"] == m["ndcg@100"] == pytest.approx(m["ndcg@10"])


def test_missing_run_entry_scores_zero():
    scores = score_relevance([_task("t1"), _task("t2")], QRELS, {"t1": ["d3", "d1", "d9"]})
    overall = _group(scores)
    assert overall.n == 2
    assert overall.metrics["ndcg@10"] == pytest.approx(0.47962 / 2, abs=1e-4)
    assert overall.metrics["recall@100"] == pytest.approx(0.25)


def test_duplicate_ids_keep_first_rank():
    deduped = score_relevance([_task("t1")], QRELS, {"t1": ["d1", "d2"]})
    with_dupes = score_relevance([_task("t1")], QRELS, {"t1": ["d1", "d2", "d1", "d1"]})
    assert _group(with_dupes).metrics == _group(deduped).metrics
    assert _group(deduped).metrics["ndcg@10"] == pytest.approx(1.0)


def test_duplicates_do_not_pull_later_docs_up():
    # d1 sits at rank 3; collapsing the duplicate must not move it to rank 2.
    scores = score_relevance([_task("t2")], {"t2": {"d1": 1}}, {"t2": ["x", "x", "d1"]})
    assert _group(scores).metrics["ndcg@10"] == pytest.approx(0.5)  # 1 / log2(3 + 1)


def test_groups_by_mode_and_perturbation():
    tasks = [_task("t1"), _task("t2", perturbation="typo")]
    scores = score_relevance(tasks, QRELS, {"t1": ["d1"], "t2": ["d5"]})
    keys = {(s.mode, s.base, s.perturbation) for s in scores}
    assert keys == {
        ("keyword", "keyword", "all"),
        ("keyword", "keyword", "original"),
        ("keyword", "keyword", "typo"),
    }
    assert _group(scores, "typo").n == 1
    assert _group(scores, "typo").metrics["success@1"] == pytest.approx(1.0)


def test_empty_run_is_all_zero():
    # Except clean@k: a run that shows nothing shows no junk.
    scores = score_relevance([_task("t1")], QRELS, {})
    m = _group(scores).metrics
    assert all(v == 0.0 for name, v in m.items() if not name.startswith("clean@"))
    assert all(m[f"clean@{k}"] == 1.0 for k in (5, 10, 20, 100))


# t1: d1 grade 2, d2 grade 1, d3 judged not relevant (0).
def test_found_is_gain_over_best_gain_in_k():
    m = clean_and_found_at_k(["t1"], QRELS, {"t1": ["d3", "d9", "d2", "d1"]}, k=2)
    assert m["found@2"] == pytest.approx(0.0)  # d3, d9 first; best possible is d1 + d2 (3)
    m = clean_and_found_at_k(["t1"], QRELS, {"t1": ["d3", "d2", "d9", "d1"]}, k=5)
    assert m["found@5"] == pytest.approx(1.0)  # d2 + d1 = 3 = best 3


def test_clean_drops_with_judged_junk_only():
    # Only the two relevant docs: no junk.
    assert clean_and_found_at_k(["t1"], QRELS, {"t1": ["d1", "d2"]}, k=5)["clean@5"] == 1.0
    # Unjudged d9 is ignored.
    assert clean_and_found_at_k(["t1"], QRELS, {"t1": ["d1", "d9"]}, k=5)["clean@5"] == 1.0
    # d1 then judged-irrelevant d3: mean grade 1, best for 2 results is (2 + 1) / 2.
    m = clean_and_found_at_k(["t1"], QRELS, {"t1": ["d1", "d3"]}, k=5)
    assert m["clean@5"] == pytest.approx(1 / 1.5)
    # Showing only the best doc is perfect, not over 1.
    assert clean_and_found_at_k(["t1"], QRELS, {"t1": ["d1"]}, k=5)["clean@5"] == 1.0


def test_clean_is_one_when_nothing_judged_is_shown():
    m = clean_and_found_at_k(["t1"], QRELS, {"t1": ["d9"]}, k=5)
    assert m == {"found@5": 0.0, "clean@5": 1.0}


def test_clean_skips_tasks_without_relevant_documents():
    qrels = {"t1": {"d1": 1}, "t2": {"d2": 0}}
    m = clean_and_found_at_k(["t1", "t2"], qrels, {"t1": ["d1"], "t2": ["d2"]}, k=5)
    assert m == {"found@5": 1.0, "clean@5": 1.0}


def test_clean_left_out_when_qrels_list_only_relevant_documents():
    qrels = {"t1": {"d1": 1}}
    m = _group(score_relevance([_task("t1")], qrels, {"t1": ["d1", "d9"]})).metrics
    assert "found@20" in m
    assert not any(name.startswith("clean@") for name in m)


def test_group_rows_are_added_after_the_existing_rows():
    tasks = [_task("t1", group="question"), _task("t2", group="keyword")]
    rows = score_relevance(tasks, QRELS, {"t1": ["d1"], "t2": ["d5"]})
    plain = [r for r in rows if r.group is None]
    grouped = [r for r in rows if r.group is not None]
    assert rows[: len(plain)] == plain
    assert {(r.mode, r.base, r.perturbation, r.group) for r in grouped} == {
        ("keyword", "keyword", "all", "question"),
        ("keyword", "keyword", "all", "keyword"),
    }
    assert sum(r.n for r in grouped) == next(r for r in plain if r.perturbation == "all").n


def test_ungrouped_tasks_produce_exactly_todays_rows():
    rows = score_relevance([_task("t1"), _task("t2")], QRELS, {"t1": ["d1"], "t2": ["d5"]})
    assert all(r.group is None for r in rows)
    dumped = [r.model_dump(exclude_none=True) for r in rows]
    keys = {"mode", "base", "perturbation", "n", "metrics", "ndcg_ci95"}
    assert all(set(d) == keys for d in dumped)


def test_judged_counts_any_grade_and_skips_empty_results():
    # t1: d3 (grade 0) and d1 judged, d9 not -> 2/3; t2 returned nothing and is left out.
    assert judged_at_k(["t1", "t2"], QRELS, {"t1": ["d3", "d1", "d9"]}) == pytest.approx(2 / 3)
    assert judged_at_k(["t1"], QRELS, {"t1": ["d9", "d9", "d1"]}) == pytest.approx(1 / 2)
    assert judged_at_k(["t1"], QRELS, {}) == 0.0
    scores = score_relevance([_task("t1")], QRELS, {"t1": ["d3", "d1", "d9"]})
    assert _group(scores).metrics["judged@10"] == pytest.approx(2 / 3)


def test_judged_looks_at_the_top_ten_only():
    run = {"t2": [f"x{i}" for i in range(10)] + ["d5"]}
    assert judged_at_k(["t2"], QRELS, run) == 0.0


def test_ci95_brackets_the_mean():
    lo, hi = ci95([0.0, 1.0, 0.0, 1.0])
    # sd = 0.57735, half-width = 1.96 * 0.57735 / 2 = 0.5658
    assert (lo, hi) == (
        pytest.approx(0.5 - 0.5658, abs=1e-3),
        pytest.approx(0.5 + 0.5658, abs=1e-3),
    )
    assert ci95([0.4]) == (0.4, 0.4)


def test_per_task_ndcg_matches_the_group_mean():
    tasks = [_task("t1"), _task("t2")]
    run = {"t1": ["d3", "d1", "d9"]}
    per_task = per_task_ndcg(tasks, QRELS, run)
    assert per_task == {"t1": pytest.approx(0.47962, abs=1e-4), "t2": 0.0}
    overall = _group(score_relevance(tasks, QRELS, run))
    assert sum(per_task.values()) / 2 == pytest.approx(overall.metrics["ndcg@10"])
    lo, hi = overall.ndcg_ci95
    assert lo < overall.metrics["ndcg@10"] < hi
