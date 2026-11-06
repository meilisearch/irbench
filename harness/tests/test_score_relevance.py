from types import SimpleNamespace

import pytest

from irbench.io import write_json, write_jsonl, write_run
from irbench.models import QueryTask
from irbench.score.relevance import (
    ci95,
    clean_and_found_at_k,
    judged_at_k,
    paired_deltas,
    per_task_ndcg,
    score_relevance,
)
from irbench.stages.score_run import score_stage


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
    in_all = [r for r in grouped if r.perturbation == "all"]
    assert grouped[: len(in_all)] == in_all
    assert {(r.mode, r.base, r.perturbation, r.group) for r in in_all} == {
        ("keyword", "keyword", "all", "question"),
        ("keyword", "keyword", "all", "keyword"),
    }
    assert sum(r.n for r in in_all) == next(r for r in plain if r.perturbation == "all").n


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


def _variant(
    qid: str, kind: str, group: str | None = None, mode: str = "keyword", base: str = "keyword"
) -> QueryTask:
    orig = qid.removesuffix(f"-{kind}")
    return QueryTask(
        task_id=f"{qid}:{mode}:{base}",
        qid=qid,
        variant_of=orig,
        perturbation=kind,
        text="x",
        mode=mode,
        base=base,
        group=group,
    )


def _plain(qid: str, group: str | None = None) -> QueryTask:
    return _variant(qid, "original", group)


def _row(rows, perturbation, group=None):
    return next(r for r in rows if r.perturbation == perturbation and r.group == group)


QRELS_Q = {
    f"{q}:keyword:keyword": {"d1": 2, "d2": 1}
    for q in ("a", "a-typo", "a-corrected", "b", "b-typo", "b-corrected")
}
RUN_Q = {
    "a:keyword:keyword": ["d1"],
    "a-typo:keyword:keyword": ["x", "d1"],
    "a-corrected:keyword:keyword": ["d1", "d2"],
    "b:keyword:keyword": ["d2"],
    "b-typo:keyword:keyword": ["d1"],
    "b-corrected:keyword:keyword": ["d2"],
}


def _tasks_q():
    return [
        _plain("a", "misspelled"),
        _variant("a-typo", "typo", "misspelled"),
        _variant("a-corrected", "corrected", "misspelled"),
        _plain("b", "clean"),
        _variant("b-typo", "typo", "clean"),
        _variant("b-corrected", "corrected", "clean"),
    ]


def test_corrected_not_in_all():
    rows = score_relevance(_tasks_q(), QRELS_Q, RUN_Q)
    assert _row(rows, "all").n == 4  # originals and typos only
    assert _row(rows, "corrected").n == 2
    assert _row(rows, "all", "misspelled").n == 2
    assert _row(rows, "corrected", "misspelled").n == 1
    assert _row(rows, "typo", "clean").n == 1
    # the headline does not move when a corrected task is added
    without = [t for t in _tasks_q() if t.perturbation != "corrected"]
    base = score_relevance(without, QRELS_Q, RUN_Q)
    assert _row(rows, "all").metrics == _row(base, "all").metrics
    assert _row(rows, "all", "misspelled").metrics == _row(base, "all", "misspelled").metrics


def test_existing_rows_unchanged():
    # Pinned from the code before reference kinds existed: these tasks have none.
    tasks = [
        _plain("a", "misspelled"),
        _variant("a-typo", "typo", "misspelled"),
        _plain("b", "clean"),
    ]
    rows = score_relevance(tasks, QRELS_Q, RUN_Q)
    old = [r for r in rows if r.group is None or r.perturbation == "all"]
    third = 1 / 3
    assert [(r.perturbation, r.group, r.n) for r in old] == [
        ("all", None, 3),
        ("original", None, 2),
        ("typo", None, 1),
        ("all", "clean", 1),
        ("all", "misspelled", 2),
    ]
    assert [r.metrics["ndcg@10"] for r in old] == pytest.approx(
        [0.5399687444, 0.5701406501, 0.4796249332, 0.3800937667, 0.6199062333]
    )
    assert [r.metrics["success@1"] for r in old] == pytest.approx([2 * third, 1, 0, 1, 0.5])
    assert [r.metrics["judged@10"] for r in old] == pytest.approx([0.8333333333, 1, 0.5, 1, 0.75])
    assert old[0].ndcg_ci95 == pytest.approx((0.3169319328, 0.7630055561))
    assert old[3].ndcg_ci95 is None


def test_paired_deltas():
    tasks = [
        _plain("a", "misspelled"),
        _variant("a-typo", "typo", "misspelled"),
        _plain("b", "misspelled"),
        _variant("b-typo", "typo", "misspelled"),
        _plain("c", "clean"),  # an original with no variant is not paired
    ]
    per_task = {
        "a:keyword:keyword": 0.2,
        "a-typo:keyword:keyword": 0.6,
        "b:keyword:keyword": 0.5,
        "b-typo:keyword:keyword": 0.3,
        "c:keyword:keyword": 0.9,
    }
    rows = paired_deltas(tasks, per_task)
    overall = next(r for r in rows if r.group is None)
    assert (overall.perturbation, overall.n) == ("typo", 2)
    assert overall.delta == pytest.approx(0.1)
    lo, hi = overall.ci95
    assert lo < 0.1 < hi
    assert (lo, hi) == pytest.approx(ci95([0.4, -0.2]))
    assert [(r.group, r.n) for r in rows] == [(None, 2), ("misspelled", 2)]


def test_paired_deltas_skips_unsupported_pairs():
    tasks = [_plain("a"), _variant("a-typo", "typo"), _variant("b-typo", "typo")]
    per_task = {
        "a:keyword:keyword": 0.5,
        "a-typo:keyword:keyword": 0.25,
        "b-typo:keyword:keyword": 1,
    }
    rows = paired_deltas(tasks, per_task)
    assert [(r.n, r.delta) for r in rows] == [(1, pytest.approx(-0.25))]


def test_per_task_excludes_reference(tmp_path):
    tasks = _tasks_q()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "na.json", {})
    write_run(run_dir / "run.trec", RUN_Q, "t")
    write_jsonl(run_dir / "checks.jsonl", [])
    write_jsonl(run_dir / "errors.jsonl", [])
    ctx = SimpleNamespace(
        run_dir=run_dir,
        tasks=tasks,
        all_tasks=tasks,
        qrels=QRELS_Q,
        spec=SimpleNamespace(run_id="r", scenario="s", engine="e", model="m", track="default"),
    )
    report = score_stage(ctx)
    per_task = report["per_task"]["ndcg@10"]
    assert not any("-corrected:" in t for t in per_task)
    assert len(per_task) == 4
    # the paired block still sees them
    corrected = [p for p in report["paired"] if p["perturbation"] == "corrected"]
    assert corrected and corrected[0]["n"] == 2 and "half" not in corrected[0]
    # and, like the relevance rows, once per half of the queries (the report shows `test`)
    halved = [p for p in corrected if "half" in p]
    assert {p["half"] for p in halved} <= {"tune", "test"}
    assert sum(p["n"] for p in halved if "group" not in p) == 2
    assert all("group" not in p or p["group"] is not None for p in report["paired"])
