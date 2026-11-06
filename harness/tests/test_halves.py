from irbench.models import QueryTask
from irbench.report.data import published_half
from irbench.score.halves import assign_halves, task_halves, topic_of
from irbench.score.relevance import score_relevance


def _task(qid: str, source: str, group: str | None = None) -> QueryTask:
    return QueryTask(
        task_id=f"{qid}:keyword:keyword",
        qid=qid,
        variant_of=source,
        perturbation="original" if qid == source else "typo",
        text="x",
        mode="keyword",
        base="keyword",
        group=group,
    )


def test_halves_are_balanced_and_stable():
    qids = [f"q{i}" for i in range(101)]
    halves = assign_halves("s", qids)
    assert sum(h == "tune" for h in halves.values()) == 51
    assert sum(h == "test" for h in halves.values()) == 50
    # Same input, any order: same split. Another scenario: another split.
    assert assign_halves("s", reversed(qids)) == halves
    assert assign_halves("other", qids) != halves


def test_variants_and_query_forms_follow_their_topic():
    tasks = [
        _task("1-keyword", "1-keyword", "keyword"),
        _task("1-question", "1-question", "question"),
        _task("1-question~typo", "1-question", "question"),
        *[_task(f"{i}-keyword", f"{i}-keyword", "keyword") for i in range(2, 20)],
    ]
    halves = task_halves("s", tasks)
    assert topic_of("1-question", "question") == "1"
    first = {halves[t.task_id] for t in tasks[:3]}
    assert len(first) == 1


def test_score_rows_per_half_and_report_keeps_the_test_half():
    tasks = [_task("a", "a"), _task("b", "b")]
    qrels = {t.task_id: {"d1": 1} for t in tasks}
    run = {"a:keyword:keyword": ["d1"], "b:keyword:keyword": ["d9"]}
    halves = {"a:keyword:keyword": "tune", "b:keyword:keyword": "test"}
    rows = score_relevance(tasks, qrels, run, halves)
    by_half = {(r.half, r.perturbation): r.metrics["ndcg@10"] for r in rows if r.group is None}
    assert by_half[(None, "all")] == 0.5
    assert by_half[("tune", "all")] == 1.0
    assert by_half[("test", "all")] == 0.0

    scores = {
        "relevance": [r.model_dump(exclude_none=True) for r in rows],
        "per_task": {"ndcg@10": {"a:keyword:keyword": 1.0, "b:keyword:keyword": 0.0}},
        "halves": halves,
    }
    shown = published_half(scores)
    assert {r["metrics"]["ndcg@10"] for r in shown["relevance"]} == {0.0}
    assert all("half" not in r for r in shown["relevance"])
    assert shown["per_task"]["ndcg@10"] == {"b:keyword:keyword": 0.0}


def test_report_keeps_the_test_half_of_the_paired_rows():
    row = {"mode": "keyword", "base": "keyword", "perturbation": "corrected", "n": 1}
    scores = {
        "relevance": [],
        "halves": {"a:keyword:keyword": "tune"},
        "paired": [
            {**row, "delta": 0.5, "ci95": [0.5, 0.5]},
            {**row, "half": "tune", "delta": 0.1, "ci95": [0.1, 0.1]},
            {**row, "half": "test", "delta": 0.9, "ci95": [0.9, 0.9]},
        ],
    }
    assert published_half(scores)["paired"] == [{**row, "delta": 0.9, "ci95": [0.9, 0.9]}]


def test_bundles_scored_before_halves_are_unchanged():
    scores = {"relevance": [{"mode": "keyword"}], "per_task": {"ndcg@10": {}}}
    assert published_half(scores) is scores


def test_sweep_variants_retrieve_only_the_tune_half(tmp_path):
    import json

    from irbench.paths import Workspace
    from irbench.runs import RunSpec
    from irbench.stages.context import RunContext

    tasks = [_task(f"q{i}", f"q{i}") for i in range(20)]
    d = Workspace(tmp_path).scenario_dir("a/b")
    d.mkdir(parents=True)
    (d / "tasks.jsonl").write_text("".join(json.dumps(t.model_dump()) + "\n" for t in tasks))

    def ctx(track: str) -> RunContext:
        return RunContext(tmp_path, RunSpec(scenario="a/b", engine="e", model="m", track=track))

    halves = task_halves("a/b", tasks)
    assert len(ctx("tuned").tasks) == 20
    sweep = ctx("tuned+x").tasks
    assert len(sweep) == 10
    assert {halves[t.task_id] for t in sweep} == {"tune"}
    assert len(ctx("tuned+x").all_tasks) == 20
