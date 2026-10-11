import json

from typer.testing import CliRunner

from irbench.cli import app
from irbench.io import read_jsonl, read_qrels, write_run
from irbench.stages.prepare import prepare
from test_prepare import fixture_client, make_root

SCENARIO = "ecommerce/product-search"


def _perfect_run(scenario_dir):
    qrels = read_qrels(scenario_dir / "qrels.tasks.tsv")
    return {
        tid: [d for d, g in sorted(grades.items(), key=lambda kv: -kv[1]) if g > 0]
        for tid, grades in qrels.items()
    }


def test_prepare_then_score_perfect_run(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    result = prepare(root, SCENARIO, client=fixture_client(wands_raw))
    write_run(tmp_path / "perfect.trec", _perfect_run(result.scenario_dir), tag="oracle")

    out = tmp_path / "score.json"
    cli = CliRunner().invoke(
        app,
        [
            "score",
            "--scenario",
            SCENARIO,
            "--run",
            str(tmp_path / "perfect.trec"),
            "--out",
            str(out),
            "--root",
            str(root),
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert "keyword/keyword: nDCG@10=1.0000" in cli.stdout
    report = json.loads(out.read_text())
    n_tasks = len(list(read_jsonl(result.scenario_dir / "tasks.jsonl")))
    overall = [g for g in report["groups"] if g["perturbation"] == "all"]
    assert sum(g["n"] for g in overall) == n_tasks
    assert all(g["metrics"]["ndcg@10"] == 1.0 for g in overall)


def test_score_rejects_unknown_task_ids(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    prepare(root, SCENARIO, client=fixture_client(wands_raw))
    write_run(tmp_path / "bad.trec", {"nope:keyword:keyword": ["0"]}, tag="x")
    cli = CliRunner().invoke(
        app,
        [
            "score",
            "--scenario",
            SCENARIO,
            "--run",
            str(tmp_path / "bad.trec"),
            "--out",
            str(tmp_path / "s.json"),
            "--root",
            str(root),
        ],
    )
    assert cli.exit_code == 1
    assert "nope:keyword:keyword" in cli.output
