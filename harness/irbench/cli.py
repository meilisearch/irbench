import json
from pathlib import Path

import typer

from irbench import __version__
from irbench.io import read_json, read_jsonl, read_qrels, read_run
from irbench.models import QueryTask
from irbench.paths import Workspace
from irbench.runs import STAGES, RunSpec
from irbench.score.relevance import score_relevance
from irbench.stages.embed import embed_scenario
from irbench.stages.prepare import prepare as run_prepare
from irbench.stages.run import run_pipeline

app = typer.Typer(no_args_is_help=True, help="IRBench: information retrieval benchmark harness.")

ROOT_OPTION = typer.Option(Path.cwd(), "--root", help="Repository root.")


@app.callback()
def main() -> None:
    """IRBench CLI."""


@app.command()
def version() -> None:
    """Print the harness version."""
    typer.echo(__version__)


@app.command()
def prepare(
    scenario: str = typer.Option(
        ..., "--scenario", help="Scenario id, e.g. ecommerce/product-search."
    ),
    root: Path = ROOT_OPTION,
) -> None:
    """Download the dataset, generate query variants and per-mode tasks."""
    r = run_prepare(root, scenario)
    typer.echo(
        f"{scenario}: {r.n_docs} docs, {r.n_queries} queries, {r.n_variants} variants, "
        f"{r.n_tasks} tasks; skipped {len(r.skipped_queries)} queries without relevant judgments"
    )
    typer.echo(f"output: {r.scenario_dir}")


@app.command()
def embed(
    scenario: str = typer.Option(..., "--scenario"),
    model: str = typer.Option(..., "--model", help="Model id from the embedding registry."),
    root: Path = ROOT_OPTION,
) -> None:
    """Embed docs and query variants with one model (cached, resumable)."""
    r = embed_scenario(root, scenario, model)
    typer.echo(f"{model}: {r.n_docs} docs, {r.n_queries} queries, {r.n_computed} newly embedded")
    typer.echo(f"docs: {r.docs_path}\nqueries: {r.queries_path}")


@app.command()
def score(
    scenario: str = typer.Option(..., "--scenario"),
    run: Path = typer.Option(..., "--run", exists=True, dir_okay=False, help="TREC run file."),
    out: Path = typer.Option(..., "--out", help="Where to write the JSON score report."),
    root: Path = ROOT_OPTION,
) -> None:
    """Score a TREC run file against the scenario's task qrels."""
    scenario_dir = Workspace(root).scenario_dir(scenario)
    tasks = [QueryTask.model_validate(r) for r in read_jsonl(scenario_dir / "tasks.jsonl")]
    qrels = read_qrels(scenario_dir / "qrels.tasks.tsv")
    results = read_run(run)

    unknown = sorted(set(results) - {t.task_id for t in tasks})
    if unknown:
        typer.echo(f"run references unknown task ids: {', '.join(unknown[:10])}", err=True)
        raise typer.Exit(code=1)

    groups = score_relevance(tasks, qrels, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"scenario": scenario, "run": str(run), "groups": [g.model_dump() for g in groups]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for g in groups:
        if g.perturbation == "all":
            typer.echo(f"{g.mode}/{g.base}: nDCG@10={g.metrics['ndcg@10']:.4f} n={g.n}")


@app.command()
def run(
    scenario: str = typer.Option(..., "--scenario"),
    engine: str = typer.Option(..., "--engine", help="Folder name under engines/."),
    model: str = typer.Option(..., "--model"),
    track: str = typer.Option("default", "--track", help="default or tuned."),
    hardware: str = typer.Option("local", "--hardware", help="Profile under hardware/."),
    host: str = typer.Option("localhost", "--host", help="Host publishing the engine port."),
    stages: str = typer.Option(",".join(STAGES), "--stages", help="Comma-separated stages."),
    root: Path = ROOT_OPTION,
) -> None:
    """Run the pipeline for one scenario × engine × model × track."""
    spec = RunSpec(
        scenario=scenario, engine=engine, model=model, track=track, hardware=hardware, host=host
    )
    names = [s.strip() for s in stages.split(",") if s.strip()]
    try:
        manifest = run_pipeline(root, spec, names)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    for name, info in manifest["stages"].items():
        line = f"{name}: {info['status']}"
        if info["status"] == "failed":
            line += f" — {info['error']}"
        typer.echo(line)
    run_dir = Workspace(root).run_dir(spec.run_id)
    if manifest["stages"].get("score", {}).get("status") == "succeeded":
        scores = read_json(run_dir / "scores.json")
        for g in scores["relevance"]:
            if g["perturbation"] == "all":
                ndcg = g["metrics"]["ndcg@10"]
                typer.echo(f"{g['mode']}/{g['base']}: nDCG@10={ndcg:.4f} n={g['n']}")
        for key, reason in scores["not_applicable"].items():
            typer.echo(f"{key.replace(':', '/')}: N/A ({reason})")
    typer.echo(f"run dir: {run_dir}")
    if manifest["status"] != "succeeded":
        raise typer.Exit(code=1)
