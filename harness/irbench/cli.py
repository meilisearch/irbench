from pathlib import Path

import typer

from irbench import __version__
from irbench.stages.embed import embed_scenario
from irbench.stages.prepare import prepare as run_prepare

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
