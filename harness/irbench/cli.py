from pathlib import Path

import typer

from irbench import __version__
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
