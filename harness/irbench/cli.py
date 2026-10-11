import typer

from irbench import __version__

app = typer.Typer(no_args_is_help=True, help="IRBench: information retrieval benchmark harness.")


@app.callback()
def main() -> None:
    """IRBench CLI."""


@app.command()
def version() -> None:
    """Print the harness version."""
    typer.echo(__version__)
