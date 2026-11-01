from typer.testing import CliRunner

from irbench.cli import app
from report_fixtures import write_bundle


def test_version_command_prints_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_cli_report_internal_flag(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    write_bundle(
        results, "meilisearch", scenario="ecommerce/internal-search", distribution="internal"
    )
    out = tmp_path / "site"
    refused = CliRunner().invoke(app, ["report", "--results", str(results), "--out", str(out)])
    assert refused.exit_code == 1
    assert "--internal" in refused.output
    assert not out.exists()
    allowed = CliRunner().invoke(
        app, ["report", "--results", str(results), "--out", str(out), "--internal"]
    )
    assert allowed.exit_code == 0
    assert (out / "index.html").exists()
