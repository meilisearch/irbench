import shutil

import yaml
from typer.testing import CliRunner

from conftest import REPO_ROOT
from irbench.cli import app
from irbench.paths import Workspace
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


def test_cli_prepare_missing_manual(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(REPO_ROOT / "scenarios", root / "scenarios")
    sha = "0" * 64
    spec = {
        "id": "wands",
        "version": "fixture",
        "license": "x",
        "source": "x",
        "files": {
            "product": {
                "manual": "Fetch the invented archives by hand.",
                "filename": "p.zip",
                "sha256": sha,
            },
            "query": {
                "manual": "Fetch the invented archives by hand.",
                "filename": "q.zip",
                "sha256": sha,
            },
        },
    }
    (root / "datasets" / "wands").mkdir(parents=True)
    (root / "datasets" / "wands" / "dataset.yaml").write_text(yaml.safe_dump(spec))

    result = CliRunner().invoke(
        app, ["prepare", "--scenario", "ecommerce/product-search", "--root", str(root)]
    )
    raw = root / ".irbench" / "data" / "wands" / "raw"
    assert result.exit_code == 1
    assert "Fetch the invented archives by hand." in result.output
    assert str(raw / "p.zip") in result.output and str(raw / "q.zip") in result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def _fixture_root(tmp_path, spec):
    root = tmp_path / "root"
    shutil.copytree(REPO_ROOT / "scenarios", root / "scenarios")
    (root / "datasets" / "wands").mkdir(parents=True)
    (root / "datasets" / "wands" / "dataset.yaml").write_text(yaml.safe_dump(spec))
    return root


def test_cli_prepare_manual_checksum_mismatch(tmp_path):
    spec = {
        "id": "wands",
        "version": "fixture",
        "license": "x",
        "source": "x",
        "files": {
            name: {"manual": "Fetch them by hand.", "filename": f"{name}.zip", "sha256": "0" * 64}
            for name in ("product", "query")
        },
    }
    root = _fixture_root(tmp_path, spec)
    raw = root / ".irbench" / "data" / "wands" / "raw"
    raw.mkdir(parents=True)
    for name in ("product", "query"):
        (raw / f"{name}.zip").write_bytes(b"invented bytes")

    result = CliRunner().invoke(
        app, ["prepare", "--scenario", "ecommerce/product-search", "--root", str(root)]
    )
    assert result.exit_code == 1
    assert str(raw / "product.zip") in result.output
    assert "does not match the pinned" in result.output
    assert "may not be set yet" in result.output and "placeholders" in result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_vectors_pack_refuses_internal(tmp_path):
    spec = {
        "id": "wands",
        "version": "fixture",
        "license": "invented terms",
        "source": "x",
        "distribution": "internal",
        "files": {"product": {"url": "https://example.invalid/p.zip", "sha256": "0" * 64}},
    }
    root = _fixture_root(tmp_path, spec)
    scenario_dir = Workspace(root).scenario_dir("ecommerce/product-search")
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "prepare.manifest.json").write_text("{}")

    result = CliRunner().invoke(
        app,
        ["vectors", "pack", "--scenario", "ecommerce/product-search", "--model", "fake"]
        + ["--root", str(root)],
    )
    assert result.exit_code == 1
    assert "internal-only" in result.output and "pack its vectors" in result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
