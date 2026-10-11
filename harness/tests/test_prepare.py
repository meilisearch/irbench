import json
import shutil

import httpx
import yaml
from typer.testing import CliRunner

from conftest import REPO_ROOT
from irbench.cli import app
from irbench.datasets.fetch import sha256_file
from irbench.io import read_jsonl, read_qrels
from irbench.stages.prepare import prepare

SCENARIO = "ecommerce/product-search"


def make_root(tmp_path, wands_raw):
    """A repo-shaped root whose dataset.yaml points at the mini fixture files."""
    root = tmp_path / "root"
    shutil.copytree(REPO_ROOT / "scenarios", root / "scenarios")
    spec = {
        "id": "wands",
        "version": "fixture",
        "license": "MIT",
        "source": "fixture",
        "files": {
            name: {"url": f"https://fixture/{name}.csv", "sha256": sha256_file(path)}
            for name, path in wands_raw.items()
        },
    }
    (root / "datasets" / "wands").mkdir(parents=True)
    (root / "datasets" / "wands" / "dataset.yaml").write_text(yaml.safe_dump(spec))
    return root


def fixture_client(wands_raw):
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.strip("/").removesuffix(".csv")
        return httpx.Response(200, content=wands_raw[name].read_bytes())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_prepare_writes_all_outputs(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    result = prepare(root, SCENARIO, client=fixture_client(wands_raw))

    assert result.n_docs == 4
    assert result.n_queries == 3
    assert result.skipped_queries == ["2"]
    tasks = list(read_jsonl(result.scenario_dir / "tasks.jsonl"))
    assert len(tasks) == result.n_tasks > 0
    qrels = read_qrels(result.scenario_dir / "qrels.tasks.tsv")
    assert {t["task_id"] for t in tasks} == set(qrels)

    manifest = json.loads((result.scenario_dir / "prepare.manifest.json").read_text())
    assert manifest["scenario"] == SCENARIO
    assert manifest["dataset"]["version"] == "fixture"
    assert manifest["seed"] == 20261009
    assert manifest["counts"]["skipped_queries"] == 1
    assert "created_at" in manifest and "irbench_version" in manifest


def test_prepare_is_deterministic(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    first = prepare(root, SCENARIO, client=fixture_client(wands_raw))
    a = (first.scenario_dir / "tasks.jsonl").read_bytes()
    second = prepare(root, SCENARIO, client=fixture_client(wands_raw))
    assert (second.scenario_dir / "tasks.jsonl").read_bytes() == a


def test_cli_prepare_uses_cached_raw_files(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    raw_dir = root / ".irbench" / "data" / "wands" / "raw"
    raw_dir.mkdir(parents=True)
    for name, path in wands_raw.items():
        shutil.copy(path, raw_dir / f"{name}.csv")  # valid checksum -> no network

    result = CliRunner().invoke(app, ["prepare", "--scenario", SCENARIO, "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "skipped 1 queries without relevant judgments" in result.stdout
