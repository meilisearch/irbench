import json
import shutil

import pytest
import yaml
from typer.testing import CliRunner

from conftest import REPO_ROOT
from irbench.cli import app
from irbench.lint import QuerySet, lint_repo

SCENARIO = "ecommerce/product-search"
QUERIES = QuerySet.of(
    ["salon chair", "sofa", "smart coffee table", "salon cair"],
    ["0", "0-typo", "0-typo:keyword:keyword"],
)


def make_repo(tmp_path, config, image="getmeili/meilisearch:v1.54.3", skip=()):
    engine = tmp_path / "engines" / "eng"
    (engine / "configs" / "ecommerce").mkdir(parents=True)
    (engine / "docker").mkdir()
    files = {
        "adapter.py": "",
        "README.md": "# eng\n",
        "docker/compose.yaml": yaml.safe_dump({"services": {"engine": {"image": image}}}),
    }
    for name, text in files.items():
        if name not in skip:
            (engine / name).write_text(text)
    (engine / "configs" / "ecommerce" / "product-search.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def lint(root):
    return lint_repo(root, queries=lambda scenario_id: QUERIES)


def rules(findings):
    return sorted(f.rule for f in findings)


def test_clean_config_has_no_findings(tmp_path):
    n, findings = lint(make_repo(tmp_path, {"default": {"index": {}, "search": {}}}))
    assert (n, findings) == (1, [])


def test_setting_without_justification_is_flagged(tmp_path):
    _, findings = lint(make_repo(tmp_path, {"default": {"index": {"ef": 64}}}))
    assert rules(findings) == ["unjustified-setting"]
    assert "index.ef" in findings[0].message and "[default]" in findings[0].message
    assert findings[0].path == "engines/eng/configs/ecommerce/product-search.yaml"


@pytest.mark.parametrize("key", ["ef", "index.ef", "index", "hnsw"])
def test_justification_by_key_path_or_prefix(tmp_path, key):
    index = {"hnsw": {"ef": 64}} if key == "hnsw" else {"ef": 64}
    config = {"default": {"index": index, "justification": {key: "documented default for v1"}}}
    assert lint(make_repo(tmp_path, config))[1] == []


def test_search_settings_need_justification(tmp_path):
    config = {"default": {"search": {"semantic": {"ef": 128}}, "justification": {"ef": "x"}}}
    assert lint(make_repo(tmp_path, config))[1] == []
    config = {"default": {"search": {"semantic": {"ef": 128}}}}
    _, findings = lint(make_repo(tmp_path / "b", config))
    assert "search.semantic.ef" in findings[0].message


def test_tuned_track_is_checked_after_merge(tmp_path):
    config = {"default": {}, "tuned": {"index": {"m": 32}}}
    _, findings = lint(make_repo(tmp_path, config))
    assert rules(findings) == ["unjustified-setting"]
    assert "[tuned]" in findings[0].message


def test_blank_justification_does_not_count(tmp_path):
    config = {"default": {"index": {"ef": 64}, "justification": {"ef": "  "}}}
    assert rules(lint(make_repo(tmp_path, config))[1]) == ["unjustified-setting"]


@pytest.mark.parametrize(
    "key,ok",
    [
        ("hybrid", True),
        ("faceted:semantic", True),
        ("*:keyword", True),
        ("semantik", False),
        ("filtered:filtered", False),
        ("*", False),
        ("keyword:", False),
    ],
)
def test_unsupported_keys_must_name_modes(tmp_path, key, ok):
    _, findings = lint(make_repo(tmp_path, {"default": {"unsupported": {key: "reason"}}}))
    assert rules(findings) == ([] if ok else ["unsupported-key"])


@pytest.mark.parametrize(
    "value",
    [
        "salon chair",  # a query text
        "Salon  Chair",  # same after normalization
        "best salon chair deals",  # a multi-word query inside a longer string
        "0-typo",  # a variant id
        "0-typo:keyword:keyword",  # a task id
    ],
)
def test_query_text_and_ids_are_rejected(tmp_path, value):
    config = {
        "default": {"search": {"keyword": {"boost": [value]}}, "justification": {"boost": "x"}}
    }
    _, findings = lint(make_repo(tmp_path, config))
    assert rules(findings) == ["query-leak"]
    assert "search.keyword.boost[0]" in findings[0].message


def test_query_text_as_a_key_is_rejected(tmp_path):
    config = {
        "default": {"index": {"synonyms": {"sofa": ["couch"]}}, "justification": {"synonyms": "x"}}
    }
    assert rules(lint(make_repo(tmp_path, config))[1]) == ["query-leak"]


@pytest.mark.parametrize("value", ["salon", "0", "chairs"])
def test_ordinary_values_are_not_leaks(tmp_path, value):
    # A word that is only part of a query, and purely numeric ids, are ordinary settings.
    config = {"default": {"search": {"keyword": {"x": value}}, "justification": {"x": "x"}}}
    assert lint(make_repo(tmp_path, config))[1] == []


def test_prose_fields_are_not_checked_for_leaks(tmp_path):
    config = {
        "default": {
            "justification": {"x": "helps salon chair queries"},
            "unsupported": {"hybrid": "salon chair"},
        }
    }
    assert lint(make_repo(tmp_path, config))[1] == []


@pytest.mark.parametrize("image", ["getmeili/meilisearch", "getmeili/meilisearch:latest", "", None])
def test_engine_image_must_pin_a_version(tmp_path, image):
    _, findings = lint(make_repo(tmp_path, {"default": {}}, image=image))
    assert rules(findings) == ["image-pin"]


def test_image_with_digest_is_accepted(tmp_path):
    image = "qdrant/qdrant:v1.19.2@sha256:" + "a" * 64
    assert lint(make_repo(tmp_path, {"default": {}}, image=image))[1] == []


@pytest.mark.parametrize(
    "compose_text",
    [
        "services:\n",  # services is null
        "services: [engine]\n",  # services is a list
        "services:\n  engine: nginx\n",  # services.engine is a string
        "- a\n",  # the whole document is a list
        "services: [\n",  # invalid YAML
    ],
)
def test_malformed_compose_is_a_finding_not_a_crash(tmp_path, compose_text):
    root = make_repo(tmp_path, {"default": {}})
    (root / "engines" / "eng" / "docker" / "compose.yaml").write_text(compose_text)
    assert rules(lint(root)[1]) == ["image-pin"]


def test_missing_engine_files_are_flagged(tmp_path):
    _, findings = lint(make_repo(tmp_path, {"default": {}}, skip=("README.md",)))
    assert rules(findings) == ["engine-files"]
    assert findings[0].path == "engines/eng/README.md"


def test_unloadable_config_is_a_finding_not_a_crash(tmp_path):
    root = make_repo(tmp_path, {"default": {}})
    (root / "engines/eng/configs/ecommerce/product-search.yaml").write_text("default: [unclosed\n")
    assert rules(lint(root)[1]) == ["config-load"]


def test_unprepared_scenario_is_a_finding(tmp_path):
    root = make_repo(tmp_path, {"default": {}})
    _, findings = lint_repo(root)  # no injected queries: reads .irbench/, which is empty
    assert rules(findings) == ["not-prepared"]
    assert "irbench prepare --scenario ecommerce/product-search" in findings[0].message


def test_repository_configs_pass_the_static_rules():
    n, findings = lint_repo(REPO_ROOT, queries=lambda s: QuerySet.of([], []))
    assert n >= 3 and findings == [], [str(f) for f in findings]


def _write_prepared(root, rows, task_ids):
    d = root / ".irbench" / "scenarios" / "ecommerce__product-search"
    d.mkdir(parents=True)
    (d / "queries.variants.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "tasks.jsonl").write_text("".join(json.dumps({"task_id": t}) + "\n" for t in task_ids))


def test_cli_exit_codes(tmp_path):
    root = make_repo(
        tmp_path,
        {"default": {"search": {"keyword": {"q": "salon chair"}}, "justification": {"q": "x"}}},
    )
    _write_prepared(root, [{"qid": "0", "text": "salon chair"}], ["0:keyword:keyword"])
    bad = CliRunner().invoke(app, ["lint", "--root", str(root)])
    assert bad.exit_code == 1
    assert "[query-leak]" in bad.output and "1 configs, 1 findings" in bad.output

    (root / "engines/eng/configs/ecommerce/product-search.yaml").write_text(
        yaml.safe_dump({"default": {}})
    )
    good = CliRunner().invoke(app, ["lint", "--root", str(root)])
    assert good.exit_code == 0 and "1 configs, 0 findings" in good.output


def test_synonyms_reason_is_prose_not_a_setting(tmp_path):
    config = {"default": {"synonyms": "the engine cannot load synonyms.txt"}}
    assert lint(make_repo(tmp_path, config))[1] == []


def make_internal(root):
    """Marks the scenario's dataset internal in a copy of the repository's own spec."""
    shutil.copytree(
        REPO_ROOT / "scenarios" / "ecommerce", root / "scenarios" / "ecommerce", dirs_exist_ok=True
    )
    spec = yaml.safe_load((REPO_ROOT / "datasets/wands/dataset.yaml").read_text())
    (root / "datasets" / "wands").mkdir(parents=True)
    (root / "datasets/wands/dataset.yaml").write_text(
        yaml.safe_dump({**spec, "distribution": "internal"})
    )
    return root


def write_manifest(root, name, manifest):
    d = root / "results" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_internal_data_results(tmp_path):
    root = make_internal(make_repo(tmp_path, {"default": {}}))
    # Flagged by the manifest alone, and by the spec for a manifest without `distribution`.
    write_manifest(
        root, "a", {"scenario": "ecommerce/gone", "dataset": {"distribution": "internal"}}
    )
    write_manifest(root, "b", {"scenario": SCENARIO, "dataset": {"id": "wands"}})
    write_manifest(root, "c", {"scenario": "ecommerce/gone", "dataset": {"id": "wands"}})
    _, findings = lint(root)
    flagged = [f for f in findings if f.rule == "internal-data"]
    assert [(f.path, f.severity) for f in flagged] == [
        ("results/a/manifest.json", "error"),
        ("results/b/manifest.json", "error"),
    ]


def test_internal_data_published_vectors(tmp_path):
    root = make_internal(make_repo(tmp_path, {"default": {}}))
    asset = {"url": "https://example.test/x", "sha256": "0" * 64, "count": 1, "dim": 2}
    entry = {"model": "m", "dataset": "wands", "dataset_version": "v", "seed": 1}
    rows = [
        {"scenario": SCENARIO, **entry, "docs": asset, "queries": asset},
        {"scenario": "knowledge-base/other", **entry, "docs": asset, "queries": asset},
    ]
    (root / "embeddings").mkdir()
    (root / "embeddings/published.yaml").write_text(yaml.safe_dump({"vectors": rows}))
    _, findings = lint(root)
    flagged = [f for f in findings if f.rule == "internal-data"]
    assert [(f.path, f.severity) for f in flagged] == [("embeddings/published.yaml", "error")]
    assert SCENARIO in flagged[0].message


def test_internal_data_docs_nav(tmp_path):
    root = make_internal(make_repo(tmp_path, {"default": {}}))
    nav = {
        "navigation": {"groups": [{"group": "S", "pages": ["scenarios/product-search", "index"]}]}
    }
    (root / "docs").mkdir()
    (root / "docs/docs.json").write_text(json.dumps(nav))
    _, findings = lint(root)
    flagged = [f for f in findings if f.rule == "internal-data"]
    assert [(f.path, f.severity) for f in flagged] == [("docs/docs.json", "error")]


def test_public_scenario_is_not_internal_data(tmp_path):
    root = make_repo(tmp_path, {"default": {}})
    shutil.copytree(REPO_ROOT / "scenarios", root / "scenarios")
    shutil.copytree(REPO_ROOT / "datasets", root / "datasets")
    write_manifest(root, "a", {"scenario": SCENARIO, "dataset": {"distribution": "public"}})
    (root / "docs").mkdir()
    (root / "docs/docs.json").write_text(
        json.dumps({"navigation": {"pages": ["scenarios/product-search"]}})
    )
    assert lint(root)[1] == []


def test_not_prepared_internal_is_notice(tmp_path):
    root = make_repo(tmp_path, {"default": {}})
    _, findings = lint_repo(root)  # public (no spec in this root): stays an error
    assert [(f.rule, f.severity) for f in findings] == [("not-prepared", "error")]

    make_internal(root)
    _, findings = lint_repo(root)
    assert [(f.rule, f.severity) for f in findings] == [("not-prepared", "notice")]
    assert str(findings[0]).startswith("notice: engines/eng/configs/")


def test_cli_lint_notices_exit_0(tmp_path):
    root = make_internal(make_repo(tmp_path, {"default": {}}))
    notice_only = CliRunner().invoke(app, ["lint", "--root", str(root)])
    assert notice_only.exit_code == 0
    assert "notice: " in notice_only.output and "[not-prepared]" in notice_only.output
    assert "1 configs, 0 findings, 1 notices" in notice_only.output

    write_manifest(root, "a", {"scenario": SCENARIO, "dataset": {"distribution": "internal"}})
    failed = CliRunner().invoke(app, ["lint", "--root", str(root)])
    assert failed.exit_code == 1
    assert "[internal-data]" in failed.output and "1 findings" in failed.output
