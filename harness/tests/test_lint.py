import json

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
