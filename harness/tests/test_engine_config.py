import pytest
import yaml

from irbench.engines.config import (
    config_diff,
    deep_merge,
    load_engine_config,
)
from irbench.models import QueryTask

SCENARIO = "ecommerce/product-search"


def _write(root, engine, raw):
    path = root / "engines" / engine / "configs" / "ecommerce" / "product-search.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw))
    return path


def _task(mode, base):
    return QueryTask(
        task_id=f"1:{mode}:{base}",
        qid="1",
        variant_of="1",
        perturbation="original",
        text="sofa",
        mode=mode,
        base=base,
    )


def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"x": 1, "y": 2}, "b": [1]}
    merged = deep_merge(base, {"a": {"y": 3}, "b": [2]})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [2]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1]}


def test_config_diff_lists_changed_paths():
    assert config_diff({"a": {"x": 1}, "b": 2}, {"a": {"x": 5}, "b": 2, "c": 3}) == {
        "a.x": {"default": 1, "effective": 5},
        "c": {"default": None, "effective": 3},
    }


def test_tuned_track_merges_over_default(tmp_path):
    _write(
        tmp_path,
        "e",
        {
            "default": {"index": {"k": 1, "keep": True}, "search": {"keyword": {"x": 1}}},
            "tuned": {"index": {"k": 2}},
        },
    )
    cfg = load_engine_config(tmp_path, "e", SCENARIO)
    assert cfg.track("tuned").index == {"k": 2, "keep": True}
    assert cfg.track("default").search == {"keyword": {"x": 1}}
    assert cfg.diff("default") == {}
    assert cfg.diff("tuned") == {"index.k": {"default": 1, "effective": 2}}
    assert cfg.hash("default") != cfg.hash("tuned")
    assert len(cfg.hash("default")) == 64


def test_missing_tuned_track_is_an_error(tmp_path):
    _write(tmp_path, "e", {"default": {}})
    with pytest.raises(ValueError, match="tuned"):
        load_engine_config(tmp_path, "e", SCENARIO).track("tuned")


def test_unknown_keys_are_rejected(tmp_path):
    _write(tmp_path, "e", {"default": {"serch": {}}})
    with pytest.raises(ValueError, match="serch"):
        load_engine_config(tmp_path, "e", SCENARIO)


def test_search_params_keyed_by_base_mode_only(tmp_path):
    _write(tmp_path, "e", {"default": {"search": {"filtered": {}}}})
    with pytest.raises(ValueError, match="filtered"):
        load_engine_config(tmp_path, "e", SCENARIO)


def test_missing_config_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="product-search.yaml"):
        load_engine_config(tmp_path, "e", SCENARIO)


def test_unsupported_reason_precedence(tmp_path):
    _write(
        tmp_path,
        "e",
        {
            "default": {
                "unsupported": {
                    "faceted:semantic": "specific",
                    "hybrid": "whole mode",
                    "*:keyword": "whole base",
                }
            }
        },
    )
    track = load_engine_config(tmp_path, "e", SCENARIO).track("default")
    assert track.unsupported_reason(_task("faceted", "semantic")) == "specific"
    assert track.unsupported_reason(_task("hybrid", "hybrid")) == "whole mode"
    assert track.unsupported_reason(_task("sorted", "keyword")) == "whole base"
    assert track.unsupported_reason(_task("sorted", "semantic")) is None
