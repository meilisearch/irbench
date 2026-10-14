import shutil

import pytest
import yaml

from conftest import REPO_ROOT
from irbench.scenario import load_scenario
from irbench.terms import load_term_groups, scenario_term_groups

SCENARIO = "ecommerce/product-search"


def test_load_product_search_scenario():
    s = load_scenario(REPO_ROOT, SCENARIO)
    assert s.dataset == "wands"
    assert s.modes == ["keyword", "semantic", "hybrid", "filtered", "sorted", "faceted"]
    assert s.index_fields.searchable[0] == "name"
    assert s.vocabulary_path.exists()
    assert s.synonyms_path.exists()


def test_term_groups_parse(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("# comment\nSofa, couch , settee\n\ntv stand, media console\n")
    assert load_term_groups(path) == [["sofa", "couch", "settee"], ["tv stand", "media console"]]


def test_term_in_two_groups_is_rejected(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("sofa, couch\ncouch, settee\n")
    with pytest.raises(ValueError, match="couch"):
        load_term_groups(path)


def test_vocab_mismatch_and_engine_synonyms_are_disjoint():
    # The synonyms file engines may load must not be the answer key for the
    # synonym perturbation, otherwise that perturbation measures nothing.
    s = load_scenario(REPO_ROOT, SCENARIO)
    vocab = {t for g in load_term_groups(s.vocabulary_path) for t in g}
    engine = {t for g in load_term_groups(s.synonyms_path) for t in g}
    assert vocab.isdisjoint(engine), vocab & engine


def test_filter_template_must_target_filterable_field(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "scenarios" / SCENARIO / "scenario.yaml").read_text())
    raw["filter_templates"].append({"field": "description", "op": "eq"})
    target = tmp_path / "scenarios" / SCENARIO
    target.mkdir(parents=True)
    (target / "scenario.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="description"):
        load_scenario(tmp_path, SCENARIO)


def _scenario_with(tmp_path, mutate):
    raw = yaml.safe_load((REPO_ROOT / "scenarios" / SCENARIO / "scenario.yaml").read_text())
    mutate(raw)
    target = tmp_path / "scenarios" / SCENARIO
    target.mkdir(parents=True)
    (target / "scenario.yaml").write_text(yaml.safe_dump(raw))
    return lambda: load_scenario(tmp_path, SCENARIO)


def test_scenario_declares_numeric_fields_and_facet_limits():
    s = load_scenario(REPO_ROOT, SCENARIO)
    assert s.index_fields.numeric == ["average_rating", "review_count"]
    assert s.max_facet_values >= 601  # WANDS has 601 distinct product_class values
    assert s.facet_verify_limit == 1000
    assert s.request_timeout_s == 10


def test_range_filter_on_non_numeric_field_is_rejected(tmp_path):
    load = _scenario_with(
        tmp_path, lambda raw: raw["index_fields"].update(numeric=["review_count"])
    )
    with pytest.raises(ValueError, match="average_rating"):
        load()


def test_sortable_field_must_be_numeric(tmp_path):
    def mutate(raw):
        raw["index_fields"]["sortable"].append("product_class")

    with pytest.raises(ValueError, match="product_class"):
        _scenario_with(tmp_path, mutate)()


def test_product_search_measures_speed_at_20_and_100():
    s = load_scenario(REPO_ROOT, SCENARIO)
    assert s.page_sizes == [20, 100]
    assert s.top_k == 100


def test_page_sizes_default_to_top_k(tmp_path):
    def drop(raw):
        raw["perf"].pop("page_sizes", None)

    assert _scenario_with(tmp_path, drop)().page_sizes == [100]


@pytest.mark.parametrize(
    "sizes, message",
    [
        ([], "non-empty"),
        ([20, 20], "strictly ascending"),
        ([100, 20], "strictly ascending"),
        ([0, 20], "between 1 and top_k"),
        ([20, 101], "between 1 and top_k"),
    ],
)
def test_page_sizes_are_validated(tmp_path, sizes, message):
    def set_sizes(raw):
        raw["perf"]["page_sizes"] = sizes

    with pytest.raises(ValueError, match=message):
        _scenario_with(tmp_path, set_sizes)()


def copy_scenario(tmp_path, edit):
    src = REPO_ROOT / "scenarios" / "ecommerce" / "product-search"
    dst = tmp_path / "scenarios" / "ecommerce" / "product-search"
    shutil.copytree(src, dst)
    path = dst / "scenario.yaml"
    path.write_text(yaml.safe_dump(edit(yaml.safe_load(path.read_text(encoding="utf-8")))))
    return tmp_path


def _no_synonyms(raw):
    raw.pop("synonyms")
    raw["perturbations"] = {"seed": 1, "kinds": ["typo", "prefix"]}
    return raw


def test_scenario_without_synonyms_or_vocabulary(tmp_path):
    root = copy_scenario(tmp_path, _no_synonyms)
    s = load_scenario(root, SCENARIO)
    assert s.synonyms_path is None and s.vocabulary_path is None
    assert scenario_term_groups(s.synonyms_path) == []


def test_synonym_kind_requires_a_vocabulary(tmp_path):
    def edit(raw):
        raw["perturbations"] = {"seed": 1, "kinds": ["typo", "synonym"]}
        return raw

    with pytest.raises(ValueError, match="vocabulary"):
        load_scenario(copy_scenario(tmp_path, edit), SCENARIO)
