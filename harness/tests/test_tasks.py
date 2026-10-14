from conftest import REPO_ROOT
from irbench.filters import matches
from irbench.models import Filter, Query
from irbench.scenario import load_scenario
from irbench.tasks import build_tasks

DOCS = {
    "0": {"id": "0", "product_class": ["Beds"], "average_rating": 4.5, "review_count": 15.0},
    "3": {"id": "3", "product_class": ["Nightstands"], "average_rating": 4.0, "review_count": 8.0},
    "1": {"id": "1", "product_class": ["Sofas", "Sectionals"], "average_rating": None},
    "2": {"id": "2", "product_class": ["Coffee & Cocktail Tables"], "average_rating": 3.0},
}
QRELS = {"0": {"0": 2, "3": 1}, "1": {"1": 2, "2": 0}, "2": {"2": 0}}


def _variants():
    return [
        Query(qid="0", text="platform bed", variant_of="0", perturbation="original"),
        Query(qid="0-typo", text="platfrom bed", variant_of="0", perturbation="typo"),
        Query(qid="1", text="velvet couch", variant_of="1", perturbation="original"),
        Query(qid="2", text="salon chair", variant_of="2", perturbation="original"),
    ]


def _build():
    return build_tasks(
        load_scenario(REPO_ROOT, "ecommerce/product-search"), _variants(), QRELS, DOCS
    )


def test_matches_eq_on_list_and_ranges():
    assert matches(DOCS["1"], Filter(field="product_class", op="eq", value="Sectionals"))
    assert not matches(DOCS["0"], Filter(field="product_class", op="eq", value="Sofas"))
    assert matches(DOCS["0"], Filter(field="average_rating", op="gte", value=4.5))
    assert matches(DOCS["2"], Filter(field="average_rating", op="lte", value=3.0))


def test_matches_missing_value_is_false():
    assert not matches(DOCS["1"], Filter(field="average_rating", op="gte", value=0.0))
    assert not matches(DOCS["1"], Filter(field="average_rating", op="lte", value=5.0))
    assert not matches({"id": "9"}, Filter(field="product_class", op="eq", value="Beds"))


def test_query_without_positives_is_excluded():
    result = _build()
    assert result.skipped_queries == ["2"]
    assert not any(t.variant_of == "2" for t in result.tasks)


def test_task_ids_and_mode_expansion():
    result = _build()
    ids = {t.task_id for t in result.tasks if t.qid == "0"}
    assert {"0:keyword:keyword", "0:semantic:semantic", "0:hybrid:hybrid"} <= ids
    assert {
        "0:filtered:keyword",
        "0:filtered:semantic",
        "0:sorted:keyword",
        "0:faceted:semantic",
    } <= ids
    faceted = next(t for t in result.tasks if t.task_id == "0:faceted:keyword")
    assert faceted.facets == ["product_class"]


def test_filtered_qrels_are_intersection_and_keep_a_positive():
    result = _build()
    for t in result.tasks:
        if t.mode != "filtered":
            continue
        grades = result.qrels[t.task_id]
        assert any(g > 0 for g in grades.values())
        assert all(matches(DOCS[d], t.filter) for d in grades)


def test_variants_share_the_source_filter_and_sort():
    result = _build()
    by_id = {t.task_id: t for t in result.tasks}
    assert by_id["0:filtered:keyword"].filter == by_id["0-typo:filtered:keyword"].filter
    assert by_id["0:sorted:keyword"].sort == by_id["0-typo:sorted:keyword"].sort


def test_unfiltered_modes_keep_source_qrels():
    result = _build()
    assert result.qrels["0-typo:keyword:keyword"] == QRELS["0"]
    assert result.qrels["0:sorted:semantic"] == QRELS["0"]


def test_tasks_carry_the_query_group_and_ungrouped_ones_have_none():
    variants = [
        Query(
            qid="0", text="platform bed", variant_of="0", perturbation="original", group="keyword"
        ),
        Query(
            qid="0-typo", text="platfrom bed", variant_of="0", perturbation="typo", group="keyword"
        ),
        Query(qid="1", text="velvet couch", variant_of="1", perturbation="original"),
    ]
    result = build_tasks(
        load_scenario(REPO_ROOT, "ecommerce/product-search"), variants, QRELS, DOCS
    )
    grouped = [t for t in result.tasks if t.variant_of == "0"]
    plain = [t for t in result.tasks if t.variant_of == "1"]
    assert grouped and plain
    assert {t.group for t in grouped} == {"keyword"}
    assert all(t.group is None for t in plain)
    assert "group" not in plain[0].model_dump(mode="json", exclude_none=True)
