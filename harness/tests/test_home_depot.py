"""The `home-depot` loader. Every string below is invented: the dataset is internal-only and
the repository is public, so no real query, product or attribute text appears here."""

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT
from irbench.datasets import NORMALIZERS, home_depot
from irbench.datasets.fetch import ManualFileMissing, sha256_file
from irbench.datasets.home_depot import grade, load_spelling, normalize_home_depot
from irbench.datasets.registry import fetch_dataset, load_dataset_spec
from irbench.io import read_jsonl, read_qrels
from irbench.stages.prepare import prepare

SCENARIO = "ecommerce/home-depot-search"
TRAIN_HEADER = ["id", "product_uid", "product_title", "search_term", "relevance"]
DESC_HEADER = ["product_uid", "product_description"]
ATTR_HEADER = ["product_uid", "name", "value"]


def _zip_csv(path: Path, member: str, header: list[str], rows: list[list[str]]) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, buffer.getvalue().encode("latin-1"))
    return path


def make_raw(
    tmp_path: Path,
    train: list[list[str]],
    descriptions: list[list[str]] | None = None,
    attributes: list[list[str]] | None = None,
    spelling: str = "SPELLING = {}\n",
) -> dict[str, Path]:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    spelling_path = raw / "spelling.py"
    spelling_path.write_text(spelling, encoding="utf-8")
    return {
        "train": _zip_csv(raw / "train.csv.zip", "train.csv", TRAIN_HEADER, train),
        "descriptions": _zip_csv(
            raw / "product_descriptions.csv.zip",
            "product_descriptions.csv",
            DESC_HEADER,
            descriptions or [],
        ),
        "attributes": _zip_csv(
            raw / "attributes.csv.zip", "attributes.csv", ATTR_HEADER, attributes or []
        ),
        "spelling": spelling_path,
    }


def run(tmp_path: Path, raw: dict[str, Path]):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    result = normalize_home_depot(raw, out)
    docs = {d["id"]: d for d in read_jsonl(result.docs)}
    queries = {q["qid"]: q for q in read_jsonl(result.queries)}
    return result, docs, queries


@pytest.mark.parametrize(
    ("average", "expected"),
    [
        (1.0, 0),
        (1.33, 0),
        (1.5, 1),
        (1.67, 1),
        (2.33, 1),
        (2.5, 2),
        (2.67, 2),
        (3.0, 2),
    ],
)
def test_grades_boundaries(average, expected):
    assert grade(average) == expected


def test_registered_under_the_dataset_id():
    assert NORMALIZERS["home-depot"] is normalize_home_depot


def test_loader_constants():
    assert home_depot.SEED == 20261031
    assert home_depot.SAMPLE_SIZE == 4000
    assert home_depot.BRAND_ATTRIBUTE == "MFG Brand Name"


def test_query_without_positive_dropped(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["1", "10", "Zorbex widget", "kelvar plate", "1.33"],
            ["2", "11", "Zorbex gadget", "kelvar plate", "1.0"],
            ["3", "10", "Zorbex widget", "quillon rod", "2.67"],
        ],
    )
    result, docs, queries = run(tmp_path, raw)
    assert set(queries) == {"hd-3"}
    assert result.dropped["queries_no_positive"] == 1
    assert result.n_queries == 1


def test_qid_is_min_row_id(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["7", "10", "Zorbex widget", "quillon rod", "2.0"],
            ["3", "11", "Zorbex gadget", "quillon rod", "3.0"],
            ["12", "12", "Zorbex gizmo", "quillon rod", "1.67"],
        ],
    )
    _, _, queries = run(tmp_path, raw)
    assert set(queries) == {"hd-3"}


def test_title_from_smallest_row(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["5", "10", "  Second title ", "quillon rod", "2.0"],
            ["2", "10", " First title ", "kelvar plate", "2.0"],
        ],
    )
    _, docs, _ = run(tmp_path, raw)
    assert docs["10"]["title"] == "First title"


def test_brand_only_from_mfg_brand_name(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["1", "10", "Zorbex widget", "quillon rod", "2.0"],
            ["2", "11", "Zorbex gadget", "quillon rod", "2.0"],
            ["3", "12", "Zorbex gizmo", "quillon rod", "2.0"],
        ],
        attributes=[
            ["10", "Color Family", "Teal"],
            ["11", "MFG Brand Name", " Vantro "],
            ["11", "MFG Brand Name", "Ostrel"],
            ["12", "Brand Name", "Not this one"],
            ["99", "MFG Brand Name", "Never judged"],
        ],
    )
    _, docs, _ = run(tmp_path, raw)
    assert docs["10"]["brand"] is None
    assert docs["11"]["brand"] == "Vantro"
    assert docs["12"]["brand"] is None
    assert "99" not in docs


def test_attribute_rows_with_blank_or_float_ids(tmp_path):
    raw = make_raw(
        tmp_path,
        [["1", "10", "Zorbex widget", "quillon rod", "2.0"]],
        attributes=[
            ["", "MFG Brand Name", "Blank id"],
            ["10.0", "MFG Brand Name", "Vantro"],
        ],
    )
    _, docs, _ = run(tmp_path, raw)
    assert docs["10"]["brand"] == "Vantro"


def test_docs_row_shape(tmp_path):
    raw = make_raw(tmp_path, [["1", "10", "Zorbex widget", "quillon rod", "2.0"]])
    _, docs, _ = run(tmp_path, raw)
    assert docs["10"] == {"id": "10", "title": "Zorbex widget", "brand": None, "description": None}


def test_description_entities_unescaped(tmp_path):
    raw = make_raw(
        tmp_path,
        [["1", "10", "Zorbex widget", "quillon rod", "2.0"]],
        descriptions=[["10", "Alpha &amp; beta\n  gamma \t  delta &lt;eps&gt;"]],
    )
    _, docs, _ = run(tmp_path, raw)
    assert docs["10"]["description"] == "Alpha & beta gamma delta <eps>"


def test_spelling_match_raw_then_normalise(tmp_path):
    raw = make_raw(
        tmp_path,
        [["1", "10", "Zorbex widget", "zorp  brackets", "2.0"]],
        spelling='SPELLING = {"zorp  brackets": "zorb   brackets"}\n',
    )
    _, _, queries = run(tmp_path, raw)
    assert queries["hd-1"] == {
        "qid": "hd-1",
        "text": "zorp brackets",
        "group": "misspelled",
        "corrected": "zorb brackets",
    }


def test_single_spaced_key_does_not_match_a_double_spaced_query(tmp_path):
    raw = make_raw(
        tmp_path,
        [["1", "10", "Zorbex widget", "zorp  brackets", "2.0"]],
        spelling='SPELLING = {"zorp brackets": "zorb brackets"}\n',
    )
    _, _, queries = run(tmp_path, raw)
    assert queries["hd-1"]["group"] == "clean"


def test_clean_query_has_no_corrected(tmp_path):
    raw = make_raw(
        tmp_path,
        [["1", "10", "Zorbex widget", "quillon rod", "2.0"]],
        spelling='SPELLING = {"other term": "another term"}\n',
    )
    _, _, queries = run(tmp_path, raw)
    assert queries["hd-1"] == {"qid": "hd-1", "text": "quillon rod", "group": "clean"}


@pytest.mark.parametrize("correction", ["", "   ", "\t", "quillon  rod", "Quillon Rod"])
def test_correction_that_changes_nothing_is_clean(tmp_path, correction):
    """A blank correction would make an empty-text variant; one equal to the query once
    whitespace and case are normalised is no misspelling."""
    raw = make_raw(
        tmp_path,
        [["1", "10", "Zorbex widget", "quillon  rod", "2.0"]],
        spelling=f"SPELLING = {{'quillon  rod': {correction!r}}}\n",
    )
    _, _, queries = run(tmp_path, raw)
    assert queries["hd-1"] == {"qid": "hd-1", "text": "quillon rod", "group": "clean"}


def test_spelling_parsed_not_executed(tmp_path):
    sentinel = tmp_path / "sentinel"
    path = tmp_path / "evil.py"
    path.write_text(
        f'import os\nSPELLING = {{"a": open({str(sentinel)!r}, "w").write("x") and "b"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_spelling(path)
    assert not sentinel.exists()

    path.write_text('SPELLING = {"a": __import__("os").getcwd()}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_spelling(path)


@pytest.mark.parametrize(
    "source",
    [
        "SPELLING = {'a': f()}\n",
        "SPELLING = {'a': 1}\n",
        "SPELLING = {1: 'a'}\n",
        "SPELLING = {'a': 'b' + 'c'}\n",
        "SPELLING = {**OTHER}\n",
        "SPELLING = ['a', 'b']\n",
        "SPELLING = {'a': 'b'}\nOTHER = {'c': 'd'}\n",
        "SPELLING = OTHER = {'a': 'b'}\n",
        "x, y = {'a': 'b'}, {}\n",
        "import os\nSPELLING = {'a': 'b'}\n",
        "print('hi')\nSPELLING = {'a': 'b'}\n",
        "SPELLING = {'a': 'b'}\nprint('hi')\n",
        "",
        "SPELLING = {'a': \n",
    ],
)
def test_spelling_anything_but_one_string_dict_is_refused(tmp_path, source):
    path = tmp_path / "spelling.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        load_spelling(path)


def test_spelling_reads_a_dict_literal(tmp_path):
    path = tmp_path / "spelling.py"
    path.write_text(
        '# -*- coding: utf-8 -*-\n"""notes"""\nSPELLING = {"a  b": "c d", "é": "e"}\n',
        encoding="utf-8",
    )
    assert load_spelling(path) == {"a  b": "c d", "é": "e"}


def test_latin1_decoded(tmp_path):
    raw = make_raw(tmp_path, [["1", "10", "Zorbex café widget", "quillon rod", "2.0"]])
    _, docs, _ = run(tmp_path, raw)
    assert docs["10"]["title"] == "Zorbex café widget"


def test_sample_stable(tmp_path, monkeypatch):
    train = [[str(i), str(100 + i), f"Product {i}", f"term number {i}", "2.0"] for i in range(1, 7)]
    raw = make_raw(tmp_path, train)
    monkeypatch.setattr(home_depot, "SAMPLE_SIZE", 3)
    _, _, first = run(tmp_path, raw)
    _, _, second = run(tmp_path, raw)
    assert list(first) == list(second)
    assert len(first) == 3
    ids = [int(q.removeprefix("hd-")) for q in first]
    assert ids == sorted(ids)

    monkeypatch.setattr(home_depot, "SAMPLE_SIZE", 6)
    _, _, kept = run(tmp_path, raw)
    assert len(kept) == 6


def test_all_usable_queries_kept_below_the_sample_size(tmp_path, monkeypatch):
    train = [[str(i), str(100 + i), f"Product {i}", f"term number {i}", "2.0"] for i in range(1, 4)]
    raw = make_raw(tmp_path, train)
    monkeypatch.setattr(home_depot, "SAMPLE_SIZE", 3)
    _, _, queries = run(tmp_path, raw)
    assert set(queries) == {"hd-1", "hd-2", "hd-3"}


def test_queries_sorted_by_numeric_qid(tmp_path):
    train = [
        ["10", "100", "Product a", "term ten", "2.0"],
        ["9", "101", "Product b", "term nine", "2.0"],
        ["100", "102", "Product c", "term hundred", "2.0"],
    ]
    _, _, queries = run(tmp_path, make_raw(tmp_path, train))
    assert list(queries) == ["hd-9", "hd-10", "hd-100"]


def test_corpus_is_judged_products(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["1", "10", "Zorbex widget", "quillon rod", "2.67"],
            ["2", "11", "Zorbex gadget", "quillon rod", "1.0"],
            ["3", "12", "Zorbex gizmo", "kelvar plate", "1.33"],
        ],
        descriptions=[["10", "judged"], ["11", "judged"], ["12", "judged"], ["77", "not judged"]],
    )
    result, docs, queries = run(tmp_path, raw)
    assert set(docs) == {"10", "11", "12"}  # 12 only judged for a dropped query: a distractor
    assert result.n_docs == 3
    qrels = read_qrels(result.qrels)
    assert qrels == {"hd-1": {"10": 2, "11": 0}}


def test_product_without_title_is_dropped_with_its_judgments(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["1", "10", "Zorbex widget", "quillon rod", "2.0"],
            ["2", "11", "   ", "quillon rod", "3.0"],
        ],
    )
    result, docs, _ = run(tmp_path, raw)
    assert set(docs) == {"10"}
    assert result.dropped["docs"] == 1
    assert result.dropped["judgments"] == 1
    assert read_qrels(result.qrels) == {"hd-1": {"10": 1}}


def test_qrels_use_the_grade_cut_points(tmp_path):
    raw = make_raw(
        tmp_path,
        [
            ["1", "10", "a", "quillon rod", "1.33"],
            ["2", "11", "b", "quillon rod", "1.67"],
            ["3", "12", "c", "quillon rod", "2.33"],
            ["4", "13", "d", "quillon rod", "2.67"],
            ["5", "14", "e", "quillon rod", "3.0"],
        ],
    )
    result, _, _ = run(tmp_path, raw)
    assert read_qrels(result.qrels) == {"hd-1": {"10": 0, "11": 1, "12": 1, "13": 2, "14": 2}}


# --- prepare end to end, on fixture zips (no network, no Kaggle) ---------------------------

SCENARIO_YAML = """
id: ecommerce/home-depot-search
dataset: home-depot
index_fields:
  searchable: [title, brand, description]
  filterable: [brand]
  sortable: []
  facets: [brand]
modes: [keyword, semantic, hybrid, filtered, faceted]
structured_bases: [keyword, semantic]
embedding_fields: [title, brand, description]
filter_templates:
  - {field: brand, op: eq}
sort_templates: []
top_k: 100
index_timeout_s: 1800
perturbations:
  seed: 20261031
  kinds: [typo, prefix, corrected]
perf:
  page_sizes: [20, 100]
"""


def make_home_depot_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    scenario_dir = root / "scenarios" / SCENARIO
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(SCENARIO_YAML, encoding="utf-8")

    raw_dir = root / ".irbench" / "data" / "home-depot" / "raw"
    raw_dir.mkdir(parents=True)
    # Queries: "zorp  brackets" is in the dictionary; "kelvar plate" and "vantro hooks" are not.
    # "vantro hooks" only has relevant products without a brand: no filtered task for it.
    train = [
        ["1", "10", "Zorbex bracket", "zorp  brackets", "2.67"],
        ["2", "11", "Zorbex bracket pro", "zorp  brackets", "2.33"],
        ["3", "10", "Zorbex bracket", "kelvar plate", "3.0"],
        ["4", "12", "Ostrel plate", "kelvar plate", "2.0"],
        ["5", "13", "Plain hook", "vantro hooks", "3.0"],
        ["6", "14", "Plain hook long", "vantro hooks", "2.5"],
    ]
    descriptions = [[str(uid), f"Description for product {uid}"] for uid in (10, 11, 12, 13, 14)]
    attributes = [
        ["10", "MFG Brand Name", "Vantro"],
        ["11", "MFG Brand Name", "Vantro"],
        ["12", "MFG Brand Name", "Ostrel"],
        ["13", "Color Family", "Teal"],
    ]
    files = make_raw(
        raw_dir.parent,
        train,
        descriptions,
        attributes,
        spelling='SPELLING = {"zorp  brackets": "zorb brackets"}\n',
    )
    # make_raw wrote under <data>/home-depot/raw already (raw_dir.parent / "raw").
    assert files["train"].parent == raw_dir

    manual = "fixture files: place them in the raw directory"
    names = {
        "train": "train.csv.zip",
        "descriptions": "product_descriptions.csv.zip",
        "attributes": "attributes.csv.zip",
        "spelling": "spelling.py",
    }
    spec = {
        "id": "home-depot",
        "version": "fixture",
        "license": "fixture",
        "source": "fixture",
        "distribution": "internal",
        "files": {
            name: {"manual": manual, "filename": filename, "sha256": sha256_file(files[name])}
            for name, filename in names.items()
        },
    }
    (root / "datasets" / "home-depot").mkdir(parents=True)
    (root / "datasets" / "home-depot" / "dataset.yaml").write_text(yaml.safe_dump(spec))
    return root


def test_prepare_home_depot_tasks(tmp_path):
    root = make_home_depot_root(tmp_path)
    result = prepare(root, SCENARIO)

    tasks = list(read_jsonl(result.scenario_dir / "tasks.jsonl"))
    corrected = [t for t in tasks if t["perturbation"] == "corrected"]
    assert corrected
    assert {t["variant_of"] for t in corrected} == {"hd-1"}  # the misspelled query only
    assert all(t["task_id"].startswith("hd-1-corrected:") for t in corrected)
    assert {t["text"] for t in corrected} == {"zorb brackets"}
    assert {t["group"] for t in tasks if t["variant_of"] == "hd-1"} == {"misspelled"}
    assert {t["group"] for t in tasks if t["variant_of"] == "hd-3"} == {"clean"}

    filtered = [t for t in tasks if t["mode"] == "filtered"]
    assert filtered
    assert all(t["filter"]["value"] != "None" for t in filtered)
    assert {t["filter"]["value"] for t in filtered} <= {"Vantro", "Ostrel"}
    # Its relevant products have no brand: no filtered task, rather than one on "None".
    assert not [t for t in filtered if t["variant_of"] == "hd-5"]
    assert [t for t in tasks if t["variant_of"] == "hd-5" and t["mode"] == "keyword"]

    manifest = json.loads((result.scenario_dir / "prepare.manifest.json").read_text())
    assert manifest["dataset"]["distribution"] == "internal"
    assert manifest["counts"]["docs"] == 5


def test_prepare_fails_before_any_work_without_the_kaggle_files(tmp_path):
    root = make_home_depot_root(tmp_path)
    raw_dir = root / ".irbench" / "data" / "home-depot" / "raw"
    (raw_dir / "attributes.csv.zip").unlink()
    with pytest.raises(ManualFileMissing, match=r"attributes\.csv\.zip"):
        prepare(root, SCENARIO)


# --- the committed dataset definition ------------------------------------------------------


def test_committed_dataset_yaml():
    spec = load_dataset_spec(REPO_ROOT, "home-depot")
    assert spec.version == "kaggle-2016"
    assert spec.distribution == "internal"
    assert spec.license == (
        "Kaggle competition rules (Home Depot Product Search Relevance), internal use only"
    )
    assert spec.source == "https://www.kaggle.com/competitions/home-depot-product-search-relevance"
    assert spec.params == {}
    assert set(spec.files) == {"train", "descriptions", "attributes", "spelling"}

    kaggle = [spec.files[n] for n in ("train", "descriptions", "attributes")]
    assert [f.filename for f in kaggle] == [
        "train.csv.zip",
        "product_descriptions.csv.zip",
        "attributes.csv.zip",
    ]
    assert all(f.url is None and f.sha256 == "0" * 64 for f in kaggle)
    # One instruction text for the three files: a missing-file error joins distinct texts.
    assert len({f.manual for f in kaggle}) == 1
    manual = kaggle[0].manual
    assert manual is not None
    assert "kaggle.com/competitions/home-depot-product-search-relevance/rules" in manual
    assert "kaggle competitions download -c home-depot-product-search-relevance -p" in manual

    spelling = spec.files["spelling"]
    assert spelling.manual is None
    assert spelling.sha256 == "76edf5ee8de0da6cd9385e5713f0866ac6f38d76aae1c80eb8b921a876615fd2"
    assert spelling.url == (
        "https://raw.githubusercontent.com/ChenglongChen/Kaggle_HomeDepot/"
        "55c1033d0af3b6cf2f033fe4bcf3e1e0ffda3445/Code/Chenglong/google_spelling_checker_dict.py"
    )


def test_committed_dataset_names_every_missing_kaggle_file(tmp_path):
    spec = load_dataset_spec(REPO_ROOT, "home-depot")
    with pytest.raises(ManualFileMissing) as excinfo:
        fetch_dataset(spec, tmp_path / "raw")
    message = str(excinfo.value)
    for name in ("train.csv.zip", "product_descriptions.csv.zip", "attributes.csv.zip"):
        assert name in message
    assert message.count("kaggle competitions download") == 1
