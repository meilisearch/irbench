import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from irbench.datasets import NORMALIZERS
from irbench.datasets.amazon_esci import normalize_amazon_esci
from irbench.datasets.registry import DatasetSpec
from irbench.io import read_jsonl, read_qrels
from irbench.stages.prepare import prepare
from test_prepare import fixture_client, make_root


def _products(path, rows):
    """`rows`: list of dicts with product_id, title, description, bullet_point, brand,
    color, locale (locale defaults to "us")."""
    table = pa.table(
        {
            "product_id": [r["product_id"] for r in rows],
            "product_title": [r.get("title") for r in rows],
            "product_description": [r.get("description") for r in rows],
            "product_bullet_point": [r.get("bullet_point") for r in rows],
            "product_brand": [r.get("brand") for r in rows],
            "product_color": [r.get("color") for r in rows],
            "product_locale": [r.get("locale", "us") for r in rows],
        }
    )
    pq.write_table(table, path)
    return path


def _examples(path, rows):
    """`rows`: list of dicts with query_id, query, product_id, esci_label; small_version
    (default 1), split (default "test"), locale (default "us")."""
    table = pa.table(
        {
            "query_id": [r["query_id"] for r in rows],
            "query": [r["query"] for r in rows],
            "product_id": [r["product_id"] for r in rows],
            "product_locale": [r.get("locale", "us") for r in rows],
            "esci_label": [r["esci_label"] for r in rows],
            "small_version": [r.get("small_version", 1) for r in rows],
            "split": [r.get("split", "test") for r in rows],
        }
    )
    pq.write_table(table, path)
    return path


def _qac(path, rows):
    """`rows`: list of dicts with final_search_term, prefixes (list[str])."""
    table = pa.table(
        {
            "final_search_term": [r["final_search_term"] for r in rows],
            "prefixes": [r["prefixes"] for r in rows],
        }
    )
    pq.write_table(table, path)
    return path


def _sources(path, mapping):
    lines = ["query_id,source"]
    for qid, source in mapping.items():
        lines.append(f"{qid},{source}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _raw(tmp_path, products, examples, qac=(), sources=None):
    d = tmp_path / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "products": _products(d / "products.parquet", products),
        "examples": _examples(d / "examples.parquet", examples),
        "qac": _qac(d / "qac.parquet", qac),
        "sources": _sources(d / "sources.csv", sources or {}),
    }


def test_us_only_and_full_corpus(tmp_path):
    products = [
        {"product_id": "P1", "title": "Phone case"},
        {"product_id": "P2", "title": "Funda de telefono", "locale": "es"},
        {"product_id": "P3", "title": "Spare widget"},  # unjudged
    ]
    examples = [
        {"query_id": 1, "query": "phone case", "product_id": "P1", "esci_label": "E"},
        {"query_id": 2, "query": "funda", "product_id": "P2", "esci_label": "E", "locale": "es"},
    ]

    full = normalize_amazon_esci(_raw(tmp_path / "a", products, examples), tmp_path / "a" / "out")
    full_ids = {d["id"] for d in read_jsonl(full.docs)}
    assert full_ids == {"P1", "P3"}

    pool = normalize_amazon_esci(
        _raw(tmp_path / "b", products, examples), tmp_path / "b" / "out", corpus="pool"
    )
    pool_ids = {d["id"] for d in read_jsonl(pool.docs)}
    assert pool_ids == {"P1"}


def test_doc_fields(tmp_path):
    products = [
        {
            "product_id": "P1",
            "title": "Soft cotton product",
            "description": "<p>Soft <b>cotton</b>&amp;wool</p>",
        },
        {"product_id": "P2", "title": "Blank description product", "description": "<p>&nbsp;</p>"},
        {"product_id": "P3", "title": "No brand widget", "brand": None},
        {"product_id": "P4", "title": ""},
    ]
    examples = [
        {"query_id": 1, "query": "notitle product", "product_id": "P4", "esci_label": "E"},
    ]
    qac = [{"final_search_term": "notitle product", "prefixes": ["notitle product"]}]
    raw = _raw(tmp_path, products, examples, qac=qac)
    result = normalize_amazon_esci(raw, tmp_path / "out")
    docs = {d["id"]: d for d in read_jsonl(result.docs)}

    assert docs["P1"]["description"] == "Soft cotton&wool"
    assert docs["P2"]["description"] is None
    assert docs["P3"]["brand"] is None
    assert "P4" not in docs
    assert result.dropped == {"docs": 1, "judgments": 1}


def test_grades(tmp_path):
    examples = [
        {"query_id": 1, "query": "exact only", "product_id": "P1", "esci_label": "E"},
        {"query_id": 1, "query": "exact only", "product_id": "P2", "esci_label": "S"},
        {"query_id": 1, "query": "exact only", "product_id": "P3", "esci_label": "C"},
        {"query_id": 1, "query": "exact only", "product_id": "P4", "esci_label": "I"},
        {"query_id": 2, "query": "only bad", "product_id": "P1", "esci_label": "C"},
        {"query_id": 2, "query": "only bad", "product_id": "P2", "esci_label": "I"},
    ]
    products = [{"product_id": f"P{i}", "title": f"Product {i}"} for i in range(1, 5)]
    qac = [{"final_search_term": "exact only", "prefixes": ["exact only"]}]
    raw = _raw(tmp_path, products, examples, qac=qac)
    result = normalize_amazon_esci(raw, tmp_path / "out")

    queries = list(read_jsonl(result.queries))
    assert {q["qid"] for q in queries} == {"esci-1"}

    qrels = read_qrels(result.qrels)
    assert qrels["esci-1"] == {"P1": 2, "P2": 1, "P3": 0, "P4": 0}
    assert "esci-2" not in qrels


def test_pool_and_sample(tmp_path):
    n = 2500
    products = [{"product_id": "P0", "title": "Shared product"}]
    examples = [
        {"query_id": i, "query": f"query {i}", "product_id": "P0", "esci_label": "E"}
        for i in range(n)
    ]
    qac = [
        {"final_search_term": f"query {i}", "prefixes": [f"query {i}"]} for i in range(n)
    ]
    sources = {i: "generated" for i in range(n)}
    raw = _raw(tmp_path / "a", products, examples, qac=qac, sources=sources)

    first = normalize_amazon_esci(raw, tmp_path / "a" / "out1")
    queries = list(read_jsonl(first.queries))
    assert len(queries) == 2000
    assert all(q["qid"].startswith("esci-") for q in queries)
    assert all(q["group"] == "generated" for q in queries)

    raw2 = _raw(tmp_path / "b", products, examples, qac=qac, sources=sources)
    second = normalize_amazon_esci(raw2, tmp_path / "b" / "out2")
    assert first.queries.read_bytes() == second.queries.read_bytes()
    assert first.qrels.read_bytes() == second.qrels.read_bytes()


def test_case_spacing_duplicates(tmp_path):
    products = [{"product_id": "P1", "title": "iPhone case"}]
    examples = [
        {"query_id": 7, "query": "iPhone  case", "product_id": "P1", "esci_label": "E"},
        {"query_id": 3, "query": "iphone case", "product_id": "P1", "esci_label": "E"},
    ]
    qac = [{"final_search_term": "iphone case", "prefixes": ["iphone case"]}]
    raw = _raw(tmp_path, products, examples, qac=qac)
    result = normalize_amazon_esci(raw, tmp_path / "out")
    qids = {q["qid"] for q in read_jsonl(result.queries)}
    assert qids == {"esci-3"}


def test_prefix_groups(tmp_path):
    products = [{"product_id": "P1", "title": "Yeti mug product"}]
    examples = [
        {"query_id": 42, "query": "yeti mug", "product_id": "P1", "esci_label": "E"},
    ]
    session_prefixes = [
        "y",
        "ye",
        "yet",
        "yeti",
        "yeti ",
        "yeti m",
        "yeti m",
        "yeti mug",
        "yettim",
    ]
    qac = [{"final_search_term": "yeti mug", "prefixes": session_prefixes}]
    raw = _raw(tmp_path, products, examples, qac=qac)
    result = normalize_amazon_esci(raw, tmp_path / "out")

    rows = list(read_jsonl(result.query_sets["prefixes"][0]))
    groups_seen = [r["group"] for r in rows]
    assert len(groups_seen) == len(set(groups_seen))  # at most one row per group
    by_group = {r["group"]: r["text"] for r in rows}
    for r in rows:
        assert r["qid"] == f"esci-42-{r['group']}"

    assert set(by_group) == {"≤25%", "25–50%", "50–75%", "complete", "diverges"}
    assert by_group["≤25%"] in {"y", "ye"}
    assert by_group["25–50%"] in {"yet", "yeti"}
    assert by_group["50–75%"] in {"yeti ", "yeti m"}
    assert by_group["complete"] == "yeti mug"
    assert by_group["diverges"] == "yettim"
    if by_group["50–75%"] == "yeti ":
        assert len(by_group["50–75%"]) == 5  # trailing space preserved


def test_prefix_qrels(tmp_path):
    products = [
        {"product_id": "P1", "title": "Yeti mug product"},
        {"product_id": "P2", "title": "Yeti mug knockoff"},
    ]
    examples = [
        {"query_id": 42, "query": "yeti mug", "product_id": "P1", "esci_label": "E"},
        {"query_id": 42, "query": "yeti mug", "product_id": "P2", "esci_label": "C"},
    ]
    qac = [
        {"final_search_term": "yeti mug", "prefixes": ["y", "yeti mug"]},
    ]
    raw = _raw(tmp_path, products, examples, qac=qac)
    result = normalize_amazon_esci(raw, tmp_path / "out")

    final_qrels = read_qrels(result.qrels)
    prefixes_queries_path, prefixes_qrels_path = result.query_sets["prefixes"]
    prefix_qrels = read_qrels(prefixes_qrels_path)
    prefix_rows = list(read_jsonl(prefixes_queries_path))

    assert prefix_rows  # at least the complete row
    for row in prefix_rows:
        assert prefix_qrels[row["qid"]] == final_qrels["esci-42"]


def test_spec_params(tmp_path, wands_raw, monkeypatch):
    assert DatasetSpec.model_fields["params"].default == {}

    root = make_root(tmp_path, wands_raw)
    spec_path = root / "datasets" / "wands" / "dataset.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["params"] = {"corpus": "pool"}
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    received: dict = {}
    real = NORMALIZERS["wands"]

    def fake(raw, out_dir, **kwargs):
        received.update(kwargs)
        return real(raw, out_dir)

    monkeypatch.setitem(NORMALIZERS, "wands", fake)
    prepare(root, "ecommerce/product-search", client=fixture_client(wands_raw))
    assert received == {"corpus": "pool"}
