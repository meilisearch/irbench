from irbench.paths import Workspace, slug


def test_slug_and_run_dir(tmp_path):
    ws = Workspace(tmp_path)
    assert slug("ecommerce/product-search") == "ecommerce__product-search"
    assert ws.run_dir("abc") == tmp_path / ".irbench" / "runs" / "abc"


def test_vector_paths(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.doc_vectors_path("wands", "org/model") == (
        tmp_path / ".irbench" / "embeddings" / "wands" / "org__model" / "docs.parquet"
    )
    assert ws.query_vectors_path("ecommerce/product-search", "org/model") == (
        tmp_path
        / ".irbench"
        / "scenarios"
        / "ecommerce__product-search"
        / "embeddings"
        / "org__model"
        / "queries.parquet"
    )
