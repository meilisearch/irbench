import numpy as np
import pytest

from irbench.embed.models import MODELS, get_embedder
from irbench.embed.vectors import read_vectors
from irbench.stages.embed import embed_scenario
from irbench.stages.prepare import prepare
from test_embed_cache import FakeEmbedder
from test_prepare import fixture_client, make_root

SCENARIO = "ecommerce/product-search"


@pytest.fixture
def prepared_root(tmp_path, wands_raw):
    root = make_root(tmp_path, wands_raw)
    prepare(root, SCENARIO, client=fixture_client(wands_raw))
    return root


def test_embed_writes_doc_and_query_vectors(prepared_root):
    result = embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    doc_ids, doc_vecs = read_vectors(result.docs_path)
    assert doc_ids == ["0", "1", "2", "3"]
    assert doc_vecs.shape == (4, 4)
    q_ids, _ = read_vectors(result.queries_path)
    assert "0" in q_ids and "0-typo" in q_ids


def test_second_run_computes_nothing(prepared_root):
    embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    again = embed_scenario(prepared_root, SCENARIO, "fake", embedder=FakeEmbedder())
    assert again.n_computed == 0


def test_interrupted_embed_resumes_from_cache(prepared_root, monkeypatch):
    monkeypatch.setattr("irbench.stages.embed.BATCH_SIZE", 1)
    monkeypatch.setattr("irbench.stages.embed.SAVE_EVERY", 1)
    first = FakeEmbedder(fail_after_calls=2)
    with pytest.raises(RuntimeError, match="API down"):
        embed_scenario(prepared_root, SCENARIO, "fake", embedder=first)
    already = {t for call in first.calls for t in call}
    assert len(already) == 2

    resumed = FakeEmbedder()
    result = embed_scenario(prepared_root, SCENARIO, "fake", embedder=resumed)
    seen_again = {t for call in resumed.calls for t in call}
    assert already.isdisjoint(seen_again)  # vectors paid for before the crash are reused
    assert result.n_computed == sum(len(c) for c in resumed.calls)


def test_embed_requires_prepare(tmp_path):
    with pytest.raises(FileNotFoundError, match="irbench prepare"):
        embed_scenario(tmp_path, SCENARIO, "fake", embedder=FakeEmbedder())


def test_model_registry():
    assert {"bge-small-en-v1.5", "text-embedding-3-small"} <= set(MODELS)
    with pytest.raises(ValueError, match="unknown model"):
        get_embedder("nope")


@pytest.mark.slow
def test_bge_small_real_model():
    emb = get_embedder("bge-small-en-v1.5")
    vecs = emb.embed(["grey velvet sofa"], "query")
    assert vecs.shape == (1, 384)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-4)
