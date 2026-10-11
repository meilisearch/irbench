from pathlib import Path

import numpy as np
import pytest

from irbench.embed.base import render_doc_text
from irbench.embed.cache import VectorCache, embed_with_cache, text_key
from irbench.embed.vectors import read_vectors, write_vectors


class FakeEmbedder:
    model_id = "fake"
    dim = 4

    def __init__(self, fail_after_calls: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_after_calls = fail_after_calls

    def embed(self, texts, kind):
        if self.fail_after_calls is not None and len(self.calls) >= self.fail_after_calls:
            raise RuntimeError("API down")
        self.calls.append(list(texts))
        vecs = np.array([[len(t), ord(t[0]), 1.0 if kind == "query" else 0.0, 1.0] for t in texts])
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)


def test_vectors_roundtrip(tmp_path):
    vecs = np.random.default_rng(0).random((3, 5), dtype=np.float32)
    write_vectors(tmp_path / "v.parquet", ["a", "b", "c"], vecs)
    ids, back = read_vectors(tmp_path / "v.parquet")
    assert ids == ["a", "b", "c"]
    np.testing.assert_array_equal(back, vecs)


def test_crash_during_save_keeps_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / "cache.parquet"
    cache = VectorCache(path)
    embed_with_cache(FakeEmbedder(), ["sofa"], "doc", cache)
    cache.save()

    def crash_midway(table, where, *args, **kwargs):
        Path(where).write_bytes(b"truncated parquet")
        raise KeyboardInterrupt

    monkeypatch.setattr("irbench.embed.vectors.pq.write_table", crash_midway)
    embed_with_cache(FakeEmbedder(), ["bed"], "doc", cache)
    with pytest.raises(KeyboardInterrupt):
        cache.save()
    monkeypatch.undo()

    assert len(VectorCache(path)) == 1  # previous save still readable


def test_text_key_depends_on_kind():
    assert text_key("query", "sofa") != text_key("doc", "sofa")


def test_cache_avoids_recomputation_and_persists(tmp_path):
    emb = FakeEmbedder()
    cache = VectorCache(tmp_path / "cache.parquet")
    first = embed_with_cache(emb, ["sofa", "bed", "sofa"], "doc", cache, batch_size=2)
    assert first.shape == (3, 4)
    assert sum(len(c) for c in emb.calls) == 2  # "sofa" embedded once
    cache.save()

    emb2 = FakeEmbedder()
    again = embed_with_cache(emb2, ["bed", "sofa"], "doc", VectorCache(tmp_path / "cache.parquet"))
    assert emb2.calls == []
    np.testing.assert_array_equal(again, first[[1, 0]])


def test_empty_text_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        embed_with_cache(FakeEmbedder(), ["ok", "  "], "doc", VectorCache(tmp_path / "c.parquet"))


def test_wrong_shape_from_embedder_is_rejected(tmp_path):
    class Bad(FakeEmbedder):
        def embed(self, texts, kind):
            return np.zeros((len(texts), 3), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        embed_with_cache(Bad(), ["a"], "doc", VectorCache(tmp_path / "c.parquet"))


def test_render_doc_text_skips_empty_and_joins_lists():
    doc = {"name": "grey sofa", "product_class": ["Sofas", "Sectionals"], "description": ""}
    assert render_doc_text(doc, ["name", "product_class", "description"]) == (
        "grey sofa. Sofas, Sectionals"
    )
    assert len(render_doc_text({"name": "x" * 5000}, ["name"])) == 2000
