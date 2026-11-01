import sys
import types

import numpy as np
import pytest

from irbench.embed.models import MODELS, SentenceTransformerEmbedder, get_embedder


class _StubSentenceTransformer:
    instances: list["_StubSentenceTransformer"] = []

    def __init__(self, name: str):
        self.name = name
        self.calls: list[list[str]] = []
        _StubSentenceTransformer.instances.append(self)

    def encode(self, inputs: list[str], normalize_embeddings: bool, batch_size: int) -> np.ndarray:
        self.calls.append(list(inputs))
        return np.ones((len(inputs), 384), dtype=np.float32)


@pytest.fixture
def stub() -> type[_StubSentenceTransformer]:
    _StubSentenceTransformer.instances = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer  # type: ignore[attr-defined]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "sentence_transformers", module)
        yield _StubSentenceTransformer


def test_doc_prefix(stub):
    embedder = get_embedder("multilingual-e5-small")
    assert embedder.model_id == "multilingual-e5-small"
    assert embedder.dim == 384
    model = stub.instances[-1]
    assert model.name == "intfloat/multilingual-e5-small"
    embedder.embed(["a"], "doc")
    embedder.embed(["a"], "query")
    assert model.calls == [["passage: a"], ["query: a"]]


def test_bge_unchanged(stub):
    embedder = get_embedder("bge-small-en-v1.5")
    model = stub.instances[-1]
    assert model.name == "BAAI/bge-small-en-v1.5"
    embedder.embed(["a"], "doc")
    embedder.embed(["a"], "query")
    assert model.calls == [
        ["a"],
        ["Represent this sentence for searching relevant passages: a"],
    ]


def test_doc_prefix_defaults_to_empty(stub):
    embedder = SentenceTransformerEmbedder("m", "org/m", 384)
    embedder.embed(["x"], "doc")
    embedder.embed(["x"], "query")
    assert stub.instances[-1].calls == [["x"], ["x"]]


def test_registry_has_e5():
    assert "multilingual-e5-small" in MODELS
