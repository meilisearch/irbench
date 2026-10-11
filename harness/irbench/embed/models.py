from collections.abc import Callable

import numpy as np

from irbench.embed.base import Embedder, TextKind


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str, hf_name: str, dim: int, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer  # optional extra: embed-local

        self.model_id = model_id
        self.dim = dim
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(hf_name)

    def embed(self, texts: list[str], kind: TextKind) -> np.ndarray:
        inputs = [self.query_prefix + t for t in texts] if kind == "query" else texts
        vectors = self._model.encode(inputs, normalize_embeddings=True, batch_size=64)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, model_id: str, dim: int):
        from openai import OpenAI  # optional extra: embed-api; reads OPENAI_API_KEY

        self.model_id = model_id
        self.dim = dim
        self._client = OpenAI()

    def embed(self, texts: list[str], kind: TextKind) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_id, input=texts)
        data = sorted(response.data, key=lambda d: d.index)
        vectors = np.asarray([d.embedding for d in data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


MODELS: dict[str, Callable[[], Embedder]] = {
    "bge-small-en-v1.5": lambda: SentenceTransformerEmbedder(
        "bge-small-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        384,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "text-embedding-3-small": lambda: OpenAIEmbedder("text-embedding-3-small", 1536),
}


def get_embedder(model_id: str) -> Embedder:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id!r}; known: {', '.join(sorted(MODELS))}")
    return MODELS[model_id]()
