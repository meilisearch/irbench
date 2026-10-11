import hashlib
from pathlib import Path

import numpy as np

from irbench.embed.base import Embedder, TextKind
from irbench.embed.vectors import read_vectors, write_vectors


def text_key(kind: TextKind, text: str) -> str:
    return hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()


class VectorCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, np.ndarray] = {}
        if path.exists():
            keys, vectors = read_vectors(path)
            self._data = dict(zip(keys, vectors))

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> np.ndarray | None:
        return self._data.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._data[key] = np.asarray(vector, dtype=np.float32)

    def save(self) -> None:
        if not self._data:
            return
        keys = list(self._data)
        write_vectors(self.path, keys, np.stack([self._data[k] for k in keys]))


def embed_with_cache(
    embedder: Embedder,
    texts: list[str],
    kind: TextKind,
    cache: VectorCache,
    batch_size: int = 64,
    save_every: int = 50,
) -> np.ndarray:
    for i, t in enumerate(texts):
        if not t.strip():
            raise ValueError(f"cannot embed empty text at position {i}")
    keys = [text_key(kind, t) for t in texts]
    text_by_key = dict(zip(keys, texts))
    missing = [k for k in dict.fromkeys(keys) if cache.get(k) is None]

    for n, start in enumerate(range(0, len(missing), batch_size), start=1):
        batch = missing[start : start + batch_size]
        vectors = embedder.embed([text_by_key[k] for k in batch], kind)
        if vectors.shape != (len(batch), embedder.dim):
            raise ValueError(
                f"{embedder.model_id}: expected shape {(len(batch), embedder.dim)}, "
                f"got {vectors.shape}"
            )
        for key, vector in zip(batch, vectors):
            cache.put(key, vector)
        if n % save_every == 0:
            cache.save()

    return np.stack([cache.get(k) for k in keys])
