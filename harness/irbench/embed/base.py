from typing import Any, Literal, Protocol

import numpy as np

TextKind = Literal["query", "doc"]


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str], kind: TextKind) -> np.ndarray: ...


def render_doc_text(doc: dict[str, Any], fields: list[str], max_chars: int = 2000) -> str:
    parts: list[str] = []
    for name in fields:
        value = doc.get(name)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        text = "" if value is None else str(value).strip()
        if text:
            parts.append(text)
    return ". ".join(parts)[:max_chars]
