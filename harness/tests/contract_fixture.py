"""100 product docs with known answers for the adapter contract tests (spec §10)."""

import numpy as np

ITEMS = ["sofa", "bed", "table", "chair", "lamp", "desk", "rug", "shelf", "stool", "mirror"]
COLORS = ["red", "blue", "green", "black", "white", "grey", "brown", "yellow", "pink", "orange"]
DIM = 8


def rating(i: int) -> float | None:
    return None if i % 7 == 0 else round(1.0 + (i * 37 % 41) / 10, 1)


def review_count(i: int) -> float | None:
    return None if i % 11 == 0 else float(i * 13 % 97)


def product_class(i: int) -> list[str]:
    classes = [ITEMS[i % 10].title() + "s"]
    if i % 20 == 0:
        classes.append("Sectionals")  # multi-valued facet field
    return classes


def contract_docs() -> tuple[list[dict], np.ndarray]:
    docs = []
    for i in range(100):
        color, item = COLORS[i // 10], ITEMS[i % 10]
        docs.append(
            {
                "id": str(i),
                "name": f"{color} {item}",
                "product_class": product_class(i),
                "category": ["Furniture", item.title()],
                # doc 99 has an empty text field (names still carry color and item)
                "description": "" if i == 99 else f"a {color} {item} for the home",
                "features": [f"color : {color}"],
                "rating_count": review_count(i),
                "average_rating": rating(i),
                "review_count": review_count(i),
            }
        )
    vectors = np.random.default_rng(7).normal(size=(100, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return docs, vectors
