from dataclasses import dataclass
from pathlib import Path

import httpx

from irbench.embed.base import Embedder, render_doc_text
from irbench.embed.cache import VectorCache, embed_with_cache
from irbench.embed.models import get_embedder
from irbench.embed.publish import download_published, find_published, local_vectors_match
from irbench.embed.vectors import write_vectors
from irbench.io import read_jsonl
from irbench.paths import Workspace
from irbench.scenario import load_scenario

BATCH_SIZE = 64
SAVE_EVERY = 50


@dataclass(frozen=True)
class EmbedResult:
    docs_path: Path
    queries_path: Path
    n_docs: int
    n_queries: int
    n_computed: int
    source: str = "computed"  # or "local" (already on disk) or "published" (just downloaded)


def embed_scenario(
    root: Path,
    scenario_id: str,
    model_id: str,
    embedder: Embedder | None = None,
    *,
    compute: bool = False,
    client: httpx.Client | None = None,
) -> EmbedResult:
    ws = Workspace(root)
    variants_path = ws.scenario_dir(scenario_id) / "queries.variants.jsonl"
    if not variants_path.exists():
        raise FileNotFoundError(
            f"{variants_path} missing: run `irbench prepare --scenario {scenario_id}` first"
        )

    # An injected embedder or --compute means "compute". Otherwise: reuse local vectors that
    # already match the prepared data (never touching the network), else consult the
    # registry and download, else compute. This keeps `irbench run` from dead-ending on a
    # registry entry that doesn't match local data once local vectors already exist (e.g.
    # after a previous `--compute`).
    if embedder is None and not compute:
        local = local_vectors_match(root, scenario_id, model_id)
        if local is not None:
            docs_path, queries_path, n_docs, n_queries = local
            return EmbedResult(docs_path, queries_path, n_docs, n_queries, 0, "local")
        entry = find_published(root, scenario_id, model_id)
        if entry is not None:
            docs_path, queries_path = download_published(root, entry, client)
            return EmbedResult(
                docs_path, queries_path, entry.docs.count, entry.queries.count, 0, "published"
            )

    scenario = load_scenario(root, scenario_id)
    emb = embedder or get_embedder(model_id)
    emb_dir = ws.embeddings_dir(scenario.dataset, model_id)
    cache = VectorCache(emb_dir / "cache.parquet")
    cached_before = len(cache)

    docs = list(read_jsonl(ws.dataset_dir(scenario.dataset) / "docs.jsonl"))
    variants = list(read_jsonl(variants_path))
    try:
        doc_vectors = embed_with_cache(
            emb,
            [render_doc_text(d, scenario.embedding_fields) for d in docs],
            "doc",
            cache,
            batch_size=BATCH_SIZE,
            save_every=SAVE_EVERY,
        )
        query_vectors = embed_with_cache(
            emb,
            [v["text"] for v in variants],
            "query",
            cache,
            batch_size=BATCH_SIZE,
            save_every=SAVE_EVERY,
        )
    finally:
        cache.save()

    docs_path = ws.doc_vectors_path(scenario.dataset, model_id)
    queries_path = ws.query_vectors_path(scenario_id, model_id)
    write_vectors(docs_path, [d[scenario.id_field] for d in docs], doc_vectors)
    write_vectors(queries_path, [v["qid"] for v in variants], query_vectors)
    return EmbedResult(
        docs_path, queries_path, len(docs), len(variants), len(cache) - cached_before
    )
