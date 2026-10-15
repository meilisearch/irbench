import json
import re

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from fake_engine import SCENARIO, fake_root, set_scenario
from irbench.cli import app
from irbench.datasets.fetch import ChecksumError, sha256_file
from irbench.embed.publish import (
    REGISTRY,
    _save_registry,
    find_published,
    load_registry,
    local_vectors_match,
    pack_vectors,
)
from irbench.embed.vectors import read_vectors, write_vectors
from irbench.io import read_jsonl
from irbench.paths import Workspace
from irbench.scenario import load_scenario
from irbench.stages.embed import embed_scenario
from test_embed_cache import FakeEmbedder


def serve(directory, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(str(request.url))
        path = directory / request.url.path.rsplit("/", 1)[-1]
        return (
            httpx.Response(200, content=path.read_bytes()) if path.exists() else httpx.Response(404)
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def vector_paths(root):
    ws, scenario = Workspace(root), load_scenario(root, SCENARIO)
    return ws.doc_vectors_path(scenario.dataset, "fake"), ws.query_vectors_path(SCENARIO, "fake")


def test_pack_writes_assets_and_registry(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    entry, files = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    docs, queries = vector_paths(root)
    docs_sha12, queries_sha12 = entry.docs.sha256[:12], entry.queries.sha256[:12]
    assert [f.name for f in files] == [
        f"ecommerce__product-search__fake__docs-{docs_sha12}.parquet",
        f"ecommerce__product-search__fake__queries-{queries_sha12}.parquet",
    ]
    assert entry.docs.url == (
        "https://github.com/meilisearch/irbench/releases/download/vectors/"
        f"ecommerce__product-search__fake__docs-{docs_sha12}.parquet"
    )
    assert entry.docs.sha256 == sha256_file(docs) and entry.queries.sha256 == sha256_file(queries)
    assert entry.docs.count == len(read_vectors(docs)[0]) and entry.docs.dim == 4
    assert entry.dataset == "wands" and entry.seed == 20261009
    assert load_registry(root) == [entry]

    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist", release="vectors-2")
    registry = load_registry(root)
    assert len(registry) == 1 and "/download/vectors-2/" in registry[0].docs.url


def test_published_vectors_are_downloaded_instead_of_computed(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    docs, queries = vector_paths(root)
    before = sha256_file(docs), sha256_file(queries)
    docs.unlink()
    queries.unlink()
    seen = []

    # "fake" is not in the model registry: computing would raise, so success means download.
    result = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))

    assert result.source == "published" and result.n_computed == 0
    assert (sha256_file(docs), sha256_file(queries)) == before
    assert len(seen) == 2


def test_local_files_matching_the_checksum_are_not_downloaded_again(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    seen = []
    embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))
    assert seen == []


def test_tampered_asset_is_refused(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    _, files = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    files[0].write_bytes(b"not parquet")
    docs, _ = vector_paths(root)
    docs.unlink()
    with pytest.raises(ChecksumError, match="--compute"):
        embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))
    assert not docs.exists()


def test_local_vectors_are_reused_despite_registry_mismatch(tmp_path, wands_raw):
    # A registry entry that wouldn't download cleanly is never even consulted when the
    # local vectors already match the prepared data.
    root = fake_root(tmp_path, wands_raw)
    entry, _ = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    _save_registry(root, [entry.model_copy(update={"seed": 1})])
    seen = []
    result = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))
    assert result.source == "local" and result.n_computed == 0
    assert seen == []


def test_seed_mismatch_is_refused_once_local_vectors_are_gone(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    entry, _ = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    _save_registry(root, [entry.model_copy(update={"seed": 1})])
    _, queries = vector_paths(root)
    queries.unlink()
    with pytest.raises(ValueError, match="--compute"):
        embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))

    # Computing locally (as the error suggests) breaks the dead end: a later call
    # without --compute reuses the freshly computed local vectors instead of looping
    # back to the same mismatched registry entry.
    embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder())
    seen = []
    again = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))
    assert again.source == "local" and seen == []


def test_local_files_with_a_different_checksum_are_kept(tmp_path, wands_raw):
    # The registry's checksum covers the published asset, not a local file that happens
    # to carry the right ids; if it's already there, it's trusted and never overwritten.
    root = fake_root(tmp_path, wands_raw)
    entry, _ = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    _, queries = vector_paths(root)
    ids, vectors = read_vectors(queries)
    vectors = vectors.copy()
    vectors[:, 0] += 1.0  # same ids, different bytes than the registered asset
    write_vectors(queries, ids, vectors)
    before = sha256_file(queries)
    assert before != entry.queries.sha256
    seen = []
    result = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))
    assert result.source == "local" and seen == []
    assert sha256_file(queries) == before


def test_malformed_registry_raises_value_error_naming_the_file(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    path = root / REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("vectors: [{scenario: foo, model: bar\n")  # unterminated flow mapping
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_registry(root)


def test_id_mismatch_removes_the_file(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    entry, files = pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    ids, vectors = read_vectors(files[1])
    write_vectors(files[1], ids[::-1], vectors[::-1])  # same vectors, wrong order
    asset = entry.queries.model_copy(update={"sha256": sha256_file(files[1])})
    _save_registry(root, [entry.model_copy(update={"queries": asset})])
    _, queries = vector_paths(root)
    queries.unlink()
    with pytest.raises(ValueError, match="do not match"):
        embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))
    assert not queries.exists()


def test_compute_flag_and_injected_embedder_ignore_the_registry(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    result = embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder())
    assert result.source == "computed"
    with pytest.raises(ValueError, match="unknown model"):
        embed_scenario(root, SCENARIO, "fake", compute=True)  # no embedder for "fake"


def test_no_registry_entry_computes(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    assert find_published(root, SCENARIO, "fake") is None
    assert embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder()).source == "computed"


def test_registry_file_is_documented(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    text = (root / REGISTRY).read_text()
    assert text.startswith("# Published vectors")
    assert yaml.safe_load(text)["vectors"][0]["model"] == "fake"


def test_cli_pack_prints_the_upload_command(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    result = CliRunner().invoke(
        app,
        [
            "vectors",
            "pack",
            "--scenario",
            SCENARIO,
            "--model",
            "fake",
            "--out",
            str(tmp_path / "dist"),
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "gh release upload vectors" in result.output
    assert "--clobber" not in result.output  # content-addressed names are never reused


def sidecar(path):
    return path.with_name(path.name + ".meta.json")


def test_compute_writes_fingerprint_sidecars(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    docs, queries = vector_paths(root)
    for path, kind in ((docs, "docs"), (queries, "queries")):
        meta = json.loads(sidecar(path).read_text())
        assert meta["model"] == "fake" and meta["kind"] == kind
        assert meta["count"] == len(read_vectors(path)[0])
        assert re.fullmatch(r"[0-9a-f]{64}", meta["texts_sha256"])


def test_download_writes_fingerprint_sidecars(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    docs, queries = vector_paths(root)
    before = [sidecar(p).read_text() for p in (docs, queries)]
    for p in (docs, queries):
        p.unlink()
        sidecar(p).unlink()
    result = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist"))
    assert result.source == "published"
    assert [sidecar(p).read_text() for p in (docs, queries)] == before


def test_matching_sidecar_is_reused_without_compute_or_network(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    seen = []
    result = embed_scenario(root, SCENARIO, "fake", client=serve(tmp_path / "dist", seen))
    assert result.source == "local" and seen == []


@pytest.mark.parametrize("which", [0, 1])
def test_missing_sidecar_is_not_reused(tmp_path, wands_raw, which):
    root = fake_root(tmp_path, wands_raw)
    sidecar(vector_paths(root)[which]).unlink()
    assert local_vectors_match(root, SCENARIO, "fake") is None
    assert embed_scenario(root, SCENARIO, "fake", embedder=FakeEmbedder()).source == "computed"
    assert local_vectors_match(root, SCENARIO, "fake") is not None


@pytest.mark.parametrize("content", ["{not json", "[]", '{"model": "fake"}', "null"])
def test_malformed_sidecar_is_not_reused(tmp_path, wands_raw, content):
    root = fake_root(tmp_path, wands_raw)
    sidecar(vector_paths(root)[0]).write_text(content)
    assert local_vectors_match(root, SCENARIO, "fake") is None


def test_wrong_model_in_sidecar_is_not_reused(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    path = sidecar(vector_paths(root)[1])
    path.write_text(json.dumps({**json.loads(path.read_text()), "model": "other"}))
    assert local_vectors_match(root, SCENARIO, "fake") is None


def test_changed_embedding_fields_are_not_reused(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    assert local_vectors_match(root, SCENARIO, "fake") is not None
    fields = load_scenario(root, SCENARIO).embedding_fields
    set_scenario(root, embedding_fields=fields[:1])
    assert local_vectors_match(root, SCENARIO, "fake") is None


def test_changed_query_variant_text_is_not_reused(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    path = Workspace(root).scenario_dir(SCENARIO) / "queries.variants.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["text"] += " changed"  # same qids, as after a different perturbation seed
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert local_vectors_match(root, SCENARIO, "fake") is None


def test_pack_refuses_stale_vectors(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    fields = load_scenario(root, SCENARIO).embedding_fields
    set_scenario(root, embedding_fields=fields[:1])
    with pytest.raises(ValueError, match="do not match the texts"):
        pack_vectors(root, SCENARIO, "fake", tmp_path / "dist")
    assert not (tmp_path / "dist").exists() or not any((tmp_path / "dist").iterdir())
