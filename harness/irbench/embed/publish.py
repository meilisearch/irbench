"""Published vectors (spec §2, §5.2): computed once, uploaded as GitHub release assets and
downloaded by `irbench embed`, so reproducing a run needs no model download or API key."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from irbench.datasets.fetch import ChecksumError, fetch_file, sha256_file
from irbench.embed.base import render_doc_text
from irbench.embed.vectors import read_vectors
from irbench.io import read_json, read_jsonl
from irbench.paths import Workspace, slug
from irbench.scenario import load_scenario

REGISTRY = Path("embeddings") / "published.yaml"
RELEASE_REPO = "meilisearch/irbench"
DEFAULT_RELEASE = "vectors"
HEADER = (
    "# Published vectors. `irbench embed` downloads these instead of computing them.\n"
    "# Written by `irbench vectors pack`; each url is an asset of a GitHub release.\n"
)


class VectorAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    sha256: str
    count: int
    dim: int


class PublishedVectors(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str
    model: str
    dataset: str
    dataset_version: str
    seed: int  # query variants depend on the perturbation seed
    docs: VectorAsset
    queries: VectorAsset


def load_registry(root: Path) -> list[PublishedVectors]:
    path = root / REGISTRY
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: malformed YAML: {e}") from e
    return [PublishedVectors.model_validate(e) for e in data.get("vectors") or []]


def find_published(root: Path, scenario_id: str, model_id: str) -> PublishedVectors | None:
    return next(
        (e for e in load_registry(root) if (e.scenario, e.model) == (scenario_id, model_id)), None
    )


def _save_registry(root: Path, entries: list[PublishedVectors]) -> None:
    path = root / REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda e: (e.scenario, e.model))
    body = yaml.safe_dump({"vectors": [e.model_dump() for e in ordered]}, sort_keys=False)
    path.write_text(HEADER + body, encoding="utf-8")


def asset_name(scenario_id: str, model_id: str, kind: str, sha12: str) -> str:
    # Content-addressed: re-packing a scenario/model never reuses an older commit's
    # asset name, so its registry entry (and the asset itself) stays valid on the release.
    return f"{slug(scenario_id)}__{slug(model_id)}__{kind}-{sha12}.parquet"


def _prepared(root: Path, scenario_id: str) -> dict[str, Any]:
    path = Workspace(root).scenario_dir(scenario_id) / "prepare.manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `irbench prepare --scenario {scenario_id}`")
    return read_json(path)


def pack_vectors(
    root: Path, scenario_id: str, model_id: str, out_dir: Path, release: str = DEFAULT_RELEASE
) -> tuple[PublishedVectors, list[Path]]:
    ws, scenario, info = (
        Workspace(root),
        load_scenario(root, scenario_id),
        _prepared(root, scenario_id),
    )
    sources = {
        "docs": ws.doc_vectors_path(scenario.dataset, model_id),
        "queries": ws.query_vectors_path(scenario_id, model_id),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, VectorAsset] = {}
    files: list[Path] = []
    for kind, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing: run `irbench embed --scenario {scenario_id} --model {model_id} "
                "--compute` first"
            )
        tmp_dest = out_dir / f"{slug(scenario_id)}__{slug(model_id)}__{kind}.tmp.parquet"
        shutil.copyfile(src, tmp_dest)
        sha256 = sha256_file(tmp_dest)
        name = asset_name(scenario_id, model_id, kind, sha256[:12])
        dest = out_dir / name
        tmp_dest.replace(dest)
        ids, vectors = read_vectors(dest)
        assets[kind] = VectorAsset(
            url=f"https://github.com/{RELEASE_REPO}/releases/download/{release}/{name}",
            sha256=sha256,
            count=len(ids),
            dim=int(vectors.shape[1]),
        )
        files.append(dest)
    entry = PublishedVectors(
        scenario=scenario_id,
        model=model_id,
        dataset=info["dataset"]["id"],
        dataset_version=info["dataset"]["version"],
        seed=info["seed"],
        **assets,
    )
    others = [e for e in load_registry(root) if (e.scenario, e.model) != (scenario_id, model_id)]
    _save_registry(root, [*others, entry])
    return entry, files


def embedding_texts(root: Path, scenario_id: str) -> dict[str, list[str]]:
    """The exact, ordered texts that get embedded for each kind: docs rendered from the
    scenario's embedding_fields, and query variant texts. Single source of truth for the
    embed stage and for the fingerprints that guard vector reuse."""
    ws, scenario = Workspace(root), load_scenario(root, scenario_id)
    return {
        "docs": [
            render_doc_text(d, scenario.embedding_fields)
            for d in read_jsonl(ws.dataset_dir(scenario.dataset) / "docs.jsonl")
        ],
        "queries": [
            v["text"] for v in read_jsonl(ws.scenario_dir(scenario_id) / "queries.variants.jsonl")
        ],
    }


def texts_fingerprint(kind: str, texts: list[str]) -> str:
    """sha256 over the kind and the ordered texts (JSON-encoded, so boundaries are unambiguous)."""
    payload = json.dumps([kind, texts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sidecar_path(vectors_path: Path) -> Path:
    return vectors_path.with_name(vectors_path.name + ".meta.json")


def write_sidecar(vectors_path: Path, model_id: str, kind: str, texts: list[str]) -> None:
    """Record what `vectors_path` was embedded from. Written after the parquet, atomically."""
    meta = {
        "model": model_id,
        "kind": kind,
        "count": len(texts),
        "texts_sha256": texts_fingerprint(kind, texts),
    }
    path = sidecar_path(vectors_path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sidecar_matches(vectors_path: Path, model_id: str, kind: str, texts: list[str]) -> bool:
    """Whether the sidecar exists, parses, and fingerprints exactly these texts for this
    model. Anything missing or malformed counts as a mismatch."""
    try:
        meta = json.loads(sidecar_path(vectors_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(meta, dict)
        and meta.get("model") == model_id
        and meta.get("kind") == kind
        and meta.get("texts_sha256") == texts_fingerprint(kind, texts)
    )


def _expected_vectors(
    root: Path, scenario_id: str, model_id: str
) -> dict[str, tuple[Path, list[str]]]:
    """Where each kind's vectors should live, and the ids they must carry (doc ids from
    docs.jsonl in order, query ids from queries.variants.jsonl in order)."""
    ws, scenario = Workspace(root), load_scenario(root, scenario_id)
    return {
        "docs": (
            ws.doc_vectors_path(scenario.dataset, model_id),
            [
                d[scenario.id_field]
                for d in read_jsonl(ws.dataset_dir(scenario.dataset) / "docs.jsonl")
            ],
        ),
        "queries": (
            ws.query_vectors_path(scenario_id, model_id),
            [v["qid"] for v in read_jsonl(ws.scenario_dir(scenario_id) / "queries.variants.jsonl")],
        ),
    }


def _ids_match(dest: Path, ids: list[str]) -> bool:
    """Whether `dest` already exists and carries exactly `ids`, in order."""
    if not dest.exists():
        return False
    got_ids, _ = read_vectors(dest)
    return got_ids == ids


def local_vectors_match(
    root: Path, scenario_id: str, model_id: str
) -> tuple[Path, Path, int, int] | None:
    """If the local doc and query vector files both already exist, their ids match the
    prepared data, and their fingerprint sidecars match the texts that would be embedded
    now, return (docs_path, queries_path, n_docs, n_queries); else None. Ids alone are not
    enough: they survive changes to the dataset version, seed and embedding_fields. Checked
    first so a scenario with local vectors never needs the registry or the network."""
    expected = _expected_vectors(root, scenario_id, model_id)
    texts = embedding_texts(root, scenario_id)
    for kind, (dest, ids) in expected.items():
        if not _ids_match(dest, ids) or not _sidecar_matches(dest, model_id, kind, texts[kind]):
            return None
    (docs_path, doc_ids), (queries_path, query_ids) = expected["docs"], expected["queries"]
    return docs_path, queries_path, len(doc_ids), len(query_ids)


def download_published(
    root: Path, entry: PublishedVectors, client: httpx.Client | None = None
) -> tuple[Path, Path]:
    info = _prepared(root, entry.scenario)
    local = (info["dataset"]["id"], info["dataset"]["version"], info["seed"])
    published = (entry.dataset, entry.dataset_version, entry.seed)
    if local != published:
        raise ValueError(
            f"published {entry.model} vectors for {entry.scenario} were built from "
            f"{published[0]}@{published[1]} with seed {published[2]}, but the prepared data is "
            f"{local[0]}@{local[1]} with seed {local[2]}; run `irbench embed --compute`"
        )
    expected = _expected_vectors(root, entry.scenario, entry.model)
    texts = embedding_texts(root, entry.scenario)
    for kind, (dest, ids) in expected.items():
        asset: VectorAsset = getattr(entry, kind)
        try:
            fetch_file(asset.url, dest, asset.sha256, client=client)
        except ChecksumError as e:
            raise ChecksumError(
                f"{e}; the published asset does not match the registry checksum; "
                "run `irbench embed --compute` to embed locally instead"
            ) from e
        got_ids, vectors = read_vectors(dest)
        if got_ids != ids or vectors.shape[1] != asset.dim:
            dest.unlink()
            raise ValueError(
                f"{asset.url}: ids or dimension do not match the prepared {kind} "
                f"({len(got_ids)} vs {len(ids)} ids, dim {vectors.shape[1]} vs {asset.dim}); "
                "run `irbench embed --compute`"
            )
        write_sidecar(dest, entry.model, kind, texts[kind])
    return expected["docs"][0], expected["queries"][0]
