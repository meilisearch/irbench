"""Published vectors (spec §2, §5.2): computed once, uploaded as GitHub release assets and
downloaded by `irbench embed`, so reproducing a run needs no model download or API key."""

import shutil
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from irbench.datasets.fetch import fetch_file, sha256_file
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
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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


def asset_name(scenario_id: str, model_id: str, kind: str) -> str:
    return f"{slug(scenario_id)}__{slug(model_id)}__{kind}.parquet"


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
        name = asset_name(scenario_id, model_id, kind)
        dest = out_dir / name
        shutil.copyfile(src, dest)
        ids, vectors = read_vectors(dest)
        assets[kind] = VectorAsset(
            url=f"https://github.com/{RELEASE_REPO}/releases/download/{release}/{name}",
            sha256=sha256_file(dest),
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


def download_published(
    root: Path, entry: PublishedVectors, client: httpx.Client | None = None
) -> tuple[Path, Path]:
    ws, scenario, info = (
        Workspace(root),
        load_scenario(root, entry.scenario),
        _prepared(root, entry.scenario),
    )
    local = (info["dataset"]["id"], info["dataset"]["version"], info["seed"])
    published = (entry.dataset, entry.dataset_version, entry.seed)
    if local != published:
        raise ValueError(
            f"published {entry.model} vectors for {entry.scenario} were built from "
            f"{published[0]}@{published[1]} with seed {published[2]}, but the prepared data is "
            f"{local[0]}@{local[1]} with seed {local[2]}; run `irbench embed --compute`"
        )
    expected = {
        "docs": (
            ws.doc_vectors_path(scenario.dataset, entry.model),
            [
                d[scenario.id_field]
                for d in read_jsonl(ws.dataset_dir(scenario.dataset) / "docs.jsonl")
            ],
        ),
        "queries": (
            ws.query_vectors_path(entry.scenario, entry.model),
            [
                v["qid"]
                for v in read_jsonl(ws.scenario_dir(entry.scenario) / "queries.variants.jsonl")
            ],
        ),
    }
    for kind, (dest, ids) in expected.items():
        asset: VectorAsset = getattr(entry, kind)
        fetch_file(asset.url, dest, asset.sha256, client=client)
        got_ids, vectors = read_vectors(dest)
        if got_ids != ids or vectors.shape[1] != asset.dim:
            dest.unlink()
            raise ValueError(
                f"{asset.url}: ids or dimension do not match the prepared {kind} "
                f"({len(got_ids)} vs {len(ids)} ids, dim {vectors.shape[1]} vs {asset.dim}); "
                "run `irbench embed --compute`"
            )
    return expected["docs"][0], expected["queries"][0]
