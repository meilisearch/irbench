import hashlib
from pathlib import Path

import httpx
import pytest

from irbench.datasets.fetch import ChecksumError, fetch_file
from irbench.datasets.registry import load_dataset_spec
from irbench.paths import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
BODY = b"product_id\tproduct_name\n0\tbed\n"
SHA = hashlib.sha256(BODY).hexdigest()


def _client(body: bytes, calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_downloads_and_verifies(tmp_path):
    calls: list[str] = []
    dest = fetch_file("https://x/p.csv", tmp_path / "raw" / "p.csv", SHA, _client(BODY, calls))
    assert dest.read_bytes() == BODY
    assert calls == ["https://x/p.csv"]


def test_fetch_skips_when_file_already_valid(tmp_path):
    dest = tmp_path / "p.csv"
    dest.write_bytes(BODY)
    calls: list[str] = []
    fetch_file("https://x/p.csv", dest, SHA, _client(BODY, calls))
    assert calls == []


def test_fetch_rejects_checksum_mismatch_and_leaves_no_file(tmp_path):
    dest = tmp_path / "p.csv"
    with pytest.raises(ChecksumError, match="expected sha256"):
        fetch_file("https://x/p.csv", dest, SHA, _client(b"tampered", []))
    assert not dest.exists()
    assert not (tmp_path / "p.csv.part").exists()


def test_wands_spec_is_pinned():
    spec = load_dataset_spec(REPO_ROOT, "wands")
    assert spec.license == "MIT"
    assert set(spec.files) == {"product", "query", "label"}
    for f in spec.files.values():
        assert "/main/" not in f.url
        assert len(f.sha256) == 64


def test_workspace_paths(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.raw_dir("wands") == tmp_path / ".irbench" / "data" / "wands" / "raw"
    assert ws.scenario_dir("ecommerce/product-search") == (
        tmp_path / ".irbench" / "scenarios" / "ecommerce__product-search"
    )
    assert ws.embeddings_dir("wands", "BAAI/bge") == (
        tmp_path / ".irbench" / "embeddings" / "wands" / "BAAI__bge"
    )
