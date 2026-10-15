import hashlib
import io
import random
import re
import zipfile
from pathlib import Path

import httpx
import pytest

from irbench.datasets import fetch as fetch_module
from irbench.datasets.fetch import (
    ChecksumError,
    MemberNotFound,
    RangeNotSupported,
    fetch_file,
    fetch_zip_member,
)
from irbench.datasets.registry import DatasetSpec, fetch_dataset, load_dataset_spec
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


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in members.items():
            z.writestr(name, body)
    return buf.getvalue()


def _range_client(blob: bytes, calls: list[str], ranges: bool = True) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("range", "full"))
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers.get("range", ""))
        if not ranges or not m:
            return httpx.Response(200, content=blob)
        start, end = int(m[1]), min(int(m[2]), len(blob) - 1)
        return httpx.Response(
            206,
            content=blob[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(blob)}"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


MEMBERS = {
    "set/a/corpus.jsonl": b'{"_id": "1"}\n' * 5000,
    "set/b/corpus.jsonl": b"other\n",
}


def test_zip_member_is_extracted_and_verified(tmp_path):
    blob, calls = _zip_bytes(MEMBERS), []
    body = MEMBERS["set/a/corpus.jsonl"]
    dest = fetch_zip_member(
        "https://x/set.zip",
        "set/a/corpus.jsonl",
        tmp_path / "a.jsonl",
        hashlib.sha256(body).hexdigest(),
        _range_client(blob, calls),
    )
    assert dest.read_bytes() == body
    assert calls and "full" not in calls  # only range requests


def test_large_member_streams_over_many_range_requests(tmp_path):
    body = random.Random(0).randbytes(1 << 20)  # 1 MB, incompressible
    blob, calls = _zip_bytes({"big.bin": body}), []
    dest = fetch_zip_member(
        "https://x/set.zip",
        "big.bin",
        tmp_path / "big.bin",
        hashlib.sha256(body).hexdigest(),
        _range_client(blob, calls),
        buffer_size=64 << 10,
    )
    assert dest.read_bytes() == body
    assert len(calls) > 10  # read in chunks, never as one download


def test_zip_member_checksum_mismatch_leaves_no_file(tmp_path):
    blob = _zip_bytes(MEMBERS)
    with pytest.raises(ChecksumError):
        fetch_zip_member(
            "https://x/set.zip",
            "set/a/corpus.jsonl",
            tmp_path / "a.jsonl",
            "0" * 64,
            _range_client(blob, []),
        )
    assert not (tmp_path / "a.jsonl").exists()
    assert not list(tmp_path.glob("*.part"))


def test_zip_member_needs_range_support(tmp_path):
    blob = _zip_bytes(MEMBERS)
    with pytest.raises(RangeNotSupported, match="https://x/set.zip"):
        fetch_zip_member(
            "https://x/set.zip",
            "set/a/corpus.jsonl",
            tmp_path / "a.jsonl",
            "0" * 64,
            _range_client(blob, [], ranges=False),
        )


def test_zip_member_already_valid_is_not_refetched(tmp_path):
    body = MEMBERS["set/b/corpus.jsonl"]
    dest = tmp_path / "b.jsonl"
    dest.write_bytes(body)
    calls: list[str] = []
    fetch_zip_member(
        "https://x/set.zip",
        "set/b/corpus.jsonl",
        dest,
        hashlib.sha256(body).hexdigest(),
        _range_client(_zip_bytes(MEMBERS), calls),
    )
    assert calls == []


def test_fetch_dataset_stores_members_under_their_path(tmp_path):
    blob = _zip_bytes(MEMBERS)
    spec = DatasetSpec.model_validate(
        {
            "id": "set",
            "version": "1",
            "license": "x",
            "source": "x",
            "files": {
                "a": {
                    "url": "https://x/set.zip",
                    "member": "set/a/corpus.jsonl",
                    "sha256": hashlib.sha256(MEMBERS["set/a/corpus.jsonl"]).hexdigest(),
                },
                "b": {
                    "url": "https://x/set.zip",
                    "member": "set/b/corpus.jsonl",
                    "sha256": hashlib.sha256(MEMBERS["set/b/corpus.jsonl"]).hexdigest(),
                },
            },
        }
    )
    paths = fetch_dataset(spec, tmp_path / "raw", _range_client(blob, []))
    assert paths["a"] == tmp_path / "raw" / "set" / "a" / "corpus.jsonl"
    assert paths["b"].read_bytes() == MEMBERS["set/b/corpus.jsonl"]


def test_zip_member_retries_a_dropped_range_request(tmp_path):
    blob, calls = _zip_bytes(MEMBERS), []
    inner = _range_client(blob, calls)
    dropped = []

    def handler(request: httpx.Request) -> httpx.Response:
        if len(calls) == 3 and not dropped:  # drop one request mid-read
            dropped.append(request)
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return inner.send(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    body = MEMBERS["set/a/corpus.jsonl"]
    dest = fetch_zip_member(
        "https://x/set.zip",
        "set/a/corpus.jsonl",
        tmp_path / "a.jsonl",
        hashlib.sha256(body).hexdigest(),
        client,
    )
    assert dropped and dest.read_bytes() == body


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(fetch_module, "_sleep", lambda seconds: None)


def test_non_range_server_body_is_never_read(tmp_path):
    read: list[int] = []

    def body():
        for _ in range(1000):
            read.append(1)
            yield b"x" * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RangeNotSupported):
        fetch_zip_member("https://x/set.zip", "a", tmp_path / "a", "0" * 64, client)
    assert read == []


def test_range_retries_back_off(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(fetch_module, "_sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ConnectError):
        fetch_zip_member("https://x/set.zip", "a", tmp_path / "a", "0" * 64, client)
    assert sleeps == [1, 2, 4]


def test_missing_zip_member_names_url_and_member(tmp_path):
    with pytest.raises(MemberNotFound, match=r"https://x/set\.zip.*set/nope\.jsonl"):
        fetch_zip_member(
            "https://x/set.zip",
            "set/nope.jsonl",
            tmp_path / "n.jsonl",
            "0" * 64,
            _range_client(_zip_bytes(MEMBERS), []),
        )
    assert not (tmp_path / "n.jsonl.part").exists()
