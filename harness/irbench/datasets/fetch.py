import hashlib
import io
import re
import zipfile
from pathlib import Path

import httpx


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(url: str, dest: Path, sha256: str, client: httpx.Client | None = None) -> Path:
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    owned = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
    finally:
        if owned:
            http.close()
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ChecksumError(f"{url}: expected sha256 {sha256}, got {actual}")
    tmp.replace(dest)
    return dest


_RANGE_ATTEMPTS = 4


class RangeNotSupported(RuntimeError):
    pass


class _HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, one HTTP range request per read."""

    def __init__(self, url: str, client: httpx.Client, max_request: int):
        self.url, self.client, self.max_request, self.pos = url, client, max_request, 0
        self.size = self._total()

    def _get(self, start: int, end: int) -> httpx.Response:
        for attempt in range(_RANGE_ATTEMPTS):
            try:
                r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
                break
            except httpx.TransportError:  # a dropped connection on a long run of reads
                if attempt == _RANGE_ATTEMPTS - 1:
                    raise
        if r.status_code != 206:
            raise RangeNotSupported(
                f"{self.url}: the server answered {r.status_code} to a range request; "
                "single files cannot be read out of this archive"
            )
        return r

    def _total(self) -> int:
        m = re.search(r"/(\d+)$", self._get(0, 0).headers.get("content-range", ""))
        if not m:
            raise RangeNotSupported(f"{self.url}: no Content-Range total in the reply")
        return int(m[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.pos + min(len(buffer), self.max_request), self.size) - 1
        data = self._get(self.pos, end).content
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_zip_member(
    url: str,
    member: str,
    dest: Path,
    sha256: str,
    client: httpx.Client | None = None,
    buffer_size: int = 8 << 20,
) -> Path:
    """Extract one member of a remote zip, reading only the bytes it needs. The member is
    streamed to disk in chunks; `buffer_size` is the size of each range request."""
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    owned = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        remote = io.BufferedReader(_HttpRangeFile(url, http, buffer_size), buffer_size=buffer_size)
        with zipfile.ZipFile(remote) as archive, archive.open(member) as src, tmp.open("wb") as out:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                out.write(chunk)
        actual = sha256_file(tmp)
        if actual != sha256:
            raise ChecksumError(f"{url}!{member}: expected sha256 {sha256}, got {actual}")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if owned:
            http.close()
    return dest
