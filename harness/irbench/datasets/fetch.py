import hashlib
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
