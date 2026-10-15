import httpx
import pytest

from irbench.stages.context import default_client


@pytest.mark.parametrize("scheme", ["postgresql", "postgres"])
def test_postgres_urls_select_the_postgres_client(scheme, monkeypatch):
    seen = []

    class Fake:
        def __init__(self, dsn):
            seen.append(dsn)

    monkeypatch.setattr("irbench.stages.context.PostgresClient", Fake)
    url = f"{scheme}://u:p@localhost:5432/db"
    assert isinstance(default_client(url), Fake) and seen == [url]


def test_http_urls_select_httpx():
    assert isinstance(default_client("http://localhost:7700"), httpx.Client)
