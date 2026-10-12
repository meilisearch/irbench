from pathlib import Path

from irbench.runs import RunSpec
from irbench.stages.context import RunContext


def _ctx(protocol: str, port: int) -> RunContext:
    fake = type("Fake", (), {"protocol": protocol, "host_port": port})
    ctx = RunContext(Path("."), RunSpec(scenario="s", engine="e", model="m", host="h"))
    ctx.__dict__["adapter_cls"] = fake
    return ctx


def test_postgres_adapter_gets_a_dsn():
    assert _ctx("postgres", 15432).base_url == "postgresql://postgres:irbench@h:15432/irbench"


def test_http_adapter_gets_an_http_url():
    assert _ctx("http", 7700).base_url == "http://h:7700"
