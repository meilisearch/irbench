import pytest

from irbench.engines.base import Request, SqlRequest, load_request
from irbench.engines.pg import bind_param


def test_sql_request_round_trip():
    r = SqlRequest(sql="SELECT id FROM t WHERE x = $1::text LIMIT $2::text::int", params=["a", 10])
    data = r.to_json()
    assert data == {"protocol": "postgres", "sql": r.sql, "params": ["a", 10]}
    assert load_request(data) == r


def test_http_files_still_load():
    data = {"protocol": "http", "method": "POST", "path": "/s", "headers": {}, "body": {"q": 1}}
    assert isinstance(load_request(data), Request)


def test_unknown_protocol_is_an_error():
    with pytest.raises(ValueError, match="protocol"):
        load_request({"protocol": "grpc"})


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (True, "true"),
        (False, "false"),
        (10, "10"),
        (0.5, "0.5"),
        (1e-7, "1e-07"),
        ("sofa", "sofa"),
        ([0.25, -1, 2.0], "[0.25,-1.0,2.0]"),
    ],
)
def test_bind_param(value, expected):
    assert bind_param(value) == expected


@pytest.mark.parametrize("value", [{"a": 1}, ["a", "b"], [True, 1.0]])
def test_bind_param_rejects_other_types(value):
    with pytest.raises(TypeError):
        bind_param(value)
