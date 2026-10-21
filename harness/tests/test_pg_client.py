import subprocess
import time

import pytest

from irbench.engines.base import EngineError
from irbench.engines.pg import PostgresClient

pytestmark = pytest.mark.docker

DSN = "postgresql://postgres:irbench@localhost:15499/irbench"


@pytest.fixture(scope="module")
def pg():
    name = "irbench-test-pg-client"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-e",
            "POSTGRES_PASSWORD=irbench",
            "-e",
            "POSTGRES_DB=irbench",
            "-p",
            "15499:5432",
            "pgvector/pgvector:0.8.7-pg17",
        ],
        check=True,
        capture_output=True,
    )
    client = PostgresClient(DSN)
    try:
        deadline = time.monotonic() + 60
        while not client.ping():
            assert time.monotonic() < deadline, "postgres did not start"
            time.sleep(1)
        yield client
    finally:
        client.close()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_execute_binds_text_params(pg):
    pg.script(
        "CREATE EXTENSION IF NOT EXISTS vector; DROP TABLE IF EXISTS t; "
        "CREATE TABLE t (id text, v vector(2), n double precision)"
    )
    pg.copy("t", ["id", "v", "n"], [["a", "[1,0]", 1.5], ["b", "[0,1]", None]])
    rows = pg.execute(
        "SELECT id FROM t WHERE n IS NULL OR n >= $1::text::float8 "
        "ORDER BY v <=> $2::text::vector LIMIT $3::text::int",
        [1.0, [0.0, 1.0], 10],
    )
    assert [r["id"] for r in rows] == ["b", "a"]


def test_sql_error_is_an_engine_error(pg):
    with pytest.raises(EngineError):
        pg.execute("SELECT * FROM no_such_table")
    assert pg.execute("SELECT 1 AS x") == [{"x": 1}]


def test_timeout_is_an_engine_error_and_connection_survives(pg):
    with pytest.raises(EngineError, match="timeout|cancel"):
        pg.execute("SELECT pg_sleep(2)", timeout_s=0.2)
    assert pg.execute("SELECT 1 AS x") == [{"x": 1}]
