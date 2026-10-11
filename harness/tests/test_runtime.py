import httpx
import pytest
import yaml

from conftest import REPO_ROOT
from irbench.runtime.docker import DockerRuntime, wait_healthy
from irbench.runtime.hardware import load_hardware


def _root(tmp_path):
    compose = tmp_path / "engines" / "meilisearch" / "docker" / "compose.yaml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services: {}\n")
    return tmp_path, compose


def _recorder(responses=None):
    calls = []

    def run(cmd):
        calls.append(cmd)
        line = " ".join(cmd)
        for prefix, out in (responses or {}).items():
            if line.startswith(prefix):
                if isinstance(out, Exception):
                    raise out
                return out
        return ""

    return run, calls


def test_hardware_profiles_load():
    assert load_hardware(REPO_ROOT, "local").engine.cpuset is None
    ref = load_hardware(REPO_ROOT, "aws-c7i-4xlarge")
    assert (ref.engine.cpuset, ref.engine.memory, ref.loadgen.cpuset) == ("0-7", "16g", "8-11")
    with pytest.raises(FileNotFoundError, match="nope"):
        load_hardware(REPO_ROOT, "nope")


def test_up_without_pinning_uses_only_the_engine_compose_file(tmp_path):
    root, compose = _root(tmp_path)
    run, calls = _recorder()
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "local"), runner=run)
    rt.up()
    assert calls == [
        [
            "docker",
            "compose",
            "-p",
            "irbench-meilisearch",
            "-f",
            str(compose),
            "up",
            "-d",
            "--wait",
            "--wait-timeout",
            "300",
        ]
    ]
    rt.down()
    assert calls[-1][-3:] == ["down", "-v", "--remove-orphans"]


def test_up_with_reference_profile_pins_cpu_and_memory(tmp_path):
    root, _ = _root(tmp_path)
    run, calls = _recorder()
    profile = load_hardware(REPO_ROOT, "aws-c7i-4xlarge")
    rt = DockerRuntime(root, "meilisearch", profile, runner=run)
    rt.up()
    override = root / ".irbench" / "runtime" / "meilisearch.override.yaml"
    assert yaml.safe_load(override.read_text()) == {
        "services": {"engine": {"cpuset": "0-7", "mem_limit": "16g", "memswap_limit": "16g"}}
    }
    assert ["-f", str(override)] == calls[0][6:8]


def test_missing_compose_file_is_reported(tmp_path):
    rt = DockerRuntime(tmp_path, "nope", load_hardware(REPO_ROOT, "local"), runner=_recorder()[0])
    with pytest.raises(FileNotFoundError, match="nope"):
        rt.up()


def test_inspection_helpers(tmp_path):
    root, _ = _root(tmp_path)
    run, _ = _recorder(
        {
            "docker inspect": "getmeili/meilisearch:v1.54.3\n",
            "docker image inspect": '["getmeili/meilisearch@sha256:abc"]\n',
            "docker exec irbench-meilisearch cat": "123456\n",
            "docker exec irbench-meilisearch du": "2048\t/meili_data\n",
        }
    )
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "local"), runner=run)
    assert rt.image_info() == {
        "image": "getmeili/meilisearch:v1.54.3",
        "digest": "getmeili/meilisearch@sha256:abc",
    }
    assert rt.memory_peak_bytes() == 123456
    assert rt.disk_bytes("/meili_data") == 2048 * 1024


def test_inspection_failures_return_none(tmp_path):
    root, _ = _root(tmp_path)
    run, _ = _recorder({"docker": RuntimeError("no such container")})
    rt = DockerRuntime(root, "meilisearch", load_hardware(REPO_ROOT, "local"), runner=run)
    assert rt.image_info() == {"image": None, "digest": None}
    assert rt.memory_peak_bytes() is None
    assert rt.disk_bytes("/x") is None


def test_wait_healthy_retries_until_200():
    statuses = iter([503, 503, 200])
    client = httpx.Client(
        base_url="http://e",
        transport=httpx.MockTransport(lambda r: httpx.Response(next(statuses))),
    )
    wait_healthy(client, "/health", timeout_s=5, interval_s=0)
    down = httpx.Client(
        base_url="http://e", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    with pytest.raises(TimeoutError, match="/health"):
        wait_healthy(down, "/health", timeout_s=0, interval_s=0)
