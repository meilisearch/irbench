import time
from dataclasses import asdict, dataclass

from irbench.io import write_json
from irbench.runtime.docker import Runtime, wait_healthy
from irbench.stages.context import RunContext

HEALTH_TIMEOUT_S = 180


@dataclass(frozen=True)
class IndexStats:
    seconds: float
    n_docs: int
    memory_peak_bytes: int | None
    disk_bytes: int | None


def index_stage(ctx: RunContext, runtime: Runtime) -> IndexStats:
    adapter, docs, vectors = ctx.adapter, ctx.docs, ctx.doc_vectors
    with ctx.client() as client:
        wait_healthy(client, adapter.health_path, HEALTH_TIMEOUT_S)
        start = time.perf_counter()
        adapter.create_index(client)
        adapter.load_documents(client, docs, vectors)
        adapter.wait_until_indexed(client, len(docs), ctx.scenario.index_timeout_s)
        seconds = time.perf_counter() - start
    stats = IndexStats(
        seconds=round(seconds, 3),
        n_docs=len(docs),
        memory_peak_bytes=runtime.memory_peak_bytes(),
        disk_bytes=runtime.disk_bytes(adapter.data_dir),
    )
    write_json(ctx.run_dir / "index.json", asdict(stats))
    return stats
