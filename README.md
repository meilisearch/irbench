# IRBench

IRBench is a public, reproducible benchmark that measures **relevance** and
**performance** of search engines and embedding models on realistic,
imperfect search scenarios. It is maintained by Meilisearch, but every
engine's configuration is declarative, isolated under its own folder, and
editable by that engine's community through a pull request; only results
produced on a pinned reference hardware profile are published, and relevance
and performance are always reported separately, broken down by query
imperfection and by performance dimension — never blended into one score or
a ranking.

Engines: Meilisearch, Elasticsearch, Qdrant, Typesense, Postgres (built-in
full-text search plus pgvector) and ParadeDB (pg_search BM25 plus pgvector). The two
Postgres engines are queried with SQL over the Postgres wire protocol.

## Quickstart

Prerequisites: Python 3.12 with [`uv`](https://docs.astral.sh/uv/), Docker,
and a Rust toolchain (for the load generator).

```bash
uv sync --extra embed-local
uv run irbench prepare --scenario ecommerce/product-search
uv run irbench run --scenario ecommerce/product-search --engine meilisearch --model bge-small-en-v1.5
uv run irbench report --results .irbench/runs --out .irbench/site
```

The first command installs dependencies, including `sentence-transformers`
for local embedding. The second prepares the `ecommerce/product-search`
scenario (dataset download, checksums, query perturbations). The third runs
the full pipeline against Meilisearch on the `local` hardware profile. The
fourth builds a static HTML report you can open from `.irbench/site/`.
Numbers produced on the `local` profile are not comparable to published
results. See the [quickstart docs](docs/quickstart.mdx) for what each step
prints and what to do next.

## Tracks and tuning

Every engine runs two tracks. `default` is the engine as shipped, with only
what the scenario schema requires. `tuned` applies documented settings chosen
per scenario within the [parity rules](docs/methodology/parity.mdx), each
explained by a `justification:` entry in the engine's config.

Tuned settings are chosen by committed sweeps
(`engines/<engine>/sweeps/<scenario>.yaml`, run as `--track tuned+<variant>`)
on one half of each scenario's queries, the `tune` half; the report publishes
scores on the other half, the `test` half, so no published number comes from
the queries the settings were picked on. Both tracks are measured for speed,
so a relevance gain can be read against what it costs in latency, throughput
and indexing time.

## Reproducing the reference run

`scripts/vm_bench.sh` runs the whole benchmark on the reference machine (AWS
c7i.4xlarge, profile [`hardware/aws-c7i-4xlarge.yaml`](hardware/aws-c7i-4xlarge.yaml)):
`setup` prepares and embeds every scenario, `sweeps` runs the default track
and every sweep variant (relevance only, three engine lanes in parallel), and
`finals` runs the tuned and default tracks with every stage, one run at a
time. Logs and progress go to `.irbench/logs/`. `scripts/sweep_deltas.py`
compares sweep variants, `scripts/finals_table.py` summarizes the final runs.

## Documentation

The full documentation site lives in [`docs/`](docs/):

- [Methodology overview](docs/methodology/overview.mdx) — the pipeline,
  dataset, modes, perturbations, metrics, and error handling.
- [Parity rules](docs/methodology/parity.mdx) — the fairness rules every
  engine configuration must follow, and how `irbench lint` checks them.
- [Performance methodology](docs/methodology/performance.mdx) — how
  latency, throughput, indexing cost and memory are measured.
- [Contributing: engine config](docs/contributing/engine-config.mdx) and
  [Contributing: new engine](docs/contributing/new-engine.mdx) — how to
  tune an existing engine or add a new one.
- [CLI reference](docs/reference/cli.mdx) and
  [Results reference](docs/reference/results.mdx).

## Results

Published results live in [`results/`](results/), one folder per run. Only
runs on a reference hardware profile are published; results from different
profiles are never compared.

## License

MIT — see [LICENSE](LICENSE).
