# irbench-loadgen

Replays the request files that IRBench's `render` stage wrote, the same ones the relevance stage executed, and measures latency and throughput against one engine. The harness runs it as the `perf` stage; you rarely call it directly.

```bash
cargo build --release --manifest-path loadgen/Cargo.toml
loadgen/target/release/irbench-loadgen --requests .irbench/runs/<run-id>/rendered \
  --base-url http://localhost:17700 --out /tmp/perf.json
```

## What it measures

1. **Warm-up:** one pass over every request, discarded.
2. **Latency:** concurrency 1, one pass per `<mode>/<base>` group. Reports p50/p95/p99/mean/max of successful requests, plus the error count.
3. **Throughput:** for each group, closed-loop steps at concurrency 1, 2, 4, … for `--step-seconds` each. A step is valid when p99 ≤ `--p99-limit-ms` and the error rate ≤ `--max-error-rate`. Stepping stops at the first invalid step. Max QPS is the highest-QPS step among the leading run of valid steps, not necessarily the last one — a dip at one concurrency followed by a higher QPS at the next must not hide the earlier peak.

A request is an error when the connection fails, it times out, the status is not 2xx, or the body can't be read. The response body is always read before the clock stops. Facet verification fetches (`*.check.json`) are not replayed.

## Postgres target

A `--base-url` of `postgresql://user:pass@host:port/db` selects the Postgres wire protocol; `http://` stays HTTP. The request files must match: a Postgres run replays `{"protocol": "postgres", "sql", "params"}` files, and the run fails up front (before connecting, from the URL scheme) with a protocol-mismatch error if any file uses the other protocol.

- **One connection per concurrent client.** The loadgen opens as many connections as the highest concurrency step; worker `w` always uses connection `w % connections`.
- **Prepared per connection.** Before the warm-up, every distinct SQL text of the workload is prepared on every connection, so measured requests never pay a Parse. Every param is bound as text (`$n::text` in the SQL, with an optional cast), by the same rule as the Python client: null is NULL, bools are `true`/`false`, numbers are decimal, strings are themselves, lists of numbers are pgvector literals `[a,b,...]`.
- **Timeout.** `--timeout-s` becomes the session `statement_timeout`; a timed-out or failed statement counts as an error. SQL errors and server-side timeouts leave the connection usable; if the connection itself is broken (backend killed, network drop) or a request exceeds `--timeout-s` plus 1 s client-side, that request is an error and the connection is replaced (statements re-prepared) before the next one. A failed reconnect is retried, after a 50 ms pause, on the next request.
- **Measured time** is the statement execution plus fetching all rows, over an already-open connection.

## Tests

`cargo test --manifest-path loadgen/Cargo.toml` runs the unit tests and the integration tests against an in-process mock engine with injected latencies and errors. The live Postgres tests are ignored by default and each skips (returns early) when its URL variable is unset:

- `IRBENCH_PG_URL` (any Postgres engine; the CI job uses pgvector on port 15432): the prepared-statement, dead-connection, timeout-margin and prime tests. Start `engines/postgres/docker/compose.yaml` first.
- `IRBENCH_PARADEDB_URL` (the ParadeDB engine, port 15433): `paradedb_keyword_statement_survives_generic_plan_switch`. It runs `ALTER DATABASE ... SET plan_cache_mode = force_custom_plan` on the database itself, so it needs `pg_search` and a superuser. Start `engines/postgres-paradedb/docker/compose.yaml` first.

```bash
IRBENCH_PG_URL=postgresql://postgres:irbench@localhost:15432/irbench \
IRBENCH_PARADEDB_URL=postgresql://postgres:irbench@localhost:15433/irbench \
  cargo test -- --ignored
```

## Limits of the measurement

- **Throughput is closed-loop.** Each simulated client waits for its response before sending the next request. During an engine stall (garbage collection, segment merge), a closed loop records at most one slow sample per client, where an open-loop client sending at a fixed rate would record every request that queued behind the stall. Throughput-step p99 is therefore closed-loop service time, and it understates tail latency under load, more so for engines with intermittent pauses. The concurrency-1 latency pass is not affected.
- **One trial per step.** Each concurrency step runs once for `--step-seconds`; a single noisy step can end stepping early. Max QPS is the best step within the leading run of valid steps.
- **Same query order every step.** Every engine replays the same requests in the same order, so engine-side result caches act the same way for all of them.
- **Connection setup differs by protocol.** Postgres connections are opened and primed (statements prepared) before the warm-up, while HTTP keep-alive connections for higher concurrency steps are opened inside measured requests. This slightly favours Postgres.
- **Postgres caps the ladder.** The loadgen opens `max(highest concurrency step, warm-up concurrency)` connections, and Postgres's `max_connections` (100 by default) is the limit: today the ladder tops out at 64, which fits; a ladder above 100 would need a larger server setting.
- **A broken engine fails the run.** If the warm-up, a latency pass, or the lowest throughput step answers nothing or errors above `--max-error-rate`, the report lists it under `failures` and the CLI exits with status 2. Latencies with no successful request are `null`, never 0.
