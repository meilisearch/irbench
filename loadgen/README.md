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
3. **Throughput:** for each group, closed-loop steps at concurrency 1, 2, 4, … for `--step-seconds` each. A step is valid when p99 ≤ `--p99-limit-ms` and the error rate ≤ `--max-error-rate`. Stepping stops at the first invalid step. Max QPS is the QPS of the last valid step.

A request is an error when the connection fails, it times out, the status is not 2xx, or the body can't be read. The response body is always read before the clock stops. Facet verification fetches (`*.check.json`) are not replayed.

## Tests

`cargo test --manifest-path loadgen/Cargo.toml` runs the unit tests and the integration tests against an in-process mock engine with injected latencies and errors.
