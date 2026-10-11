# IRBench Plan 3 — Rust Load Generator and Perf Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure engine performance by replaying the exact rendered request files that the relevance stage used. The new `perf` stage runs a Rust load generator through three phases: a warm-up pass, a latency pass at concurrency 1, and stepped-concurrency throughput. It reports p50/p95/p99/mean latency per mode and the maximum QPS under the spec's p99 ≤ 100 ms / errors ≤ 0.1% rule.

**Architecture:** `loadgen/` is a standalone Rust crate (`irbench-loadgen`): a library (request loading, latency recording, closed-loop runner, benchmark phases) plus a thin CLI that writes `perf.json`. The Python harness gets a `perf` stage between `relevance` and `score`. It runs the release binary against the engine container started by `up`, pins it to the hardware profile's loadgen CPU set with `taskset` on Linux, and folds a summary into `scores.json` and the manifest. Engines never see anything the relevance stage didn't send: the loadgen reads `rendered/<mode>/<base>/<qid>.json` and skips the `*.check.json` facet verification fetches.

**Tech Stack:** Rust 1.99 (edition 2024): tokio 1.53, reqwest 0.13, hdrhistogram 7.6, clap 4.6, serde 1.0, serde_json 1.0, anyhow 1.0, walkdir 2.5; axum 0.8 as the test-only mock server. Python 3.12 harness (existing).

**Spec:** `docs/superpowers/specs/2026-10-09-irbench-design.md`

**Plan series:** Plan 3 of 4 for v0.1. Plans 1 and 2 built prepare → score and the engine adapters; Plan 2 is in PR #1 (`plan2-engines`) and this plan branches from it. Plan 4: static HTML report, Mintlify docs, CI (parity lint, contract tests, loadgen tests), publishing vectors, `trec_eval` cross-check.

## Global Constraints

- Perf replays the **same rendered request files** as relevance (spec §2 "requests as data"); it never re-renders. Facet verification fetches (`*.check.json`) are not part of the workload.
- Phases (spec §5.7):
  - **warm-up:** one full pass over every rendered request, discarded.
  - **latency:** concurrency 1, one full pass per group (`<mode>/<base>`); report p50/p95/p99/mean (plus max and count) per group and over all groups.
  - **throughput:** step concurrency 1, 2, 4, 8, 16, 32, 64. Max QPS is the QPS of the highest step with p99 ≤ 100 ms and error rate ≤ 0.1%.
- Throughput is closed-loop: each of N workers waits for its response before sending the next request. Requests cycle in a fixed order. A step stops issuing at its deadline; in-flight requests finish and count. QPS = successful responses / step wall time.
- A request is an error when the transport fails, it times out (`request_timeout_s`, 10 s), the status is not 2xx, or the body can't be read. Latencies are recorded for successful requests only; errors are counted separately. The full response body is always read before the clock stops.
- Stepping stops at the first invalid step. Steps above it are not run, and max QPS comes from the last valid step before it. If the first step is already invalid, max QPS is `null`.
- Loadgen CPU pinning comes from the hardware profile (`loadgen.cpuset`, `8-11` on `aws-c7i-4xlarge`), applied with `taskset -c`. If a profile pins the loadgen and `taskset` is missing, the stage fails; it never runs unpinned silently.
- Perf settings live in the scenario (`perf:` block), not in engine configs, so every engine runs the identical workload.
- Rust code passes `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`. `Cargo.lock` is committed; `loadgen/target/` is gitignored.
- Python stays ruff-clean (`uv run ruff check .`, `uv run ruff format --check harness engines`).
- Commits must NOT contain `Co-Authored-By` lines (user rule).

## Review Focus

1. **The engine returns 5xx or times out during a throughput step.** Those requests count as errors, not as fast successes. An error rate above 0.1% makes the step invalid and stops stepping. Test: Task 4 `errors_invalidate_the_step_and_stop_stepping`.
2. **Concurrency is higher than the number of requests in a group,** or a step ends while requests are in flight. No panic or division by zero; QPS is computed from completed requests over wall time. Test: Task 3 `concurrency_above_request_count_cycles_without_panicking`.
3. **Warm-up results leak into the latency numbers.** The latency count must equal exactly one pass per group. Test: Task 4 `warmup_is_excluded_from_latency`.
4. **A throughput group the engine doesn't support** (e.g. Qdrant has no `keyword/keyword` files). It is listed as skipped, never an error and never zeros. Test: Task 4 `missing_throughput_group_is_skipped`.
5. **A profile pins the loadgen but `taskset` isn't available** (macOS). The stage fails with a clear reason instead of running unpinned. Test: Task 5 `test_pinned_profile_without_taskset_fails`.

---

## File Structure

```
.gitignore                                   # + loadgen/target/
loadgen/
  Cargo.toml, Cargo.lock
  README.md                                  # what is measured, CLI, how to build
  src/lib.rs                                 # module exports
  src/requests.rs                            # rendered files -> PreparedRequest (grouped)
  src/stats.rs                               # Recorder (HDR histogram), LatencySummary, StepResult, max_qps
  src/runner.rs                              # Target (HTTP client), closed-loop run(), Stop
  src/benchmark.rs                           # Config, Report, run_benchmark (warm-up, latency, throughput)
  src/main.rs                                # clap CLI -> perf.json
  tests/common/mod.rs                        # axum mock server + fixture writer
  tests/runner.rs                            # runner timing and error tests
  tests/benchmark.rs                         # phase tests + CLI end-to-end
harness/irbench/
  scenario.py                                # + PerfSpec, Scenario.perf
  runs.py                                    # STAGES gains "perf" (after relevance)
  stages/perf.py                             # loadgen command, perf_stage, summarize_perf
  stages/run.py                              # perf stage wiring, loadgen_runner injection, outputs
  stages/score_run.py                        # scores.json gains "perf"
  cli.py                                     # prints perf lines
harness/tests/
  fake_engine.py                             # + fake_loadgen runner, fake binary in fake_root
  test_perf_stage.py
  test_run_pipeline.py                       # e2e includes perf
scenarios/ecommerce/product-search/scenario.yaml   # + perf block
```

---

### Task 1: Loadgen crate and rendered-request loading

**Files:**
- Create: `loadgen/Cargo.toml`, `loadgen/src/lib.rs`, `loadgen/src/requests.rs`, `loadgen/src/main.rs` (placeholder `fn main() {}` replaced in Task 4)
- Modify: `.gitignore`
- Test: unit tests inside `loadgen/src/requests.rs`

**Interfaces:**
- Consumes: the rendered-request file format from Plan 2 (`Request.to_json()`): `{"protocol": "http", "method", "path", "headers": {}, "body": <json> | "content": "<raw>"}` under `rendered/<mode>/<base>/<qid>.json`.
- Produces:
  - `pub struct PreparedRequest { pub group: String, pub method: reqwest::Method, pub path: String, pub headers: Vec<(String, String)>, pub body: Option<Vec<u8>> }` (Clone, Debug).
  - `pub fn load_requests(dir: &Path) -> anyhow::Result<Vec<PreparedRequest>>`. Files come in sorted path order. `group` = `"<mode>/<base>"`. `*.check.json` and files not at depth 3 are skipped. A JSON `body` is serialized once, with `Content-Type: application/json` added unless the headers already set one. `content` is sent verbatim. Errors: non-`http` protocol, unparsable file or method, and a directory with no requests.

- [ ] **Step 1: Create the crate**

`loadgen/Cargo.toml`:

```toml
[package]
name = "irbench-loadgen"
version = "0.1.0"
edition = "2024"
license = "MIT"
publish = false
description = "Replays IRBench rendered requests to measure engine latency and throughput"

[dependencies]
anyhow = "1.0.104"
clap = { version = "4.6.7", features = ["derive"] }
hdrhistogram = "7.6.0"
reqwest = { version = "0.13.5", default-features = false }
serde = { version = "1.0.229", features = ["derive"] }
serde_json = "1.0.154"
tokio = { version = "1.53.2", features = ["rt-multi-thread", "macros", "time", "sync"] }
walkdir = "2.5.0"

[dev-dependencies]
axum = "0.8.9"
tokio = { version = "1.53.2", features = ["rt-multi-thread", "macros", "time", "sync", "net"] }
```

`reqwest` is built without default features: IRBench only talks plain HTTP to local containers, so it needs no TLS stack. If `cargo build` reports that `reqwest` 0.13 needs an HTTP version feature, add `features = ["http1"]`. Record that as a ruling.

`loadgen/src/lib.rs`:

```rust
//! IRBench load generator: replays rendered request files against one engine.

pub mod requests;
```

`loadgen/src/main.rs` (replaced in Task 4):

```rust
fn main() {}
```

Append to `.gitignore`:

```
loadgen/target/
```

- [ ] **Step 2: Write the failing tests** — the `#[cfg(test)]` module at the bottom of `loadgen/src/requests.rs` (create the file with only this module and the `use` lines for now)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("irbench-loadgen-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    #[test]
    fn loads_grouped_requests_and_skips_check_files() {
        let dir = tmp("grouped");
        let body = r#"{"protocol":"http","method":"POST","path":"/i/search","headers":{},"body":{"q":"sofa"}}"#;
        write(&dir, "keyword/keyword/1.json", body);
        write(&dir, "faceted/semantic/2.json", body);
        write(&dir, "faceted/semantic/2.check.json", body);
        write(&dir, "stray.json", body);

        let reqs = load_requests(&dir).unwrap();
        let groups: Vec<_> = reqs.iter().map(|r| r.group.as_str()).collect();
        assert_eq!(groups, ["faceted/semantic", "keyword/keyword"]);
        let r = &reqs[1];
        assert_eq!(r.method, reqwest::Method::POST);
        assert_eq!(r.path, "/i/search");
        assert_eq!(r.body.as_deref(), Some(br#"{"q":"sofa"}"#.as_slice()));
        assert_eq!(r.headers, [("Content-Type".to_string(), "application/json".to_string())]);
    }

    #[test]
    fn raw_content_is_sent_verbatim_with_its_own_headers() {
        let dir = tmp("raw");
        write(
            &dir,
            "keyword/keyword/1.json",
            r#"{"protocol":"http","method":"POST","path":"/_bulk","headers":{"Content-Type":"application/x-ndjson"},"content":"{}\n"}"#,
        );
        let r = &load_requests(&dir).unwrap()[0];
        assert_eq!(r.body.as_deref(), Some(b"{}\n".as_slice()));
        assert_eq!(r.headers, [("Content-Type".to_string(), "application/x-ndjson".to_string())]);
    }

    #[test]
    fn get_without_body_has_no_body() {
        let dir = tmp("get");
        write(&dir, "semantic/semantic/1.json", r#"{"protocol":"http","method":"GET","path":"/health","headers":{},"body":null}"#);
        let r = &load_requests(&dir).unwrap()[0];
        assert!(r.body.is_none() && r.headers.is_empty());
    }

    #[test]
    fn rejects_other_protocols_and_empty_dirs() {
        let dir = tmp("pg");
        write(&dir, "keyword/keyword/1.json", r#"{"protocol":"postgres","method":"POST","path":"/","headers":{}}"#);
        assert!(load_requests(&dir).unwrap_err().to_string().contains("postgres"));
        let empty = tmp("empty");
        assert!(load_requests(&empty).unwrap_err().to_string().contains("no rendered requests"));
    }
}
```

Add `pub mod requests;` usage is already in `lib.rs`. Above the test module, put only:

```rust
use std::path::Path;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --manifest-path loadgen/Cargo.toml requests 2>&1 | tail -15`
Expected: FAIL to compile — `cannot find function load_requests` / `cannot find type PreparedRequest`. (The first build downloads crates; that is expected.)

- [ ] **Step 4: Implement** — `loadgen/src/requests.rs` above the test module

```rust
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// One rendered request, ready to send: the body is serialized once at load time.
#[derive(Debug, Clone)]
pub struct PreparedRequest {
    /// `<mode>/<base>`, e.g. `keyword/keyword` or `filtered/semantic`.
    pub group: String,
    pub method: reqwest::Method,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct RenderedFile {
    protocol: String,
    method: String,
    path: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    body: Option<serde_json::Value>,
    #[serde(default)]
    content: Option<String>,
}

/// Every `<mode>/<base>/<qid>.json` under `dir`, in sorted path order.
/// Facet verification fetches (`*.check.json`) are not part of the workload.
pub fn load_requests(dir: &Path) -> Result<Vec<PreparedRequest>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(dir).sort_by_file_name() {
        let entry = entry.with_context(|| format!("walking {}", dir.display()))?;
        let name = entry.file_name().to_string_lossy();
        if !entry.file_type().is_file() || !name.ends_with(".json") || name.ends_with(".check.json") {
            continue;
        }
        let path = entry.path();
        let parts: Vec<String> = path
            .strip_prefix(dir)?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        if parts.len() != 3 {
            continue;
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: RenderedFile =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if file.protocol != "http" {
            bail!("{}: protocol {:?} is not supported", path.display(), file.protocol);
        }
        let method = file
            .method
            .parse::<reqwest::Method>()
            .with_context(|| format!("{}: bad method {:?}", path.display(), file.method))?;
        let mut headers: Vec<(String, String)> = file.headers.into_iter().collect();
        let body = match (file.content, file.body) {
            (Some(content), _) => Some(content.into_bytes()),
            (None, None | Some(serde_json::Value::Null)) => None,
            (None, Some(value)) => {
                if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
                    headers.push(("Content-Type".to_string(), "application/json".to_string()));
                }
                Some(serde_json::to_vec(&value)?)
            }
        };
        out.push(PreparedRequest {
            group: format!("{}/{}", parts[0], parts[1]),
            method,
            path: file.path,
            headers,
            body,
        });
    }
    if out.is_empty() {
        bail!("no rendered requests under {}", dir.display());
    }
    Ok(out)
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --manifest-path loadgen/Cargo.toml requests 2>&1 | tail -5 && cargo fmt --manifest-path loadgen/Cargo.toml --check && cargo clippy --manifest-path loadgen/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 4 passed; fmt silent; clippy `Finished` with no warnings. Run `cargo fmt --manifest-path loadgen/Cargo.toml` first if fmt reports a diff.

- [ ] **Step 6: Commit**

```bash
git add .gitignore loadgen
git commit -m "feat: loadgen crate loading rendered requests by mode and base"
```

---

### Task 2: Latency recorder, step validity and max QPS

**Files:**
- Create: `loadgen/src/stats.rs`
- Modify: `loadgen/src/lib.rs` (`pub mod stats;`)
- Test: unit tests inside `loadgen/src/stats.rs`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `pub struct Recorder` (HDR histogram of microseconds, 1 µs–60 s, 3 significant digits, plus an error count): `new()`, `Default`, `record_ok(micros: u64)`, `record_error()`, `merge(&Recorder)`, `successes() -> u64`, `errors() -> u64`, `total() -> u64`, `error_rate() -> f64` (0 when empty), `summary() -> LatencySummary`.
  - `#[derive(Serialize)] pub struct LatencySummary { count, errors, mean_ms, p50_ms, p95_ms, p99_ms, max_ms }`. `count` = successful requests. All `_ms` fields are 0.0 when count is 0.
  - `pub struct Limits { pub p99_ms: f64, pub max_error_rate: f64 }` (Serialize, Clone).
  - `#[derive(Serialize)] pub struct StepResult { concurrency, seconds, requests, errors, qps, error_rate, p50_ms, p95_ms, p99_ms, valid }`, built by `StepResult::new(concurrency, seconds, &Recorder, &Limits)`. `qps` = successes / seconds. `valid` = successes > 0 && p99 ≤ limit && error_rate ≤ max.
  - `pub fn max_qps(steps: &[StepResult]) -> Option<&StepResult>`: the last step of the leading run of valid steps.

- [ ] **Step 1: Write the failing tests** — create `loadgen/src/stats.rs` containing only this module, and add `pub mod stats;` to `lib.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Limits = Limits { p99_ms: 100.0, max_error_rate: 0.001 };

    fn recorder(latencies_ms: &[u64], errors: u64) -> Recorder {
        let mut r = Recorder::new();
        for ms in latencies_ms {
            r.record_ok(ms * 1000);
        }
        for _ in 0..errors {
            r.record_error();
        }
        r
    }

    #[test]
    fn summary_reports_percentiles_in_milliseconds() {
        let mut lat = vec![10; 90];
        lat.extend([50; 10]);
        let s = recorder(&lat, 2).summary();
        assert_eq!((s.count, s.errors), (100, 2));
        assert!((s.p50_ms - 10.0).abs() < 0.1, "{s:?}");
        assert!((s.p95_ms - 50.0).abs() < 0.1, "{s:?}");
        assert!((s.mean_ms - 14.0).abs() < 0.1, "{s:?}");
        assert!((s.max_ms - 50.0).abs() < 0.1, "{s:?}");
    }

    #[test]
    fn empty_recorder_is_all_zero() {
        let r = Recorder::new();
        assert_eq!(r.error_rate(), 0.0);
        let s = r.summary();
        assert_eq!((s.count, s.p99_ms, s.mean_ms), (0, 0.0, 0.0));
    }

    #[test]
    fn merge_adds_counts_and_errors() {
        let mut a = recorder(&[1, 2], 1);
        a.merge(&recorder(&[3], 2));
        assert_eq!((a.successes(), a.errors(), a.total()), (3, 3, 6));
        assert!((a.error_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn step_validity_follows_p99_and_error_rate() {
        let fast = StepResult::new(4, 2.0, &recorder(&[10; 1000], 0), &LIMITS);
        assert!(fast.valid);
        assert!((fast.qps - 500.0).abs() < 1e-9);
        assert_eq!((fast.requests, fast.errors), (1000, 0));
        let slow = StepResult::new(8, 1.0, &recorder(&[150; 100], 0), &LIMITS);
        assert!(!slow.valid);
        let failing = StepResult::new(8, 1.0, &recorder(&[10; 998], 2), &LIMITS);
        assert!(!failing.valid, "0.2% errors exceeds 0.1%");
        let nothing = StepResult::new(1, 1.0, &Recorder::new(), &LIMITS);
        assert!(!nothing.valid);
    }

    #[test]
    fn max_qps_is_the_last_valid_step_before_the_first_invalid_one() {
        let step = |c, valid, qps| StepResult {
            concurrency: c, seconds: 1.0, requests: 1, errors: 0, qps, error_rate: 0.0,
            p50_ms: 1.0, p95_ms: 1.0, p99_ms: 1.0, valid,
        };
        let steps = [step(1, true, 100.0), step(2, true, 180.0), step(4, false, 300.0), step(8, true, 400.0)];
        assert_eq!(max_qps(&steps).map(|s| s.concurrency), Some(2));
        assert!(max_qps(&[step(1, false, 10.0)]).is_none());
        assert!(max_qps(&[]).is_none());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path loadgen/Cargo.toml stats 2>&1 | tail -15`
Expected: FAIL to compile — `cannot find type Recorder`, `Limits`, `StepResult`.

- [ ] **Step 3: Implement** — `loadgen/src/stats.rs` above the test module

```rust
use hdrhistogram::Histogram;
use serde::Serialize;

/// Latency histogram (microseconds) of successful requests, plus an error count.
pub struct Recorder {
    hist: Histogram<u64>,
    errors: u64,
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            hist: Histogram::new_with_bounds(1, 60_000_000, 3).expect("static histogram bounds"),
            errors: 0,
        }
    }

    pub fn record_ok(&mut self, micros: u64) {
        self.hist.saturating_record(micros.max(1));
    }

    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    pub fn merge(&mut self, other: &Recorder) {
        self.hist.add(&other.hist).expect("histograms share bounds");
        self.errors += other.errors;
    }

    pub fn successes(&self) -> u64 {
        self.hist.len()
    }

    pub fn errors(&self) -> u64 {
        self.errors
    }

    pub fn total(&self) -> u64 {
        self.successes() + self.errors
    }

    pub fn error_rate(&self) -> f64 {
        match self.total() {
            0 => 0.0,
            total => self.errors as f64 / total as f64,
        }
    }

    fn ms_at(&self, quantile: f64) -> f64 {
        self.hist.value_at_quantile(quantile) as f64 / 1000.0
    }

    pub fn summary(&self) -> LatencySummary {
        if self.successes() == 0 {
            return LatencySummary { errors: self.errors, ..LatencySummary::default() };
        }
        LatencySummary {
            count: self.successes(),
            errors: self.errors,
            mean_ms: self.hist.mean() / 1000.0,
            p50_ms: self.ms_at(0.50),
            p95_ms: self.ms_at(0.95),
            p99_ms: self.ms_at(0.99),
            max_ms: self.hist.max() as f64 / 1000.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct LatencySummary {
    pub count: u64,
    pub errors: u64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

/// The spec's validity rule for one throughput step.
#[derive(Debug, Clone, Serialize)]
pub struct Limits {
    pub p99_ms: f64,
    pub max_error_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepResult {
    pub concurrency: usize,
    pub seconds: f64,
    pub requests: u64,
    pub errors: u64,
    pub qps: f64,
    pub error_rate: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub valid: bool,
}

impl StepResult {
    pub fn new(concurrency: usize, seconds: f64, rec: &Recorder, limits: &Limits) -> Self {
        let s = rec.summary();
        let error_rate = rec.error_rate();
        let qps = if seconds > 0.0 { s.count as f64 / seconds } else { 0.0 };
        Self {
            concurrency,
            seconds,
            requests: rec.total(),
            errors: rec.errors(),
            qps,
            error_rate,
            p50_ms: s.p50_ms,
            p95_ms: s.p95_ms,
            p99_ms: s.p99_ms,
            valid: s.count > 0 && s.p99_ms <= limits.p99_ms && error_rate <= limits.max_error_rate,
        }
    }
}

/// Spec §5.7: the highest step with p99 ≤ limit and errors ≤ limit. Stepping
/// stops at the first invalid step, so this is the last step of the leading valid run.
pub fn max_qps(steps: &[StepResult]) -> Option<&StepResult> {
    steps.iter().take_while(|s| s.valid).last()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --manifest-path loadgen/Cargo.toml stats 2>&1 | tail -5 && cargo fmt --manifest-path loadgen/Cargo.toml --check && cargo clippy --manifest-path loadgen/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 5 passed; fmt and clippy clean.

HDR histograms keep 3 significant digits, so `p50_ms` for 10 ms can read 10.007. That is why the tests compare with a 0.1 ms tolerance; keep it.

- [ ] **Step 5: Commit**

```bash
git add loadgen
git commit -m "feat: loadgen latency recorder, step validity and max QPS rule"
```

---

### Task 3: Closed-loop runner against a mock engine

**Files:**
- Create: `loadgen/src/runner.rs`, `loadgen/tests/common/mod.rs`, `loadgen/tests/runner.rs`
- Modify: `loadgen/src/lib.rs` (`pub mod runner;`)

**Interfaces:**
- Consumes: `PreparedRequest` (Task 1), `Recorder` (Task 2).
- Produces:
  - `pub struct Target`: `Target::new(base_url: &str, timeout: Duration, max_connections: usize) -> anyhow::Result<Self>`. Its private `send(&PreparedRequest) -> bool` returns true only for a 2xx with a fully read body.
  - `pub enum Stop { Pass, Deadline(Duration) }`.
  - `pub async fn run(target: Arc<Target>, requests: Arc<Vec<PreparedRequest>>, concurrency: usize, stop: Stop) -> (Recorder, f64)`. Workers claim request indexes from a shared counter: `Pass` stops after one pass, `Deadline` cycles until the deadline. Returns the merged recorder and the elapsed seconds. An empty request list returns an empty recorder immediately.
  - Test helpers (`tests/common/mod.rs`): `spawn_mock() -> Mock { base_url: String, hits: Arc<AtomicUsize> }` (an axum server on 127.0.0.1:0; each request's JSON body `{"delay_ms": u64, "status": u16}` sets the delay and status), `write_group(dir, group, specs: &[(u64, u16)])`, `tmp_dir(name)`.

- [ ] **Step 1: Write the test helpers** — `loadgen/tests/common/mod.rs`

```rust
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;

pub struct Mock {
    pub base_url: String,
    pub hits: Arc<AtomicUsize>,
}

/// Each request's JSON body says how long to wait and which status to return.
async fn handler(State(hits): State<Arc<AtomicUsize>>, body: Bytes) -> (StatusCode, &'static str) {
    hits.fetch_add(1, Ordering::SeqCst);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    let delay = v["delay_ms"].as_u64().unwrap_or(0);
    let status = v["status"].as_u64().unwrap_or(200) as u16;
    tokio::time::sleep(Duration::from_millis(delay)).await;
    (StatusCode::from_u16(status).unwrap(), "{\"hits\":[]}")
}

pub async fn spawn_mock() -> Mock {
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new().fallback(handler).with_state(hits.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Mock { base_url: format!("http://{addr}"), hits }
}

pub fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("irbench-loadgen-it-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One rendered request file per spec entry under `<dir>/<group>/`.
pub fn write_group(dir: &Path, group: &str, specs: &[(u64, u16)]) {
    let target = dir.join(group);
    std::fs::create_dir_all(&target).unwrap();
    for (i, (delay_ms, status)) in specs.iter().enumerate() {
        let file = serde_json::json!({
            "protocol": "http", "method": "POST", "path": "/search", "headers": {},
            "body": {"delay_ms": delay_ms, "status": status},
        });
        std::fs::write(target.join(format!("{i:04}.json")), file.to_string()).unwrap();
    }
}
```

- [ ] **Step 2: Write the failing tests** — `loadgen/tests/runner.rs`

```rust
mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use common::{spawn_mock, tmp_dir, write_group};
use irbench_loadgen::requests::load_requests;
use irbench_loadgen::runner::{Stop, Target, run};

async fn setup(name: &str, specs: &[(u64, u16)]) -> (common::Mock, Arc<Target>, Arc<Vec<irbench_loadgen::requests::PreparedRequest>>) {
    let mock = spawn_mock().await;
    let dir = tmp_dir(name);
    write_group(&dir, "keyword/keyword", specs);
    let requests = Arc::new(load_requests(&dir).unwrap());
    let target = Arc::new(Target::new(&mock.base_url, Duration::from_secs(5), 64).unwrap());
    (mock, target, requests)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_pass_sends_each_request_once_and_measures_injected_latency() {
    let mut specs = vec![(5, 200); 90];
    specs.extend([(50, 200); 10]);
    let (mock, target, requests) = setup("pass", &specs).await;

    let (rec, _) = run(target, requests, 1, Stop::Pass).await;

    assert_eq!(mock.hits.load(Ordering::SeqCst), 100);
    let s = rec.summary();
    assert_eq!((s.count, s.errors), (100, 0));
    assert!(s.p50_ms >= 5.0 && s.p50_ms < 15.0, "p50 {s:?}");
    assert!(s.p95_ms >= 50.0 && s.p95_ms < 90.0, "p95 {s:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn non_2xx_responses_are_errors_not_latencies() {
    let mut specs = vec![(1, 200); 95];
    specs.extend([(1, 500); 5]);
    let (_mock, target, requests) = setup("errors", &specs).await;
    let (rec, _) = run(target, requests, 4, Stop::Pass).await;
    assert_eq!((rec.successes(), rec.errors()), (95, 5));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timeouts_are_errors() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("timeout");
    write_group(&dir, "keyword/keyword", &[(300, 200), (1, 200)]);
    let requests = Arc::new(load_requests(&dir).unwrap());
    let target = Arc::new(Target::new(&mock.base_url, Duration::from_millis(100), 4).unwrap());
    let (rec, _) = run(target, requests, 1, Stop::Pass).await;
    assert_eq!((rec.successes(), rec.errors()), (1, 1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deadline_mode_is_closed_loop_and_cycles_requests() {
    let (_mock, target, requests) = setup("deadline", &[(20, 200); 10]).await;
    let (rec, seconds) = run(target, requests, 4, Stop::Deadline(Duration::from_secs(1))).await;
    let qps = rec.successes() as f64 / seconds;
    // 4 workers × 1 request per ~20 ms ≈ 200 QPS; scheduling noise only lowers it.
    assert!(rec.successes() > 10, "requests must cycle past one pass");
    assert!(qps > 100.0 && qps <= 210.0, "qps {qps}");
    assert!(seconds >= 1.0 && seconds < 1.5, "seconds {seconds}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_above_request_count_cycles_without_panicking() {
    let (_mock, target, requests) = setup("tiny", &[(1, 200); 3]).await;
    let (pass, _) = run(target.clone(), requests.clone(), 64, Stop::Pass).await;
    assert_eq!(pass.total(), 3);
    let (step, seconds) = run(target, requests, 64, Stop::Deadline(Duration::from_millis(300))).await;
    assert!(step.successes() > 3 && seconds > 0.0);
    let (empty, _) = run(
        Arc::new(Target::new("http://127.0.0.1:9", Duration::from_secs(1), 1).unwrap()),
        Arc::new(Vec::new()),
        8,
        Stop::Deadline(Duration::from_millis(50)),
    )
    .await;
    assert_eq!(empty.total(), 0);
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --manifest-path loadgen/Cargo.toml --test runner 2>&1 | tail -15`
Expected: FAIL to compile — `unresolved import irbench_loadgen::runner`.

- [ ] **Step 4: Implement** — `loadgen/src/runner.rs`, and add `pub mod runner;` to `lib.rs`

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::requests::PreparedRequest;
use crate::stats::Recorder;

/// The engine under test: one pooled HTTP client and a base URL.
pub struct Target {
    client: reqwest::Client,
    base_url: String,
}

impl Target {
    pub fn new(base_url: &str, timeout: Duration, max_connections: usize) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(max_connections)
            .build()?;
        Ok(Self { client, base_url: base_url.trim_end_matches('/').to_string() })
    }

    /// True for a 2xx response whose body was read completely.
    async fn send(&self, req: &PreparedRequest) -> bool {
        let mut builder = self.client.request(req.method.clone(), format!("{}{}", self.base_url, req.path));
        for (name, value) in &req.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        if let Some(body) = &req.body {
            builder = builder.body(body.clone());
        }
        match builder.send().await {
            Ok(response) => {
                let ok = response.status().is_success();
                response.bytes().await.is_ok() && ok
            }
            Err(_) => false,
        }
    }
}

pub enum Stop {
    /// Send every request exactly once.
    Pass,
    /// Cycle through the requests until this much time has passed.
    Deadline(Duration),
}

/// Closed-loop load: `concurrency` workers each send the next request and wait
/// for its response before sending another. Returns the merged recorder and the
/// wall time, including requests still in flight at the deadline.
pub async fn run(
    target: Arc<Target>,
    requests: Arc<Vec<PreparedRequest>>,
    concurrency: usize,
    stop: Stop,
) -> (Recorder, f64) {
    let start = Instant::now();
    let total = requests.len();
    if total == 0 {
        return (Recorder::new(), 0.0);
    }
    let deadline = match stop {
        Stop::Pass => None,
        Stop::Deadline(d) => Some(start + d),
    };
    let next = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::with_capacity(concurrency.max(1));
    for _ in 0..concurrency.max(1) {
        let (target, requests, next) = (target.clone(), requests.clone(), next.clone());
        workers.push(tokio::spawn(async move {
            let mut rec = Recorder::new();
            loop {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if deadline.is_none() && i >= total {
                    break;
                }
                let t0 = Instant::now();
                if target.send(&requests[i % total]).await {
                    rec.record_ok(t0.elapsed().as_micros() as u64);
                } else {
                    rec.record_error();
                }
            }
            rec
        }));
    }
    let mut merged = Recorder::new();
    for worker in workers {
        merged.merge(&worker.await.expect("load worker panicked"));
    }
    (merged, start.elapsed().as_secs_f64())
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --manifest-path loadgen/Cargo.toml --test runner 2>&1 | tail -5 && cargo fmt --manifest-path loadgen/Cargo.toml --check && cargo clippy --manifest-path loadgen/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -3`
Expected: 5 passed; fmt and clippy clean.

The timing assertions have wide bounds on purpose. A tight bound that fails on a loaded laptop is a flaky test, not a stricter one. If `deadline_mode_is_closed_loop_and_cycles_requests` fails because QPS is above 210, the runner is not waiting for responses, which is a real bug: debug it, don't widen the bound.

- [ ] **Step 6: Commit**

```bash
git add loadgen
git commit -m "feat: closed-loop loadgen runner with timeout and error accounting"
```

---

### Task 4: Benchmark phases, JSON report and CLI

**Files:**
- Create: `loadgen/src/benchmark.rs`, `loadgen/tests/benchmark.rs`, `loadgen/README.md`
- Modify: `loadgen/src/lib.rs` (`pub mod benchmark;`), `loadgen/src/main.rs`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `pub struct Config { requests_dir: PathBuf, base_url: String, concurrency: Vec<usize>, groups: Vec<String>, step_seconds: f64, limits: Limits, timeout: Duration, warmup_concurrency: usize }`. An empty `groups` means every group gets a throughput run.
  - `#[derive(Serialize)] pub struct Report { base_url, requests: usize, limits: Limits, warmup: PhaseInfo, latency: BTreeMap<String, LatencySummary>, throughput: BTreeMap<String, GroupThroughput>, skipped_groups: Vec<String> }`. `latency` holds one key per group plus `"all"`. `PhaseInfo { requests, errors, seconds }`. `GroupThroughput { steps: Vec<StepResult>, max_qps: Option<f64>, max_qps_concurrency: Option<usize> }`.
  - `pub async fn run_benchmark(cfg: &Config) -> anyhow::Result<Report>`.
  - CLI `irbench-loadgen --requests <dir> --base-url <url> --out <perf.json> [--concurrency 1,2,4,8,16,32,64] [--groups a/b,c/d] [--step-seconds 20] [--p99-limit-ms 100] [--max-error-rate 0.001] [--timeout-s 10] [--warmup-concurrency 4]`. It writes the report as pretty JSON and prints one line per phase to stderr.

- [ ] **Step 1: Write the failing tests** — `loadgen/tests/benchmark.rs`

```rust
mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::{spawn_mock, tmp_dir, write_group};
use irbench_loadgen::benchmark::{Config, run_benchmark};
use irbench_loadgen::stats::Limits;

fn config(dir: std::path::PathBuf, base_url: &str) -> Config {
    Config {
        requests_dir: dir,
        base_url: base_url.to_string(),
        concurrency: vec![1, 2],
        groups: vec![],
        step_seconds: 0.3,
        limits: Limits { p99_ms: 100.0, max_error_rate: 0.001 },
        timeout: Duration::from_secs(5),
        warmup_concurrency: 4,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn latency_per_group_follows_the_injected_distribution() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("dist");
    let mut keyword = vec![(5, 200); 90];
    keyword.extend([(50, 200); 10]);
    write_group(&dir, "keyword/keyword", &keyword);
    write_group(&dir, "semantic/semantic", &[(20, 200); 20]);

    let report = run_benchmark(&config(dir, &mock.base_url)).await.unwrap();

    let kw = &report.latency["keyword/keyword"];
    assert_eq!(kw.count, 100);
    assert!(kw.p50_ms >= 5.0 && kw.p50_ms < 15.0, "{kw:?}");
    assert!(kw.p99_ms >= 50.0 && kw.p99_ms < 90.0, "{kw:?}");
    let sem = &report.latency["semantic/semantic"];
    assert!(sem.p50_ms >= 20.0 && sem.p50_ms < 35.0, "{sem:?}");
    assert_eq!(report.latency["all"].count, 120);
    assert_eq!(report.requests, 120);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn warmup_is_excluded_from_latency() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("warmup");
    write_group(&dir, "keyword/keyword", &[(1, 200); 30]);
    let mut cfg = config(dir, &mock.base_url);
    cfg.groups = vec!["none/none".to_string()]; // no throughput phase traffic

    let report = run_benchmark(&cfg).await.unwrap();

    assert_eq!(report.warmup.requests, 30);
    assert_eq!(report.latency["keyword/keyword"].count, 30);
    assert_eq!(mock.hits.load(Ordering::SeqCst), 60, "one warm-up pass + one latency pass");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn throughput_steps_and_max_qps() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("steps");
    write_group(&dir, "keyword/keyword", &[(10, 200); 20]);

    let report = run_benchmark(&config(dir, &mock.base_url)).await.unwrap();

    let t = &report.throughput["keyword/keyword"];
    assert_eq!(t.steps.iter().map(|s| s.concurrency).collect::<Vec<_>>(), [1, 2]);
    assert!(t.steps.iter().all(|s| s.valid));
    assert_eq!(t.max_qps_concurrency, Some(2));
    assert_eq!(t.max_qps, Some(t.steps[1].qps));
    assert!(t.steps[1].qps > t.steps[0].qps);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn errors_invalidate_the_step_and_stop_stepping() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("failing");
    let mut specs = vec![(1, 200); 95];
    specs.extend([(1, 503); 5]);
    write_group(&dir, "keyword/keyword", &specs);

    let report = run_benchmark(&config(dir, &mock.base_url)).await.unwrap();

    let t = &report.throughput["keyword/keyword"];
    assert_eq!(t.steps.len(), 1, "stepping stops at the first invalid step");
    assert!(!t.steps[0].valid && t.steps[0].error_rate > 0.01);
    assert_eq!((t.max_qps, t.max_qps_concurrency), (None, None));
    assert_eq!(report.latency["keyword/keyword"].errors, 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn missing_throughput_group_is_skipped() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("skipped");
    write_group(&dir, "semantic/semantic", &[(1, 200); 5]);
    let mut cfg = config(dir, &mock.base_url);
    cfg.groups = vec!["keyword/keyword".to_string(), "semantic/semantic".to_string()];

    let report = run_benchmark(&cfg).await.unwrap();

    assert_eq!(report.skipped_groups, ["keyword/keyword"]);
    assert!(report.throughput.contains_key("semantic/semantic"));
    assert!(!report.throughput.contains_key("keyword/keyword"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cli_writes_the_report() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("cli");
    write_group(&dir, "keyword/keyword", &[(1, 200); 5]);
    let out = dir.join("perf.json");
    let base_url = mock.base_url.clone();
    let requests = dir.clone();
    let out_path = out.clone();
    let status = tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_irbench-loadgen"))
            .args(["--requests", requests.to_str().unwrap(), "--base-url", &base_url])
            .args(["--out", out_path.to_str().unwrap(), "--concurrency", "1", "--step-seconds", "0.2"])
            .status()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(status.success());
    let report: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    assert_eq!(report["latency"]["keyword/keyword"]["count"], 5);
    assert_eq!(report["limits"]["p99_ms"], 100.0);
    assert_eq!(report["throughput"]["keyword/keyword"]["steps"][0]["concurrency"], 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path loadgen/Cargo.toml --test benchmark 2>&1 | tail -15`
Expected: FAIL to compile — `unresolved import irbench_loadgen::benchmark`.

- [ ] **Step 3: Implement**

`loadgen/src/benchmark.rs` (and `pub mod benchmark;` in `lib.rs`):

```rust
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde::Serialize;

use crate::requests::{PreparedRequest, load_requests};
use crate::runner::{Stop, Target, run};
use crate::stats::{LatencySummary, Limits, Recorder, StepResult, max_qps};

pub struct Config {
    pub requests_dir: PathBuf,
    pub base_url: String,
    pub concurrency: Vec<usize>,
    /// Groups that get a throughput run; empty means every group.
    pub groups: Vec<String>,
    pub step_seconds: f64,
    pub limits: Limits,
    pub timeout: Duration,
    pub warmup_concurrency: usize,
}

#[derive(Debug, Serialize)]
pub struct PhaseInfo {
    pub requests: u64,
    pub errors: u64,
    pub seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct GroupThroughput {
    pub steps: Vec<StepResult>,
    pub max_qps: Option<f64>,
    pub max_qps_concurrency: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub base_url: String,
    pub requests: usize,
    pub limits: Limits,
    pub warmup: PhaseInfo,
    pub latency: BTreeMap<String, LatencySummary>,
    pub throughput: BTreeMap<String, GroupThroughput>,
    pub skipped_groups: Vec<String>,
}

/// Spec §5.7: warm-up pass (discarded), latency pass at concurrency 1 per group,
/// then stepped closed-loop throughput per group until a step breaks the limits.
pub async fn run_benchmark(cfg: &Config) -> Result<Report> {
    let all = load_requests(&cfg.requests_dir)?;
    let max_conn = cfg.concurrency.iter().copied().max().unwrap_or(1).max(cfg.warmup_concurrency);
    let target = Arc::new(Target::new(&cfg.base_url, cfg.timeout, max_conn)?);

    let mut by_group: BTreeMap<String, Vec<PreparedRequest>> = BTreeMap::new();
    for req in &all {
        by_group.entry(req.group.clone()).or_default().push(req.clone());
    }

    let (warm, warm_seconds) =
        run(target.clone(), Arc::new(all.clone()), cfg.warmup_concurrency, Stop::Pass).await;
    eprintln!("warm-up: {} requests, {} errors, {warm_seconds:.1}s", warm.total(), warm.errors());

    let mut latency = BTreeMap::new();
    let mut overall = Recorder::new();
    for (group, requests) in &by_group {
        let (rec, _) = run(target.clone(), Arc::new(requests.clone()), 1, Stop::Pass).await;
        let summary = rec.summary();
        eprintln!("latency {group}: p50={:.2}ms p99={:.2}ms errors={}", summary.p50_ms, summary.p99_ms, summary.errors);
        overall.merge(&rec);
        latency.insert(group.clone(), summary);
    }
    latency.insert("all".to_string(), overall.summary());

    let wanted: Vec<String> =
        if cfg.groups.is_empty() { by_group.keys().cloned().collect() } else { cfg.groups.clone() };
    let mut throughput = BTreeMap::new();
    let mut skipped_groups = Vec::new();
    let step = Duration::from_secs_f64(cfg.step_seconds);
    for group in wanted {
        let Some(requests) = by_group.get(&group) else {
            skipped_groups.push(group);
            continue;
        };
        let requests = Arc::new(requests.clone());
        let mut steps = Vec::new();
        for &concurrency in &cfg.concurrency {
            let (rec, seconds) = run(target.clone(), requests.clone(), concurrency, Stop::Deadline(step)).await;
            let result = StepResult::new(concurrency, seconds, &rec, &cfg.limits);
            eprintln!(
                "throughput {group} c={concurrency}: {:.0} qps p99={:.2}ms errors={:.3}% valid={}",
                result.qps, result.p99_ms, result.error_rate * 100.0, result.valid
            );
            let valid = result.valid;
            steps.push(result);
            if !valid {
                break;
            }
        }
        let best = max_qps(&steps).map(|s| (s.qps, s.concurrency));
        throughput.insert(
            group,
            GroupThroughput {
                max_qps: best.map(|(qps, _)| qps),
                max_qps_concurrency: best.map(|(_, c)| c),
                steps,
            },
        );
    }

    Ok(Report {
        base_url: cfg.base_url.clone(),
        requests: all.len(),
        limits: cfg.limits.clone(),
        warmup: PhaseInfo { requests: warm.total(), errors: warm.errors(), seconds: warm_seconds },
        latency,
        throughput,
        skipped_groups,
    })
}
```

`loadgen/src/main.rs`:

```rust
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use irbench_loadgen::benchmark::{Config, run_benchmark};
use irbench_loadgen::stats::Limits;

/// Replay IRBench rendered requests against one engine and write perf.json.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// The run's `rendered/` directory (`<mode>/<base>/<qid>.json`).
    #[arg(long)]
    requests: PathBuf,
    /// Engine base URL, e.g. http://localhost:17700.
    #[arg(long)]
    base_url: String,
    /// Where to write the JSON report.
    #[arg(long)]
    out: PathBuf,
    /// Throughput concurrency steps.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32,64")]
    concurrency: Vec<usize>,
    /// Groups (`<mode>/<base>`) that get a throughput run; empty means all.
    #[arg(long, value_delimiter = ',')]
    groups: Vec<String>,
    #[arg(long, default_value_t = 20.0)]
    step_seconds: f64,
    #[arg(long, default_value_t = 100.0)]
    p99_limit_ms: f64,
    #[arg(long, default_value_t = 0.001)]
    max_error_rate: f64,
    #[arg(long, default_value_t = 10.0)]
    timeout_s: f64,
    #[arg(long, default_value_t = 4)]
    warmup_concurrency: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config {
        requests_dir: args.requests,
        base_url: args.base_url,
        concurrency: args.concurrency,
        groups: args.groups.into_iter().filter(|g| !g.is_empty()).collect(),
        step_seconds: args.step_seconds,
        limits: Limits { p99_ms: args.p99_limit_ms, max_error_rate: args.max_error_rate },
        timeout: Duration::from_secs_f64(args.timeout_s),
        warmup_concurrency: args.warmup_concurrency,
    };
    let report = run_benchmark(&config).await?;
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    Ok(())
}
```

`loadgen/README.md`:

````markdown
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
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --manifest-path loadgen/Cargo.toml 2>&1 | grep -E "test result|FAILED|panicked" && cargo fmt --manifest-path loadgen/Cargo.toml --check && cargo clippy --manifest-path loadgen/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -3`
Expected: every `test result: ok`: 9 unit tests, 5 runner tests, 6 benchmark tests; fmt and clippy clean.

- [ ] **Step 5: Build the release binary**

Run: `cargo build --release --manifest-path loadgen/Cargo.toml 2>&1 | tail -2 && loadgen/target/release/irbench-loadgen --help | head -5`
Expected: `Finished release`; the help text starts with "Replay IRBench rendered requests against one engine and write perf.json".

- [ ] **Step 6: Commit**

```bash
git add loadgen
git commit -m "feat: loadgen benchmark phases, perf.json report and CLI"
```

---

### Task 5: The `perf` stage in the harness

**Files:**
- Create: `harness/irbench/stages/perf.py`, `harness/tests/test_perf_stage.py`
- Modify: `harness/irbench/scenario.py`, `harness/irbench/runs.py`, `harness/irbench/stages/run.py`, `harness/irbench/stages/score_run.py`, `harness/irbench/cli.py`, `harness/tests/fake_engine.py`, `harness/tests/test_run_pipeline.py`, `scenarios/ecommerce/product-search/scenario.yaml`

**Interfaces:**
- Consumes: `RunContext` (Plan 2: `rendered_dir`, `base_url`, `run_dir`, `scenario`), `HardwareProfile.loadgen.cpuset` (Plan 2), `run_command` / `Runner` (Plan 2 `runtime/docker.py`), and the `perf.json` shape from Task 4.
- Produces:
  - `PerfSpec` (pydantic): `concurrency: list[int] = [1, 2, 4, 8, 16, 32, 64]`, `step_seconds: float = 20.0`, `p99_limit_ms: float = 100.0`, `max_error_rate: float = 0.001`, `throughput_groups: list[str] = ["keyword/keyword", "semantic/semantic", "hybrid/hybrid"]`, `warmup_concurrency: int = 4`. Exposed as `Scenario.perf: PerfSpec = PerfSpec()`.
  - `STAGES = ("prepare", "embed", "up", "index", "render", "relevance", "perf", "score", "down")`. `STAGE_OUTPUTS["perf"] = ("perf.json",)`.
  - `irbench.stages.perf`:
    - `LOADGEN_PATH = Path("loadgen/target/release/irbench-loadgen")`
    - `loadgen_command(ctx, profile, taskset: str | None) -> list[str]`
    - `perf_stage(ctx, profile, runner=run_command) -> dict`, which writes `perf.json` through the binary and returns `summarize_perf(report)`
    - `summarize_perf(report) -> dict[str, dict]`: per group `{p50_ms, p95_ms, p99_ms, mean_ms, errors, max_qps, max_qps_concurrency}`; a group without a throughput run has `max_qps` = None.
  - `run_pipeline(..., loadgen_runner: Runner | None = None)`.
  - `scores.json` gains `"perf": summarize_perf(perf.json)` when `perf.json` exists, else `null`.
  - The CLI prints `perf <group>: p50=… p99=… max_qps=… (c=…)` per group.
  - Test helpers: `fake_loadgen(report=None) -> Runner`; `fake_root` also creates an empty file at `LOADGEN_PATH`.

- [ ] **Step 1: Write the failing tests**

Append to `harness/tests/fake_engine.py`:

```python
SAMPLE_PERF = {
    "base_url": "http://localhost:9999",
    "requests": 10,
    "limits": {"p99_ms": 100.0, "max_error_rate": 0.001},
    "warmup": {"requests": 10, "errors": 0, "seconds": 0.1},
    "latency": {
        "keyword/keyword": {"count": 5, "errors": 0, "mean_ms": 2.0, "p50_ms": 1.5,
                            "p95_ms": 3.0, "p99_ms": 4.0, "max_ms": 4.0},
        "all": {"count": 10, "errors": 0, "mean_ms": 2.5, "p50_ms": 2.0,
                "p95_ms": 4.0, "p99_ms": 5.0, "max_ms": 6.0},
    },
    "throughput": {
        "keyword/keyword": {"steps": [], "max_qps": 812.5, "max_qps_concurrency": 16},
    },
    "skipped_groups": ["hybrid/hybrid"],
}


def fake_loadgen(report=None, calls=None):
    """A Runner standing in for the Rust binary: writes perf.json where --out points."""

    def run(cmd):
        if calls is not None:
            calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report or SAMPLE_PERF))
        return ""

    return run
```

Add `from pathlib import Path` and `from irbench.stages.perf import LOADGEN_PATH` to its imports. At the end of `fake_root`, before `return root`:

```python
    binary = root / LOADGEN_PATH
    binary.parent.mkdir(parents=True)
    binary.write_text("")
```

Create `harness/tests/test_perf_stage.py`:

```python
import pytest

from conftest import REPO_ROOT
from fake_engine import FAKE_SPEC, SAMPLE_PERF, fake_loadgen, fake_root
from irbench.io import read_json
from irbench.runtime.hardware import load_hardware
from irbench.scenario import load_scenario
from irbench.stages.context import RunContext
from irbench.stages.perf import LOADGEN_PATH, loadgen_command, perf_stage, summarize_perf
from irbench.stages.render import render_stage

LOCAL = load_hardware(REPO_ROOT, "local")
REFERENCE = load_hardware(REPO_ROOT, "aws-c7i-4xlarge")


def test_scenario_declares_the_spec_perf_protocol():
    p = load_scenario(REPO_ROOT, "ecommerce/product-search").perf
    assert p.concurrency == [1, 2, 4, 8, 16, 32, 64]
    assert (p.p99_limit_ms, p.max_error_rate) == (100.0, 0.001)
    assert p.throughput_groups == ["keyword/keyword", "semantic/semantic", "hybrid/hybrid"]


def test_command_replays_the_rendered_dir_with_scenario_settings(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    cmd = loadgen_command(ctx, LOCAL, taskset=None)
    assert cmd[0] == str(ctx.root / LOADGEN_PATH)
    args = dict(zip(cmd[1::2], cmd[2::2]))
    assert args["--requests"] == str(ctx.rendered_dir)
    assert args["--base-url"] == "http://localhost:9999"
    assert args["--out"] == str(ctx.run_dir / "perf.json")
    assert args["--concurrency"] == "1,2,4,8,16,32,64"
    assert args["--groups"] == "keyword/keyword,semantic/semantic,hybrid/hybrid"
    assert (args["--p99-limit-ms"], args["--max-error-rate"], args["--timeout-s"]) == (
        "100.0", "0.001", "10.0",
    )


def test_reference_profile_pins_the_loadgen_cpuset(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    cmd = loadgen_command(ctx, REFERENCE, taskset="/usr/bin/taskset")
    assert cmd[:3] == ["/usr/bin/taskset", "-c", "8-11"]


def test_pinned_profile_without_taskset_fails(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    with pytest.raises(RuntimeError, match="taskset"):
        loadgen_command(ctx, REFERENCE, taskset=None)


def test_perf_stage_runs_the_binary_and_summarizes(tmp_path, wands_raw):
    ctx = RunContext(fake_root(tmp_path, wands_raw), FAKE_SPEC)
    render_stage(ctx)
    calls = []
    summary = perf_stage(ctx, LOCAL, runner=fake_loadgen(calls=calls))
    assert len(calls) == 1
    assert read_json(ctx.run_dir / "perf.json") == SAMPLE_PERF
    assert summary["keyword/keyword"] == {
        "p50_ms": 1.5, "p95_ms": 3.0, "p99_ms": 4.0, "mean_ms": 2.0, "errors": 0,
        "max_qps": 812.5, "max_qps_concurrency": 16,
    }
    assert summary["all"]["max_qps"] is None


def test_perf_needs_rendered_requests_and_the_binary(tmp_path, wands_raw):
    root = fake_root(tmp_path, wands_raw)
    ctx = RunContext(root, FAKE_SPEC)
    with pytest.raises(FileNotFoundError, match="render"):
        perf_stage(ctx, LOCAL, runner=fake_loadgen())
    render_stage(ctx)
    (root / LOADGEN_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="cargo build --release"):
        perf_stage(ctx, LOCAL, runner=fake_loadgen())


def test_summary_of_a_group_without_throughput():
    report = {**SAMPLE_PERF, "throughput": {}}
    assert summarize_perf(report)["keyword/keyword"]["max_qps"] is None
```

In `harness/tests/test_run_pipeline.py`:
- Import `fake_loadgen` from `fake_engine`.
- Pass `loadgen_runner=fake_loadgen()` to `run_pipeline` in `test_pipeline_end_to_end_with_fake_engine`.
- Append to that test:

```python
    assert manifest["stages"]["perf"]["result"]["keyword/keyword"]["max_qps"] == 812.5
    assert scores["perf"]["keyword/keyword"]["max_qps_concurrency"] == 16
```

Then append a CLI test:

```python
def test_cli_run_prints_perf(tmp_path, wands_raw, monkeypatch):
    root = fake_root(tmp_path, wands_raw)
    monkeypatch.setattr("irbench.stages.run.DockerRuntime", lambda *a: FakeRuntime())
    monkeypatch.setattr("irbench.stages.run.default_client", _server(root).client_factory)
    monkeypatch.setattr("irbench.stages.run.run_command", fake_loadgen())
    args = ["run", "--scenario", SCENARIO, "--engine", "fake", "--model", "fake"]
    args += ["--stages", "up,index,render,relevance,perf,score,down", "--root", str(root)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "perf keyword/keyword: p50=1.50ms p99=4.00ms max_qps=812 (c=16)" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest harness/tests/test_perf_stage.py harness/tests/test_run_pipeline.py -q 2>&1 | grep -E "^E |passed|failed" | head -5`
Expected: FAIL — `ModuleNotFoundError: No module named 'irbench.stages.perf'` (raised while importing `fake_engine`).

- [ ] **Step 3: Implement**

`harness/irbench/scenario.py`: add before `class Scenario`, and add `perf: PerfSpec = PerfSpec()` after `max_error_rate` in `Scenario`:

```python
class PerfSpec(BaseModel):
    """Spec §5.7 load protocol, identical for every engine."""

    concurrency: list[int] = [1, 2, 4, 8, 16, 32, 64]
    step_seconds: float = 20.0
    p99_limit_ms: float = 100.0
    max_error_rate: float = 0.001
    throughput_groups: list[str] = ["keyword/keyword", "semantic/semantic", "hybrid/hybrid"]
    warmup_concurrency: int = 4
```

`scenarios/ecommerce/product-search/scenario.yaml`: append

```yaml
# Load protocol for the perf stage (spec §5.7), identical for every engine.
perf:
  concurrency: [1, 2, 4, 8, 16, 32, 64]
  step_seconds: 20
  p99_limit_ms: 100
  max_error_rate: 0.001
  # Throughput runs on the retrieval bases; latency covers every mode.
  throughput_groups: [keyword/keyword, semantic/semantic, hybrid/hybrid]
  warmup_concurrency: 4
```

`harness/irbench/runs.py`:

```python
STAGES = ("prepare", "embed", "up", "index", "render", "relevance", "perf", "score", "down")
```

`harness/irbench/stages/perf.py`:

```python
import shutil
from pathlib import Path
from typing import Any

from irbench.io import read_json
from irbench.runtime.docker import Runner, run_command
from irbench.runtime.hardware import HardwareProfile
from irbench.stages.context import RunContext

LOADGEN_PATH = Path("loadgen/target/release/irbench-loadgen")
BUILD_HINT = "cargo build --release --manifest-path loadgen/Cargo.toml"


def loadgen_command(ctx: RunContext, profile: HardwareProfile, taskset: str | None) -> list[str]:
    p = ctx.scenario.perf
    cmd = [
        str(ctx.root / LOADGEN_PATH),
        "--requests", str(ctx.rendered_dir),
        "--base-url", ctx.base_url,
        "--out", str(ctx.run_dir / "perf.json"),
        "--concurrency", ",".join(str(c) for c in p.concurrency),
        "--groups", ",".join(p.throughput_groups),
        "--step-seconds", str(p.step_seconds),
        "--p99-limit-ms", str(p.p99_limit_ms),
        "--max-error-rate", str(p.max_error_rate),
        "--timeout-s", str(ctx.scenario.request_timeout_s),
        "--warmup-concurrency", str(p.warmup_concurrency),
    ]
    cpuset = profile.loadgen.cpuset
    if cpuset:
        if taskset is None:
            raise RuntimeError(
                f"hardware profile {profile.id!r} pins the loadgen to CPUs {cpuset}, but taskset "
                "is not available (Linux only); refusing to measure unpinned"
            )
        cmd = [taskset, "-c", cpuset, *cmd]
    return cmd


def summarize_perf(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = {}
    for group, lat in report["latency"].items():
        t = report["throughput"].get(group, {})
        summary[group] = {
            "p50_ms": lat["p50_ms"],
            "p95_ms": lat["p95_ms"],
            "p99_ms": lat["p99_ms"],
            "mean_ms": lat["mean_ms"],
            "errors": lat["errors"],
            "max_qps": t.get("max_qps"),
            "max_qps_concurrency": t.get("max_qps_concurrency"),
        }
    return summary


def perf_stage(
    ctx: RunContext, profile: HardwareProfile, runner: Runner = run_command
) -> dict[str, dict[str, Any]]:
    """Replay the rendered requests with the Rust loadgen (spec §5.7)."""
    if not ctx.rendered_dir.exists():
        raise FileNotFoundError(f"{ctx.rendered_dir} missing: run the render stage first")
    binary = ctx.root / LOADGEN_PATH
    if not binary.exists():
        raise FileNotFoundError(f"{binary} missing: build it with `{BUILD_HINT}`")
    runner(loadgen_command(ctx, profile, shutil.which("taskset")))
    return summarize_perf(read_json(ctx.run_dir / "perf.json"))
```

`harness/irbench/stages/run.py`:
- Import `Runner` alongside `DockerRuntime, Runtime`, import `run_command`, and import `perf_stage` from `irbench.stages.perf`.
- Add `"perf": ("perf.json",),` to `STAGE_OUTPUTS`, between `relevance` and `score`.
- Add the keyword parameter `loadgen_runner: Runner | None = None` after `embedder`.
- In `execute`, before the `score` branch:

```python
        if stage == "perf":
            return perf_stage(ctx, profile, runner=loadgen_runner or run_command)
```

`harness/irbench/stages/score_run.py`: import `summarize_perf` from `irbench.stages.perf`. Add to `report` after `"errors"`:

```python
        "perf": summarize_perf(read_json(run_dir / "perf.json"))
        if (run_dir / "perf.json").exists()
        else None,
```

`harness/irbench/cli.py`: in `run`, after the constraint-violation loop, inside the `score succeeded` block:

```python
        for group, p in (scores.get("perf") or {}).items():
            if group == "all":
                continue
            qps = "n/a" if p["max_qps"] is None else f"{p['max_qps']:.0f} (c={p['max_qps_concurrency']})"
            typer.echo(
                f"perf {group}: p50={p['p50_ms']:.2f}ms p99={p['p99_ms']:.2f}ms max_qps={qps}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q 2>&1 | tail -2 && uv run ruff format harness >/dev/null && uv run ruff check . && uv run ruff format --check harness engines`
Expected: all pass (147 before this task + 7 in `test_perf_stage.py` + 1 new CLI test = 155 passed); ruff clean.

The monkeypatch target `irbench.stages.run.run_command` only works because `run.py` imports the name `run_command` and looks it up when `execute` runs. Keep `runner=loadgen_runner or run_command` inside `execute`, not as a default argument.

- [ ] **Step 5: Commit**

```bash
git add harness scenarios
git commit -m "feat: perf stage replaying rendered requests with the Rust loadgen"
```

---

### Task 6: Real perf runs on all three engines

**Files:** none expected. A run that exposes a bug gets fixed test-first in the task that owns the code, with a ruling recorded in the ledger.

**Interfaces:**
- Consumes: the whole pipeline; the cached `bge-small-en-v1.5` vectors; the release binary from Task 4.
- Produces: `perf.json` and `scores.json` with a `perf` section for the three runs under `.irbench/runs/ecommerce__product-search__<engine>__bge-small-en-v1.5__default__local/`.

- [ ] **Step 1: Build and run**

```bash
cargo build --release --manifest-path loadgen/Cargo.toml
for e in meilisearch elasticsearch qdrant; do
  uv run irbench run --scenario ecommerce/product-search --engine $e --model bge-small-en-v1.5 \
    --stages up,index,render,relevance,perf,score,down 2>&1 | tail -25
done
```

Run it in the background (expect 20–40 minutes in total). Expected: every stage `succeeded` for each engine; each prints `perf <group>: …` lines for every group the engine supports. Qdrant's `perf.json` lists `keyword/keyword` and `hybrid/hybrid` under `skipped_groups`.

- [ ] **Step 2: Sanity-check**

```bash
uv run python - <<'EOF'
import json, pathlib
for d in sorted(pathlib.Path(".irbench/runs").glob("ecommerce__product-search__*__bge-small-en-v1.5__default__local")):
    p = json.loads((d / "perf.json").read_text())
    print(f"== {d.name.split('__')[2]}  requests={p['requests']}  warmup errors={p['warmup']['errors']}  skipped={p['skipped_groups']}")
    for g, lat in p["latency"].items():
        t = p["throughput"].get(g)
        qps = "" if t is None else f" max_qps={t['max_qps']} (c={t['max_qps_concurrency']}) steps={[(s['concurrency'], round(s['qps']), round(s['p99_ms'], 1), s['valid']) for s in t['steps']]}"
        print(f"  {g:20} n={lat['count']} err={lat['errors']} p50={lat['p50_ms']:.2f} p99={lat['p99_ms']:.2f}{qps}")
EOF
```

Expected:
- **Request count:** equals the number of non-check rendered files for that engine.
- **Errors:** latency and warm-up errors are 0. Any error is a finding; report it.
- **Latency:** p50 at concurrency 1 is below the 100 ms limit for the base modes.
- **Throughput:** QPS rises from c=1 to c=2.
- **Stopping rule:** every throughput group either reaches c=64 or stops at its first invalid step with the reason visible (p99 or errors).

On this local machine the loadgen and the engines share CPUs, so these numbers check that the method works. They are not publishable. Record the table in the final message, labelled `local`.

- [ ] **Step 3: Report**

No commit. Put the Step 2 table, any non-zero error counts and any rulings in the final message.

---

## Self-Review

**Spec coverage**

| Spec section | Where |
|---|---|
| §2 "requests as data": Rust load generator replays the identical requests | Tasks 1, 5 (reads `rendered/`, skips check fetches) |
| §4 `loadgen/` Rust crate, standalone binary invoked by the harness | Tasks 1–5 |
| §5.7 warm-up pass discarded | Task 4 `warmup_is_excluded_from_latency` |
| §5.7 latency at concurrency 1: p50/p95/p99/mean | Tasks 2, 4 |
| §5.7 throughput steps 1…64; max QPS = highest step with p99 ≤ 100 ms, error rate ≤ 0.1% | Tasks 2, 4, scenario `perf` block in Task 5 |
| §8 loadgen pinned to a disjoint 4-vCPU cpuset | Task 5 (`taskset -c` from `loadgen.cpuset`, fails if unavailable) |
| §9 perf: errors counted; a step above 0.1% errors is invalid | Tasks 2, 3, 4 |
| §10 loadgen test: mock server with injected latency distribution, percentiles within tolerance | Tasks 3, 4 (axum mock, per-request delay/status) |
| §11 report: latency percentiles, QPS curve | Task 5 writes `perf.json` steps (the curve) and the `scores.json` perf summary; HTML is Plan 4 |
| Pinecone relevance-only | Not reachable in v0.1 (no Pinecone adapter); the perf stage has no special case to remove later |
| §5 Postgres wire protocol in loadgen | Later (no Postgres adapter in v0.1); `load_requests` rejects non-`http` protocols explicitly |

**Spec interpretation, decided here (ledger it when Task 4 runs):** "max QPS is the highest step with p99 ≤ 100 ms and error rate ≤ 0.1%". Stepping stops at the first invalid step, so max QPS is the QPS of the last valid step before it. Higher steps are not run, because their results under an already-saturated engine aren't meaningful and running them costs minutes per group.

**Placeholder scan.** Every code step has its code. Task 1's reqwest feature fallback and Task 6's outcomes depend on reality; each states the exact action and the ruling format.

**Type consistency.**
- `PreparedRequest`, `Recorder`, `Limits`, `StepResult`, `max_qps`, `Target`, `Stop`, `run`, `Config`, `Report`, `GroupThroughput` and `PhaseInfo` keep the same names and fields across Tasks 1–4.
- The JSON field names Task 5 reads (`latency.<group>.p50_ms/p95_ms/p99_ms/mean_ms/errors`, `throughput.<group>.max_qps/max_qps_concurrency`, `skipped_groups`) match the Task 4 `Serialize` structs.
- The CLI flags in `loadgen_command` match the clap `Args` field names in kebab case.
