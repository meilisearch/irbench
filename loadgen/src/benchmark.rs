use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use serde::Serialize;

use crate::requests::{Kind, PreparedRequest, load_requests};
use crate::runner::{Stop, Target, protocol_for_url, run};
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
    /// Phases where the engine did not answer reliably. Non-empty means the
    /// measurement is not valid and the CLI exits non-zero.
    pub failures: Vec<String>,
}

/// An engine that errors above the limit (or answers nothing) in a phase that
/// should be easy is broken, not slow.
fn check_phase(name: &str, rec: &Recorder, limits: &Limits, failures: &mut Vec<String>) {
    if rec.successes() == 0 {
        failures.push(format!(
            "{name}: no successful requests ({} errors)",
            rec.errors()
        ));
    } else if rec.error_rate() > limits.max_error_rate {
        failures.push(format!(
            "{name}: error rate {:.3}% > {:.3}% ({} errors)",
            rec.error_rate() * 100.0,
            limits.max_error_rate * 100.0,
            rec.errors()
        ));
    }
}

/// Spec §5.7: warm-up pass (discarded), latency pass at concurrency 1 per group,
/// then stepped closed-loop throughput per group until a step breaks the limits.
pub async fn run_benchmark(cfg: &Config) -> Result<Report> {
    let all = load_requests(&cfg.requests_dir)?;
    let max_conn = cfg
        .concurrency
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(cfg.warmup_concurrency);
    let url_protocol = protocol_for_url(&cfg.base_url);
    if let Some(req) = all.iter().find(|r| r.kind.protocol() != url_protocol) {
        bail!(
            "rendered requests use protocol {} but --base-url selects the {} protocol",
            req.kind.protocol(),
            url_protocol
        );
    }
    let target = Arc::new(Target::connect(&cfg.base_url, cfg.timeout, max_conn).await?);
    let mut sqls: Vec<String> = all
        .iter()
        .filter_map(|r| match &r.kind {
            Kind::Postgres { sql, .. } => Some(sql.clone()),
            Kind::Http { .. } => None,
        })
        .collect();
    sqls.sort();
    sqls.dedup();
    target.prime(&sqls).await?;

    let mut by_group: BTreeMap<String, Vec<PreparedRequest>> = BTreeMap::new();
    for req in &all {
        by_group
            .entry(req.group.clone())
            .or_default()
            .push(req.clone());
    }

    let mut failures = Vec::new();
    let (warm, warm_seconds) = run(
        target.clone(),
        Arc::new(all.clone()),
        cfg.warmup_concurrency,
        Stop::Pass,
    )
    .await;
    eprintln!(
        "warm-up: {} requests, {} errors, {warm_seconds:.1}s",
        warm.total(),
        warm.errors()
    );
    check_phase("warm-up", &warm, &cfg.limits, &mut failures);

    let mut latency = BTreeMap::new();
    let mut overall = Recorder::new();
    for (group, requests) in &by_group {
        let (rec, _) = run(target.clone(), Arc::new(requests.clone()), 1, Stop::Pass).await;
        let summary = rec.summary();
        eprintln!(
            "latency {group}: p50={:?}ms p99={:?}ms errors={}",
            summary.p50_ms, summary.p99_ms, summary.errors
        );
        check_phase(
            &format!("latency {group}"),
            &rec,
            &cfg.limits,
            &mut failures,
        );
        overall.merge(&rec);
        latency.insert(group.clone(), summary);
    }
    latency.insert("all".to_string(), overall.summary());

    let wanted: Vec<String> = if cfg.groups.is_empty() {
        by_group.keys().cloned().collect()
    } else {
        cfg.groups.clone()
    };
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
            let (rec, seconds) = run(
                target.clone(),
                requests.clone(),
                concurrency,
                Stop::Deadline(step),
            )
            .await;
            let result = StepResult::new(concurrency, seconds, &rec, &cfg.limits);
            eprintln!(
                "throughput {group} c={concurrency}: {:.0} qps p99={:?}ms errors={:.3}% {}",
                result.qps,
                result.p99_ms,
                result.error_rate * 100.0,
                result.invalid_reason.as_deref().unwrap_or("valid")
            );
            if steps.is_empty() {
                // The lowest step failing on errors means the engine broke, not saturated.
                check_phase(
                    &format!("throughput {group} c={concurrency}"),
                    &rec,
                    &cfg.limits,
                    &mut failures,
                );
            }
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
        warmup: PhaseInfo {
            requests: warm.total(),
            errors: warm.errors(),
            seconds: warm_seconds,
        },
        latency,
        throughput,
        skipped_groups,
        failures,
    })
}
