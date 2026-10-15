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
        limits: Limits {
            p99_ms: args.p99_limit_ms,
            max_error_rate: args.max_error_rate,
        },
        timeout: Duration::from_secs_f64(args.timeout_s),
        warmup_concurrency: args.warmup_concurrency,
    };
    let report = run_benchmark(&config).await?;
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing {}", args.out.display()))?;
    if !report.failures.is_empty() {
        for failure in &report.failures {
            eprintln!("FAILED {failure}");
        }
        std::process::exit(2);
    }
    Ok(())
}
