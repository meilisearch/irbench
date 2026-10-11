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
        limits: Limits {
            p99_ms: 100.0,
            max_error_rate: 0.001,
        },
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
    assert_eq!(
        mock.hits.load(Ordering::SeqCst),
        60,
        "one warm-up pass + one latency pass"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn throughput_steps_and_max_qps() {
    let mock = spawn_mock().await;
    let dir = tmp_dir("steps");
    write_group(&dir, "keyword/keyword", &[(10, 200); 20]);

    let report = run_benchmark(&config(dir, &mock.base_url)).await.unwrap();

    let t = &report.throughput["keyword/keyword"];
    assert_eq!(
        t.steps.iter().map(|s| s.concurrency).collect::<Vec<_>>(),
        [1, 2]
    );
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
    cfg.groups = vec![
        "keyword/keyword".to_string(),
        "semantic/semantic".to_string(),
    ];

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
            .args([
                "--requests",
                requests.to_str().unwrap(),
                "--base-url",
                &base_url,
            ])
            .args([
                "--out",
                out_path.to_str().unwrap(),
                "--concurrency",
                "1",
                "--step-seconds",
                "0.2",
            ])
            .status()
            .unwrap()
    })
    .await
    .unwrap();
    assert!(status.success());
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap();
    assert_eq!(report["latency"]["keyword/keyword"]["count"], 5);
    assert_eq!(report["limits"]["p99_ms"], 100.0);
    assert_eq!(
        report["throughput"]["keyword/keyword"]["steps"][0]["concurrency"],
        1
    );
}
