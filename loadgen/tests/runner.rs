mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use common::{spawn_mock, tmp_dir, write_group};
use irbench_loadgen::requests::{PreparedRequest, load_requests};
use irbench_loadgen::runner::{Stop, Target, run};

async fn setup(
    name: &str,
    specs: &[(u64, u16)],
) -> (common::Mock, Arc<Target>, Arc<Vec<PreparedRequest>>) {
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
    assert!((5.0..15.0).contains(&s.p50_ms.unwrap()), "p50 {s:?}");
    assert!((50.0..90.0).contains(&s.p95_ms.unwrap()), "p95 {s:?}");
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
    assert!((1.0..1.5).contains(&seconds), "seconds {seconds}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_above_request_count_cycles_without_panicking() {
    let (_mock, target, requests) = setup("tiny", &[(1, 200); 3]).await;
    let (pass, _) = run(target.clone(), requests.clone(), 64, Stop::Pass).await;
    assert_eq!(pass.total(), 3);
    let (step, seconds) = run(
        target,
        requests,
        64,
        Stop::Deadline(Duration::from_millis(300)),
    )
    .await;
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
