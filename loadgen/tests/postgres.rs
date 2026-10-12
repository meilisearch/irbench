//! Live Postgres test: IRBENCH_PG_URL=postgresql://postgres:irbench@localhost:15432/irbench cargo test -- --ignored
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn postgres_target_runs_prepared_statements() {
    let Ok(url) = std::env::var("IRBENCH_PG_URL") else {
        return;
    };
    let target = irbench_loadgen::runner::Target::connect(&url, Duration::from_secs(5), 2)
        .await
        .unwrap();
    let ok = irbench_loadgen::requests::PreparedRequest::postgres_for_test(
        "x/x",
        "SELECT $1::text::int + 1 AS n",
        vec![Some("41".into())],
    );
    let bad = irbench_loadgen::requests::PreparedRequest::postgres_for_test(
        "x/x",
        "SELECT * FROM no_such_table",
        vec![],
    );
    assert!(target.send(0, &ok).await);
    assert!(target.send(1, &ok).await);
    assert!(!target.send(0, &bad).await);
    assert!(
        target.send(0, &ok).await,
        "connection still usable after an error"
    );
}
