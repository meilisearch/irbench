//! Live Postgres test: IRBENCH_PG_URL=postgresql://postgres:irbench@localhost:15432/irbench cargo test -- --ignored
use std::time::Duration;

/// The kill test terminates every loadgen backend, so live tests run one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
#[ignore]
async fn postgres_target_runs_prepared_statements() {
    let _serial = SERIAL.lock().await;
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

fn pg_url() -> Option<String> {
    std::env::var("IRBENCH_PG_URL").ok()
}

#[tokio::test]
#[ignore]
async fn dead_connection_is_replaced_after_one_failed_send() {
    let _serial = SERIAL.lock().await;
    use irbench_loadgen::requests::PreparedRequest;
    let Some(url) = pg_url() else { return };
    let target = irbench_loadgen::runner::Target::connect(&url, Duration::from_secs(5), 2)
        .await
        .unwrap();
    let ok = PreparedRequest::postgres_for_test(
        "x/x",
        "SELECT $1::text::int + 1 AS n",
        vec![Some("41".into())],
    );
    assert!(target.send(0, &ok).await);
    assert!(target.send(1, &ok).await);

    let (admin, conn) = tokio_postgres::connect(&url, tokio_postgres::NoTls)
        .await
        .unwrap();
    tokio::spawn(conn);
    admin
        .execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
             WHERE application_name = 'irbench-loadgen'",
            &[],
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(!target.send(0, &ok).await, "the dead connection fails once");
    assert!(target.send(0, &ok).await, "then the slot is reconnected");
    assert!(!target.send(1, &ok).await);
    assert!(target.send(1, &ok).await);
}

#[tokio::test]
#[ignore]
async fn slow_statement_is_an_error_within_the_timeout_margin() {
    let _serial = SERIAL.lock().await;
    use irbench_loadgen::requests::PreparedRequest;
    let Some(url) = pg_url() else { return };
    let target = irbench_loadgen::runner::Target::connect(&url, Duration::from_millis(200), 1)
        .await
        .unwrap();
    let slow = PreparedRequest::postgres_for_test("x/x", "SELECT pg_sleep(5)", vec![]);
    let ok = PreparedRequest::postgres_for_test("x/x", "SELECT 1", vec![]);
    let t0 = std::time::Instant::now();
    assert!(!target.send(0, &slow).await);
    assert!(
        t0.elapsed() < Duration::from_millis(1500),
        "{:?}",
        t0.elapsed()
    );
    assert!(target.send(0, &ok).await);
}

#[tokio::test]
#[ignore]
async fn prime_prepares_statements_on_every_connection() {
    let _serial = SERIAL.lock().await;
    use irbench_loadgen::requests::PreparedRequest;
    let Some(url) = pg_url() else { return };
    let target = irbench_loadgen::runner::Target::connect(&url, Duration::from_secs(5), 3)
        .await
        .unwrap();
    target.prime(&["SELECT 1".to_string()]).await.unwrap();
    assert!(
        target
            .prime(&["SELECT * FROM no_such_table".to_string()])
            .await
            .is_err()
    );
    let ok = PreparedRequest::postgres_for_test("x/x", "SELECT 1", vec![]);
    assert!(target.send(2, &ok).await);
}

/// ParadeDB only (IRBENCH_PG_URL on the paradedb engine, port 15433, with the database's
/// `plan_cache_mode = force_custom_plan` applied): the boost cast on a bound parameter must
/// keep working past the switch to a generic plan.
#[tokio::test]
#[ignore]
async fn paradedb_keyword_statement_survives_generic_plan_switch() {
    let _serial = SERIAL.lock().await;
    use irbench_loadgen::requests::PreparedRequest;
    let Some(url) = pg_url() else { return };
    let (admin, conn) = tokio_postgres::connect(&url, tokio_postgres::NoTls)
        .await
        .unwrap();
    tokio::spawn(conn);
    admin
        .batch_execute(
            "CREATE EXTENSION IF NOT EXISTS pg_search;
             DROP TABLE IF EXISTS pdb_plan_t;
             CREATE TABLE pdb_plan_t (id text PRIMARY KEY, name text, description text);
             INSERT INTO pdb_plan_t SELECT i::text, 'white table ' || i, 'a sofa' FROM generate_series(1, 200) i;
             CREATE INDEX ON pdb_plan_t USING bm25 (id, name, description);",
        )
        .await
        .unwrap();
    let target = irbench_loadgen::runner::Target::connect(&url, Duration::from_secs(5), 1)
        .await
        .unwrap();
    let kw = PreparedRequest::postgres_for_test(
        "x/x",
        "SELECT id FROM pdb_plan_t WHERE (name ||| $1::text::pdb.boost(2) OR description ||| $1::text::pdb.boost(1)) \
         ORDER BY pdb.score(id) DESC LIMIT $2::text::int",
        vec![Some("white table".into()), Some("10".into())],
    );
    for i in 0..30 {
        assert!(target.send(0, &kw).await, "execution {}", i + 1);
    }
    admin.batch_execute("DROP TABLE pdb_plan_t").await.unwrap();
}
