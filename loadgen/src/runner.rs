use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tokio_postgres::NoTls;
use tokio_postgres::types::ToSql;

use crate::requests::{Kind, PreparedRequest};
use crate::stats::Recorder;

/// One Postgres connection with the statements prepared on it. `broken` is
/// set when a reconnect failed; the next request retries it.
pub struct PgConn {
    client: tokio_postgres::Client,
    stmts: HashMap<String, tokio_postgres::Statement>,
    broken: bool,
}

/// How long to wait before retrying a failed reconnect.
const RECONNECT_BACKOFF: Duration = Duration::from_millis(50);
/// Client-side margin on top of the server-side `statement_timeout`.
const CLIENT_MARGIN: Duration = Duration::from_secs(1);

/// The engine under test: a pooled HTTP client, or one Postgres connection per
/// concurrent client.
pub enum Target {
    Http {
        client: reqwest::Client,
        base_url: String,
    },
    Postgres {
        config: Box<tokio_postgres::Config>,
        timeout: Duration,
        conns: Vec<Mutex<PgConn>>,
    },
}

/// `"http"` or `"postgres"`, from the URL scheme alone.
pub fn protocol_for_url(url: &str) -> &'static str {
    if url.starts_with("postgresql://") || url.starts_with("postgres://") {
        "postgres"
    } else {
        "http"
    }
}

async fn pg_connect(config: &tokio_postgres::Config) -> Result<tokio_postgres::Client> {
    let (client, connection) = config
        .connect(NoTls)
        .await
        .context("connecting to postgres")?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(client)
}

impl Target {
    /// `postgresql://` / `postgres://` opens `max_connections` Postgres
    /// connections; anything else is an HTTP base URL.
    pub async fn connect(url: &str, timeout: Duration, max_connections: usize) -> Result<Self> {
        if protocol_for_url(url) == "postgres" {
            let mut config = tokio_postgres::Config::from_str(url)
                .with_context(|| "parsing the postgres URL")?;
            config.options(format!("-c statement_timeout={}", timeout.as_millis()));
            config.application_name("irbench-loadgen");
            // Also bounds every reconnect: a black-holed address must not stall a worker.
            config.connect_timeout(timeout);
            let mut conns = Vec::with_capacity(max_connections.max(1));
            for _ in 0..max_connections.max(1) {
                conns.push(Mutex::new(PgConn {
                    client: pg_connect(&config).await?,
                    stmts: HashMap::new(),
                    broken: false,
                }));
            }
            return Ok(Target::Postgres {
                config: Box::new(config),
                timeout,
                conns,
            });
        }
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(max_connections)
            .build()?;
        Ok(Target::Http {
            client,
            base_url: url.trim_end_matches('/').to_string(),
        })
    }

    pub fn protocol(&self) -> &'static str {
        match self {
            Target::Http { .. } => "http",
            Target::Postgres { .. } => "postgres",
        }
    }

    /// Prepares every SQL text on every Postgres connection so measured
    /// requests never pay a Parse round trip. No-op for HTTP.
    pub async fn prime(&self, sqls: &[String]) -> Result<()> {
        if let Target::Postgres { conns, .. } = self {
            for conn in conns {
                let mut conn = conn.lock().await;
                for sql in sqls {
                    if !conn.stmts.contains_key(sql) {
                        let stmt = conn
                            .client
                            .prepare(sql)
                            .await
                            .with_context(|| format!("preparing {sql}"))?;
                        conn.stmts.insert(sql.clone(), stmt);
                    }
                }
            }
        }
        Ok(())
    }

    /// Replaces a broken connection and re-prepares the statements it had.
    /// On failure the slot stays `broken` and the next request retries.
    async fn reconnect(config: &tokio_postgres::Config, conn: &mut PgConn) {
        conn.broken = true;
        let Ok(client) = pg_connect(config).await else {
            return;
        };
        let mut stmts = HashMap::new();
        for sql in conn.stmts.keys() {
            match client.prepare(sql).await {
                Ok(stmt) => {
                    stmts.insert(sql.clone(), stmt);
                }
                Err(_) => return,
            }
        }
        conn.client = client;
        conn.stmts = stmts;
        conn.broken = false;
    }

    /// True for a 2xx HTTP response whose body was read completely, or a
    /// Postgres statement that ran without error. `worker` picks the Postgres
    /// connection; a request of the wrong protocol is a failure.
    pub async fn send(&self, worker: usize, req: &PreparedRequest) -> bool {
        match (self, &req.kind) {
            (
                Target::Http { client, base_url },
                Kind::Http {
                    method,
                    path,
                    headers,
                    body,
                },
            ) => {
                let mut builder = client.request(method.clone(), format!("{base_url}{path}"));
                for (name, value) in headers {
                    builder = builder.header(name.as_str(), value.as_str());
                }
                if let Some(body) = body {
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
            (
                Target::Postgres {
                    config,
                    timeout,
                    conns,
                },
                Kind::Postgres { sql, params },
            ) => {
                let mut conn = conns[worker % conns.len()].lock().await;
                if conn.broken {
                    // A previous reconnect failed: try again, but this request is lost.
                    tokio::time::sleep(RECONNECT_BACKOFF).await;
                    Self::reconnect(config, &mut conn).await;
                    return false;
                }
                let refs: Vec<&(dyn ToSql + Sync)> =
                    params.iter().map(|p| p as &(dyn ToSql + Sync)).collect();
                let attempt = tokio::time::timeout(*timeout + CLIENT_MARGIN, async {
                    let stmt = match conn.stmts.get(sql) {
                        Some(stmt) => stmt.clone(),
                        None => conn.client.prepare(sql).await?,
                    };
                    let rows = conn.client.query(&stmt, &refs).await;
                    rows.map(|_| stmt)
                })
                .await;
                match attempt {
                    Ok(Ok(stmt)) => {
                        conn.stmts.entry(sql.clone()).or_insert(stmt);
                        true
                    }
                    Ok(Err(err)) => {
                        if err.is_closed() || conn.client.is_closed() {
                            Self::reconnect(config, &mut conn).await;
                        }
                        false
                    }
                    Err(_elapsed) => {
                        // An in-flight query on a stalled connection can't be trusted.
                        Self::reconnect(config, &mut conn).await;
                        false
                    }
                }
            }
            _ => false,
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
    for worker in 0..concurrency.max(1) {
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
                if target.send(worker, &requests[i % total]).await {
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
