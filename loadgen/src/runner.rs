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

/// One Postgres connection with the statements prepared on it.
pub struct PgConn {
    client: tokio_postgres::Client,
    stmts: HashMap<String, tokio_postgres::Statement>,
}

/// The engine under test: a pooled HTTP client, or one Postgres connection per
/// concurrent client.
pub enum Target {
    Http {
        client: reqwest::Client,
        base_url: String,
    },
    Postgres {
        conns: Vec<Mutex<PgConn>>,
    },
}

impl Target {
    /// `postgresql://` / `postgres://` opens `max_connections` Postgres
    /// connections; anything else is an HTTP base URL.
    pub async fn connect(url: &str, timeout: Duration, max_connections: usize) -> Result<Self> {
        if url.starts_with("postgresql://") || url.starts_with("postgres://") {
            let mut config = tokio_postgres::Config::from_str(url)
                .with_context(|| "parsing the postgres URL")?;
            config.options(format!("-c statement_timeout={}", timeout.as_millis()));
            let mut conns = Vec::with_capacity(max_connections.max(1));
            for _ in 0..max_connections.max(1) {
                let (client, connection) = config
                    .connect(NoTls)
                    .await
                    .context("connecting to postgres")?;
                tokio::spawn(async move {
                    let _ = connection.await;
                });
                conns.push(Mutex::new(PgConn {
                    client,
                    stmts: HashMap::new(),
                }));
            }
            return Ok(Target::Postgres { conns });
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
            (Target::Postgres { conns }, Kind::Postgres { sql, params }) => {
                let mut conn = conns[worker % conns.len()].lock().await;
                let stmt = match conn.stmts.get(sql) {
                    Some(stmt) => stmt.clone(),
                    None => match conn.client.prepare(sql).await {
                        Ok(stmt) => {
                            conn.stmts.insert(sql.clone(), stmt.clone());
                            stmt
                        }
                        Err(_) => return false,
                    },
                };
                let refs: Vec<&(dyn ToSql + Sync)> =
                    params.iter().map(|p| p as &(dyn ToSql + Sync)).collect();
                conn.client.query(&stmt, &refs).await.is_ok()
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
