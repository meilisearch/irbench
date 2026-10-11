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
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// True for a 2xx response whose body was read completely.
    async fn send(&self, req: &PreparedRequest) -> bool {
        let mut builder = self
            .client
            .request(req.method.clone(), format!("{}{}", self.base_url, req.path));
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
