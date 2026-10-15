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
    Mock {
        base_url: format!("http://{addr}"),
        hits,
    }
}

pub fn tmp_dir(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("irbench-loadgen-it-{}-{name}", std::process::id()));
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
