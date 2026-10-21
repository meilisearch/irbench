use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

/// What to send: an HTTP request or a parameterized Postgres statement.
#[derive(Debug, Clone)]
pub enum Kind {
    Http {
        method: reqwest::Method,
        path: String,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
    },
    /// Every param is bound as text (`Option<String>`); the SQL casts as needed.
    Postgres {
        sql: String,
        params: Vec<Option<String>>,
    },
}

impl Kind {
    pub fn protocol(&self) -> &'static str {
        match self {
            Kind::Http { .. } => "http",
            Kind::Postgres { .. } => "postgres",
        }
    }
}

/// One rendered request, ready to send: the body is serialized once at load time.
#[derive(Debug, Clone)]
pub struct PreparedRequest {
    /// `<mode>/<base>`, e.g. `keyword/keyword` or `filtered/semantic`.
    pub group: String,
    pub kind: Kind,
}

impl PreparedRequest {
    /// Test helper: a Postgres request without going through a request file.
    pub fn postgres_for_test(group: &str, sql: &str, params: Vec<Option<String>>) -> Self {
        Self {
            group: group.to_string(),
            kind: Kind::Postgres {
                sql: sql.to_string(),
                params,
            },
        }
    }
}

#[derive(Deserialize)]
struct RenderedFile {
    protocol: String,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    body: Option<Value>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    sql: Option<String>,
    #[serde(default)]
    params: Option<Vec<Value>>,
}

/// Binds a JSON param as Postgres text, mirroring the Python client's rule:
/// null -> NULL, bool -> true/false, numbers -> decimal, string -> itself,
/// list of numbers -> pgvector literal `[a,b,...]` with floats.
pub fn bind_param(value: &Value) -> Result<Option<String>> {
    Ok(match value {
        Value::Null => None,
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Number(n) => parts.push(format!("{:?}", n.as_f64().unwrap_or(f64::NAN))),
                    other => bail!("cannot bind list element {other} (lists must hold numbers)"),
                }
            }
            Some(format!("[{}]", parts.join(",")))
        }
        other => bail!("cannot bind param {other}"),
    })
}

fn http_kind(path: &Path, file: RenderedFile) -> Result<Kind> {
    let method = file
        .method
        .with_context(|| format!("{}: http request without method", path.display()))?;
    let url_path = file
        .path
        .with_context(|| format!("{}: http request without path", path.display()))?;
    let method = method
        .parse::<reqwest::Method>()
        .with_context(|| format!("{}: bad method {method:?}", path.display()))?;
    let mut headers: Vec<(String, String)> = file.headers.into_iter().collect();
    let body = match (file.content, file.body) {
        (Some(content), _) => Some(content.into_bytes()),
        (None, None | Some(Value::Null)) => None,
        (None, Some(value)) => {
            if !headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            {
                headers.push(("Content-Type".to_string(), "application/json".to_string()));
            }
            Some(serde_json::to_vec(&value)?)
        }
    };
    Ok(Kind::Http {
        method,
        path: url_path,
        headers,
        body,
    })
}

/// Every `<mode>/<base>/<qid>.json` under `dir`, in sorted path order.
/// Facet verification fetches (`*.check.json`) are not part of the workload.
pub fn load_requests(dir: &Path) -> Result<Vec<PreparedRequest>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(dir).sort_by_file_name() {
        let entry = entry.with_context(|| format!("walking {}", dir.display()))?;
        let name = entry.file_name().to_string_lossy();
        if !entry.file_type().is_file() || !name.ends_with(".json") || name.ends_with(".check.json")
        {
            continue;
        }
        let path = entry.path();
        let parts: Vec<String> = path
            .strip_prefix(dir)?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        if parts.len() != 3 {
            continue;
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: RenderedFile =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let kind = match file.protocol.as_str() {
            "http" => http_kind(path, file)?,
            "postgres" => {
                let sql = file
                    .sql
                    .with_context(|| format!("{}: postgres request without sql", path.display()))?;
                let params = file
                    .params
                    .unwrap_or_default()
                    .iter()
                    .map(bind_param)
                    .collect::<Result<Vec<_>>>()
                    .with_context(|| format!("{}: binding params", path.display()))?;
                Kind::Postgres { sql, params }
            }
            other => bail!("{}: protocol {other:?} is not supported", path.display()),
        };
        out.push(PreparedRequest {
            group: format!("{}/{}", parts[0], parts[1]),
            kind,
        });
    }
    if out.is_empty() {
        bail!("no rendered requests under {}", dir.display());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("irbench-loadgen-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, rel: &str, text: &str) {
        let path = dir.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    type HttpParts = (
        reqwest::Method,
        String,
        Vec<(String, String)>,
        Option<Vec<u8>>,
    );

    fn http(r: &PreparedRequest) -> HttpParts {
        match &r.kind {
            Kind::Http {
                method,
                path,
                headers,
                body,
            } => (method.clone(), path.clone(), headers.clone(), body.clone()),
            other => panic!("expected http, got {other:?}"),
        }
    }

    #[test]
    fn loads_grouped_requests_and_skips_check_files() {
        let dir = tmp("grouped");
        let body = r#"{"protocol":"http","method":"POST","path":"/i/search","headers":{},"body":{"q":"sofa"}}"#;
        write(&dir, "keyword/keyword/1.json", body);
        write(&dir, "faceted/semantic/2.json", body);
        write(&dir, "faceted/semantic/2.check.json", body);
        write(&dir, "stray.json", body);

        let reqs = load_requests(&dir).unwrap();
        let groups: Vec<_> = reqs.iter().map(|r| r.group.as_str()).collect();
        assert_eq!(groups, ["faceted/semantic", "keyword/keyword"]);
        let (method, path, headers, body) = http(&reqs[1]);
        assert_eq!(method, reqwest::Method::POST);
        assert_eq!(path, "/i/search");
        assert_eq!(body.as_deref(), Some(br#"{"q":"sofa"}"#.as_slice()));
        assert_eq!(
            headers,
            [("Content-Type".to_string(), "application/json".to_string())]
        );
    }

    #[test]
    fn raw_content_is_sent_verbatim_with_its_own_headers() {
        let dir = tmp("raw");
        write(
            &dir,
            "keyword/keyword/1.json",
            r#"{"protocol":"http","method":"POST","path":"/_bulk","headers":{"Content-Type":"application/x-ndjson"},"content":"{}\n"}"#,
        );
        let (_, _, headers, body) = http(&load_requests(&dir).unwrap()[0]);
        assert_eq!(body.as_deref(), Some(b"{}\n".as_slice()));
        assert_eq!(
            headers,
            [(
                "Content-Type".to_string(),
                "application/x-ndjson".to_string()
            )]
        );
    }

    #[test]
    fn get_without_body_has_no_body() {
        let dir = tmp("get");
        write(
            &dir,
            "semantic/semantic/1.json",
            r#"{"protocol":"http","method":"GET","path":"/health","headers":{},"body":null}"#,
        );
        let (_, _, headers, body) = http(&load_requests(&dir).unwrap()[0]);
        assert!(body.is_none() && headers.is_empty());
    }

    #[test]
    fn rejects_other_protocols_and_empty_dirs() {
        let dir = tmp("grpc");
        write(
            &dir,
            "keyword/keyword/1.json",
            r#"{"protocol":"grpc","method":"POST","path":"/","headers":{}}"#,
        );
        assert!(
            load_requests(&dir)
                .unwrap_err()
                .to_string()
                .contains("grpc")
        );
        let empty = tmp("empty");
        assert!(
            load_requests(&empty)
                .unwrap_err()
                .to_string()
                .contains("no rendered requests")
        );
    }

    #[test]
    fn loads_postgres_files_and_binds_params_as_text() {
        let dir = tmp("pgload");
        write(
            &dir,
            "keyword/keyword/0.json",
            r#"{"protocol":"postgres","sql":"SELECT id FROM t WHERE x = $1::text LIMIT $3::text::int","params":["sofa",[0.5,-1],100]}"#,
        );
        let reqs = load_requests(&dir).unwrap();
        match &reqs[0].kind {
            Kind::Postgres { sql, params } => {
                assert!(sql.starts_with("SELECT id"));
                assert_eq!(
                    params,
                    &vec![
                        Some("sofa".into()),
                        Some("[0.5,-1.0]".into()),
                        Some("100".into())
                    ]
                );
            }
            _ => panic!("expected a postgres request"),
        }
    }

    #[test]
    fn bind_param_rules() {
        use serde_json::json;
        assert_eq!(bind_param(&json!(null)).unwrap(), None);
        assert_eq!(bind_param(&json!(true)).unwrap().as_deref(), Some("true"));
        assert_eq!(bind_param(&json!(10)).unwrap().as_deref(), Some("10"));
        assert_eq!(bind_param(&json!(0.5)).unwrap().as_deref(), Some("0.5"));
        assert!(bind_param(&json!({"a": 1})).is_err());
        assert!(bind_param(&json!(["a"])).is_err());
    }
}
