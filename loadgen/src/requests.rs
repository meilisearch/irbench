use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// One rendered request, ready to send: the body is serialized once at load time.
#[derive(Debug, Clone)]
pub struct PreparedRequest {
    /// `<mode>/<base>`, e.g. `keyword/keyword` or `filtered/semantic`.
    pub group: String,
    pub method: reqwest::Method,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct RenderedFile {
    protocol: String,
    method: String,
    path: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    body: Option<serde_json::Value>,
    #[serde(default)]
    content: Option<String>,
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
        if file.protocol != "http" {
            bail!(
                "{}: protocol {:?} is not supported",
                path.display(),
                file.protocol
            );
        }
        let method = file
            .method
            .parse::<reqwest::Method>()
            .with_context(|| format!("{}: bad method {:?}", path.display(), file.method))?;
        let mut headers: Vec<(String, String)> = file.headers.into_iter().collect();
        let body = match (file.content, file.body) {
            (Some(content), _) => Some(content.into_bytes()),
            (None, None | Some(serde_json::Value::Null)) => None,
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
        out.push(PreparedRequest {
            group: format!("{}/{}", parts[0], parts[1]),
            method,
            path: file.path,
            headers,
            body,
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
        let r = &reqs[1];
        assert_eq!(r.method, reqwest::Method::POST);
        assert_eq!(r.path, "/i/search");
        assert_eq!(r.body.as_deref(), Some(br#"{"q":"sofa"}"#.as_slice()));
        assert_eq!(
            r.headers,
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
        let r = &load_requests(&dir).unwrap()[0];
        assert_eq!(r.body.as_deref(), Some(b"{}\n".as_slice()));
        assert_eq!(
            r.headers,
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
        let r = &load_requests(&dir).unwrap()[0];
        assert!(r.body.is_none() && r.headers.is_empty());
    }

    #[test]
    fn rejects_other_protocols_and_empty_dirs() {
        let dir = tmp("pg");
        write(
            &dir,
            "keyword/keyword/1.json",
            r#"{"protocol":"postgres","method":"POST","path":"/","headers":{}}"#,
        );
        assert!(
            load_requests(&dir)
                .unwrap_err()
                .to_string()
                .contains("postgres")
        );
        let empty = tmp("empty");
        assert!(
            load_requests(&empty)
                .unwrap_err()
                .to_string()
                .contains("no rendered requests")
        );
    }
}
