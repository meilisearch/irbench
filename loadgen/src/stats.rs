use hdrhistogram::Histogram;
use serde::Serialize;

/// Latency histogram (microseconds) of successful requests, plus an error count.
pub struct Recorder {
    hist: Histogram<u64>,
    errors: u64,
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            hist: Histogram::new_with_bounds(1, 60_000_000, 3).expect("static histogram bounds"),
            errors: 0,
        }
    }

    pub fn record_ok(&mut self, micros: u64) {
        self.hist.saturating_record(micros.max(1));
    }

    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    pub fn merge(&mut self, other: &Recorder) {
        self.hist.add(&other.hist).expect("histograms share bounds");
        self.errors += other.errors;
    }

    pub fn successes(&self) -> u64 {
        self.hist.len()
    }

    pub fn errors(&self) -> u64 {
        self.errors
    }

    pub fn total(&self) -> u64 {
        self.successes() + self.errors
    }

    pub fn error_rate(&self) -> f64 {
        match self.total() {
            0 => 0.0,
            total => self.errors as f64 / total as f64,
        }
    }

    fn ms_at(&self, quantile: f64) -> Option<f64> {
        Some(self.hist.value_at_quantile(quantile) as f64 / 1000.0)
    }

    /// Latencies are `None` when nothing succeeded: an engine that answers
    /// nothing must never look like one that answers in 0 ms.
    pub fn summary(&self) -> LatencySummary {
        if self.successes() == 0 {
            return LatencySummary {
                errors: self.errors,
                ..LatencySummary::default()
            };
        }
        LatencySummary {
            count: self.successes(),
            errors: self.errors,
            mean_ms: Some(self.hist.mean() / 1000.0),
            p50_ms: self.ms_at(0.50),
            p95_ms: self.ms_at(0.95),
            p99_ms: self.ms_at(0.99),
            max_ms: Some(self.hist.max() as f64 / 1000.0),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct LatencySummary {
    pub count: u64,
    pub errors: u64,
    pub mean_ms: Option<f64>,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub max_ms: Option<f64>,
}

/// The spec's validity rule for one throughput step.
#[derive(Debug, Clone, Serialize)]
pub struct Limits {
    pub p99_ms: f64,
    pub max_error_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepResult {
    pub concurrency: usize,
    pub seconds: f64,
    pub requests: u64,
    pub errors: u64,
    pub qps: f64,
    pub error_rate: f64,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub valid: bool,
    /// Why the step is invalid: no successes, error rate, or p99 over the limit.
    pub invalid_reason: Option<String>,
}

impl StepResult {
    pub fn new(concurrency: usize, seconds: f64, rec: &Recorder, limits: &Limits) -> Self {
        let s = rec.summary();
        let error_rate = rec.error_rate();
        let qps = if seconds > 0.0 {
            s.count as f64 / seconds
        } else {
            0.0
        };
        let invalid_reason = match s.p99_ms {
            None => Some("no successful requests".to_string()),
            Some(_) if error_rate > limits.max_error_rate => Some(format!(
                "error rate {:.3}% > {:.3}%",
                error_rate * 100.0,
                limits.max_error_rate * 100.0
            )),
            Some(p99) if p99 > limits.p99_ms => {
                Some(format!("p99 {p99:.1} ms > {} ms", limits.p99_ms))
            }
            Some(_) => None,
        };
        Self {
            concurrency,
            seconds,
            requests: rec.total(),
            errors: rec.errors(),
            qps,
            error_rate,
            p50_ms: s.p50_ms,
            p95_ms: s.p95_ms,
            p99_ms: s.p99_ms,
            valid: invalid_reason.is_none(),
            invalid_reason,
        }
    }
}

/// Spec §5.7: the best throughput among steps with p99 ≤ limit and errors ≤ limit.
/// Stepping stops at the first invalid step, so only the leading valid run counts;
/// within it the highest QPS wins, even if a later valid step dipped.
pub fn max_qps(steps: &[StepResult]) -> Option<&StepResult> {
    steps
        .iter()
        .take_while(|s| s.valid)
        .max_by(|a, b| a.qps.total_cmp(&b.qps))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Limits = Limits {
        p99_ms: 100.0,
        max_error_rate: 0.001,
    };

    fn recorder(latencies_ms: &[u64], errors: u64) -> Recorder {
        let mut r = Recorder::new();
        for ms in latencies_ms {
            r.record_ok(ms * 1000);
        }
        for _ in 0..errors {
            r.record_error();
        }
        r
    }

    #[test]
    fn summary_reports_percentiles_in_milliseconds() {
        let mut lat = vec![10; 90];
        lat.extend([50; 10]);
        let s = recorder(&lat, 2).summary();
        assert_eq!((s.count, s.errors), (100, 2));
        assert!((s.p50_ms.unwrap() - 10.0).abs() < 0.1, "{s:?}");
        assert!((s.p95_ms.unwrap() - 50.0).abs() < 0.1, "{s:?}");
        assert!((s.mean_ms.unwrap() - 14.0).abs() < 0.1, "{s:?}");
        assert!((s.max_ms.unwrap() - 50.0).abs() < 0.1, "{s:?}");
    }

    #[test]
    fn empty_recorder_is_all_zero() {
        let r = Recorder::new();
        assert_eq!(r.error_rate(), 0.0);
        let s = r.summary();
        // No successful request: no latency to report, never a 0 ms that reads as "fast".
        assert_eq!((s.count, s.p99_ms, s.mean_ms), (0, None, None));
    }

    #[test]
    fn merge_adds_counts_and_errors() {
        let mut a = recorder(&[1, 2], 1);
        a.merge(&recorder(&[3], 2));
        assert_eq!((a.successes(), a.errors(), a.total()), (3, 3, 6));
        assert!((a.error_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn step_validity_follows_p99_and_error_rate() {
        let fast = StepResult::new(4, 2.0, &recorder(&[10; 1000], 0), &LIMITS);
        assert!(fast.valid);
        assert!((fast.qps - 500.0).abs() < 1e-9);
        assert_eq!((fast.requests, fast.errors), (1000, 0));
        assert_eq!(fast.invalid_reason, None);
        let slow = StepResult::new(8, 1.0, &recorder(&[150; 100], 0), &LIMITS);
        assert!(!slow.valid);
        assert!(slow.invalid_reason.as_deref().unwrap().starts_with("p99"));
        let failing = StepResult::new(8, 1.0, &recorder(&[10; 998], 2), &LIMITS);
        assert!(!failing.valid, "0.2% errors exceeds 0.1%");
        assert!(
            failing
                .invalid_reason
                .as_deref()
                .unwrap()
                .starts_with("error rate")
        );
        let nothing = StepResult::new(1, 1.0, &Recorder::new(), &LIMITS);
        assert!(!nothing.valid);
        assert_eq!(
            nothing.invalid_reason.as_deref(),
            Some("no successful requests")
        );
        assert_eq!(nothing.p99_ms, None);
    }

    #[test]
    fn max_qps_is_the_best_valid_step_before_the_first_invalid_one() {
        let step = |c, valid: bool, qps| StepResult {
            concurrency: c,
            seconds: 1.0,
            requests: 1,
            errors: 0,
            qps,
            error_rate: 0.0,
            p50_ms: Some(1.0),
            p95_ms: Some(1.0),
            p99_ms: Some(1.0),
            valid,
            invalid_reason: (!valid).then(|| "p99".to_string()),
        };
        let steps = [
            step(1, true, 100.0),
            step(2, true, 180.0),
            step(4, false, 300.0),
            step(8, true, 400.0),
        ];
        assert_eq!(max_qps(&steps).map(|s| s.concurrency), Some(2));
        // A dip at a higher step must not hide the peak (real Meilisearch keyword run).
        let dip = [
            step(1, true, 343.0),
            step(16, true, 1754.0),
            step(32, true, 1697.0),
            step(64, true, 1674.0),
        ];
        assert_eq!(max_qps(&dip).map(|s| s.concurrency), Some(16));
        assert!(max_qps(&[step(1, false, 10.0)]).is_none());
        assert!(max_qps(&[]).is_none());
    }
}
