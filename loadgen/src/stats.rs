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

    fn ms_at(&self, quantile: f64) -> f64 {
        self.hist.value_at_quantile(quantile) as f64 / 1000.0
    }

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
            mean_ms: self.hist.mean() / 1000.0,
            p50_ms: self.ms_at(0.50),
            p95_ms: self.ms_at(0.95),
            p99_ms: self.ms_at(0.99),
            max_ms: self.hist.max() as f64 / 1000.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct LatencySummary {
    pub count: u64,
    pub errors: u64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
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
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub valid: bool,
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
            valid: s.count > 0 && s.p99_ms <= limits.p99_ms && error_rate <= limits.max_error_rate,
        }
    }
}

/// Spec §5.7: the highest step with p99 ≤ limit and errors ≤ limit. Stepping
/// stops at the first invalid step, so this is the last step of the leading valid run.
pub fn max_qps(steps: &[StepResult]) -> Option<&StepResult> {
    steps.iter().take_while(|s| s.valid).last()
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
        assert!((s.p50_ms - 10.0).abs() < 0.1, "{s:?}");
        assert!((s.p95_ms - 50.0).abs() < 0.1, "{s:?}");
        assert!((s.mean_ms - 14.0).abs() < 0.1, "{s:?}");
        assert!((s.max_ms - 50.0).abs() < 0.1, "{s:?}");
    }

    #[test]
    fn empty_recorder_is_all_zero() {
        let r = Recorder::new();
        assert_eq!(r.error_rate(), 0.0);
        let s = r.summary();
        assert_eq!((s.count, s.p99_ms, s.mean_ms), (0, 0.0, 0.0));
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
        let slow = StepResult::new(8, 1.0, &recorder(&[150; 100], 0), &LIMITS);
        assert!(!slow.valid);
        let failing = StepResult::new(8, 1.0, &recorder(&[10; 998], 2), &LIMITS);
        assert!(!failing.valid, "0.2% errors exceeds 0.1%");
        let nothing = StepResult::new(1, 1.0, &Recorder::new(), &LIMITS);
        assert!(!nothing.valid);
    }

    #[test]
    fn max_qps_is_the_last_valid_step_before_the_first_invalid_one() {
        let step = |c, valid, qps| StepResult {
            concurrency: c,
            seconds: 1.0,
            requests: 1,
            errors: 0,
            qps,
            error_rate: 0.0,
            p50_ms: 1.0,
            p95_ms: 1.0,
            p99_ms: 1.0,
            valid,
        };
        let steps = [
            step(1, true, 100.0),
            step(2, true, 180.0),
            step(4, false, 300.0),
            step(8, true, 400.0),
        ];
        assert_eq!(max_qps(&steps).map(|s| s.concurrency), Some(2));
        assert!(max_qps(&[step(1, false, 10.0)]).is_none());
        assert!(max_qps(&[]).is_none());
    }
}
