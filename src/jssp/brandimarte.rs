//! Brandimarte FJSSP benchmark file loader.
//!
//! Format (Brandimarte 1993, "Routing and scheduling in a flexible job shop
//! by tabu search", Annals of Operations Research):
//!
//! ```text
//! <n_jobs> <n_machines> <avg_eligibility>
//! <job_1>
//! <job_2>
//! ...
//! ```
//!
//! Each job line:
//! ```text
//! <n_ops>  <op_1>  <op_2>  ...  <op_n>
//! ```
//!
//! Each op spec:
//! ```text
//! <n_eligible_machines>  (<machine_id> <proc_time>)+
//! ```
//!
//! Machine IDs in the file are 1-indexed; we normalise to 0-indexed.
//!
//! Due dates are not part of the file. We synthesise them using the
//! TWK convention (실험설계서_수정 §7-1):
//!
//! ```text
//! due_j = release_j + tightness · DDT · total_processing_j
//! ```

use std::fs;
use std::path::Path;

use crate::jssp::instance::{Instance, Job, Machine, Operation};

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "brandimarte parse error: {}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parameters used when constructing due dates / metadata on top of a
/// raw Brandimarte file.
#[derive(Debug, Clone, Copy)]
pub struct LoadParams {
    /// Base TWK tightness factor applied to total processing time.
    /// 1.5 is the lower end of the dynamic-JSSP literature; smaller =
    /// tighter due dates.
    pub tightness: f64,
    /// Multiplier on `tightness` (실험설계서_수정 §7-1). 1.0 = no change,
    /// 0.7 = tightened, 1.5 = loosened.
    pub ddt: f64,
}

impl Default for LoadParams {
    fn default() -> Self {
        Self { tightness: 1.5, ddt: 1.0 }
    }
}

/// Parse a Brandimarte text file into an `Instance`.
pub fn load_from_path(path: &Path, params: &LoadParams) -> Result<Instance, ParseError> {
    let raw = fs::read_to_string(path).map_err(|e| ParseError {
        message: format!("cannot read {}: {e}", path.display()),
    })?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("brandimarte");
    parse(&raw, stem, params)
}

/// Parse a Brandimarte instance from an in-memory string.
pub fn parse(src: &str, name: &str, params: &LoadParams) -> Result<Instance, ParseError> {
    let mut tokens = src
        .lines()
        .map(|l| match l.find('#') { Some(i) => &l[..i], None => l })
        .flat_map(|l| l.split_ascii_whitespace())
        .peekable();

    let n_jobs = next_usize(&mut tokens, "n_jobs")?;
    let n_machines = next_usize(&mut tokens, "n_machines")?;
    // Third number is average eligibility (a float). Consume it.
    let _avg_elig = next_token(&mut tokens, "avg_eligibility")?;

    let machines: Vec<Machine> = (0..n_machines as u32).map(|id| Machine { id }).collect();
    let mut jobs: Vec<Job> = Vec::with_capacity(n_jobs);

    for j in 0..n_jobs as u32 {
        let n_ops = next_usize(&mut tokens, "n_ops")?;
        let mut operations: Vec<Operation> = Vec::with_capacity(n_ops);
        for op_idx in 0..n_ops {
            let n_elig = next_usize(&mut tokens, "n_eligible")?;
            if n_elig == 0 {
                return Err(ParseError {
                    message: format!("job {j} op {op_idx} has 0 eligible machines"),
                });
            }
            let mut eligible: Vec<u32> = Vec::with_capacity(n_elig);
            let mut times: Vec<f64> = Vec::with_capacity(n_elig);
            for _ in 0..n_elig {
                let m1 = next_usize(&mut tokens, "machine_id")?;
                if m1 == 0 || m1 > n_machines {
                    return Err(ParseError {
                        message: format!(
                            "job {j} op {op_idx} references machine {m1} \
                             (must be in 1..={n_machines})"
                        ),
                    });
                }
                let p = next_f64(&mut tokens, "proc_time")?;
                eligible.push((m1 - 1) as u32);
                times.push(p);
            }

            // Full FJSSP fidelity: store all (machine, time) pairs. The
            // canonical `processing_time` is the mean — used by job-level
            // aggregates where the dispatched machine isn't yet known.
            let processing_time = times.iter().sum::<f64>() / times.len() as f64;
            operations.push(Operation {
                job: j,
                idx: op_idx as u32,
                processing_time,
                eligible_machines: eligible,
                processing_times: times,
            });
        }

        let release_time = 0.0;
        let total_p: f64 = operations.iter().map(|o| o.processing_time).sum();
        let due_date = release_time + params.tightness * params.ddt * total_p;

        jobs.push(Job {
            id: j,
            release_time,
            due_date,
            urgent: false,
            tardiness_penalty: 1.0,
            operations,
            material_shortage_risk: 0.0,
            inbound_delay_time: 0.0,
        });
    }

    if tokens.peek().is_some() {
        // Extra tokens — file is longer than expected. Not necessarily an
        // error (some variants include trailing metadata) but worth warning.
        // We do not fail; the simulator only needs what we parsed.
    }

    Ok(Instance {
        name: name.to_string(),
        jobs,
        machines,
    })
}

fn next_token<'a, I: Iterator<Item = &'a str>>(
    it: &mut std::iter::Peekable<I>,
    what: &str,
) -> Result<&'a str, ParseError> {
    it.next().ok_or_else(|| ParseError {
        message: format!("unexpected EOF while reading {what}"),
    })
}

fn next_usize<'a, I: Iterator<Item = &'a str>>(
    it: &mut std::iter::Peekable<I>,
    what: &str,
) -> Result<usize, ParseError> {
    let t = next_token(it, what)?;
    t.parse::<usize>().map_err(|e| ParseError {
        message: format!("invalid {what} '{t}': {e}"),
    })
}

fn next_f64<'a, I: Iterator<Item = &'a str>>(
    it: &mut std::iter::Peekable<I>,
    what: &str,
) -> Result<f64, ParseError> {
    let t = next_token(it, what)?;
    t.parse::<f64>().map_err(|e| ParseError {
        message: format!("invalid {what} '{t}': {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny hand-crafted instance: 2 jobs, 3 machines, average eligibility 1.5.
    /// Job 0: op0 (m1, 5), op1 (m2 or m3 — 4 / 3)
    /// Job 1: op0 (m2, 6), op1 (m1, 2)
    const TINY: &str = "\
2 3 1.5
2  1 1 5   2 2 4 3 3
2  1 2 6   1 1 2
";

    #[test]
    fn parses_tiny() {
        let inst = parse(TINY, "tiny", &LoadParams::default()).unwrap();
        assert_eq!(inst.n_jobs(), 2);
        assert_eq!(inst.n_machines(), 3);

        let j0 = &inst.jobs[0];
        assert_eq!(j0.operations.len(), 2);
        // op0: 1 eligible machine -> machine_id 1 in file -> 0 here, proc 5
        assert_eq!(j0.operations[0].eligible_machines, vec![0]);
        assert_eq!(j0.operations[0].processing_times, vec![5.0]);
        assert_eq!(j0.operations[0].proc_time_on(0), 5.0);
        // op1: 2 eligible -> [1, 2], times [4, 3]; mean = 3.5
        assert_eq!(j0.operations[1].eligible_machines, vec![1, 2]);
        assert_eq!(j0.operations[1].processing_times, vec![4.0, 3.0]);
        assert_eq!(j0.operations[1].proc_time_on(1), 4.0);
        assert_eq!(j0.operations[1].proc_time_on(2), 3.0);
        assert_eq!(j0.operations[1].processing_time, 3.5);

        let j1 = &inst.jobs[1];
        assert_eq!(j1.operations[0].eligible_machines, vec![1]);
        assert_eq!(j1.operations[0].proc_time_on(1), 6.0);
        assert_eq!(j1.operations[1].eligible_machines, vec![0]);
        assert_eq!(j1.operations[1].proc_time_on(0), 2.0);
    }

    #[test]
    fn ddt_tightens_due_date() {
        let lax = LoadParams { tightness: 1.5, ddt: 1.5 };
        let tight = LoadParams { tightness: 1.5, ddt: 0.7 };
        let lax_inst = parse(TINY, "lax", &lax).unwrap();
        let tight_inst = parse(TINY, "tight", &tight).unwrap();
        for (l, t) in lax_inst.jobs.iter().zip(tight_inst.jobs.iter()) {
            assert!(l.due_date > t.due_date,
                "lax DDT must yield looser due dates");
        }
    }

    #[test]
    fn rejects_out_of_range_machine() {
        let bad = "1 2 1.0\n1  1 5 99\n";
        assert!(parse(bad, "bad", &LoadParams::default()).is_err());
    }
}
