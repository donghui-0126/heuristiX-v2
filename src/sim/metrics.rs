use serde::Serialize;

use crate::jssp::{Instance, JobId};
use crate::sim::state::ShopState;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Metrics {
    pub n_jobs: usize,
    pub n_completed_jobs: usize,
    pub n_urgent_jobs: usize,
    pub n_urgent_tardy: usize,

    pub makespan: f64,
    pub mean_flow_time: f64,
    pub mean_tardiness: f64,
    pub max_tardiness: f64,
    pub total_tardiness: f64,
    pub tardy_rate: f64,
    pub urgent_tardy_rate: f64,
    pub urgent_mean_tardiness: f64,

    pub mean_machine_utilization: f64,
    pub total_machine_idle: f64,
    /// Idle time attributable to machine being down.
    pub idle_breakdown: f64,
    /// Idle time when machine was up but had no ready op (= starvation,
    /// often caused by upstream part delay).
    pub idle_starved: f64,
    /// Machine Idle Time per 실험설계서_수정 §8 — sum across machines of
    /// continuous "no work feasible" intervals. Equals `idle_starved` in
    /// the current setup (S0/S1/S2 do not emit breakdowns).
    pub mit: f64,
    /// Percentage of Tardy Jobs (PTJ) per 실험설계서_수정 §8 — tardy_rate × 100.
    pub ptj_pct: f64,

    /// Composite total cost (sum of tardiness × penalty across jobs).
    pub total_tardiness_cost: f64,

    /// Time-weighted mean fraction of waiting jobs that were feasible.
    /// 1.0 = no part-blocking ever; lower under S1/S2/S5.
    pub feasible_job_ratio: f64,

    /// Per-job completion times for downstream stability/gap-ratio
    /// comparisons. Empty entry = job never completed.
    pub completion_times: Vec<JobCompletion>,

    pub horizon: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobCompletion {
    pub job: u32,
    pub release: f64,
    pub due: f64,
    pub urgent: bool,
    /// Some(t) if the job's last op finished, None otherwise.
    pub finish: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreWeights {
    pub tardiness: f64,
    pub makespan: f64,
    pub urgent_tardiness: f64,
    pub idle: f64,
}
impl Default for ScoreWeights {
    fn default() -> Self { Self { tardiness: 1.0, makespan: 0.5, urgent_tardiness: 5.0, idle: 0.1 } }
}

impl ScoreWeights {
    /// Per-scenario weights from 창종설 보고서 §6-2 sketch. Returns
    /// `Default::default()` for unknown names.
    pub fn for_scenario(name: &str) -> Self {
        match name.to_ascii_uppercase().as_str() {
            // S1: machine idle dominates (parts late → starved fleet).
            "S1" => Self { tardiness: 1.0, makespan: 0.5, urgent_tardiness: 2.0, idle: 0.5 },
            // S2: similar idle pressure but lower since parts still arrive on time.
            "S2" => Self { tardiness: 1.0, makespan: 0.5, urgent_tardiness: 2.0, idle: 0.3 },
            // S3: urgent_tardiness dominates.
            "S3" => Self { tardiness: 1.0, makespan: 0.3, urgent_tardiness: 8.0, idle: 0.05 },
            // S4: due-date pressure → tardiness dominates.
            "S4" => Self { tardiness: 2.0, makespan: 0.3, urgent_tardiness: 4.0, idle: 0.05 },
            // S5: balanced — every component matters.
            "S5" => Self { tardiness: 1.0, makespan: 0.5, urgent_tardiness: 5.0, idle: 0.2 },
            _    => Self::default(),
        }
    }
}

impl Metrics {
    pub fn score(&self, w: ScoreWeights) -> f64 {
        -(w.tardiness * self.mean_tardiness
            + w.makespan * self.makespan
            + w.urgent_tardiness * self.urgent_mean_tardiness
            + w.idle * self.total_machine_idle)
    }
}

pub fn finalize(state: &ShopState, instance: &Instance) -> Metrics {
    let n_jobs = instance.n_jobs();
    let mut completion_finish: Vec<f64> = Vec::with_capacity(n_jobs);
    let mut tardiness: Vec<f64> = Vec::with_capacity(n_jobs);
    let mut flow_times: Vec<f64> = Vec::with_capacity(n_jobs);
    let mut urgent_tard: Vec<f64> = Vec::new();
    let mut total_tardiness_cost = 0.0;
    let mut completion_times: Vec<JobCompletion> = Vec::with_capacity(n_jobs);

    for job in &instance.jobs {
        let last_op = job.operations.len() as u32 - 1;
        let key = crate::sim::state::OpKey { job: job.id, op: last_op };
        let rec = match state.ops.get(&key) { Some(r) => r, None => continue };
        let finish = rec.finish_time;
        completion_times.push(JobCompletion {
            job: job.id,
            release: job.release_time,
            due: job.due_date,
            urgent: job.urgent,
            finish,
        });
        if let Some(end) = finish {
            completion_finish.push(end);
            let t = (end - job.due_date).max(0.0);
            tardiness.push(t);
            flow_times.push(end - job.release_time);
            total_tardiness_cost += t * job.tardiness_penalty;
            if job.urgent { urgent_tard.push(t); }
        }
    }

    let n_completed = completion_finish.len();
    let makespan = completion_finish.iter().cloned().fold(0.0_f64, f64::max);
    let mean_flow = mean(&flow_times);
    let mean_t = mean(&tardiness);
    let max_t = tardiness.iter().cloned().fold(0.0_f64, f64::max);
    let total_t: f64 = tardiness.iter().sum();
    let n_tardy = tardiness.iter().filter(|t| **t > 1e-9).count();
    let n_urgent = urgent_tard.len();
    let n_urgent_tardy = urgent_tard.iter().filter(|t| **t > 1e-9).count();

    // Machine util / idle accounting (idle split into breakdown vs starved).
    let horizon = makespan.max(state.now);
    let mut total_busy = 0.0;
    let mut total_idle = 0.0;
    let mut total_idle_breakdown = 0.0;
    for (_, mr) in &state.machines {
        total_busy += mr.busy_total;
        total_idle += mr.idle_total;
        total_idle_breakdown += mr.idle_breakdown;
    }
    let m = state.machines.len() as f64;
    let mean_util = if horizon > 0.0 { total_busy / (m * horizon) } else { 0.0 };
    let idle_starved = (total_idle - total_idle_breakdown).max(0.0);

    // Feasible-job-ratio: time-weighted average over [0, horizon].
    // Flush any pending interval first using the cached last ratio.
    let pending = (horizon - state.feasible_last_t).max(0.0);
    let feasible_integral = state.feasible_integral + pending * state.feasible_last_ratio;
    let feasible_job_ratio = if horizon > 0.0 { feasible_integral / horizon } else { 1.0 };

    Metrics {
        n_jobs,
        n_completed_jobs: n_completed,
        n_urgent_jobs: n_urgent,
        n_urgent_tardy,
        makespan,
        mean_flow_time: mean_flow,
        mean_tardiness: mean_t,
        max_tardiness: max_t,
        total_tardiness: total_t,
        tardy_rate: if n_completed == 0 { 0.0 } else { n_tardy as f64 / n_completed as f64 },
        urgent_tardy_rate: if n_urgent == 0 { 0.0 } else { n_urgent_tardy as f64 / n_urgent as f64 },
        urgent_mean_tardiness: mean(&urgent_tard),
        mean_machine_utilization: mean_util,
        total_machine_idle: total_idle,
        idle_breakdown: total_idle_breakdown,
        idle_starved,
        mit: idle_starved,
        ptj_pct: if n_completed == 0 { 0.0 } else { (n_tardy as f64 / n_completed as f64) * 100.0 },
        total_tardiness_cost,
        feasible_job_ratio,
        completion_times,
        horizon,
    }
}

/// Schedule stability: mean |C_new - C_orig| over jobs that completed in
/// both runs (창종설 보고서 §6-1). Returns None if no comparable jobs.
pub fn schedule_stability(orig: &Metrics, new: &Metrics) -> Option<f64> {
    use std::collections::HashMap;
    let orig_map: HashMap<u32, f64> = orig.completion_times.iter()
        .filter_map(|c| c.finish.map(|f| (c.job, f)))
        .collect();
    let mut diffs = Vec::new();
    for c in &new.completion_times {
        if let (Some(f_new), Some(f_orig)) = (c.finish, orig_map.get(&c.job)) {
            diffs.push((f_new - f_orig).abs());
        }
    }
    if diffs.is_empty() { None } else { Some(diffs.iter().sum::<f64>() / diffs.len() as f64) }
}

/// Gap ratio (창종설 보고서 §6-3): `(obj_rule - obj_baseline) / obj_baseline × 100`.
/// Negative = rule beats baseline. Returns None if baseline obj is ≤ 0.
pub fn gap_ratio(rule_obj: f64, baseline_obj: f64) -> Option<f64> {
    if baseline_obj.abs() < 1e-9 { None }
    else { Some((rule_obj - baseline_obj) / baseline_obj * 100.0) }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}

// Re-export so callers can use Metrics::finalize-like helpers if they want.
pub fn _link(_: JobId) {}
