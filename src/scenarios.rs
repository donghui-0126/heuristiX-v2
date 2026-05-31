//! Disruption-scenario construction.
//!
//! 실험설계서_수정 §4: only three scenarios are in scope.
//!   S0  Normal             — no external shock.
//!   S1  Part Delay         — a fraction of jobs has its first operation's
//!                            part_available_time shifted into the future,
//!                            once, at a uniformly random instant in
//!                            [0.1·T_est, 0.3·T_est].
//!   S2  Urgent Order       — a single new urgent job arrives at a
//!                            uniformly random instant in
//!                            [0.2·T_est, 0.5·T_est], cloning the routing
//!                            of a randomly chosen existing job.
//!
//! T_est is the trivial makespan lower bound (max over jobs of total
//! processing time).

use rand::seq::SliceRandom;
use rand::Rng;

use crate::disruption::DisruptionEvent;
use crate::jssp::{Instance, Job, Operation};

#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    S0Normal,
    S1PartDelay,
    S2UrgentOrder,
}

/// Tunable parameters for scenario construction. Defaults match the
/// "moderate" column of the design spec.
#[derive(Debug, Clone, Copy)]
pub struct ScenarioParams {
    /// S1 — fraction of jobs whose first-op part is delayed.
    /// Spec levels: {0.10, 0.20, 0.40}.
    pub part_delay_ratio: f64,

    /// S1 — delay factor k applied to instance-mean total processing time.
    /// part_available_time ← event_time + round(k · mean_total_processing).
    /// Spec levels: {0.5, 1.0, 2.0}.
    pub part_delay_k: f64,

    /// S2 — due-date tightness for the inserted urgent job, expressed as
    /// a ratio against the average total processing time of existing jobs.
    /// due_date ← release_time + urgent_due_ratio · mean_total_processing.
    /// Spec levels: {0.3, 0.5, 1.0}.
    pub urgent_due_ratio: f64,

    /// Due-date tightening factor (TWK + DDT) applied at instance generation.
    /// Smaller = tighter due dates. Kept here so scenario-aware runs can plumb
    /// it through to the generator (see §7-1).
    pub ddt: f64,

    /// FJSSP routing flexibility passed through to dynamically inserted
    /// jobs so urgent inserts inherit the same routing freedom as the
    /// initial instance. See `GenParams.flexibility`.
    pub flexibility: f64,
}

impl Default for ScenarioParams {
    fn default() -> Self {
        Self {
            part_delay_ratio: 0.20,
            part_delay_k: 1.0,
            urgent_due_ratio: 0.5,
            ddt: 1.0,
            flexibility: 0.0,
        }
    }
}

impl Scenario {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_uppercase().as_str() {
            "S0" | "NORMAL" => Some(Self::S0Normal),
            "S1" | "PART"   => Some(Self::S1PartDelay),
            "S2" | "URGENT" => Some(Self::S2UrgentOrder),
            _ => None,
        }
    }

    /// Build a list of disruption events for this instance.
    pub fn build<R: Rng + ?Sized>(
        &self,
        instance: &Instance,
        params: &ScenarioParams,
        rng: &mut R,
    ) -> Vec<DisruptionEvent> {
        let t_est = expected_makespan_lower_bound(instance);
        match self {
            Self::S0Normal => vec![],
            Self::S1PartDelay => {
                let onset = rng.gen_range(0.1 * t_est..=0.3 * t_est);
                part_delay_event(instance, onset, params.part_delay_ratio, params.part_delay_k, rng)
            }
            Self::S2UrgentOrder => {
                let onset = rng.gen_range(0.2 * t_est..=0.5 * t_est);
                vec![urgent_insert_event(instance, onset, params.urgent_due_ratio, rng)]
            }
        }
    }
}

/// S1 — at `onset`, push out part_available_time for the head op of a
/// random subset of jobs by `k · mean_total_processing`.
fn part_delay_event<R: Rng + ?Sized>(
    inst: &Instance,
    onset: f64,
    fraction_of_jobs: f64,
    k: f64,
    rng: &mut R,
) -> Vec<DisruptionEvent> {
    if inst.n_jobs() == 0 {
        return Vec::new();
    }
    let n_pick = ((inst.n_jobs() as f64) * fraction_of_jobs).round() as usize;
    let n_pick = n_pick.max(1).min(inst.n_jobs());

    let mean_total: f64 = inst.jobs.iter().map(|j| j.total_processing()).sum::<f64>()
        / inst.n_jobs() as f64;
    let offset = (k * mean_total).round();

    let mut ids: Vec<u32> = (0..inst.n_jobs() as u32).collect();
    ids.shuffle(rng);
    ids.into_iter()
        .take(n_pick)
        .map(|j| DisruptionEvent::PartArrival {
            at: onset + offset,
            job: j,
            op: 0,
        })
        .collect()
}

/// S2 — at `onset`, insert a single urgent job whose routing structure is
/// cloned from a random existing job. Due date is tight: release_time +
/// `urgent_due_ratio · mean_total_processing_of_existing_jobs`.
fn urgent_insert_event<R: Rng + ?Sized>(
    inst: &Instance,
    onset: f64,
    urgent_due_ratio: f64,
    rng: &mut R,
) -> DisruptionEvent {
    let template_idx = rng.gen_range(0..inst.n_jobs());
    let template = &inst.jobs[template_idx];
    let new_id = inst.n_jobs() as u32;

    let operations: Vec<Operation> = template
        .operations
        .iter()
        .enumerate()
        .map(|(i, op)| Operation {
            job: new_id,
            idx: i as u32,
            processing_time: op.processing_time,
            eligible_machines: op.eligible_machines.clone(),
            processing_times: op.processing_times.clone(),
        })
        .collect();

    let mean_total: f64 = inst.jobs.iter().map(|j| j.total_processing()).sum::<f64>()
        / inst.n_jobs() as f64;
    let due_date = onset + urgent_due_ratio * mean_total;

    DisruptionEvent::JobInsert {
        job: Job {
            id: new_id,
            release_time: onset,
            due_date,
            urgent: true,
            tardiness_penalty: 5.0,
            operations,
            material_shortage_risk: 0.0,
            inbound_delay_time: 0.0,
        },
    }
}

fn expected_makespan_lower_bound(inst: &Instance) -> f64 {
    inst.jobs
        .iter()
        .map(|j| j.total_processing())
        .fold(0.0_f64, f64::max)
}
