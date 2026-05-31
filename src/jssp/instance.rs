//! Static JSSP instance: jobs, operations, machines.
//!
//! Conventions (Taillard-style):
//!   - Time unit: minutes (f64).
//!   - Each job has an ordered list of operations.
//!   - Each operation has a list of eligible machines and a *parallel* list
//!     of processing times — `processing_times[i]` is the time when the op
//!     is run on `eligible_machines[i]`. For strict JSSP both lists have
//!     length 1. The struct field `processing_time` keeps the canonical
//!     (mean) value used by job-level aggregates that don't know which
//!     machine will be picked yet.

pub type JobId = u32;
pub type MachineId = u32;
pub type OperationIdx = u32;

#[derive(Debug, Clone)]
pub struct Operation {
    pub job: JobId,
    pub idx: OperationIdx,            // 0-based position in the job
    /// Canonical processing time = mean over `processing_times` (or the
    /// single value when length 1). Used by job-level totals where the
    /// dispatched machine is not yet known.
    pub processing_time: f64,
    pub eligible_machines: Vec<MachineId>,
    /// Per-eligible-machine processing time. Length matches
    /// `eligible_machines`. For strict JSSP this is `[processing_time]`.
    pub processing_times: Vec<f64>,
}

impl Operation {
    /// Processing time when this op runs on `machine_id`. Falls back to
    /// the canonical `processing_time` if `machine_id` is not eligible.
    pub fn proc_time_on(&self, machine_id: MachineId) -> f64 {
        for (i, &m) in self.eligible_machines.iter().enumerate() {
            if m == machine_id {
                return self.processing_times.get(i).copied()
                    .unwrap_or(self.processing_time);
            }
        }
        self.processing_time
    }

    /// Mean processing time across all eligible machines. Used as the
    /// canonical estimate for job-level totals.
    pub fn mean_proc_time(&self) -> f64 {
        if self.processing_times.is_empty() { return self.processing_time; }
        self.processing_times.iter().sum::<f64>() / self.processing_times.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: JobId,
    pub release_time: f64,            // arrival time
    pub due_date: f64,
    pub urgent: bool,
    pub tardiness_penalty: f64,
    pub operations: Vec<Operation>,

    // ---- Supply-chain disruption fields (창종설 보고서 §3-2) ----
    /// Material shortage risk in [0,1]. Updated at runtime by
    /// `DisruptionEvent::MaterialShortage`. 0 = no concern, 1 = critical.
    pub material_shortage_risk: f64,
    /// Total inbound delay imposed on this job's parts (minutes).
    /// Sum across operations for which a `PartArrival` event was injected.
    /// Read-only feature for LLM rules; not used in feasibility check.
    pub inbound_delay_time: f64,
}

impl Job {
    pub fn total_processing(&self) -> f64 {
        self.operations.iter().map(|o| o.processing_time).sum()
    }
}

#[derive(Debug, Clone)]
pub struct Machine {
    pub id: MachineId,
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub name: String,
    pub jobs: Vec<Job>,
    pub machines: Vec<Machine>,
}

impl Instance {
    pub fn n_jobs(&self) -> usize { self.jobs.len() }
    pub fn n_machines(&self) -> usize { self.machines.len() }
}
