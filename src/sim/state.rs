use std::collections::HashMap;

use crate::jssp::{Instance, JobId, MachineId, OperationIdx};

/// Identifier for an operation: (job, idx within job).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpKey {
    pub job: JobId,
    pub op: OperationIdx,
}

/// Lifecycle state of an operation. The engine drives transitions forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpStatus {
    /// Job not yet released, or predecessor not done. Cannot be dispatched.
    Blocked,
    /// All preconditions met (predecessor done + part available + job released).
    /// Sitting in the ready queue waiting for a machine.
    Ready,
    /// Currently being processed on a machine.
    Running,
    /// Finished.
    Done,
}

/// Per-operation runtime fields.
#[derive(Debug, Clone)]
pub struct OpRecord {
    pub status: OpStatus,
    /// The earliest time this op may start (max of: predecessor completion,
    /// part availability, job release, machine availability — the last is
    /// computed at dispatch time, the first three live here).
    pub part_available_time: f64,
    pub start_time: Option<f64>,
    pub finish_time: Option<f64>,
    /// FJSSP: the machine the op was actually assigned to at dispatch time.
    /// `None` while Blocked/Ready (machine choice not yet made).
    pub assigned_machine: Option<MachineId>,
}

/// Per-machine runtime fields.
#[derive(Debug, Clone)]
pub struct MachineRecord {
    pub busy_with: Option<OpKey>,
    pub busy_total: f64,         // accumulated busy minutes
    pub idle_total: f64,         // total non-busy minutes (= breakdown + starved)
    /// Subset of `idle_total` attributable to machine being down.
    pub idle_breakdown: f64,
    pub down: bool,
    pub last_change: f64,
}

impl MachineRecord {
    fn new() -> Self {
        Self {
            busy_with: None,
            busy_total: 0.0,
            idle_total: 0.0,
            idle_breakdown: 0.0,
            down: false,
            last_change: 0.0,
        }
    }
    pub fn is_idle(&self) -> bool { self.busy_with.is_none() && !self.down }
}

/// Mutable shop-floor state. Owned by the engine.
pub struct ShopState {
    pub now: f64,
    pub instance: Instance,
    pub ops: HashMap<OpKey, OpRecord>,
    pub machines: HashMap<MachineId, MachineRecord>,

    /// Next operation index that must run for each job (= number of ops already
    /// completed). Used to find each job's "current" op without scanning.
    pub next_op_idx: HashMap<JobId, u32>,
    pub job_arrived: HashMap<JobId, bool>,

    // ---- Supply-chain aggregate state (창종설 보고서 §3-2) ----
    /// 0 = none, 1 = mild, 2 = severe. Stepped up when PartArrival events
    /// inject delays; decays back as parts arrive.
    pub supply_delay_level: f64,
    /// Fraction of currently-waiting (not Done) jobs flagged urgent.
    pub urgent_order_ratio: f64,
    /// Composite shock intensity in [0,1].
    pub disruption_level: f64,
    /// Mean inbound delay across jobs that have any delay (minutes).
    pub avg_inbound_delay: f64,

    // ---- Feasible-Job-Ratio integrator (창종설 보고서 §6-1) ----
    /// Time-weighted integral of (feasible jobs / waiting jobs).
    /// Final ratio = integral / horizon at finalize.
    pub feasible_integral: f64,
    /// Most recent sample time used to advance the integral.
    pub feasible_last_t: f64,
    /// Cached current ratio (feasible / waiting at the last update).
    pub feasible_last_ratio: f64,
}

impl ShopState {
    pub fn new(instance: Instance) -> Self {
        let mut ops = HashMap::new();
        let mut next_op_idx = HashMap::new();
        let mut job_arrived = HashMap::new();
        for j in &instance.jobs {
            next_op_idx.insert(j.id, 0);
            job_arrived.insert(j.id, j.release_time <= 0.0);
            for (i, _) in j.operations.iter().enumerate() {
                let key = OpKey { job: j.id, op: i as u32 };
                ops.insert(key, OpRecord {
                    status: OpStatus::Blocked,
                    part_available_time: j.release_time,
                    start_time: None,
                    finish_time: None,
                    assigned_machine: None,
                });
            }
        }
        let machines = instance.machines.iter()
            .map(|m| (m.id, MachineRecord::new()))
            .collect();

        Self {
            now: 0.0,
            instance,
            ops,
            machines,
            next_op_idx,
            job_arrived,
            supply_delay_level: 0.0,
            urgent_order_ratio: 0.0,
            disruption_level: 0.0,
            avg_inbound_delay: 0.0,
            feasible_integral: 0.0,
            feasible_last_t: 0.0,
            feasible_last_ratio: 0.0,
        }
    }

    /// Snapshot the (feasible / waiting) ratio at `now` and accumulate the
    /// time-weighted integral with the *previous* ratio over the elapsed
    /// interval. Call this before any state change that affects feasibility:
    /// JobArrival, PartArrival, OpComplete, Breakdown, Repair, MaterialShortage.
    pub fn tick_feasibility(&mut self, now: f64) {
        // Advance the integral over [last_t, now] using the LAST ratio
        // (the value that held for that whole interval).
        let dt = (now - self.feasible_last_t).max(0.0);
        self.feasible_integral += dt * self.feasible_last_ratio;
        self.feasible_last_t = now;

        // Recompute current ratio.
        let mut waiting = 0u32;
        let mut feasible = 0u32;
        for job in &self.instance.jobs {
            let last = job.operations.len() as u32 - 1;
            let last_done = self.ops.get(&OpKey { job: job.id, op: last })
                .map(|r| r.status == OpStatus::Done).unwrap_or(false);
            if last_done { continue; }
            waiting += 1;
            // Feasible if head op is NOT blocked. Ready/Running both count
            // as "currently dispatchable or being dispatched".
            let next = *self.next_op_idx.get(&job.id).unwrap_or(&0);
            let n_ops = job.operations.len() as u32;
            if next < n_ops {
                let head = OpKey { job: job.id, op: next };
                if let Some(r) = self.ops.get(&head) {
                    if !matches!(r.status, OpStatus::Blocked) {
                        feasible += 1;
                    }
                }
            }
        }
        self.feasible_last_ratio = if waiting == 0 {
            1.0
        } else {
            feasible as f64 / waiting as f64
        };
    }

    /// Recompute supply-chain aggregate features from per-job state.
    /// Called by the engine whenever a disruption event lands or a job
    /// completes, so rule expressions see fresh values.
    pub fn recompute_supply_aggregates(&mut self) {
        let n_jobs = self.instance.jobs.len() as f64;
        if n_jobs == 0.0 { return; }

        // Active = not yet Done (last op).
        let mut waiting_total = 0u32;
        let mut waiting_urgent = 0u32;
        let mut delayed_count = 0u32;
        let mut delay_sum = 0.0;
        let mut max_mat_risk = 0.0_f64;
        let mut mean_mat_risk = 0.0_f64;
        let mut active_for_mat = 0u32;

        for job in &self.instance.jobs {
            let last = job.operations.len() as u32 - 1;
            let done = self.ops.get(&OpKey { job: job.id, op: last })
                .map(|r| r.status == OpStatus::Done).unwrap_or(false);
            if done { continue; }

            waiting_total += 1;
            if job.urgent { waiting_urgent += 1; }

            if job.inbound_delay_time > 0.0 {
                delayed_count += 1;
                delay_sum += job.inbound_delay_time;
            }
            mean_mat_risk += job.material_shortage_risk;
            max_mat_risk = max_mat_risk.max(job.material_shortage_risk);
            active_for_mat += 1;
        }

        self.urgent_order_ratio = if waiting_total > 0 {
            waiting_urgent as f64 / waiting_total as f64
        } else { 0.0 };

        self.avg_inbound_delay = if delayed_count > 0 {
            delay_sum / delayed_count as f64
        } else { 0.0 };

        // supply_delay_level: bin avg_inbound_delay into {0, 1, 2}
        // thresholds (minutes) align with 창종설 보고서 §3-3 delay_severity bins.
        self.supply_delay_level = if self.avg_inbound_delay <= 0.0 { 0.0 }
            else if self.avg_inbound_delay < 60.0 { 1.0 }
            else { 2.0 };

        let mean_mat = if active_for_mat > 0 { mean_mat_risk / active_for_mat as f64 } else { 0.0 };

        // Composite disruption_level in [0,1]: weighted blend of three signals.
        // Weights chosen so each component contributes ≤ 1/3 individually.
        let supply_term = (self.supply_delay_level / 2.0).min(1.0);
        let urgent_term = self.urgent_order_ratio.min(1.0);
        let mat_term = max_mat_risk.max(mean_mat).min(1.0);
        self.disruption_level = (supply_term + urgent_term + mat_term) / 3.0;
    }

    /// Processing time of the op on the given machine. For strict JSSP
    /// it matches the canonical `processing_time`; for FJSSP it varies
    /// per dispatched machine.
    pub fn op_processing(&self, key: OpKey, machine: MachineId) -> f64 {
        let job = &self.instance.jobs[key.job as usize];
        job.operations[key.op as usize].proc_time_on(machine)
    }

    /// Convenience: the first eligible machine of an op. For strict JSSP
    /// this is THE machine; for FJSSP it's just the canonical one.
    pub fn op_machine(&self, key: OpKey) -> MachineId {
        let job = &self.instance.jobs[key.job as usize];
        job.operations[key.op as usize].eligible_machines[0]
    }

    /// All machines this op may be dispatched to (FJSSP routing flexibility).
    pub fn op_eligible_machines(&self, key: OpKey) -> &[MachineId] {
        let job = &self.instance.jobs[key.job as usize];
        &job.operations[key.op as usize].eligible_machines
    }

    pub fn job(&self, j: JobId) -> &crate::jssp::Job {
        &self.instance.jobs[j as usize]
    }

    /// Return the current "head" operation key for a job (the next op that needs to run),
    /// or None if the job is complete.
    pub fn head_op(&self, j: JobId) -> Option<OpKey> {
        let next = *self.next_op_idx.get(&j)?;
        let n_ops = self.instance.jobs[j as usize].operations.len() as u32;
        if next >= n_ops { None } else { Some(OpKey { job: j, op: next }) }
    }

    /// Iterate all ready operation keys (status == Ready).
    pub fn ready_keys(&self) -> Vec<OpKey> {
        self.ops.iter()
            .filter_map(|(k, r)| if r.status == OpStatus::Ready { Some(*k) } else { None })
            .collect()
    }
}
