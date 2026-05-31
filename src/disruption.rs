//! External supply-chain disruption events. The scenario layer constructs
//! a `Vec<DisruptionEvent>`; the engine injects them into its event queue at
//! bootstrap.

use crate::jssp::{Job, JobId, MachineId, OperationIdx};

#[derive(Debug, Clone)]
pub enum DisruptionEvent {
    /// A part required by (job, op) becomes available at `at`.
    /// Until that time the operation cannot start, even if its predecessor
    /// is done and its machine is idle.
    PartArrival { at: f64, job: JobId, op: OperationIdx },

    /// A new urgent / dynamic job is inserted into the system. Its
    /// `release_time` determines when it actually becomes considerable.
    JobInsert { job: Job },

    /// A job's due date is shifted (typically earlier) at runtime.
    DueShift { at: f64, job: JobId, new_due: f64 },

    /// A machine breaks down at `at` and is repaired at `repair_at`.
    Breakdown { at: f64, machine: MachineId, repair_at: f64 },

    /// Material shortage risk on a job rises to `level` ∈ [0,1] at `at`
    /// (창종설 보고서 §3-3 ③). Does not block dispatch; surfaces as a
    /// rule input variable.
    MaterialShortage { at: f64, job: JobId, level: f64 },
}
