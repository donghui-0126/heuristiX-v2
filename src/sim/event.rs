use std::cmp::Ordering;

use crate::jssp::{JobId, MachineId, OperationIdx};

/// Simulation events. Time-ordered via a min-heap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventKind {
    /// A job arrives (release_time reached). For dynamic insert / urgent surge.
    JobArrival { job: JobId },
    /// An operation finishes on its machine.
    OpComplete { job: JobId, op: OperationIdx, machine: MachineId },
    /// A part needed by an operation becomes available.
    PartArrival { job: JobId, op: OperationIdx },
    /// A machine breakdown begins.
    Breakdown { machine: MachineId },
    /// A previously broken-down machine returns to service.
    Repair { machine: MachineId },
    /// A job's due date is shifted (earlier) at runtime.
    DueShift { job: JobId, new_due: f64 },
    /// A job's material-shortage risk is updated to `level` at runtime.
    MaterialShortage { job: JobId, level: f64 },
    /// Render hint for UI (not used in headless mode).
    RenderTick,
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub time: f64,
    pub seq: u64,           // tiebreaker for equal timestamps
    pub kind: EventKind,
}

impl Eq for Event {}
impl PartialEq for Event {
    fn eq(&self, o: &Self) -> bool { self.time == o.time && self.seq == o.seq }
}
impl Ord for Event {
    fn cmp(&self, o: &Self) -> Ordering {
        self.time.partial_cmp(&o.time).unwrap_or(Ordering::Equal).then(self.seq.cmp(&o.seq))
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
