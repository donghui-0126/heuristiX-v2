use crate::jssp::{Job, Operation};
use crate::rules::PriorityRule;
use crate::sim::engine::MachineView;
use crate::sim::state::ShopState;

/// Convention: higher score = higher priority. For "earliest X" rules we
/// negate the underlying value so that smaller becomes larger.

pub struct Fifo;
impl PriorityRule for Fifo {
    fn name(&self) -> &str { "FIFO" }
    fn score(&self, j: &Job, _o: &Operation, _m: &MachineView, _s: &ShopState) -> f64 {
        -j.release_time
    }
}

pub struct Edd;
impl PriorityRule for Edd {
    fn name(&self) -> &str { "EDD" }
    fn score(&self, j: &Job, _o: &Operation, _m: &MachineView, _s: &ShopState) -> f64 {
        -j.due_date
    }
}

pub struct Spt;
impl PriorityRule for Spt {
    fn name(&self) -> &str { "SPT" }
    fn score(&self, _j: &Job, o: &Operation, m: &MachineView, _s: &ShopState) -> f64 {
        // FJSSP: per-machine time. For strict JSSP this equals the canonical
        // processing_time.
        -o.proc_time_on(m.id)
    }
}

pub struct CriticalRatio;
impl PriorityRule for CriticalRatio {
    fn name(&self) -> &str { "CR" }
    fn score(&self, j: &Job, _o: &Operation, _m: &MachineView, s: &ShopState) -> f64 {
        // Remaining processing = sum of processing_time for ops not yet Done.
        let rem: f64 = (0..j.operations.len() as u32)
            .filter_map(|i| {
                let key = crate::sim::state::OpKey { job: j.id, op: i };
                let r = s.ops.get(&key)?;
                if r.status == crate::sim::state::OpStatus::Done { None } else {
                    Some(j.operations[i as usize].processing_time)
                }
            })
            .sum();
        let rem = rem.max(1e-6);
        let cr = (j.due_date - s.now) / rem;
        -cr
    }
}

pub struct UrgencyFirst;
impl PriorityRule for UrgencyFirst {
    fn name(&self) -> &str { "Urgency" }
    fn score(&self, j: &Job, _o: &Operation, _m: &MachineView, _s: &ShopState) -> f64 {
        if j.urgent { 1.0 } else { 0.0 }
    }
}

pub fn by_name(name: &str) -> Option<Box<dyn PriorityRule>> {
    match name.to_ascii_uppercase().as_str() {
        "FIFO" => Some(Box::new(Fifo)),
        "EDD"  => Some(Box::new(Edd)),
        "SPT"  => Some(Box::new(Spt)),
        "CR"   => Some(Box::new(CriticalRatio)),
        "URGENCY" | "URG" => Some(Box::new(UrgencyFirst)),
        _ => None,
    }
}
