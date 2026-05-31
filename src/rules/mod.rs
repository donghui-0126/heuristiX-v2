pub mod baselines;
pub mod expr;

use crate::jssp::{Job, Operation};
use crate::sim::engine::MachineView;
use crate::sim::state::ShopState;

/// A dispatching rule scores a (job, op, machine) candidate; the engine picks argmax.
pub trait PriorityRule: Send + Sync {
    fn name(&self) -> &str;
    fn score(&self, job: &Job, op: &Operation, machine: &MachineView, state: &ShopState) -> f64;
}

pub use baselines::*;
pub use expr::ExprRule;
