use crate::sim::engine::EngineSnapshot;
use crate::sim::{Metrics, SimulationEngine};

/// UI-Off runner. Tight loop, no rendering — runs at full Rust speed.
pub fn run(mut engine: SimulationEngine) -> Metrics {
    engine.run_to_end()
}

/// Same as `run`, but also returns a post-run snapshot so callers can
/// render the realised schedule (per-op start/end/machine).
pub fn run_with_snapshot(mut engine: SimulationEngine) -> (Metrics, EngineSnapshot) {
    let metrics = engine.run_to_end();
    let snap = engine.snapshot();
    (metrics, snap)
}
