//! Synthetic JSSP instance generator.
//!
//! Defaults follow the Taillard convention: each job visits every machine
//! exactly once, the visit order is a random permutation, and processing
//! times are i.i.d. Uniform(1, 99) minutes.
//!
//! Due dates are generated with the standard tightness factor:
//!     due_j = release_j + tightness * total_processing_j
//! with `tightness ∈ [1.3, 2.5]` per the dynamic-JSSP literature.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::jssp::instance::{Instance, Job, Machine, Operation};

#[derive(Debug, Clone)]
pub struct GenParams {
    pub n_jobs: usize,
    pub n_machines: usize,
    pub p_min: f64,
    pub p_max: f64,
    pub tightness: f64,
    pub urgent_p: f64,
    /// Optional dynamic arrival rate (jobs/min). If `None`, all jobs release at t=0.
    pub arrival_rate: Option<f64>,
    /// FJSSP routing flexibility ∈ [0, 1].
    ///   0.0  → each op has exactly 1 eligible machine (strict JSSP)
    ///   0.3  → each op has ~30% of machines as alternatives
    ///   1.0  → every op can run on any machine
    /// Sub-1 values round up so every op has at least 1 eligible machine.
    pub flexibility: f64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            n_jobs: 10,
            n_machines: 5,
            p_min: 1.0,
            p_max: 99.0,
            tightness: 1.6,
            urgent_p: 0.10,
            arrival_rate: None,
            flexibility: 0.0,
        }
    }
}

/// Pick a random subset of machine IDs for one operation, given a
/// flexibility factor. Returns at least one machine.
pub fn pick_eligible_machines<R: Rng + ?Sized>(
    n_machines: usize,
    flexibility: f64,
    rng: &mut R,
) -> Vec<u32> {
    let f = flexibility.clamp(0.0, 1.0);
    let n_eligible = ((n_machines as f64) * f).round() as usize;
    let n_eligible = n_eligible.max(1).min(n_machines);
    let mut pool: Vec<u32> = (0..n_machines as u32).collect();
    pool.shuffle(rng);
    pool.into_iter().take(n_eligible).collect()
}

pub fn generate<R: Rng + ?Sized>(name: &str, params: &GenParams, rng: &mut R) -> Instance {
    let machines: Vec<Machine> = (0..params.n_machines as u32).map(|id| Machine { id }).collect();

    let mut jobs = Vec::with_capacity(params.n_jobs);
    let mut next_release = 0.0_f64;

    for j in 0..params.n_jobs as u32 {
        // Number of operations per job: one per machine (Taillard convention).
        let n_ops = params.n_machines;

        let release_time = match params.arrival_rate {
            None => 0.0,
            Some(lambda) => {
                // Exponential inter-arrival.
                next_release += -((1.0 - rng.gen::<f64>()).ln()) / lambda;
                next_release
            }
        };

        // For strict JSSP (Taillard convention): one machine permutation
        // for the whole job so each machine is visited exactly once.
        let jssp_route: Vec<u32> = if params.flexibility <= 0.0 {
            let mut order: Vec<u32> = (0..params.n_machines as u32).collect();
            order.shuffle(rng);
            order
        } else {
            Vec::new()
        };

        let operations: Vec<Operation> = (0..n_ops)
            .map(|i| {
                let eligible = if params.flexibility <= 0.0 {
                    vec![jssp_route[i]]
                } else {
                    pick_eligible_machines(params.n_machines, params.flexibility, rng)
                };
                // FJSSP: each (op, machine) pair gets its own time. We draw a
                // base time then jitter it ±20% per machine, so faster
                // machines (drawn smaller) reward routing.
                let base = rng.gen_range(params.p_min..=params.p_max);
                let processing_times: Vec<f64> = if eligible.len() == 1 {
                    vec![base]
                } else {
                    eligible.iter().map(|_| {
                        let jitter = rng.gen_range(0.8..=1.2);
                        (base * jitter).max(1.0)
                    }).collect()
                };
                let mean_pt = processing_times.iter().sum::<f64>()
                    / processing_times.len() as f64;
                Operation {
                    job: j,
                    idx: i as u32,
                    processing_time: mean_pt,
                    eligible_machines: eligible,
                    processing_times,
                }
            })
            .collect();

        let total_p: f64 = operations.iter().map(|o| o.processing_time).sum();
        let urgent = rng.gen::<f64>() < params.urgent_p;
        let due_date = release_time + params.tightness * total_p;
        // Urgent jobs get tighter due dates and bigger penalty
        let (due_date, tardiness_penalty) = if urgent {
            (release_time + 0.7 * params.tightness * total_p, 5.0)
        } else {
            (due_date, 1.0)
        };

        jobs.push(Job {
            id: j,
            release_time,
            due_date,
            urgent,
            tardiness_penalty,
            operations,
            material_shortage_risk: 0.0,
            inbound_delay_time: 0.0,
        });
    }

    Instance { name: name.into(), jobs, machines }
}
