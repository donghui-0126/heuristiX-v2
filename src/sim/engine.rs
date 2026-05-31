use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::disruption::DisruptionEvent;
use crate::jssp::Instance;
use crate::rules::PriorityRule;
use crate::sim::clock::SimClock;
use crate::sim::event::{Event, EventKind};
use crate::sim::metrics::{self, Metrics};
use crate::sim::state::{OpKey, OpStatus, ShopState};

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub horizon: f64,         // max sim minutes (safety bound)
    pub seed: u64,
    pub render_tick: Option<f64>,
}
impl Default for SimConfig {
    fn default() -> Self { Self { horizon: 1e6, seed: 42, render_tick: None } }
}

pub struct SimulationEngine {
    pub clock: SimClock,
    pub state: ShopState,
    pub rule: Box<dyn PriorityRule>,
    pub config: SimConfig,
    events: BinaryHeap<Reverse<Event>>,
    next_seq: u64,
}

impl SimulationEngine {
    pub fn new(
        instance: Instance,
        rule: Box<dyn PriorityRule>,
        disruptions: Vec<DisruptionEvent>,
        config: SimConfig,
    ) -> Self {
        let state = ShopState::new(instance);
        let mut engine = Self {
            clock: SimClock::new(),
            state,
            rule,
            config,
            events: BinaryHeap::with_capacity(1024),
            next_seq: 0,
        };
        engine.bootstrap(disruptions);
        engine
    }

    fn bootstrap(&mut self, disruptions: Vec<DisruptionEvent>) {
        // Schedule a JobArrival for each job at its release_time (so dynamically
        // arriving jobs go through the same path as t=0 jobs).
        let n_jobs = self.state.instance.jobs.len();
        for j in 0..n_jobs {
            let job = &self.state.instance.jobs[j];
            let t = job.release_time.max(0.0);
            self.push(t, EventKind::JobArrival { job: job.id });
        }

        // Inject all disruption events.
        for d in disruptions {
            match d {
                DisruptionEvent::PartArrival { at, job, op } => {
                    if let Some(rec) = self.state.ops.get_mut(&OpKey { job, op }) {
                        // Record per-op part arrival time AND accumulate the
                        // imposed delay onto the job, so rules can read
                        // job.inbound_delay_time as a feature.
                        let prev = rec.part_available_time;
                        rec.part_available_time = rec.part_available_time.max(at);
                        let added = (rec.part_available_time - prev).max(0.0);
                        if added > 0.0 {
                            if let Some(j) = self.state.instance.jobs.get_mut(job as usize) {
                                j.inbound_delay_time += added;
                            }
                        }
                    }
                    self.push(at, EventKind::PartArrival { job, op });
                }
                DisruptionEvent::DueShift { at, job, new_due } => {
                    self.push(at, EventKind::DueShift { job, new_due });
                }
                DisruptionEvent::Breakdown { at, machine, repair_at } => {
                    self.push(at, EventKind::Breakdown { machine });
                    self.push(repair_at, EventKind::Repair { machine });
                }
                DisruptionEvent::MaterialShortage { at, job, level } => {
                    self.push(at, EventKind::MaterialShortage { job, level });
                }
                DisruptionEvent::JobInsert { job } => {
                    let release = job.release_time;
                    let job_id = job.id;
                    // Inject job into instance and create op records.
                    self.state.instance.jobs.push(job.clone());
                    self.state.next_op_idx.insert(job.id, 0);
                    self.state.job_arrived.insert(job.id, false);
                    for (i, _) in job.operations.iter().enumerate() {
                        self.state.ops.insert(OpKey { job: job.id, op: i as u32 },
                            crate::sim::state::OpRecord {
                                status: OpStatus::Blocked,
                                part_available_time: job.release_time,
                                start_time: None,
                                finish_time: None,
                                assigned_machine: None,
                            });
                    }
                    self.push(release, EventKind::JobArrival { job: job_id });
                }
            }
        }
        self.state.recompute_supply_aggregates();
        // Seed the feasibility integrator at t=0 (initial ratio).
        self.state.tick_feasibility(0.0);

        if let Some(dt) = self.config.render_tick {
            let mut t = dt;
            while t < self.config.horizon { self.push(t, EventKind::RenderTick); t += dt; }
        }
    }

    fn push(&mut self, time: f64, kind: EventKind) {
        let seq = self.next_seq; self.next_seq += 1;
        self.events.push(Reverse(Event { time, seq, kind }));
    }

    pub fn run_to_end(&mut self) -> Metrics {
        while let Some(Reverse(ev)) = self.events.pop() {
            if ev.time > self.config.horizon { break; }
            self.advance(ev);
            if self.all_jobs_done() { break; }
        }
        self.finalize()
    }

    /// Process exactly one event. Returns whether more work remains.
    pub fn step(&mut self) -> StepOutcome {
        let Some(Reverse(ev)) = self.events.pop() else { return StepOutcome::Done };
        if ev.time > self.config.horizon { return StepOutcome::Done; }
        let kind = ev.kind;
        self.advance(ev);
        if matches!(kind, EventKind::RenderTick) { return StepOutcome::RenderFrame; }
        if self.all_jobs_done() { return StepOutcome::Done; }
        StepOutcome::Progressed
    }

    /// Time of the next event in the queue, if any.
    pub fn next_event_time(&self) -> Option<f64> {
        self.events.peek().map(|Reverse(e)| e.time)
    }

    fn all_jobs_done(&self) -> bool {
        self.state.instance.jobs.iter().all(|j| {
            let last = j.operations.len() as u32 - 1;
            self.state.ops.get(&OpKey { job: j.id, op: last })
                .map(|r| r.status == OpStatus::Done)
                .unwrap_or(false)
        })
    }

    fn advance(&mut self, ev: Event) {
        self.clock.advance_to(ev.time);
        self.state.now = ev.time;

        match ev.kind {
            EventKind::JobArrival { job } => {
                self.state.tick_feasibility(ev.time);
                self.state.job_arrived.insert(job, true);
                self.refresh_readiness_for_job(job);
                self.state.recompute_supply_aggregates();
                self.try_dispatch();
            }
            EventKind::PartArrival { job, op } => {
                self.state.tick_feasibility(ev.time);
                if let Some(rec) = self.state.ops.get_mut(&OpKey { job, op }) {
                    if rec.part_available_time > ev.time { /* already future-set */ }
                    else { rec.part_available_time = ev.time; }
                }
                self.refresh_readiness_for_op(OpKey { job, op });
                self.state.recompute_supply_aggregates();
                self.try_dispatch();
            }
            EventKind::OpComplete { job, op, machine } => {
                self.state.tick_feasibility(ev.time);
                self.handle_op_complete(OpKey { job, op }, machine);
                self.state.recompute_supply_aggregates();
                self.try_dispatch();
            }
            EventKind::Breakdown { machine } => {
                self.state.tick_feasibility(ev.time);
                let mr = self.state.machines.get_mut(&machine).unwrap();
                // If the machine was mid-op when it broke, flush busy minutes
                // up to now so they don't get lost (the op continues without
                // preemption — a deliberate simplification of the report's
                // "rescheduling_trigger" model).
                if mr.busy_with.is_some() {
                    mr.busy_total += ev.time - mr.last_change;
                } else {
                    // It was idle-and-up; flush starved idle to idle_total.
                    let dt = ev.time - mr.last_change;
                    if dt > 0.0 { mr.idle_total += dt; }
                }
                mr.down = true;
                mr.last_change = ev.time;
            }
            EventKind::Repair { machine } => {
                self.state.tick_feasibility(ev.time);
                let mr = self.state.machines.get_mut(&machine).unwrap();
                let dt = ev.time - mr.last_change;
                mr.down = false;
                mr.idle_total += dt;
                mr.idle_breakdown += dt;
                mr.last_change = ev.time;
                self.try_dispatch();
            }
            EventKind::DueShift { job, new_due } => {
                if let Some(j) = self.state.instance.jobs.get_mut(job as usize) {
                    j.due_date = new_due;
                }
            }
            EventKind::MaterialShortage { job, level } => {
                if let Some(j) = self.state.instance.jobs.get_mut(job as usize) {
                    // Monotonically rising risk; multiple events on same job
                    // pick the worst level so far.
                    j.material_shortage_risk = j.material_shortage_risk.max(level).min(1.0);
                }
                self.state.recompute_supply_aggregates();
                self.try_dispatch();
            }
            EventKind::RenderTick => {}
        }
    }

    /// Move blocked → ready transitions where possible, for a single job's head op.
    fn refresh_readiness_for_job(&mut self, job: u32) {
        if let Some(head) = self.state.head_op(job) {
            self.refresh_readiness_for_op(head);
        }
    }

    fn refresh_readiness_for_op(&mut self, key: OpKey) {
        let now = self.clock.now();
        let arrived = *self.state.job_arrived.get(&key.job).unwrap_or(&false);
        let head = self.state.head_op(key.job) == Some(key);
        let rec = match self.state.ops.get_mut(&key) { Some(r) => r, None => return };
        if rec.status != OpStatus::Blocked { return; }
        if arrived && head && rec.part_available_time <= now {
            rec.status = OpStatus::Ready;
        }
    }

    fn handle_op_complete(&mut self, key: OpKey, machine: u32) {
        let now = self.clock.now();
        // Mark op done.
        if let Some(rec) = self.state.ops.get_mut(&key) {
            rec.status = OpStatus::Done;
            rec.finish_time = Some(now);
        }
        // Free machine.
        let mr = self.state.machines.get_mut(&machine).unwrap();
        mr.busy_total += now - mr.last_change;
        mr.busy_with = None;
        mr.last_change = now;

        // Advance job pointer.
        let next = self.state.next_op_idx.get(&key.job).copied().unwrap_or(0) + 1;
        self.state.next_op_idx.insert(key.job, next);

        // The newly exposed head op (if any) might be ready.
        self.refresh_readiness_for_job(key.job);
    }

    /// Match idle machines to ready operations using the priority rule.
    fn try_dispatch(&mut self) {
        let now = self.clock.now();
        // Collect machine ids first to avoid borrow issues.
        let machine_ids: Vec<u32> = self.state.instance.machines.iter().map(|m| m.id).collect();

        for mid in machine_ids {
            // Skip non-idle machines.
            let idle = self.state.machines.get(&mid).map(|r| r.is_idle()).unwrap_or(false);
            if !idle { continue; }

            // Candidate ready ops eligible for this machine (FJSSP: an op
            // can be eligible for multiple machines).
            let mut best: Option<(OpKey, f64)> = None;
            for key in self.state.ready_keys() {
                if !self.state.op_eligible_machines(key).contains(&mid) { continue; }
                let job = self.state.job(key.job).clone();
                let op = job.operations[key.op as usize].clone();
                let machine_view = MachineView { id: mid };
                let s = self.rule.score(&job, &op, &machine_view, &self.state);
                if best.map(|(_, sb)| s > sb).unwrap_or(true) {
                    best = Some((key, s));
                }
            }

            if let Some((key, _)) = best {
                self.start_op_on_machine(key, mid, now);
            } else {
                // No ready op for this machine — accumulate starved idle time.
                let mr = self.state.machines.get_mut(&mid).unwrap();
                let dt = now - mr.last_change;
                if dt > 0.0 { mr.idle_total += dt; mr.last_change = now; }
            }
        }
    }

    fn start_op_on_machine(&mut self, key: OpKey, machine: u32, now: f64) {
        let p = self.state.op_processing(key, machine);

        // Account for any starvation since last change.
        let mr = self.state.machines.get_mut(&machine).unwrap();
        let dt = now - mr.last_change;
        if dt > 0.0 { mr.idle_total += dt; }
        mr.busy_with = Some(key);
        mr.last_change = now;

        // Move op to running. Record which machine we actually dispatched
        // it to (matters for FJSSP where it's a runtime choice).
        let rec = self.state.ops.get_mut(&key).unwrap();
        rec.status = OpStatus::Running;
        rec.start_time = Some(now);
        rec.assigned_machine = Some(machine);

        // Schedule completion.
        self.push(now + p, EventKind::OpComplete { job: key.job, op: key.op, machine });
    }

    pub fn finalize(&self) -> Metrics {
        metrics::finalize(&self.state, &self.state.instance)
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        let mut ops = Vec::with_capacity(self.state.ops.len());
        for job in &self.state.instance.jobs {
            for (idx, op) in job.operations.iter().enumerate() {
                let key = OpKey { job: job.id, op: idx as u32 };
                let rec = match self.state.ops.get(&key) { Some(r) => r, None => continue };
                // For FJSSP: prefer the actually-assigned machine (running/done);
                // for not-yet-dispatched ops show first eligible as canonical.
                let machine = rec.assigned_machine.unwrap_or(op.eligible_machines[0]);
                let status = match rec.status {
                    OpStatus::Blocked => "Blocked",
                    OpStatus::Ready   => "Ready",
                    OpStatus::Running => "Running",
                    OpStatus::Done    => "Done",
                };
                ops.push(OpSnap {
                    job: job.id,
                    op: idx as u32,
                    machine,
                    processing_time: op.processing_time,
                    part_available_time: rec.part_available_time,
                    start: rec.start_time,
                    end: rec.finish_time,
                    status: status.to_string(),
                });
            }
        }

        let machines: Vec<MachineSnap> = self.state.instance.machines.iter().map(|m| {
            let mr = self.state.machines.get(&m.id).unwrap();
            MachineSnap {
                id: m.id,
                down: mr.down,
                busy_with: mr.busy_with.map(|k| (k.job, k.op)),
            }
        }).collect();

        let jobs: Vec<JobSnap> = self.state.instance.jobs.iter().map(|j| {
            let last = j.operations.len() as u32 - 1;
            let done = self.state.ops.get(&OpKey { job: j.id, op: last })
                .map(|r| r.status == OpStatus::Done).unwrap_or(false);
            JobSnap {
                id: j.id,
                release: j.release_time,
                due: j.due_date,
                urgent: j.urgent,
                done,
            }
        }).collect();

        EngineSnapshot {
            now: self.clock.now(),
            rule_name: self.rule.name().to_string(),
            n_jobs: self.state.instance.jobs.len(),
            n_machines: self.state.instance.machines.len(),
            machines,
            jobs,
            ops,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OpSnap {
    pub job: u32,
    pub op: u32,
    pub machine: u32,
    pub processing_time: f64,
    pub part_available_time: f64,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub status: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MachineSnap {
    pub id: u32,
    pub down: bool,
    pub busy_with: Option<(u32, u32)>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct JobSnap {
    pub id: u32,
    pub release: f64,
    pub due: f64,
    pub urgent: bool,
    pub done: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineSnapshot {
    pub now: f64,
    pub rule_name: String,
    pub n_jobs: usize,
    pub n_machines: usize,
    pub machines: Vec<MachineSnap>,
    pub jobs: Vec<JobSnap>,
    pub ops: Vec<OpSnap>,
}

/// Lightweight view passed to priority rules for the candidate machine.
#[derive(Debug, Clone, Copy)]
pub struct MachineView {
    pub id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepOutcome {
    Progressed,
    RenderFrame,
    Done,
}
