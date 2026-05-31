use anyhow::{anyhow, Result};
use evalexpr::{context_map, eval_with_context, ContextWithMutableFunctions, Function, HashMapContext, Value};

use crate::jssp::{Job, Operation};
use crate::rules::PriorityRule;
use crate::sim::engine::MachineView;
use crate::sim::state::{OpKey, OpStatus, ShopState};

/// LLM-emitted priority expression.
///
/// Bindings exposed to the formula (창종설 보고서 §4-2 priority signature):
///   Job:        release, due, slack, urgent (1/0), penalty,
///               total_proc, remaining_proc,
///               part_avail, time_to_avail, mat_risk, inbound_delay
///   Op:         proc, op_idx
///   Machine:    machine_id, machine_queue, mach_util, mach_down
///   State:      now, n_ready, n_running, n_jobs,
///               supply_delay_level, urgent_ratio, disruption_level,
///               avg_inbound_delay
///
/// Helper functions (numeric in / numeric out, 0/1 for booleans):
///   iff(c, t, e)       — c≠0 → t else e
///   gt(a, b), lt(a, b), eq(a, b)  — 1.0 / 0.0
///   max_(a, b), min_(a, b)
///   clamp(x, lo, hi)
///   exp_(x)            — natural exponential (for ATC-style rules)
///
/// Convention: higher score = higher priority. Feasibility (part_avail ≤ now)
/// is enforced by the engine before the rule is called, so a rule may rely on
/// `part_avail` and `time_to_avail` purely as features.
pub struct ExprRule {
    name: String,
    expr: String,
}

fn as_f64(v: &Value) -> evalexpr::EvalexprResult<f64> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(evalexpr::EvalexprError::expected_number(other.clone())),
    }
}

fn pair(v: &Value) -> evalexpr::EvalexprResult<(f64, f64)> {
    let t = v.as_fixed_len_tuple(2)?;
    Ok((as_f64(&t[0])?, as_f64(&t[1])?))
}

fn triple(v: &Value) -> evalexpr::EvalexprResult<(f64, f64, f64)> {
    let t = v.as_fixed_len_tuple(3)?;
    Ok((as_f64(&t[0])?, as_f64(&t[1])?, as_f64(&t[2])?))
}

/// Register the helper functions on a context. Called for both the startup
/// compile-check context and the per-score context so they stay in sync.
fn install_helpers(ctx: &mut HashMapContext) -> Result<()> {
    ctx.set_function("iff".into(), Function::new(|v| {
        let (c, t, e) = triple(v)?;
        Ok(Value::Float(if c.abs() > f64::EPSILON { t } else { e }))
    }))?;
    ctx.set_function("gt".into(), Function::new(|v| {
        let (a, b) = pair(v)?; Ok(Value::Float(if a > b { 1.0 } else { 0.0 }))
    }))?;
    ctx.set_function("lt".into(), Function::new(|v| {
        let (a, b) = pair(v)?; Ok(Value::Float(if a < b { 1.0 } else { 0.0 }))
    }))?;
    ctx.set_function("eq".into(), Function::new(|v| {
        let (a, b) = pair(v)?;
        Ok(Value::Float(if (a - b).abs() < 1e-9 { 1.0 } else { 0.0 }))
    }))?;
    ctx.set_function("max_".into(), Function::new(|v| {
        let (a, b) = pair(v)?; Ok(Value::Float(a.max(b)))
    }))?;
    ctx.set_function("min_".into(), Function::new(|v| {
        let (a, b) = pair(v)?; Ok(Value::Float(a.min(b)))
    }))?;
    ctx.set_function("clamp".into(), Function::new(|v| {
        // Swap bounds if the LLM passed lo > hi (otherwise f64::clamp panics).
        // Also coerce NaN to 0.0 so a malformed rule degrades to -inf score
        // via the engine's downstream handling rather than crashing.
        let (x, lo, hi) = triple(v)?;
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        let x = if x.is_nan() { 0.0 } else { x };
        Ok(Value::Float(x.max(lo).min(hi)))
    }))?;
    // exp_(x) — natural exponential. Needed for the ATC dispatching rule
    // (Vepsalainen & Morton 1987).
    ctx.set_function("exp_".into(), Function::new(|v| {
        Ok(Value::Float(as_f64(v)?.exp()))
    }))?;
    Ok(())
}

impl ExprRule {
    pub fn new(name: impl Into<String>, expr: impl Into<String>) -> Result<Self> {
        let rule = Self { name: name.into(), expr: expr.into() };
        // Compile-time sanity eval against a dummy context (catch typos at startup).
        let mut ctx: HashMapContext = context_map! {
            "release" => 0.0, "due" => 1.0, "slack" => 1.0, "urgent" => 0i64,
            "penalty" => 1.0, "total_proc" => 1.0, "remaining_proc" => 1.0,
            "part_avail" => 0.0, "time_to_avail" => 0.0,
            "mat_risk" => 0.0, "inbound_delay" => 0.0,
            "proc" => 1.0, "op_idx" => 0i64,
            "machine_id" => 0i64, "machine_queue" => 0i64,
            "mach_util" => 0.0, "mach_down" => 0i64,
            "now" => 0.0, "n_ready" => 0i64, "n_running" => 0i64, "n_jobs" => 1i64,
            "supply_delay_level" => 0.0, "urgent_ratio" => 0.0,
            "disruption_level" => 0.0, "avg_inbound_delay" => 0.0,
            // Aliases — common LLM hallucinations. Same values as the
            // canonical names, accepted as a safety net.
            "urgency" => 0i64,
            "processing_time" => 1.0,
            "due_date" => 1.0, "due_time" => 1.0,
            "material_shortage_risk" => 0.0,
            "machine_utilization" => 0.0,
            "machine_breakdown_flag" => 0i64,
            "remaining_processing_time" => 1.0,
            // Spec §3-2 variable names (실험설계서_수정). Map to the
            // canonical engine values so the LLM can write expressions
            // using the spec wording directly.
            "release_time" => 0.0,
            "remaining_pt" => 1.0,
            "part_available_time" => 0.0,
            "urgent_order_flag" => 0i64,
            "eligible_machines" => 1i64,
            "n_eligible" => 1i64,
            "machine_available_time" => 0.0,
            "current_time" => 0.0,
        }?;
        install_helpers(&mut ctx)?;
        eval_with_context(&rule.expr, &ctx)
            .map_err(|e| anyhow!("priority expression failed compile-check: {e}"))?;
        Ok(rule)
    }
}

impl PriorityRule for ExprRule {
    fn name(&self) -> &str { &self.name }

    fn score(&self, job: &Job, op: &Operation, machine: &MachineView, s: &ShopState) -> f64 {
        let total_proc = job.total_processing();
        // FJSSP: per-machine time for the candidate op. For ops further
        // down the pipeline we don't yet know which machine they'll go to,
        // so `remaining_proc` uses each op's mean across eligible machines.
        let proc_here = op.proc_time_on(machine.id);
        let remaining_proc: f64 = (0..job.operations.len() as u32)
            .filter_map(|i| {
                let key = OpKey { job: job.id, op: i };
                let r = s.ops.get(&key)?;
                if r.status == OpStatus::Done { None } else { Some(job.operations[i as usize].processing_time) }
            })
            .sum();
        let n_ready = s.ops.values().filter(|r| r.status == OpStatus::Ready).count() as i64;
        let n_running = s.ops.values().filter(|r| r.status == OpStatus::Running).count() as i64;

        // Per-op part-arrival time (feasibility feature for the rule). For ops
        // already past blocking, this equals the op's release-time floor.
        let key = OpKey { job: job.id, op: op.idx };
        let part_avail = s.ops.get(&key).map(|r| r.part_available_time).unwrap_or(0.0);
        let time_to_avail = (part_avail - s.now).max(0.0);

        // Machine-level signals.
        let mach = s.machines.get(&machine.id);
        let mach_down = mach.map(|m| m.down as i64).unwrap_or(0);
        let total_time = (s.now).max(1e-6);
        let mach_util = mach.map(|m| (m.busy_total / total_time).min(1.0)).unwrap_or(0.0);
        // Per-machine queue: count Ready ops eligible for this machine.
        let machine_queue = s.ops.iter().filter(|(k, r)| {
            r.status == OpStatus::Ready
                && s.instance.jobs[k.job as usize].operations[k.op as usize]
                    .eligible_machines.contains(&machine.id)
        }).count() as i64;

        let mut ctx: HashMapContext = context_map! {
            "release" => job.release_time,
            "due" => job.due_date,
            "slack" => job.due_date - s.now,
            "urgent" => job.urgent as i64,
            "penalty" => job.tardiness_penalty,
            "total_proc" => total_proc,
            "remaining_proc" => remaining_proc,
            "part_avail" => part_avail,
            "time_to_avail" => time_to_avail,
            "mat_risk" => job.material_shortage_risk,
            "inbound_delay" => job.inbound_delay_time,
            "proc" => proc_here,
            "op_idx" => op.idx as i64,
            "machine_id" => machine.id as i64,
            "machine_queue" => machine_queue,
            "mach_util" => mach_util,
            "mach_down" => mach_down,
            "now" => s.now,
            "n_ready" => n_ready,
            "n_running" => n_running,
            "n_jobs" => s.instance.jobs.len() as i64,
            "supply_delay_level" => s.supply_delay_level,
            "urgent_ratio" => s.urgent_order_ratio,
            "disruption_level" => s.disruption_level,
            "avg_inbound_delay" => s.avg_inbound_delay,
            // Aliases (must match the startup compile-check context).
            "urgency" => job.urgent as i64,
            "processing_time" => proc_here,
            "due_date" => job.due_date,
            "due_time" => job.due_date,
            "material_shortage_risk" => job.material_shortage_risk,
            "machine_utilization" => mach_util,
            "machine_breakdown_flag" => mach_down,
            "remaining_processing_time" => remaining_proc,
            // Spec §3-2 names.
            "release_time" => job.release_time,
            "remaining_pt" => remaining_proc,
            "part_available_time" => part_avail,
            "urgent_order_flag" => job.urgent as i64,
            "eligible_machines" => op.eligible_machines.len() as i64,
            "n_eligible" => op.eligible_machines.len() as i64,
            // Candidate machine is idle at scoring time → available now.
            "machine_available_time" => s.now,
            "current_time" => s.now,
        }.unwrap();
        // Failure to install helpers here would mean a malformed expr; fall
        // through to the eval which will return the same error path.
        let _ = install_helpers(&mut ctx);

        match eval_with_context(&self.expr, &ctx) {
            Ok(Value::Float(f)) => f,
            Ok(Value::Int(i))   => i as f64,
            _ => f64::MIN,
        }
    }
}
