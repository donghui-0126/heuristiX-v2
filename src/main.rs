use anyhow::{anyhow, Result};
use clap::Parser;
use rand::SeedableRng;

use std::path::PathBuf;

use heuristix::jssp::brandimarte::{self, LoadParams};
use heuristix::jssp::generator::{generate, GenParams};
use heuristix::jssp::Instance;
use heuristix::rules::{baselines, ExprRule, PriorityRule};
use heuristix::runner::{headless, web::WebRunner};
use heuristix::scenarios::{Scenario, ScenarioParams};
use heuristix::sim::{SimConfig, SimulationEngine};

#[derive(Parser, Debug)]
#[command(name = "heuristix", about = "DES job-shop scheduler under supply-chain disruption")]
struct Cli {
    /// Scenario: S0 (normal) / S1 (part delay) / S2 (urgent order).
    #[arg(long, default_value = "S0")]
    scenario: String,

    /// Dispatching rule: FIFO / EDD / SPT / CR / URGENCY, or "expr"
    #[arg(long, default_value = "FIFO")]
    rule: String,

    /// Priority expression (used when --rule expr).
    #[arg(long)]
    expr: Option<String>,

    /// Number of jobs in the synthetic instance.
    #[arg(long, default_value_t = 10)]
    jobs: usize,

    /// Number of machines in the synthetic instance.
    #[arg(long, default_value_t = 5)]
    machines: usize,

    /// Replications.
    #[arg(long, default_value_t = 1)]
    replications: u32,

    /// Base RNG seed; replication k uses seed + k.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Launch the live Web Gantt UI (browser at http://127.0.0.1:PORT).
    #[arg(long)]
    web: bool,

    /// Web port.
    #[arg(long, default_value_t = 7878)]
    port: u16,

    /// Walltime ms per simulated minute (web mode). Smaller = faster.
    #[arg(long, default_value_t = 50)]
    ui_speed: u64,

    // ---- Scenario parameters (실험설계서_수정 §4) ----
    /// S1 — fraction of jobs whose first-op part is delayed. Spec: {0.10, 0.20, 0.40}.
    #[arg(long = "part-delay-ratio", default_value_t = 0.20)]
    part_delay_ratio: f64,
    /// S1 — delay multiplier k against mean total processing. Spec: {0.5, 1.0, 2.0}.
    #[arg(long = "part-delay-k", default_value_t = 1.0)]
    part_delay_k: f64,
    /// S2 — tightness ratio for the inserted urgent job's due date. Spec: {0.3, 0.5, 1.0}.
    #[arg(long = "urgent-due-ratio", default_value_t = 0.5)]
    urgent_due_ratio: f64,
    /// Due-date tightening factor (TWK + DDT) applied at instance generation.
    /// 1.0 = no tightening. Lower = tighter due dates.
    #[arg(long, default_value_t = 1.0)]
    ddt: f64,

    /// FJSSP routing flexibility in [0, 1].
    ///   0.0 = strict JSSP (each op has 1 machine)
    ///   0.3 = each op has ~30% of machines as alternatives
    ///   1.0 = any machine for any op
    #[arg(long, default_value_t = 0.0)]
    flexibility: f64,

    /// Load a Brandimarte-format FJSSP instance from disk instead of
    /// generating one synthetically. When set, `--jobs`/`--machines`/
    /// `--flexibility` are ignored (the file decides those). `--ddt` is
    /// still applied to the synthesised due dates.
    #[arg(long = "instance-file")]
    instance_file: Option<PathBuf>,

    /// TWK base tightness used when synthesising due dates for a loaded
    /// instance. effective_tightness = `tightness × ddt`.
    #[arg(long = "instance-tightness", default_value_t = 1.5)]
    instance_tightness: f64,

    /// Run this rule (e.g. FIFO) on the same instance/scenario for gap-ratio.
    /// Output JSON gains a `gap_ratio` field (vs total_tardiness).
    /// 창종설 보고서 §6-3.
    #[arg(long = "gap-baseline")]
    gap_baseline: Option<String>,

    /// Run the same rule under this scenario (typically S0) for schedule
    /// stability. Output JSON gains a `schedule_stability` field. §6-1.
    #[arg(long = "stability-baseline")]
    stability_baseline: Option<String>,

    /// If set, write the realised schedule (per-op start/end/machine) to
    /// this JSON path after the FIRST replication. Useful for Gantt
    /// rendering / comparison demos.
    #[arg(long = "dump-schedule")]
    dump_schedule: Option<PathBuf>,
}

impl Cli {
    fn scenario_params(&self) -> ScenarioParams {
        ScenarioParams {
            part_delay_ratio: self.part_delay_ratio,
            part_delay_k: self.part_delay_k,
            urgent_due_ratio: self.urgent_due_ratio,
            ddt: self.ddt,
            flexibility: self.flexibility,
        }
    }
}

fn build_rule(name: &str, expr: Option<&str>) -> Result<Box<dyn PriorityRule>> {
    if name.eq_ignore_ascii_case("expr") {
        let e = expr.ok_or_else(|| anyhow!("--rule expr requires --expr <formula>"))?;
        return Ok(Box::new(ExprRule::new("LLM-Expr", e)?));
    }
    baselines::by_name(name).ok_or_else(|| anyhow!("unknown rule: {name}"))
}

fn build_instance(
    cli: &Cli,
    name: &str,
    rng: &mut rand::rngs::StdRng,
) -> Result<Instance> {
    if let Some(path) = &cli.instance_file {
        let params = LoadParams {
            tightness: cli.instance_tightness,
            ddt: cli.ddt,
        };
        return brandimarte::load_from_path(path, &params)
            .map_err(|e| anyhow!("{e}"));
    }
    let params = GenParams {
        n_jobs: cli.jobs,
        n_machines: cli.machines,
        flexibility: cli.flexibility,
        ..Default::default()
    };
    Ok(generate(name, &params, rng))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let scenario = Scenario::from_name(&cli.scenario)
        .ok_or_else(|| anyhow!("unknown scenario: {}", cli.scenario))?;

    if cli.web {
        // Validate rule expression at startup so a typo doesn't only show on restart.
        let _probe = build_rule(&cli.rule, cli.expr.as_deref())?;
        drop(_probe);

        let rule_name = cli.rule.clone();
        let expr_str = cli.expr.clone();
        let scenario_name = cli.scenario.clone();
        let n_jobs = cli.jobs;
        let n_machines = cli.machines;
        let seed = cli.seed;
        let flexibility = cli.flexibility;
        let scen_params = cli.scenario_params();

        let instance_file = cli.instance_file.clone();
        let instance_tightness = cli.instance_tightness;
        let ddt_for_load = cli.ddt;
        let factory = move || {
            let mut rng_inst = rand::rngs::StdRng::seed_from_u64(seed);
            let mut rng_dis  = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF));
            let instance = if let Some(path) = instance_file.as_ref() {
                brandimarte::load_from_path(
                    path,
                    &LoadParams { tightness: instance_tightness, ddt: ddt_for_load },
                ).expect("failed to load Brandimarte instance")
            } else {
                let params = GenParams { n_jobs, n_machines, flexibility, ..Default::default() };
                generate("synth-web", &params, &mut rng_inst)
            };
            let scenario = Scenario::from_name(&scenario_name).unwrap();
            let disruptions = scenario.build(&instance, &scen_params, &mut rng_dis);
            let rule = build_rule(&rule_name, expr_str.as_deref())
                .expect("rule rebuild failed (validated at startup)");
            let cfg = SimConfig { seed, ..Default::default() };
            SimulationEngine::new(instance, rule, disruptions, cfg)
        };

        let runner = WebRunner::new(factory, cli.scenario.clone(), cli.port, cli.ui_speed);
        runner.run()?;
        return Ok(());
    }

    let scen_params = cli.scenario_params();
    for k in 0..cli.replications {
        let seed = cli.seed.wrapping_add(k as u64);

        // Helper to build a fresh (instance, disruptions) pair for a given
        // scenario, using identical RNG seeding so rule comparisons are CRN.
        let build_run = |scen: Scenario, rule_name: &str,
                         dump_schedule_to: Option<&PathBuf>|
            -> Result<heuristix::sim::Metrics>
        {
            let mut rng_inst = rand::rngs::StdRng::seed_from_u64(seed);
            let mut rng_dis = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF));
            let instance = build_instance(&cli, &format!("synth-{k}"), &mut rng_inst)?;
            let disruptions = scen.build(&instance, &scen_params, &mut rng_dis);
            let rule = build_rule(rule_name, cli.expr.as_deref())?;
            let cfg = SimConfig { seed, ..Default::default() };
            let engine = SimulationEngine::new(instance, rule, disruptions, cfg);
            if let Some(path) = dump_schedule_to {
                let (m, snap) = headless::run_with_snapshot(engine);
                std::fs::write(path, serde_json::to_string(&snap)?)?;
                Ok(m)
            } else {
                Ok(headless::run(engine))
            }
        };

        // Only dump on the first replication.
        let dump_first = if k == 0 { cli.dump_schedule.as_ref() } else { None };
        let m = build_run(scenario, &cli.rule, dump_first)?;

        // Optional gap-ratio: same scenario, different rule.
        let gap = if let Some(base_rule) = &cli.gap_baseline {
            let m_base = build_run(scenario, base_rule, None)?;
            heuristix::sim::gap_ratio(m.total_tardiness, m_base.total_tardiness)
        } else { None };

        // Optional schedule stability: same rule, different (calmer) scenario.
        let stability = if let Some(base_scen_name) = &cli.stability_baseline {
            let base_scen = Scenario::from_name(base_scen_name)
                .ok_or_else(|| anyhow!("unknown stability-baseline scenario: {base_scen_name}"))?;
            let m_base = build_run(base_scen, &cli.rule, None)?;
            heuristix::sim::schedule_stability(&m_base, &m)
        } else { None };

        println!("{}", serde_json::to_string(&serde_json::json!({
            "scenario": cli.scenario,
            "rule": cli.rule,
            "rep": k,
            "metrics": m,
            "gap_ratio": gap,
            "schedule_stability": stability,
        }))?);
    }
    Ok(())
}
