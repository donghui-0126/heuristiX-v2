"""DSevolve-style heuristic portfolio (창종설 §8 + DSevolve Huang 2026).

Idea: evolve a specialist dispatching rule per scenario, then compose
them into a single portfolio expression that switches between them at
runtime using the supply-chain state signals exposed in §3-2:

    if disruption_level > τ_disrupt:      use rule trained on S5
    elif supply_delay_level > 0:          use rule trained on S1
    elif urgent_ratio > τ_urgent:         use rule trained on S3
    elif mat_risk > τ_mat:                use rule trained on S2
    elif slack < proc * 1.5:              use rule trained on S4   (tight slack proxy)
    else:                                 use rule trained on S0

The composed portfolio is a single evalexpr expression (nested `iff`)
that the simulator can run via `--rule expr --expr "..."`.

Run:
    # End-to-end: evolve each scenario, compose, evaluate.
    python -m evolution.portfolio --iterations 3 --replications 5

    # Use externally-evolved rules:
    python -m evolution.portfolio --rules-file runs/per_scenario.json

    # Save the composed portfolio for later reuse:
    python -m evolution.portfolio --out runs/portfolio_v1.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

from .baselines import BASELINES
from .evolve import _best_baseline_score, _classify_for_memory, _evaluate_population
from .llm import build_llm
from .memory import MemoryBank
from .robustness import SCENARIOS, evaluate_rule
from .simulator import RunResult, Simulator, weights_for


# Switching thresholds. Chosen to match the bins emitted by
# `ShopState::recompute_supply_aggregates` so the portfolio reacts to
# the same step-changes the simulator produces.
THRESHOLDS = {
    "disruption": 0.45,
    "supply_delay": 0.5,
    "urgent_ratio": 0.20,
    "mat_risk": 0.30,
    "slack_tightness": 1.5,  # head op is "tight" when slack < proc * τ
}


# ---- per-scenario evolution (small loop) ------------------------------

def evolve_one_scenario(
    scenario: str,
    *,
    iterations: int,
    sim_kwargs: dict,
    llm_provider: str = "mock",
    llm_model: str | None = None,
    success_threshold: float = 0.05,
) -> tuple[str, RunResult]:
    """Run a small evolution loop on one scenario; return (best_expr, best_result).

    Mirrors `evolve.evolve` but trimmed: no I/O, single-scenario, returns
    the artefact in-memory."""
    sim = Simulator(scenario=scenario, gap_baseline="FIFO", **sim_kwargs)
    gen_llm = build_llm(llm_provider, model=llm_model)
    reflect_llm = gen_llm
    memory = MemoryBank()
    population: list[str] = list(BASELINES.values())
    for _ in range(3):
        _, c = gen_llm.generate_rule(scenario, [], memory, operation="explore")
        population.append(c)

    cache: dict[str, RunResult] = {}
    best: RunResult | None = None

    for _ in range(iterations):
        results = _evaluate_population(sim, population, cache)
        results.sort(key=lambda r: r.primary_objective)
        elite = results[:3]
        _, baseline_score = _best_baseline_score(sim, cache)
        successes, failures = _classify_for_memory(results, baseline_score, success_threshold)
        memory.add_many(reflect_llm.reflect(scenario, successes, failures))
        if best is None or elite[0].primary_objective < best.primary_objective:
            best = elite[0]
        new = []
        for _ in range(2):
            _, c = gen_llm.generate_rule(scenario, elite, memory, operation="explore")
            new.append(c)
        _, c = gen_llm.generate_rule(scenario, elite, memory, operation="crossover")
        new.append(c)
        _, c = gen_llm.generate_rule(scenario, elite, memory, operation="modify")
        new.append(c)
        _, c = gen_llm.generate_rule(scenario, elite, memory, operation="simplify")
        new.append(c)
        population = [r.expr for r in elite] + new

    assert best is not None
    return best.expr, best


# ---- portfolio composition --------------------------------------------

def compose_portfolio(rules: dict[str, str], thresholds: dict[str, float] | None = None) -> str:
    """Compose a single DSL expression that dispatches based on runtime
    state signals. Missing scenarios fall back to S0 or to ATC."""
    th = thresholds or THRESHOLDS

    # Sensible fallbacks: ATC is our strongest single rule, S0 default.
    default_s0 = rules.get("S0", BASELINES["ATC"])
    r_s1 = rules.get("S1", default_s0)
    r_s2 = rules.get("S2", default_s0)
    r_s3 = rules.get("S3", default_s0)
    r_s4 = rules.get("S4", default_s0)
    r_s5 = rules.get("S5", default_s0)

    # Nested iff hierarchy. We test the most specific signals first so
    # they aren't shadowed by the composite disruption_level.
    return (
        f"iff(gt(disruption_level, {th['disruption']}),"
        f"    ({r_s5}),"
        f"    iff(gt(supply_delay_level, {th['supply_delay']}),"
        f"        ({r_s1}),"
        f"        iff(gt(urgent_ratio, {th['urgent_ratio']}),"
        f"            ({r_s3}),"
        f"            iff(gt(mat_risk, {th['mat_risk']}),"
        f"                ({r_s2}),"
        f"                iff(lt((due - now), proc * {th['slack_tightness']}),"
        f"                    ({r_s4}),"
        f"                    ({default_s0}))))))"
    )


# ---- driver -----------------------------------------------------------

def _load_rules_file(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "rules" in raw:
        raw = raw["rules"]
    if not isinstance(raw, dict):
        raise SystemExit(f"--rules-file must be a JSON object mapping scenario→expr; got {type(raw).__name__}")
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def run_portfolio(args: argparse.Namespace) -> dict:
    sim_kwargs = dict(
        jobs=args.jobs, machines=args.machines, replications=args.replications,
        seed=args.seed, eave=args.eave, n_add=args.n_add, ddt=args.ddt,
        breakdown_rate=args.breakdown_rate,
    )

    # --- 1) gather per-scenario specialist rules ---
    if args.rules_file:
        rules = _load_rules_file(Path(args.rules_file))
        print(f"loaded {len(rules)} per-scenario rules from {args.rules_file}")
        per_scenario_results = None
    else:
        rules = {}
        per_scenario_results = {}
        print(f"evolving {len(SCENARIOS)} specialist rules ({args.iterations} iterations each) "
              f"using {args.provider}/{args.model or 'default'}…")
        for scen in SCENARIOS:
            t0 = time.time()
            expr, result = evolve_one_scenario(
                scen,
                iterations=args.iterations,
                sim_kwargs=sim_kwargs,
                llm_provider=args.provider,
                llm_model=args.model,
                success_threshold=args.success_threshold,
            )
            rules[scen] = expr
            per_scenario_results[scen] = {
                "expr": expr,
                "primary_objective": result.primary_objective,
                "mean_tardiness": result.mean_tardiness,
                "feasible_job_ratio": result.feasible_job_ratio,
                "gap_vs_fifo": result.gap_ratio,
            }
            dt = time.time() - t0
            print(f"  {scen}: obj={result.primary_objective:.0f}  ({dt:.1f}s)  expr: {expr[:80]}{'…' if len(expr) > 80 else ''}")

    # --- 2) compose portfolio ---
    portfolio_expr = compose_portfolio(rules)
    print(f"\nportfolio length: {len(portfolio_expr)} chars")

    # --- 3) evaluate portfolio across all scenarios ---
    print("evaluating portfolio across S0–S5…")
    rep = evaluate_rule(
        portfolio_expr, "Portfolio",
        jobs=args.jobs, machines=args.machines, replications=args.replications,
        seed=args.seed, eave=args.eave, n_add=args.n_add,
        ddt=args.ddt, breakdown_rate=args.breakdown_rate,
    )

    # --- 4) baseline comparison: ATC robustness for context ---
    atc_rep = evaluate_rule(
        BASELINES["ATC"], "ATC",
        jobs=args.jobs, machines=args.machines, replications=args.replications,
        seed=args.seed, eave=args.eave, n_add=args.n_add,
        ddt=args.ddt, breakdown_rate=args.breakdown_rate,
    )

    # --- 5) report ---
    print(f"\n{'Scenario':<5}  {'Portfolio':>10}  {'ATC':>10}  {'Δ vs ATC':>10}")
    deltas = []
    for scen in SCENARIOS:
        p_s = rep.per_scenario[scen]["score"]
        a_s = atc_rep.per_scenario[scen]["score"]
        d = (p_s - a_s) / a_s * 100
        deltas.append(d)
        print(f"{scen:<5}  {p_s:>10.0f}  {a_s:>10.0f}  {d:>9.1f}%")
    print(f"\n  Portfolio: mean={rep.mean_score:.0f}  stddev={rep.score_stddev:.0f}")
    print(f"  ATC      : mean={atc_rep.mean_score:.0f}  stddev={atc_rep.score_stddev:.0f}")
    print(f"  Δ        : mean={(rep.mean_score - atc_rep.mean_score):+.0f}  "
          f"stddev={(rep.score_stddev - atc_rep.score_stddev):+.0f}")

    return {
        "rules_per_scenario": rules,
        "per_scenario_evolution": per_scenario_results,
        "portfolio_expr": portfolio_expr,
        "portfolio_report": dataclasses.asdict(rep),
        "atc_report": dataclasses.asdict(atc_rep),
        "thresholds": THRESHOLDS,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="DSevolve-style portfolio composition")
    p.add_argument("--rules-file", default=None,
                   help="optional JSON mapping scenario→expr; skips per-scenario evolution")
    p.add_argument("--iterations", type=int, default=3,
                   help="evolution iterations per scenario (only when --rules-file is absent)")
    p.add_argument("--provider", choices=["mock", "anthropic", "openai"], default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--success-threshold", type=float, default=0.05)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--machines", type=int, default=6)
    p.add_argument("--replications", type=int, default=5)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--eave", type=float, default=75.0)
    p.add_argument("--n-add", type=int, default=5)
    p.add_argument("--ddt", type=float, default=0.7)
    p.add_argument("--breakdown-rate", type=float, default=0.2)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    out = run_portfolio(args)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nportfolio saved → {args.out}")


if __name__ == "__main__":
    main()
