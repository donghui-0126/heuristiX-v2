"""4-cell ablation for memory retrieval (Phase 4 — C1/C2 novelty).

Compares four retrieval mechanisms on the same scenario with everything
else held constant:

  vanilla cosine         — ReasoningBank-style (text embedding top-k)
  state-conditioned      — C1: nearest neighbour in disruption-state space
  contrastive pair       — C2: matched success ↔ failure pairs
  state + contrastive    — C1 + C2 combined (recommended novelty angle)

The keyword heuristic (current default) is included as a 5th cell for
context. All 5 conditions share the same instance, seed, LLM, and
operator schedule (E1/E2/M1/M2). Differences in final best score
isolate the effect of retrieval.

Run:
    python -m evolution.retrieval_ablation \
        --scenario S5 --iterations 4 --reps 8 \
        --model gpt-4o-mini --seed 9000

Output:
    runs/retrieval_ablation.json — raw per-cell results
    RETRIEVAL_ABLATION.md         — markdown report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from .baselines import BASELINES
from .embedding import embed
from .evolve import (
    SCENARIO_STATE, _best_baseline_score, _classify_for_memory,
    _evaluate_population, _memory_text,
)
from .llm import build_llm
from .memory import MemoryBank
from .simulator import RunResult, Simulator


RETRIEVAL_MODES = ["keyword", "cosine", "state", "contrastive", "state_contrastive"]


def _evolve_once(
    *,
    scenario: str,
    retrieval: str,
    iterations: int,
    sim: Simulator,
    llm,
    success_threshold: float,
) -> dict:
    """Run a single evolution loop with the given retrieval mode."""
    memory = MemoryBank()
    population: list[str] = list(BASELINES.values())
    current_state = SCENARIO_STATE.get(scenario)
    needs_embed = retrieval in ("cosine", "contrastive")
    query_embedding = embed(
        f"Scenario {scenario}: dispatching rule for supply-chain disruption"
    ) if needs_embed else None

    gen_kwargs = dict(
        retrieval_mode=retrieval,
        current_state=current_state,
        query_embedding=query_embedding,
    )

    # Initial population enrichment.
    for _ in range(3):
        _, code = llm.generate_rule(scenario, [], memory, operation="explore", **gen_kwargs)
        population.append(code)

    cache: dict[str, RunResult] = {}
    best: RunResult | None = None
    convergence: list[float] = []

    for _ in range(iterations):
        results = _evaluate_population(sim, population, cache)
        results.sort(key=lambda r: r.primary_objective)
        elite = results[:3]
        _, baseline_score = _best_baseline_score(sim, cache)
        successes, failures = _classify_for_memory(results, baseline_score, success_threshold)
        lessons = llm.reflect(scenario, successes, failures)
        for lesson in lessons:
            if lesson.state_signature is None:
                lesson.state_signature = current_state
            if lesson.embedding is None and needs_embed:
                lesson.embedding = embed(_memory_text(lesson))
        memory.add_many(lessons)

        if best is None or elite[0].primary_objective < best.primary_objective:
            best = elite[0]
        convergence.append(best.primary_objective)

        new = []
        for _ in range(2):
            _, c = llm.generate_rule(scenario, elite, memory, operation="explore", **gen_kwargs)
            new.append(c)
        for _ in range(2):
            _, c = llm.generate_rule(scenario, elite, memory, operation="crossover", **gen_kwargs)
            new.append(c)
        _, c = llm.generate_rule(scenario, elite, memory, operation="modify", **gen_kwargs)
        new.append(c)
        _, c = llm.generate_rule(scenario, elite, memory, operation="simplify", **gen_kwargs)
        new.append(c)
        population = [r.expr for r in elite] + new

    assert best is not None
    return {
        "retrieval": retrieval,
        "best_expr": best.expr,
        "best_score": best.primary_objective,
        "best_gap_vs_fifo": best.gap_ratio,
        "best_mean_tardiness": best.mean_tardiness,
        "convergence": convergence,
        "memory_size": len(memory.items),
        "memory_items": [dataclasses.asdict(m) for m in memory.items],
    }


def render_report(results: dict, args) -> str:
    md: list[str] = []
    md.append("# Retrieval Ablation — Phase 4 (C1 + C2)")
    md.append("")
    md.append(f"*Scenario {args.scenario}, {args.iterations} iterations, "
              f"{args.reps} replications per evaluation, model `{args.model}`, "
              f"seed = {args.seed}*")
    md.append("")
    md.append("## Cells")
    md.append("")
    md.append("- **keyword** — scenario tag + |Δ| ranking (current default)")
    md.append("- **cosine** — text-embedding top-k (ReasoningBank baseline)")
    md.append("- **state** — runtime state-vector nearest neighbour (C1 only)")
    md.append("- **contrastive** — matched (success, failure) pair (C2 only)")
    md.append("- **state_contrastive** — C1 + C2 combined (recommended)")
    md.append("")

    md.append("## Final best score per cell (lower = better)")
    md.append("")
    md.append("| Retrieval mode | Best score | gap vs FIFO | mem items | best expr (excerpt) |")
    md.append("| --- | --- | --- | --- | --- |")
    for mode in RETRIEVAL_MODES:
        if mode not in results:
            continue
        r = results[mode]
        expr_short = r["best_expr"][:60] + ("…" if len(r["best_expr"]) > 60 else "")
        gap = "—" if r["best_gap_vs_fifo"] is None else f"{r['best_gap_vs_fifo']:+.1f}%"
        md.append(f"| **{mode}** | {r['best_score']:.0f} | {gap} | {r['memory_size']} | `{expr_short}` |")
    md.append("")

    md.append("## Convergence (per-iteration best primary objective)")
    md.append("")
    headers = ["Retrieval"] + [f"iter {i+1}" for i in range(args.iterations)]
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for mode in RETRIEVAL_MODES:
        if mode not in results:
            continue
        conv = results[mode]["convergence"]
        cells = [mode] + [f"{c:.0f}" for c in conv]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Best cell summary.
    valid = {k: v for k, v in results.items() if k in RETRIEVAL_MODES}
    if valid:
        best_mode = min(valid.items(), key=lambda kv: kv[1]["best_score"])
        worst_mode = max(valid.items(), key=lambda kv: kv[1]["best_score"])
        md.append("## Summary")
        md.append("")
        md.append(f"- **Best**: `{best_mode[0]}` ({best_mode[1]['best_score']:.0f})")
        md.append(f"- **Worst**: `{worst_mode[0]}` ({worst_mode[1]['best_score']:.0f})")
        gap = (worst_mode[1]["best_score"] - best_mode[1]["best_score"]) / worst_mode[1]["best_score"] * 100
        md.append(f"- **Spread**: {gap:.1f}% between best and worst retrieval mode")
        md.append("")
        # C1 + C2 verdict
        sc = valid.get("state_contrastive")
        cos = valid.get("cosine")
        kw = valid.get("keyword")
        if sc and cos:
            d = (sc["best_score"] - cos["best_score"]) / cos["best_score"] * 100
            verdict = "✅ beats" if d < -1.0 else ("≈" if abs(d) <= 1.0 else "⚠ trails")
            md.append(f"- **C1+C2 vs ReasoningBank baseline**: {d:+.1f}% ({verdict})")
        if sc and kw:
            d = (sc["best_score"] - kw["best_score"]) / kw["best_score"] * 100
            verdict = "✅ beats" if d < -1.0 else ("≈" if abs(d) <= 1.0 else "⚠ trails")
            md.append(f"- **C1+C2 vs keyword heuristic**: {d:+.1f}% ({verdict})")
        md.append("")
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="S5", choices=["S0", "S1", "S2", "S3", "S4", "S5"])
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--machines", type=int, default=6)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--eave", type=float, default=75.0)
    p.add_argument("--n-add", type=int, default=5)
    p.add_argument("--ddt", type=float, default=0.7)
    p.add_argument("--breakdown-rate", type=float, default=0.2)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic", "mock"])
    p.add_argument("--success-threshold", type=float, default=0.05)
    p.add_argument("--modes", nargs="+", default=RETRIEVAL_MODES,
                   help="subset of retrieval modes to run")
    p.add_argument("--raw-out", default="runs/retrieval_ablation.json")
    p.add_argument("--report-out", default="RETRIEVAL_ABLATION.md")
    args = p.parse_args()

    llm = build_llm(args.provider, model=args.model)
    sim = Simulator(
        scenario=args.scenario, jobs=args.jobs, machines=args.machines,
        replications=args.reps, seed=args.seed,
        eave=args.eave, n_add=args.n_add, ddt=args.ddt,
        breakdown_rate=args.breakdown_rate, gap_baseline="FIFO",
    )

    results: dict = {}
    t_start = time.time()
    for mode in args.modes:
        print(f"=== {mode} ===")
        t0 = time.time()
        results[mode] = _evolve_once(
            scenario=args.scenario, retrieval=mode,
            iterations=args.iterations, sim=sim, llm=llm,
            success_threshold=args.success_threshold,
        )
        print(f"  best={results[mode]['best_score']:.0f}  ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed/60:.1f} min")

    repo = Path(__file__).resolve().parent.parent
    raw = repo / args.raw_out
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))
    print(f"raw → {raw}")

    rep = repo / args.report_out
    rep.write_text(render_report(results, args))
    print(f"report → {rep}")


if __name__ == "__main__":
    main()
