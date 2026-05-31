"""Main evolution orchestrator (창종설 보고서 §4-4 진화 루프).

Pseudocode:
    population = baselines + LLM_init(5)
    for it in range(max_iters):
        for rule in population: scores[rule] = simulate(rule, scenario)
        elite = top_k(population, scores)
        reflection = LLM_S.reflect(elite, failures)
        memory.update(reflection)
        new = LLM_A.{explore, crossover, modify}(elite, memory)
        population = elite + new

Run:
    python -m evolution.evolve --scenario S5 --iterations 5 --mock
    python -m evolution.evolve --scenario S5 --iterations 10 \
        --provider anthropic --model claude-opus-4-7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import BASELINES
from .embedding import embed
from .llm import build_llm
from .memory import MemoryBank, MemoryItem, StateSignature
from .simulator import RunResult, Simulator


# Representative state signatures for each scenario (실험설계서_수정 §4).
SCENARIO_STATE: dict[str, StateSignature] = {
    "S0": StateSignature(0.05, 0.0, 0.05, 0.0, 0.0),
    "S1": StateSignature(0.45, 1.0, 0.05, 60.0, 0.0),
    "S2": StateSignature(0.30, 0.0, 0.35, 0.0, 0.0),
}


def _memory_text(item: MemoryItem) -> str:
    """Compose the text we embed for cosine retrieval."""
    return f"{item.title}\n{item.description}\n{item.content}"


def _best_baseline_score(sim: Simulator, cache: dict[str, RunResult]) -> tuple[str, float]:
    """Find the best baseline rule on the current scenario.

    Used by LLM-as-judge (창종설 §5-3): the 5% threshold is measured
    against the strongest fixed rule, not just FIFO."""
    best_name, best_score = None, float("inf")
    for name, expr in BASELINES.items():
        if expr in cache:
            r = cache[expr]
        else:
            r = sim.run(expr)
            cache[expr] = r
        s = r.primary_objective
        if s < best_score:
            best_name, best_score = name, s
    return best_name, best_score


def _classify_for_memory(
    results: list[RunResult],
    baseline_score: float,
    threshold: float,
) -> tuple[list[RunResult], list[RunResult]]:
    """LLM-as-judge classification (창종설 §5-3 memory_update):
      success := primary_objective <= baseline_score × (1 - threshold)
      failure := primary_objective >= baseline_score × (1 + threshold)
    Neutral rules (within ±threshold of baseline) are not stored."""
    successes, failures = [], []
    lo = baseline_score * (1.0 - threshold)
    hi = baseline_score * (1.0 + threshold)
    for r in results:
        if r.primary_objective <= lo:
            successes.append(r)
        elif r.primary_objective >= hi:
            failures.append(r)
    return successes, failures


def _evaluate_population(
    sim: Simulator, population: list[str], cache: dict[str, RunResult]
) -> list[RunResult]:
    results: list[RunResult] = []
    for expr in population:
        if expr in cache:
            results.append(cache[expr])
            continue
        try:
            r = sim.run(expr)
        except RuntimeError as e:
            # Treat malformed/divergent expressions as worst-case so they
            # drop out of the elite naturally.
            print(f"  [skip] simulator error for: {expr[:80]}\n    {e}")
            r = RunResult(
                expr=expr, scenario=sim.scenario, replications=0,
                mean_tardiness=1e9, makespan=1e9, urgent_mean_tardiness=1e9,
            )
        cache[expr] = r
        results.append(r)
    return results


def _print_population(label: str, results: list[RunResult]) -> None:
    print(f"\n--- {label} (n={len(results)}) ---")
    for i, r in enumerate(results[:10], start=1):
        gap = f" gap={r.gap_ratio:+.1f}%" if r.gap_ratio is not None else ""
        print(f"  #{i}  obj={r.primary_objective:8.1f}  feas={r.feasible_job_ratio:.2f}{gap}  "
              f"expr: {r.expr[:80]}{'…' if len(r.expr) > 80 else ''}")


def evolve(args: argparse.Namespace) -> RunResult:
    sim = Simulator(
        scenario=args.scenario,
        jobs=args.jobs,
        machines=args.machines,
        replications=args.replications,
        seed=args.seed,
        part_delay_ratio=args.part_delay_ratio,
        part_delay_k=args.part_delay_k,
        urgent_due_ratio=args.urgent_due_ratio,
        ddt=args.ddt,
        flexibility=args.flexibility,
        gap_baseline="FIFO",
    )

    # Dual-expert (EvoDR §4-4): LLM-A generates, LLM-S reflects. They may
    # share the same client (default), or be different providers/models.
    gen_llm = build_llm(args.gen_provider or args.provider,
                       model=args.gen_model or args.model)
    if args.reflect_provider or args.reflect_model:
        reflect_llm = build_llm(args.reflect_provider or args.provider,
                               model=args.reflect_model or args.model)
        print(f"  LLM-A (generator):  {args.gen_provider or args.provider} / "
              f"{args.gen_model or args.model or 'default'}")
        print(f"  LLM-S (reflector):  {args.reflect_provider or args.provider} / "
              f"{args.reflect_model or args.model or 'default'}")
    else:
        reflect_llm = gen_llm

    memory = MemoryBank()
    if args.memory_in:
        memory = MemoryBank.load(Path(args.memory_in))
        print(f"loaded {len(memory.items)} memories from {args.memory_in}")

    # Retrieval-mode helpers (Phase-4 C1/C2 ablation).
    current_state = SCENARIO_STATE.get(args.scenario)
    scenario_query_text = f"Scenario {args.scenario}: dispatching rule for supply-chain disruption"
    query_embedding = embed(scenario_query_text) if args.retrieval in ("cosine", "contrastive") else None

    variant = args.variant
    use_memory = (variant == "P3")
    # For P1/P2 the memory bank passed to the prompt builder is always
    # empty; reflection results are discarded.
    prompt_memory = lambda: (memory if use_memory else MemoryBank())

    # Initial population: spec §6-2 — B1~B5 (5 rules) + 5 LLM-generated = 10.
    # We deliberately exclude the literature extras (WMDD/MDD/COVERT/...) so
    # the LLM has to discover penalty/due-date sophistication itself rather
    # than picking up an already-strong rule for free.
    spec_b1_b5 = ["FIFO", "EDD", "SPT", "CR", "Urgency"]
    population: list[str] = [BASELINES[name] for name in spec_b1_b5]
    user_baseline_names: list[str] = []

    # Optional: user-defined baselines from a JSON workspace file. These
    # join the initial population, displacing LLM-init slots so the total
    # stays around 10. (heuristiX research-platform integration.)
    if args.extra_baselines_json:
        from pathlib import Path as _P
        p = _P(args.extra_baselines_json)
        if p.exists():
            try:
                extra = json.loads(p.read_text())
                for name, expr in extra.items():
                    population.append(expr)
                    user_baseline_names.append(name)
                print(f"  loaded {len(extra)} user baselines from {p}: "
                      f"{', '.join(user_baseline_names)}")
            except Exception as e:
                print(f"  [warn] failed to load --extra-baselines-json: {e}")

    n_llm_init = max(0, 5 - len(user_baseline_names))
    for _ in range(n_llm_init):
        _, code = gen_llm.generate_rule(
            args.scenario, [], prompt_memory(), operation="explore",
            retrieval_mode=args.retrieval,
            current_state=current_state,
            query_embedding=query_embedding,
            variant=variant,
        )
        population.append(code)

    cache: dict[str, RunResult] = {}
    best: RunResult | None = None
    convergence: list[float] = []

    for it in range(args.iterations):
        print(f"\n========== iteration {it+1}/{args.iterations} "
              f"(scenario={args.scenario}, variant={variant}) ==========")

        results = _evaluate_population(sim, population, cache)
        results.sort(key=lambda r: r.primary_objective)
        _print_population("ranked", results)

        elite = results[: args.elite_k]

        best_baseline, baseline_score = _best_baseline_score(sim, cache)

        if use_memory:
            # LLM-as-judge classification against the best fixed baseline
            # (실험설계서_수정 §6-3 memory_update). Only rules meaningfully
            # beating or trailing the baseline get stored.
            successes, failures = _classify_for_memory(
                results, baseline_score, args.success_threshold
            )
            print(f"\njudge: baseline={best_baseline} (score={baseline_score:.0f}), "
                  f"threshold=±{args.success_threshold*100:.0f}% → "
                  f"{len(successes)} success / {len(failures)} failure / "
                  f"{len(results) - len(successes) - len(failures)} neutral")
            lessons = reflect_llm.reflect(
                args.scenario, successes, failures, variant=variant,
            )
            for lesson in lessons:
                if lesson.state_signature is None:
                    lesson.state_signature = current_state
                if lesson.embedding is None and args.retrieval in ("cosine", "contrastive"):
                    lesson.embedding = embed(_memory_text(lesson))
            memory.add_many(lessons)
            print(f"memory: +{len(lessons)} items (total={len(memory.items)})")
        else:
            print(f"\nbaseline={best_baseline} (score={baseline_score:.0f}); "
                  f"variant={variant} — memory disabled")

        if best is None or elite[0].primary_objective < best.primary_objective:
            best = elite[0]
        convergence.append(best.primary_objective)

        gen_kwargs = dict(
            retrieval_mode=args.retrieval,
            current_state=current_state,
            query_embedding=query_embedding,
            variant=variant,
        )
        new_exprs: list[str] = []
        for _ in range(2):
            _, c = gen_llm.generate_rule(args.scenario, elite, prompt_memory(),
                                         operation="explore", **gen_kwargs)
            new_exprs.append(c)
        for _ in range(2):
            _, c = gen_llm.generate_rule(args.scenario, elite, prompt_memory(),
                                         operation="crossover", **gen_kwargs)
            new_exprs.append(c)
        _, c = gen_llm.generate_rule(args.scenario, elite, prompt_memory(),
                                     operation="modify", **gen_kwargs)
        new_exprs.append(c)
        _, c = gen_llm.generate_rule(args.scenario, elite, prompt_memory(),
                                     operation="simplify", **gen_kwargs)
        new_exprs.append(c)

        population = [r.expr for r in elite] + new_exprs

    print(f"\n========== best ==========\n  obj={best.primary_objective:.1f}")
    print(f"  expr: {best.expr}")
    if best.gap_ratio is not None:
        print(f"  gap_vs_FIFO: {best.gap_ratio:+.1f}%")

    if args.memory_out:
        memory.save(Path(args.memory_out))
        print(f"\nmemory saved → {args.memory_out}")
    if args.best_out:
        Path(args.best_out).write_text(json.dumps({
            "scenario": args.scenario,
            "variant": args.variant,
            "expr": best.expr,
            "primary_objective": best.primary_objective,
            "gap_ratio": best.gap_ratio,
            "feasible_job_ratio": best.feasible_job_ratio,
            "convergence": convergence,
            "iterations": args.iterations,
            "replications": args.replications,
        }, indent=2, ensure_ascii=False))
        print(f"best rule saved → {args.best_out}")

    return best


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-driven dispatching-rule evolution")
    p.add_argument("--scenario", default="S1", choices=["S0", "S1", "S2"])
    p.add_argument("--variant", default="P3", choices=["P1", "P2", "P3"],
                   help="실험설계서_수정 §5-2: P1 hides disruption variables; "
                        "P2 exposes them but no memory; P3 = P2 + memory bank.")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--elite-k", type=int, default=3, dest="elite_k")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--machines", type=int, default=6)
    p.add_argument("--replications", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--part-delay-ratio", type=float, default=0.20)
    p.add_argument("--part-delay-k", type=float, default=1.0)
    p.add_argument("--urgent-due-ratio", type=float, default=0.5)
    p.add_argument("--ddt", type=float, default=1.0)
    p.add_argument("--flexibility", type=float, default=0.0)
    p.add_argument("--extra-baselines-json", default=None,
                   help="optional JSON {name: evalexpr} to add to initial population "
                        "(used by the hub research platform).")
    p.add_argument("--provider", choices=["mock", "anthropic", "openai"], default="mock",
                   help="default provider for both LLM-A (generator) and LLM-S (reflector)")
    p.add_argument("--model", default=None,
                   help="default model; defaults: anthropic→claude-opus-4-7, openai→gpt-5")
    # EvoDR-style dual-expert overrides (창종설 §4-4 LLM-A + LLM-S).
    p.add_argument("--gen-provider", default=None,
                   help="override provider for LLM-A (rule generator)")
    p.add_argument("--gen-model", default=None,
                   help="override model for LLM-A (rule generator)")
    p.add_argument("--reflect-provider", default=None,
                   help="override provider for LLM-S (reflector). Use a smaller/cheaper model here.")
    p.add_argument("--reflect-model", default=None,
                   help="override model for LLM-S (reflector)")
    p.add_argument("--success-threshold", type=float, default=0.05,
                   help="gap-vs-best-baseline magnitude for LLM-as-judge success/failure "
                        "classification (창종설 §5-3 default 0.05)")
    p.add_argument("--retrieval",
                   choices=["keyword", "cosine", "state", "contrastive", "state_contrastive"],
                   default="keyword",
                   help="memory retrieval mode (Phase-4 C1/C2 ablation). "
                        "keyword=current heuristic; cosine=ReasoningBank baseline; "
                        "state=C1 only; contrastive=C2 only; state_contrastive=C1+C2.")
    p.add_argument("--memory-in", default=None, help="optional path to load prior memories")
    p.add_argument("--memory-out", default=None, help="optional path to save memories after run")
    p.add_argument("--best-out", default=None, help="optional path to save best rule JSON")
    args = p.parse_args()
    evolve(args)


if __name__ == "__main__":
    main()
