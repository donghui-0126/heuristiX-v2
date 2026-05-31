"""Cross-scenario robustness evaluation (실험설계서_수정 §7).

For a single rule (or a list of rules), run it under S0/S1/S2 and compute:
  - per-scenario metrics + score
  - mean score across scenarios
  - stddev of score across scenarios (the "robustness" number — lower is
    better, meaning the rule is consistent across shocks)
  - per-scenario gap_ratio vs FIFO

Run:
    python -m evolution.robustness --expr "0.0 - slack + 5*mat_risk"
    python -m evolution.robustness --rule-file runs/s5_best.json --replications 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .baselines import BASELINES
from .simulator import RunResult, Simulator, weights_for


SCENARIOS = ["S0", "S1", "S2"]


@dataclass
class RobustnessReport:
    expr: str
    label: str
    per_scenario: dict[str, dict]      # scenario → flat metrics dict
    mean_score: float                  # average of scenario-weighted scores
    score_stddev: float                # robustness — lower is better
    mean_gap_vs_fifo: float | None     # average gap_ratio across scenarios

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_rule(
    expr: str,
    label: str,
    *,
    jobs: int = 12,
    machines: int = 6,
    replications: int = 5,
    seed: int = 1000,
    part_delay_ratio: float = 0.20,
    part_delay_k: float = 1.0,
    urgent_due_ratio: float = 0.5,
    ddt: float = 1.0,
) -> RobustnessReport:
    per_scenario: dict[str, dict] = {}
    scores: list[float] = []
    gaps: list[float] = []

    for scen in SCENARIOS:
        sim = Simulator(
            scenario=scen, jobs=jobs, machines=machines,
            replications=replications, seed=seed,
            part_delay_ratio=part_delay_ratio,
            part_delay_k=part_delay_k,
            urgent_due_ratio=urgent_due_ratio,
            ddt=ddt,
            gap_baseline="FIFO" if expr != BASELINES["FIFO"] else None,
        )
        r = sim.run(expr)
        score = r.score(weights_for(scen))
        scores.append(score)
        if r.gap_ratio is not None:
            gaps.append(r.gap_ratio)
        per_scenario[scen] = {
            "score": score,
            "weights": asdict(weights_for(scen)),
            "mean_tardiness": r.mean_tardiness,
            "total_tardiness": r.total_tardiness,
            "makespan": r.makespan,
            "urgent_mean_tardiness": r.urgent_mean_tardiness,
            "feasible_job_ratio": r.feasible_job_ratio,
            "idle_breakdown": r.idle_breakdown,
            "idle_starved": r.idle_starved,
            "gap_ratio_vs_fifo": r.gap_ratio,
            "tardiness_stddev": r.tardiness_stddev,
        }

    return RobustnessReport(
        expr=expr,
        label=label,
        per_scenario=per_scenario,
        mean_score=statistics.mean(scores),
        score_stddev=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        mean_gap_vs_fifo=statistics.mean(gaps) if gaps else None,
    )


def _load_rule(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve --expr / --rule-file / --baseline into (expr, label)."""
    if args.expr:
        return args.expr, args.label or "custom"
    if args.rule_file:
        data = json.loads(Path(args.rule_file).read_text())
        return data["expr"], args.label or Path(args.rule_file).stem
    if args.baseline:
        if args.baseline not in BASELINES:
            raise SystemExit(f"unknown baseline: {args.baseline}. Pick from {list(BASELINES)}")
        return BASELINES[args.baseline], args.baseline
    raise SystemExit("provide one of --expr / --rule-file / --baseline")


def _print_report(rep: RobustnessReport) -> None:
    print(f"\n=== Robustness: {rep.label} ===")
    print(f"expr: {rep.expr}")
    print(f"\n{'Scenario':<5}  {'Score':>10}  {'MeanTard':>9}  {'Makespan':>9}  "
          f"{'UrgTard':>8}  {'Feas':>5}  {'GapVsFIFO':>10}")
    for scen in SCENARIOS:
        s = rep.per_scenario[scen]
        gap = f"{s['gap_ratio_vs_fifo']:+.1f}%" if s['gap_ratio_vs_fifo'] is not None else "—"
        print(f"{scen:<5}  {s['score']:>10.1f}  {s['mean_tardiness']:>9.1f}  "
              f"{s['makespan']:>9.1f}  {s['urgent_mean_tardiness']:>8.1f}  "
              f"{s['feasible_job_ratio']:>5.2f}  {gap:>10}")
    print(f"\n  mean score        : {rep.mean_score:.1f}")
    print(f"  robustness (stdev): {rep.score_stddev:.1f}   (lower = more consistent)")
    if rep.mean_gap_vs_fifo is not None:
        print(f"  mean gap vs FIFO  : {rep.mean_gap_vs_fifo:+.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-scenario robustness for a dispatching rule")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--expr", help="evalexpr string for the rule")
    src.add_argument("--rule-file", help="JSON file with {expr: ...} (e.g. produced by --best-out)")
    src.add_argument("--baseline", help="name of a baseline rule (FIFO/EDD/SPT/CR/Urgency)")
    p.add_argument("--label", default=None, help="display label for this rule")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--machines", type=int, default=6)
    p.add_argument("--replications", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--part-delay-ratio", type=float, default=0.20)
    p.add_argument("--part-delay-k", type=float, default=1.0)
    p.add_argument("--urgent-due-ratio", type=float, default=0.5)
    p.add_argument("--ddt", type=float, default=1.0)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    expr, label = _load_rule(args)
    rep = evaluate_rule(
        expr, label,
        jobs=args.jobs, machines=args.machines, replications=args.replications,
        seed=args.seed,
        part_delay_ratio=args.part_delay_ratio,
        part_delay_k=args.part_delay_k,
        urgent_due_ratio=args.urgent_due_ratio,
        ddt=args.ddt,
    )
    _print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
        print(f"\nreport saved → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
