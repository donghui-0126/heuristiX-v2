"""Smoke test: run B1-B10 on Mk01 under S0, S1, S2 and print AT results."""

from __future__ import annotations

import sys
from copy import deepcopy

from sim.data.loader import load_instance, generate_due_dates, estimate_makespan
from sim.core.simulator import FJSSPSimulator
from sim.rules.baseline import RULES, BASELINE_IDS
from sim.scenarios.s0_normal import S0Normal
from sim.scenarios.s1_part_delay import S1PartDelay
from sim.scenarios.s2_urgent_order import S2UrgentOrder


def run_one(rule_id, fn, jobs, n_machines, scenario_obj, t_est, seed=42):
    j = generate_due_dates(deepcopy(jobs), ddt=1.5, seed=seed)
    events = scenario_obj.build_events(j, n_machines, t_est, seed=seed)
    sim = FJSSPSimulator(j, n_machines, fn, scenario_events=events)
    r = sim.run()
    return r.at, r.mit_total, r.ptj, r.makespan


def main():
    print("Loading Mk01 ...")
    jobs, n_machines = load_instance("mk01")
    t_est = estimate_makespan(jobs, n_machines, seed=0)
    print(f"  n_jobs={len(jobs)}  n_machines={n_machines}  T_est={t_est:.1f}\n")

    scenarios = {
        "S0": S0Normal(),
        "S1 (ratio=0.20, k=1.0)": S1PartDelay(affected_ratio=0.20, delay_k=1.0),
        "S2 (ddf=0.5)": S2UrgentOrder(due_date_factor=0.5),
    }

    header = f"{'Rule':<5}  {'AT':>8}  {'MIT':>8}  {'PTJ%':>7}  {'Span':>8}"
    for scenario_name, scenario_obj in scenarios.items():
        print(f"{'─'*55}")
        print(f"Scenario: {scenario_name}")
        print(header)
        for bid in BASELINE_IDS:
            fn = RULES[bid]
            at, mit, ptj, span = run_one(bid, fn, jobs, n_machines, scenario_obj, t_est)
            print(f"{bid:<5}  {at:8.3f}  {mit:8.2f}  {ptj:6.1f}%  {span:8.1f}")
        print()

    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
