"""Entry point for running experiments.

Quick baseline-only test
------------------------
    python run_experiment.py --scenario S0 --instances mk01 --baselines-only

Full S1 experiment (B1-B10 + P2)
---------------------------------
    python run_experiment.py --scenario S1 --s1-ratio 0.20 --s1-k 1.0 --p2
"""

from __future__ import annotations

import argparse

from sim.experiment.config import ExperimentConfig
from sim.experiment.runner import ExperimentRunner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="S0", choices=["S0", "S1", "S2"])
    p.add_argument("--instances", nargs="+",
                   default=[f"mk{i:02d}" for i in range(1, 11)])
    p.add_argument("--ddt", type=float, default=1.5)
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--p1", action="store_true")
    p.add_argument("--p2", action="store_true")
    p.add_argument("--p3", action="store_true")
    # S1
    p.add_argument("--s1-ratio", type=float, default=0.20, choices=[0.10, 0.20, 0.40])
    p.add_argument("--s1-k", type=float, default=1.0, choices=[0.5, 1.0, 2.0])
    # S2
    p.add_argument("--s2-ddf", type=float, default=0.5, choices=[0.3, 0.5, 1.0])
    return p.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig(
        instances=args.instances,
        ddt=args.ddt,
        scenario=args.scenario,
        s1_affected_ratio=args.s1_ratio,
        s1_delay_k=args.s1_k,
        s2_due_date_factor=args.s2_ddf,
        n_final_seeds=args.n_seeds,
        run_baselines=True,
        run_p1=args.p1,
        run_p2=args.p2,
        run_p3=args.p3,
        verbose=True,
    )

    runner = ExperimentRunner(cfg)
    results = runner.run()

    print("\n\nSummary")
    print("=" * 60)
    for inst, inst_res in results.items():
        print(f"\n{inst.upper()}")
        for method, stats in inst_res.items():
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            ari = stats.get("ari", 0)
            print(f"  {method:4s}  AT={mean:.3f} ± {std:.3f}  ARI={ari:+.1f}%")


if __name__ == "__main__":
    main()
