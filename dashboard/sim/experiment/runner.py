"""Experiment runner: evaluates baselines and LLM methods across instances and seeds."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..core.simulator import FJSSPSimulator
from ..data.loader import load_instance, generate_due_dates, estimate_makespan
from ..rules.baseline import RULES as BASELINE_RULES
from ..scenarios.s0_normal import S0Normal
from ..scenarios.s1_part_delay import S1PartDelay
from ..scenarios.s2_urgent_order import S2UrgentOrder
from ..evaluation.metrics import average_tardiness, summarise_runs, average_relative_improvement
from ..evaluation.stats import wilcoxon_test
from ..llm.evolution import EoHEvolution
from ..llm.experience import ExperienceStore
from .config import ExperimentConfig


class ExperimentRunner:

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.results_dir = Path(cfg.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.store = ExperienceStore(self.results_dir / "experience_store.json")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        all_results = {}

        for instance_name in self.cfg.instances:
            if self.cfg.verbose:
                print(f"\n{'='*60}")
                print(f"Instance: {instance_name.upper()}")
                print(f"{'='*60}")

            result = self._run_instance(instance_name)
            all_results[instance_name] = result

            # Save incrementally
            self._save_results(all_results)

        return all_results

    # ------------------------------------------------------------------
    # Per-instance
    # ------------------------------------------------------------------

    def _run_instance(self, name: str) -> Dict:
        jobs, n_machines = load_instance(name)
        t_est = estimate_makespan(jobs, n_machines)

        scenario_cls, scenario_kwargs = self._build_scenario()
        scenario_obj = scenario_cls(**scenario_kwargs)
        scenario_tag = self._scenario_tag()
        scenario_desc = self._scenario_description()

        # Factory: scenario_factory(seed) -> list[ScenarioEvent]
        def scenario_factory(seed: int):
            return scenario_obj.build_events(jobs, n_machines, t_est, seed=seed)

        results: Dict = {}

        # ---- Baselines ----
        if self.cfg.run_baselines:
            baseline_ats: Dict[str, List[float]] = {}
            for bid, fn in BASELINE_RULES.items():
                ats = self._evaluate_rule(fn, jobs, n_machines, scenario_factory,
                                          self.cfg.n_final_seeds, self.cfg.seed_offset)
                baseline_ats[bid] = ats
                summary = summarise_runs(ats)
                if self.cfg.verbose:
                    print(f"  {bid:4s}  AT={summary['mean']:.3f} ± {summary['std']:.3f}")
                results[bid] = summary

            best_baseline_mean = min(v["mean"] for v in results.values())

            # ARI
            for bid in BASELINE_RULES:
                results[bid]["ari"] = average_relative_improvement(
                    best_baseline_mean, results[bid]["mean"]
                )

            # Wilcoxon vs best baseline
            best_bid = min(BASELINE_RULES, key=lambda b: results[b]["mean"])
            for bid in BASELINE_RULES:
                if bid == best_bid:
                    continue
                stat, p = wilcoxon_test(baseline_ats[best_bid], baseline_ats[bid])
                results[bid]["wilcoxon_vs_best"] = {"stat": stat, "p": p}

        else:
            baseline_ats = {}
            best_baseline_mean = None

        # ---- P1 / P2 / P3 ----
        for method_id, use_ext, use_exp in [
            ("P1", False, False),
            ("P2", True,  False),
            ("P3", True,  True),
        ]:
            flag = getattr(self.cfg, f"run_{method_id.lower()}")
            if not flag:
                continue

            if self.cfg.verbose:
                print(f"\n  Running {method_id} evolution ({self.cfg.n_iter} iter)...")

            evo = EoHEvolution(
                jobs=jobs,
                n_machines=n_machines,
                scenario_factory=scenario_factory,
                scenario_tag=scenario_tag,
                scenario_description=scenario_desc,
                use_external_vars=use_ext,
                use_experience=use_exp,
                experience_store=self.store,
                ddt=self.cfg.ddt,
                seed_offset=1000,     # separated from final eval seeds
            )
            evo_result = evo.run()
            best_fn = evo_result.best_rule.fn

            ats = self._evaluate_rule(best_fn, jobs, n_machines, scenario_factory,
                                      self.cfg.n_final_seeds, self.cfg.seed_offset)
            summary = summarise_runs(ats)
            summary["rule_id"] = evo_result.best_rule.rule_id
            summary["rule_code"] = evo_result.best_rule.code
            summary["iteration_best_at"] = evo_result.iteration_best_at

            if best_baseline_mean is not None:
                summary["ari"] = average_relative_improvement(best_baseline_mean, summary["mean"])
                if self.cfg.run_baselines:
                    best_bid = min(BASELINE_RULES, key=lambda b: results[b]["mean"])
                    stat, p = wilcoxon_test(baseline_ats[best_bid], ats)
                    summary["wilcoxon_vs_best_baseline"] = {"stat": stat, "p": p}

            results[method_id] = summary
            if self.cfg.verbose:
                print(f"  {method_id}   AT={summary['mean']:.3f} ± {summary['std']:.3f}  "
                      f"ARI={summary.get('ari', 0):.1f}%")

        return results

    # ------------------------------------------------------------------
    # Rule evaluation (100 seeds)
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        fn: Callable,
        jobs,
        n_machines: int,
        scenario_factory,
        n_seeds: int,
        seed_offset: int,
    ) -> List[float]:
        ats = []
        for seed in range(seed_offset, seed_offset + n_seeds):
            j = generate_due_dates(deepcopy(jobs), ddt=self.cfg.ddt, seed=seed)
            events = scenario_factory(seed)
            sim = FJSSPSimulator(j, n_machines, fn, scenario_events=events)
            try:
                ats.append(sim.run().at)
            except Exception:
                ats.append(float("inf"))
        return ats

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    def _build_scenario(self) -> Tuple:
        s = self.cfg.scenario
        if s == "S0":
            return S0Normal, {}
        elif s == "S1":
            return S1PartDelay, {
                "affected_ratio": self.cfg.s1_affected_ratio,
                "delay_k": self.cfg.s1_delay_k,
            }
        elif s == "S2":
            return S2UrgentOrder, {"due_date_factor": self.cfg.s2_due_date_factor}
        else:
            raise ValueError(f"Unknown scenario: {s!r}")

    def _scenario_tag(self) -> str:
        s = self.cfg.scenario
        if s == "S1":
            return f"S1_ratio{self.cfg.s1_affected_ratio}_k{self.cfg.s1_delay_k}"
        if s == "S2":
            return f"S2_ddf{self.cfg.s2_due_date_factor}"
        return "S0"

    def _scenario_description(self) -> str:
        s = self.cfg.scenario
        if s == "S0":
            return "Normal operation (no disruption)."
        if s == "S1":
            return (
                f"Part-delay scenario: {self.cfg.s1_affected_ratio*100:.0f}% of jobs affected, "
                f"delay_k={self.cfg.s1_delay_k}."
            )
        if s == "S2":
            return (
                f"Urgent-order scenario: one urgent job inserted mid-run, "
                f"due_date_factor={self.cfg.s2_due_date_factor}."
            )
        return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_results(self, data: Dict) -> None:
        tag = self._scenario_tag()
        path = self.results_dir / f"results_{tag}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
