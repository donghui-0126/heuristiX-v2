"""Subprocess wrapper around the Rust heuristiX CLI.

Each `Simulator.run(...)` call shells out to `cargo run --release --quiet`,
parses the per-replication JSON, and returns aggregated metrics.

The simulator is treated as a black box: the Python side never inspects
the DES internals — it only feeds it expressions and reads back metrics.
"""

from __future__ import annotations

import json
import shlex
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---- Scenario-specific score weights (창종설 보고서 §6-2) -----------------

@dataclass(frozen=True)
class ScoreWeights:
    """Composite-objective weights. Lower is always better when applied
    via `RunResult.score(weights)`.

    Mirrors `Rust::ScoreWeights::for_scenario`. Keep them in sync if you
    change one side."""
    tardiness: float = 1.0
    makespan: float = 0.5
    urgent_tardiness: float = 5.0
    idle: float = 0.1


SCENARIO_WEIGHTS: dict[str, ScoreWeights] = {
    # 실험설계서_수정 §8: AT is *the* primary objective. Evolution fitness
    # matches the reporting metric — all non-tardiness weights are 0.
    "S0": ScoreWeights(tardiness=1.0, makespan=0.0, urgent_tardiness=0.0, idle=0.0),
    "S1": ScoreWeights(tardiness=1.0, makespan=0.0, urgent_tardiness=0.0, idle=0.0),
    "S2": ScoreWeights(tardiness=1.0, makespan=0.0, urgent_tardiness=0.0, idle=0.0),
}


def weights_for(scenario: str) -> ScoreWeights:
    return SCENARIO_WEIGHTS.get(scenario.upper(), ScoreWeights())


@dataclass
class RunResult:
    """Aggregated metrics over `replications` runs of a single rule."""
    expr: str
    scenario: str
    replications: int

    # Means across replications.
    mean_tardiness: float = 0.0
    total_tardiness: float = 0.0
    makespan: float = 0.0
    urgent_mean_tardiness: float = 0.0
    feasible_job_ratio: float = 1.0
    idle_breakdown: float = 0.0
    idle_starved: float = 0.0

    # Stddev across replications (for robustness reporting).
    tardiness_stddev: float = 0.0
    makespan_stddev: float = 0.0

    # Optional comparison fields populated when --gap-baseline / --stability-
    # baseline were passed.
    gap_ratio: Optional[float] = None
    schedule_stability: Optional[float] = None

    # Raw per-rep dicts for debugging.
    reps: list[dict] = field(default_factory=list)

    def score(self, weights: ScoreWeights | None = None) -> float:
        """Composite scenario objective. Lower is better.

        If `weights` is None, uses scenario-specific defaults from
        `SCENARIO_WEIGHTS` (창종설 §6-2)."""
        w = weights if weights is not None else weights_for(self.scenario)
        return (
            w.tardiness * self.mean_tardiness
            + w.makespan * self.makespan
            + w.urgent_tardiness * self.urgent_mean_tardiness
            + w.idle * (self.idle_breakdown + self.idle_starved)
        )

    @property
    def primary_objective(self) -> float:
        """Convenience: scenario-default score. Lower is better."""
        return self.score()


@dataclass
class Simulator:
    """Configuration shared across simulator invocations within one
    evolution run. Mutating `scenario`, `jobs`, etc. between calls is
    intended."""
    scenario: str = "S0"
    jobs: int = 12
    machines: int = 6
    replications: int = 5
    seed: int = 1000

    # Scenario knobs (실험설계서_수정 §4).
    part_delay_ratio: float = 0.20    # S1 levels: {0.10, 0.20, 0.40}
    part_delay_k: float = 1.0         # S1 levels: {0.5, 1.0, 2.0}
    urgent_due_ratio: float = 0.5     # S2 levels: {0.3, 0.5, 1.0}
    ddt: float = 1.0
    flexibility: float = 0.0          # FJSSP routing flexibility ∈ [0, 1]

    # Optional comparison flags. None disables; otherwise a baseline name.
    gap_baseline: Optional[str] = None
    stability_baseline: Optional[str] = None  # scenario name like "S0"

    # If True, build once with `cargo build --release` and use the produced
    # binary directly — avoids cargo's per-invocation overhead.
    use_compiled_binary: bool = True
    _binary_built: bool = False

    def _binary_path(self) -> Path:
        return REPO_ROOT / "target" / "release" / "heuristix"

    def _ensure_binary(self) -> None:
        if not self.use_compiled_binary:
            return
        if self._binary_built and self._binary_path().exists():
            return
        subprocess.run(
            ["cargo", "build", "--release", "--quiet"],
            cwd=REPO_ROOT, check=True,
        )
        self._binary_built = True

    def _build_argv(self, expr: str) -> list[str]:
        if self.use_compiled_binary:
            argv = [str(self._binary_path())]
        else:
            argv = ["cargo", "run", "--release", "--quiet", "--"]
        argv += [
            "--rule", "expr",
            "--expr", expr,
            "--scenario", self.scenario,
            "--jobs", str(self.jobs),
            "--machines", str(self.machines),
            "--replications", str(self.replications),
            "--seed", str(self.seed),
            "--part-delay-ratio", str(self.part_delay_ratio),
            "--part-delay-k", str(self.part_delay_k),
            "--urgent-due-ratio", str(self.urgent_due_ratio),
            "--ddt", str(self.ddt),
            "--flexibility", str(self.flexibility),
        ]
        if self.gap_baseline:
            argv += ["--gap-baseline", self.gap_baseline]
        if self.stability_baseline:
            argv += ["--stability-baseline", self.stability_baseline]
        return argv

    def run(self, expr: str) -> RunResult:
        """Execute `expr` under the current scenario config and return
        aggregated metrics. Raises CalledProcessError on simulator failure."""
        self._ensure_binary()
        argv = self._build_argv(expr)
        proc = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"simulator failed (exit {proc.returncode})\n"
                f"argv: {shlex.join(argv)}\n"
                f"stderr: {proc.stderr}"
            )

        reps = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            reps.append(json.loads(line))
        if not reps:
            raise RuntimeError(f"no JSON output from simulator (stderr={proc.stderr})")

        result = RunResult(expr=expr, scenario=self.scenario, replications=len(reps), reps=reps)
        m = lambda key: [r["metrics"][key] for r in reps]
        result.mean_tardiness = statistics.mean(m("mean_tardiness"))
        result.total_tardiness = statistics.mean(m("total_tardiness"))
        result.makespan = statistics.mean(m("makespan"))
        result.urgent_mean_tardiness = statistics.mean(m("urgent_mean_tardiness"))
        result.feasible_job_ratio = statistics.mean(m("feasible_job_ratio"))
        result.idle_breakdown = statistics.mean(m("idle_breakdown"))
        result.idle_starved = statistics.mean(m("idle_starved"))
        if len(reps) > 1:
            result.tardiness_stddev = statistics.stdev(m("mean_tardiness"))
            result.makespan_stddev = statistics.stdev(m("makespan"))

        gaps = [r["gap_ratio"] for r in reps if r.get("gap_ratio") is not None]
        if gaps:
            result.gap_ratio = statistics.mean(gaps)
        stabs = [r["schedule_stability"] for r in reps if r.get("schedule_stability") is not None]
        if stabs:
            result.schedule_stability = statistics.mean(stabs)

        return result
