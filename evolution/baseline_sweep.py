"""Baseline-matrix sanity sweep for the new scenario taxonomy.

실험설계서_수정 §5-1 (B1~B5) × §4 parameter levels × seed 100.
Confirms that:
  (a) the three scenarios (S0/S1/S2) differentiate baseline rules,
  (b) the spec's 3-level parameter grid actually shifts metrics,
  (c) FJSSP flexibility changes the picture as the report predicts.

This is a no-LLM check — pure DES, quick (~30s on a laptop).

Run:
    python3 -m evolution.baseline_sweep
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .baselines import BASELINES
from .simulator import Simulator


# ----- knobs ---------------------------------------------------------------

# 실험설계서_수정 §5-1 B1~B5.
B1_B5 = ["FIFO", "EDD", "SPT", "CR", "Urgency"]

# §4-2 / §4-3.
PART_DELAY_RATIOS = [0.10, 0.20, 0.40]
PART_DELAY_KS = [0.5, 1.0, 2.0]
URGENT_DUE_RATIOS = [0.3, 0.5, 1.0]

# §7-2.
N_REPS = 100
BASE_SEED = 1000

# 보고서 모티브: FJSSP routing flexibility — strict JSSP vs. moderate FJSSP.
FLEX_LEVELS = [0.0, 0.5]

# Instance size: kept at the project default (close to Brandimarte Mk02).
JOBS, MACHINES = 12, 6


# ----- helpers -------------------------------------------------------------

def _agg(reps: list[dict], key: str) -> tuple[float, float]:
    """Return (mean, stddev) of metrics[key] across replications."""
    xs = [r["metrics"][key] for r in reps]
    return (
        statistics.mean(xs),
        statistics.stdev(xs) if len(xs) > 1 else 0.0,
    )


def _cell(sim: Simulator, baseline: str) -> dict:
    r = sim.run(BASELINES[baseline])
    at_mean, at_std = _agg(r.reps, "mean_tardiness")
    pt_mean, _ = _agg(r.reps, "tardy_rate")
    mk_mean, mk_std = _agg(r.reps, "makespan")
    return {
        "at_mean": at_mean,
        "at_stddev": at_std,
        "ptj_mean_pct": pt_mean * 100.0,   # 보고서 §8 PTJ in %
        "makespan_mean": mk_mean,
        "makespan_stddev": mk_std,
    }


def _key_s0(flex: float) -> str:
    return f"S0|flex={flex}"


def _key_s1(ratio: float, k: float, flex: float) -> str:
    return f"S1|r={ratio}|k={k}|flex={flex}"


def _key_s2(due_ratio: float, flex: float) -> str:
    return f"S2|d={due_ratio}|flex={flex}"


def run_matrix() -> dict:
    """Return raw['cells'][cell_key][baseline] = metrics dict."""
    out: dict = {
        "config": {
            "jobs": JOBS,
            "machines": MACHINES,
            "n_reps": N_REPS,
            "base_seed": BASE_SEED,
            "baselines": B1_B5,
        },
        "cells": {},
    }

    def add(cell_key: str, sim: Simulator) -> None:
        out["cells"][cell_key] = {b: _cell(sim, b) for b in B1_B5}

    t0 = time.time()
    for flex in FLEX_LEVELS:
        sim_s0 = Simulator(
            scenario="S0", jobs=JOBS, machines=MACHINES,
            replications=N_REPS, seed=BASE_SEED,
        )
        # CLI flag for flexibility goes through the binary's --flexibility,
        # which the Simulator dataclass doesn't currently model — we pipe it
        # through by mutating part_delay_* values still no-op for S0, then
        # call the binary directly for the flex argument.
        # Simpler: skip the wrapper and shell out manually for S0/S1/S2 below.
        pass

    # Manual shell-out keeps the script self-contained and explicit.
    import subprocess
    binary = Path(__file__).resolve().parent.parent / "target" / "release" / "heuristix"
    if not binary.exists():
        subprocess.run(["cargo", "build", "--release", "--quiet"],
                       cwd=binary.parent.parent.parent, check=True)

    def run_one(scenario: str, flex: float,
                part_ratio: float | None, part_k: float | None,
                urgent_ratio: float | None) -> list[dict]:
        argv = [
            str(binary),
            "--scenario", scenario,
            "--rule", "expr",
            "--jobs", str(JOBS),
            "--machines", str(MACHINES),
            "--replications", str(N_REPS),
            "--seed", str(BASE_SEED),
            "--flexibility", str(flex),
        ]
        if part_ratio is not None:
            argv += ["--part-delay-ratio", str(part_ratio),
                     "--part-delay-k", str(part_k)]
        if urgent_ratio is not None:
            argv += ["--urgent-due-ratio", str(urgent_ratio)]
        # Set --expr per-baseline outside the helper.
        return argv

    def collect(argv_base: list[str]) -> dict[str, dict]:
        cells = {}
        for name in B1_B5:
            argv = argv_base + ["--expr", BASELINES[name]]
            proc = subprocess.run(argv, capture_output=True, text=True, check=True)
            reps = [json.loads(line) for line in proc.stdout.splitlines()
                    if line.startswith("{")]
            at = [r["metrics"]["mean_tardiness"] for r in reps]
            ptj = [r["metrics"]["tardy_rate"] * 100.0 for r in reps]
            mk = [r["metrics"]["makespan"] for r in reps]
            cells[name] = {
                "at_mean": statistics.mean(at),
                "at_stddev": statistics.stdev(at) if len(at) > 1 else 0.0,
                "ptj_mean_pct": statistics.mean(ptj),
                "makespan_mean": statistics.mean(mk),
                "makespan_stddev": statistics.stdev(mk) if len(mk) > 1 else 0.0,
            }
        return cells

    n_cells = 0
    for flex in FLEX_LEVELS:
        # S0 — no scenario params.
        argv = run_one("S0", flex, None, None, None)
        out["cells"][_key_s0(flex)] = collect(argv)
        n_cells += 1

        # S1 — 3×3 grid.
        for ratio in PART_DELAY_RATIOS:
            for k in PART_DELAY_KS:
                argv = run_one("S1", flex, ratio, k, None)
                out["cells"][_key_s1(ratio, k, flex)] = collect(argv)
                n_cells += 1

        # S2 — 3 levels.
        for due in URGENT_DUE_RATIOS:
            argv = run_one("S2", flex, None, None, due)
            out["cells"][_key_s2(due, flex)] = collect(argv)
            n_cells += 1

    out["wall_seconds"] = time.time() - t0
    out["n_cells"] = n_cells
    return out


# ----- reporting -----------------------------------------------------------

def _winner(cells: dict, key: str) -> str:
    return min(cells.keys(), key=lambda b: cells[b][key])


def render_markdown(raw: dict) -> str:
    lines: list[str] = []
    cfg = raw["config"]
    lines.append("# Baseline sanity sweep (실험설계서_수정 §5-1 × §4)")
    lines.append("")
    lines.append(
        f"- jobs/machines: **{cfg['jobs']}/{cfg['machines']}**"
        f"  · reps/cell: **{cfg['n_reps']}** (seed base {cfg['base_seed']})"
        f"  · baselines: {', '.join(cfg['baselines'])}"
    )
    lines.append(
        f"- wall time: **{raw['wall_seconds']:.1f}s** across "
        f"{raw['n_cells']} cells × {len(cfg['baselines'])} baselines = "
        f"{raw['n_cells'] * len(cfg['baselines'])} sim invocations."
    )
    lines.append("")
    lines.append("Each cell = mean AT ± stddev over 100 seeds; PTJ in %.")
    lines.append("")

    # Section: per-cell table.
    for cell_key, cells in raw["cells"].items():
        lines.append(f"## `{cell_key}`")
        lines.append("")
        lines.append("| Rule | AT (mean ± std) | PTJ % | Makespan |")
        lines.append("|------|-----------------|-------|----------|")
        at_winner = _winner(cells, "at_mean")
        for name in cfg["baselines"]:
            c = cells[name]
            marker = " ★" if name == at_winner else ""
            lines.append(
                f"| **{name}**{marker} | {c['at_mean']:.1f} ± {c['at_stddev']:.1f} "
                f"| {c['ptj_mean_pct']:.1f} | {c['makespan_mean']:.1f} |"
            )
        lines.append("")

    # Section: cross-cell AT pivot (winners per scenario × params).
    lines.append("## Winner-per-cell pivot (AT)")
    lines.append("")
    lines.append("| Cell | Winner | AT | Worst | AT | Spread |")
    lines.append("|------|--------|----|-------|----|--------|")
    for cell_key, cells in raw["cells"].items():
        win = min(cells.items(), key=lambda kv: kv[1]["at_mean"])
        worst = max(cells.items(), key=lambda kv: kv[1]["at_mean"])
        spread = worst[1]["at_mean"] - win[1]["at_mean"]
        lines.append(
            f"| `{cell_key}` | {win[0]} | {win[1]['at_mean']:.1f} "
            f"| {worst[0]} | {worst[1]['at_mean']:.1f} | {spread:.1f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    raw = run_matrix()

    out_dir = Path(__file__).resolve().parent.parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "baseline_sanity.json"
    md_path = out_dir / "baseline_sanity.md"

    raw_path.write_text(json.dumps(raw, indent=2))
    md_path.write_text(render_markdown(raw))

    print(f"raw  → {raw_path}")
    print(f"md   → {md_path}")
    print(f"wall → {raw['wall_seconds']:.1f}s "
          f"({raw['n_cells']} cells × {len(raw['config']['baselines'])} rules)")


if __name__ == "__main__":
    main()
