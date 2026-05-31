"""Full P1/P2/P3 × S0/S1/S2 battery driver (실험설계서_수정 §5-2 + §7-2).

9 combinations = 3 scenarios × 3 LLM variants. For each:
  - 20 evolution iterations
  - 100 replications per fitness evaluation
  - same base seed so cross-variant comparisons are paired

Outputs:
  runs/p123_battery/<scen>_<variant>_best.json
  runs/p123_battery/<scen>_<variant>.log
  runs/p123_battery/<scen>_P3_memory.json
  runs/p123_battery/REPORT.md     (after all 9 complete)

Run:
    python3 -m evolution.p123_battery
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "runs" / "p123_battery_v3_full"

# 실험설계서_수정 §4 mid-level parameters.
SCENARIO_PARAMS: dict[str, list[str]] = {
    "S0": [],
    "S1": ["--part-delay-ratio", "0.2", "--part-delay-k", "1.0"],
    # Tightest level of spec §4-3 (납기계수 ∈ {0.3, 0.5, 1.0}) so the urgent
    # job becomes hard enough to differentiate P1/P2/P3.
    "S2": ["--urgent-due-ratio", "0.3"],
}

COMMON_ARGS = [
    "--iterations", "20",
    "--replications", "100",
    "--jobs", "12",
    "--machines", "6",
    "--seed", "1000",
    # Full FJSSP — any machine for any op (n_eligible = n_machines).
    "--flexibility", "1.0",
    "--provider", "openai",
    "--model", "gpt-4o-mini",
    "--success-threshold", "0.05",
]


def _argv(scen: str, variant: str) -> list[str]:
    out_best = OUT_DIR / f"{scen}_{variant}_best.json"
    argv = [
        sys.executable, "-m", "evolution.evolve",
        "--scenario", scen,
        "--variant", variant,
        "--best-out", str(out_best),
    ] + COMMON_ARGS + SCENARIO_PARAMS[scen]
    if variant == "P3":
        argv += ["--memory-out", str(OUT_DIR / f"{scen}_P3_memory.json")]
    return argv


def run_one(scen: str, variant: str) -> dict:
    argv = _argv(scen, variant)
    log_path = OUT_DIR / f"{scen}_{variant}.log"
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                              cwd=REPO_ROOT, check=False)
    wall = time.time() - t0
    return {
        "scenario": scen,
        "variant": variant,
        "exit_code": proc.returncode,
        "wall_seconds": wall,
        "log": str(log_path),
    }


def render_report(results: list[dict]) -> str:
    lines = ["# P1/P2/P3 × S0/S1/S2 battery report",
             "",
             "*실험설계서_수정 §5-2 (P1/P2/P3) × §4 (S0/S1/S2) × §7-2 (100 reps).*",
             ""]

    total_wall = sum(r["wall_seconds"] for r in results)
    lines.append(f"Total wall time: **{total_wall/60:.1f} min** across {len(results)} combos.")
    lines.append("")

    lines.append("## 1. Best rule per (scenario, variant)")
    lines.append("")
    lines.append("| Scen | Variant | obj | gap vs FIFO | feas | wall (s) |")
    lines.append("|------|---------|-----|-------------|------|----------|")
    for r in results:
        if r["exit_code"] != 0:
            lines.append(f"| {r['scenario']} | {r['variant']} | **FAILED (exit {r['exit_code']})** |  |  | {r['wall_seconds']:.0f} |")
            continue
        bp = OUT_DIR / f"{r['scenario']}_{r['variant']}_best.json"
        if not bp.exists():
            lines.append(f"| {r['scenario']} | {r['variant']} | (no best.json) |  |  | {r['wall_seconds']:.0f} |")
            continue
        b = json.loads(bp.read_text())
        gap = f"{b.get('gap_ratio'):+.1f}%" if b.get('gap_ratio') is not None else "—"
        lines.append(
            f"| {r['scenario']} | {r['variant']} | {b['primary_objective']:.1f} | "
            f"{gap} | {b['feasible_job_ratio']:.2f} | {r['wall_seconds']:.0f} |"
        )
    lines.append("")

    lines.append("## 2. Convergence (best-so-far obj per iter)")
    lines.append("")
    for r in results:
        if r["exit_code"] != 0:
            continue
        bp = OUT_DIR / f"{r['scenario']}_{r['variant']}_best.json"
        if not bp.exists():
            continue
        b = json.loads(bp.read_text())
        conv = b.get("convergence", [])
        if not conv:
            continue
        deltas = " → ".join(f"{x:.0f}" for x in conv)
        lines.append(f"- **{r['scenario']} / {r['variant']}**: {deltas}")
    lines.append("")

    lines.append("## 3. Best expressions")
    lines.append("")
    for r in results:
        if r["exit_code"] != 0:
            continue
        bp = OUT_DIR / f"{r['scenario']}_{r['variant']}_best.json"
        if not bp.exists():
            continue
        b = json.loads(bp.read_text())
        lines.append(f"### {r['scenario']} / {r['variant']}")
        lines.append("```")
        lines.append(b['expr'])
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combos: list[tuple[str, str]] = [
        (s, v) for s in ("S0", "S1", "S2") for v in ("P1", "P2", "P3")
    ]
    # All 9 combos run in parallel — OpenAI gpt-4o-mini default limits
    # are 500 RPM / 200K TPM, and our peak is ~130 RPM, so the API is
    # not the bottleneck. Each subprocess opens its own client.
    max_workers = len(combos)
    print(f"Launching {len(combos)} combos in parallel (max_workers={max_workers}) → {OUT_DIR}")
    t_start = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(run_one, scen, variant): (scen, variant)
            for scen, variant in combos
        }
        for fut in concurrent.futures.as_completed(future_map):
            scen, variant = future_map[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "scenario": scen, "variant": variant,
                    "exit_code": -1, "wall_seconds": 0.0,
                    "error": repr(e), "log": "",
                }
            marker = "OK" if r["exit_code"] == 0 else f"FAIL({r['exit_code']})"
            print(f"  {scen}/{variant}: {marker}  {r['wall_seconds']:.0f}s "
                  f"  [{len(results) + 1}/{len(combos)} done]", flush=True)
            results.append(r)
            (OUT_DIR / "REPORT.md").write_text(render_report(results))

    (OUT_DIR / "REPORT.md").write_text(render_report(results))
    (OUT_DIR / "battery_meta.json").write_text(json.dumps({
        "total_wall_seconds": time.time() - t_start,
        "results": results,
    }, indent=2))
    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min.")
    print(f"Report: {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
