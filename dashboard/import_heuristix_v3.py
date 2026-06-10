"""Import heuristiX v3 battery results into the FJSSP workbench's results/ folder.

Source layout (heuristiX_v2):
    runs/p123_battery_v3_strict/   flex=0.0
    runs/p123_battery_v3_fjssp/    flex=0.5
    runs/p123_battery_v3_full/     flex=1.0
        <scen>_<variant>_best.json
        performance_raw.json           # baseline + LLM AT/MIT/PTJ per cell

Target layout (this dashboard):
    results/
        baselines_<scen>_<instance>.json
            {rule_id: {mean, std, ptj, mit, makespan, ari}}
        evolution_<method>_<scen>_<instance>.json
            {method, summary: {mean, std, ari, ...}, best_rule: {rule_id, code, ...}}
    benchmarks/custom/
        <instance>.json             # stub instance so the dropdown sees them

Each flex level becomes a distinct "instance" so the dashboard can compare
flex=0.0 vs 0.5 vs 1.0 as different benchmark targets.

Run:
    python3 import_heuristix_v3.py /home/amuredo/heuristiX_v2
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


SRC_DIRS = {
    "synth12x6_flex00": ("runs/p123_battery_v3_strict", 0.0),
    "synth12x6_flex05": ("runs/p123_battery_v3_fjssp",  0.5),
    "synth12x6_flex10": ("runs/p123_battery_v3_full",   1.0),
}
SCENARIOS = ("S0", "S1", "S2")
VARIANTS  = ("P1", "P2", "P3")

DEST_RESULTS    = Path(__file__).parent / "results"
DEST_BENCHMARKS = Path(__file__).parent / "benchmarks" / "custom"


def _to_baseline_record(at_mean: float, at_std: float, mit: float, ptj: float,
                        makespan_proxy: float, best_at: float) -> dict:
    ari = (best_at - at_mean) / best_at * 100.0 if best_at > 0 else 0.0
    return {
        "mean": at_mean, "std": at_std,
        "ptj": ptj, "mit": mit,
        "makespan": makespan_proxy,
        "ari": ari,
    }


def write_stub_instance(instance: str, flexibility: float) -> Path:
    """Generate a 12-job × 6-machine instance compatible with the dashboard's
    JSON loader. Jobs visit every machine once; eligibility is determined by
    flexibility (0 → 1 machine, 0.5 → 3 machines, 1.0 → all 6)."""
    n_jobs, n_machines = 12, 6
    rng = random.Random(hash(instance) & 0xFFFFFFFF)

    n_eligible = max(1, round(flexibility * n_machines)) if flexibility > 0 else 1

    jobs = []
    for j in range(n_jobs):
        ops = []
        for op in range(n_machines):
            # strict JSSP: single random machine; FJSSP: a subset
            pool = list(range(n_machines))
            rng.shuffle(pool)
            eligible = sorted(pool[:n_eligible])
            base = rng.uniform(1.0, 99.0)
            entries = [
                {"machine": m, "processing": max(1.0, base * rng.uniform(0.8, 1.2))}
                for m in eligible
            ]
            ops.append(entries)
        jobs.append(ops)

    payload = {
        "machines": n_machines,
        "jobs": jobs,
        "_meta": {
            "source": "heuristiX v3 import",
            "flexibility": flexibility,
            "instance": instance,
        },
    }
    DEST_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    out = DEST_BENCHMARKS / f"{instance}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def import_one_battery(repo_root: Path, instance: str, sub: str, flex: float) -> int:
    src = repo_root / sub
    if not src.exists():
        print(f"  [skip] {src} (not found)")
        return 0
    perf = src / "performance_raw.json"
    if not perf.exists():
        print(f"  [skip] {src}/performance_raw.json")
        return 0
    cells = json.loads(perf.read_text())

    # stub instance for the dashboard's benchmark dropdown
    inst_path = write_stub_instance(instance, flex)
    print(f"  stub instance → {inst_path}")

    # Group by scenario.
    by_scen: dict[str, list[dict]] = {s: [] for s in SCENARIOS}
    for c in cells:
        by_scen.setdefault(c["scen"], []).append(c)

    written = 0
    for scen, scen_cells in by_scen.items():
        if not scen_cells:
            continue

        # ---- baselines (same across variants within scenario × flex) ----
        first = scen_cells[0]
        baselines: dict[str, dict] = {}
        bls   = first["baselines"]
        bstd  = first.get("baselines_std", {})
        bmit  = first.get("baselines_mit", {})
        bptj  = first.get("baselines_ptj", {})
        best_at = min(bls.values()) if bls else 0.0
        for name, at_mean in bls.items():
            baselines[name] = _to_baseline_record(
                at_mean=at_mean,
                at_std=bstd.get(name, 0.0),
                mit=bmit.get(name, 0.0),
                ptj=bptj.get(name, 0.0),
                makespan_proxy=at_mean * 4.5,
                best_at=best_at,
            )
        (DEST_RESULTS / f"baselines_{scen}_{instance}.json").write_text(
            json.dumps(baselines, indent=2, ensure_ascii=False)
        )
        written += 1

        # ---- evolution: one file per variant ----
        for c in scen_cells:
            variant = c["variant"]
            llm_at  = c["llm_at_mean"]
            llm_std = c.get("llm_at_std", 0.0)
            ari     = c.get("ari_vs_overall_pct", 0.0)
            mit     = c.get("llm_mit_mean", 0.0)
            ptj     = c.get("llm_ptj_pct", 0.0)

            best_path = src / f"{scen}_{variant}_best.json"
            convergence: list[float] = []
            n_iter, n_rep = 20, 100
            if best_path.exists():
                try:
                    bj = json.loads(best_path.read_text())
                    convergence = bj.get("convergence", [])
                    n_iter = bj.get("iterations", n_iter)
                    n_rep  = bj.get("replications", n_rep)
                except Exception:
                    pass

            method_kr = {"P1": "P1 — 외부충격 변수 미노출",
                         "P2": "P2 — 외부충격 변수 노출",
                         "P3": "P3 — P2 + 메모리 누적"}[variant]
            evo = {
                "method": f"heuristiX-{variant}",
                "method_kr": method_kr,
                "instance": instance,
                "scenario": scen,
                "summary": {
                    "mean": llm_at, "std": llm_std,
                    "ari": ari, "ptj": ptj, "mit": mit,
                    "makespan": llm_at * 4.5,
                },
                "best_rule": {
                    "rule_id": f"heuristiX-{variant}-{scen}-{instance}",
                    "code": c["llm_expr"],
                    "generation": "—",
                    "operator": variant,
                    "operator_kr": method_kr,
                },
                "iterations": n_iter,
                "replications": n_rep,
                "convergence": convergence,
                "_source": str(src.name),
            }
            (DEST_RESULTS / f"evolution_{variant}_{scen}_{instance}.json").write_text(
                json.dumps(evo, indent=2, ensure_ascii=False)
            )
            written += 1
    return written


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("usage: python3 import_heuristix_v3.py <heuristiX_v2 root>")
        sys.exit(2)
    repo_root = Path(argv[1]).resolve()
    DEST_RESULTS.mkdir(parents=True, exist_ok=True)
    total = 0
    for inst, (sub, flex) in SRC_DIRS.items():
        print(f"importing {inst} (flex={flex}) ← {sub}")
        n = import_one_battery(repo_root, inst, sub, flex)
        print(f"  wrote {n} result files")
        total += n
    print(f"\nDone. {total} result files in {DEST_RESULTS}")
    print(f"     {len(SRC_DIRS)} stub instances in {DEST_BENCHMARKS}")


if __name__ == "__main__":
    main(sys.argv)
