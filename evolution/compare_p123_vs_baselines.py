"""LLM-evolved rules vs. fixed B1~B5 head-to-head, with primary metric AT.

Runs each best.json from runs/p123_battery/ on its native scenario/params
with the same 100 reps used in the evolution, then compares mean tardiness
(AT, 실험설계서_수정 §8 primary) against B1~B5 evaluated under matching
parameters.

Output: runs/p123_battery/PERFORMANCE.md
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .baselines import BASELINES
from .simulator import Simulator


REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY = REPO_ROOT / "runs" / "p123_battery_v3_full"
# Prior battery (strict JSSP, pure-AT fitness) — for side-by-side comparison.
PRIOR_BATTERY = REPO_ROOT / "runs" / "p123_battery_v3_full"

B1_B5 = ["FIFO", "EDD", "SPT", "CR", "Urgency"]
# B1~B5 + literature heavies (WMDD, COVERT, ATC + 4 classic).
ALL_BASELINES = list(B1_B5) + ["WMDD", "COVERT", "ATC", "LPT", "MWKR", "LWKR", "MDD"]

SCEN_CONFIG = {
    "S0": dict(),
    "S1": dict(part_delay_ratio=0.2, part_delay_k=1.0),
    "S2": dict(urgent_due_ratio=0.3),
}


def _agg(sim: Simulator, expr: str) -> dict:
    """Return AT/MIT/PTJ aggregates (mean ± stddev) for `expr`."""
    r = sim.run(expr)
    def stats(key: str) -> tuple[float, float]:
        xs = [rep["metrics"][key] for rep in r.reps]
        return (statistics.mean(xs),
                statistics.stdev(xs) if len(xs) > 1 else 0.0)
    at_m, at_s = stats("mean_tardiness")
    mit_m, _ = stats("mit")
    ptj_m, _ = stats("ptj_pct")
    return {"at_mean": at_m, "at_std": at_s, "mit_mean": mit_m, "ptj_mean": ptj_m}


def _at(sim: Simulator, expr: str) -> tuple[float, float]:
    d = _agg(sim, expr)
    return d["at_mean"], d["at_std"]


def evaluate_cell(scen: str, variant: str) -> dict:
    bp = BATTERY / f"{scen}_{variant}_best.json"
    if not bp.exists():
        return {"error": f"missing {bp.name}"}
    best = json.loads(bp.read_text())

    sim = Simulator(
        scenario=scen, jobs=12, machines=6, replications=100, seed=1000,
        flexibility=1.0, **SCEN_CONFIG[scen],
    )

    print(f"  evaluating {scen}/{variant} best…", flush=True)
    llm = _agg(sim, best["expr"])
    llm_at_mean, llm_at_std = llm["at_mean"], llm["at_std"]
    llm_mit, llm_ptj = llm["mit_mean"], llm["ptj_mean"]

    print(f"  evaluating all {len(ALL_BASELINES)} baselines on {scen}…", flush=True)
    baselines: dict[str, tuple[float, float]] = {}
    baseline_extras: dict[str, dict] = {}
    for name in ALL_BASELINES:
        d = _agg(sim, BASELINES[name])
        baselines[name] = (d["at_mean"], d["at_std"])
        baseline_extras[name] = d

    # Strongest *overall* baseline (full literature set).
    bb_all_name, (bb_all_mean, bb_all_std) = min(baselines.items(), key=lambda kv: kv[1][0])
    # Strongest within the spec's B1~B5.
    b1b5_dict = {n: baselines[n] for n in B1_B5}
    bb_spec_name, (bb_spec_mean, bb_spec_std) = min(b1b5_dict.items(), key=lambda kv: kv[1][0])

    # ARI per 실험설계서_수정 §8: positive ⇒ LLM improvement over baseline.
    ari_overall = (bb_all_mean - llm_at_mean) / bb_all_mean * 100.0
    ari_spec = (bb_spec_mean - llm_at_mean) / bb_spec_mean * 100.0
    return {
        "scen": scen,
        "variant": variant,
        "llm_expr": best["expr"],
        "llm_at_mean": llm_at_mean,
        "llm_at_std": llm_at_std,
        "llm_mit_mean": llm_mit,
        "llm_ptj_pct": llm_ptj,
        "baselines": {n: m for n, (m, s) in baselines.items()},
        "baselines_std": {n: s for n, (m, s) in baselines.items()},
        "baselines_mit": {n: baseline_extras[n]["mit_mean"] for n in ALL_BASELINES},
        "baselines_ptj": {n: baseline_extras[n]["ptj_mean"] for n in ALL_BASELINES},
        "best_overall_name": bb_all_name,
        "best_overall_mean": bb_all_mean,
        "best_overall_std": bb_all_std,
        "ari_vs_overall_pct": ari_overall,
        "best_spec_name": bb_spec_name,
        "best_spec_mean": bb_spec_mean,
        "best_spec_std": bb_spec_std,
        "ari_vs_spec_pct": ari_spec,
        # Back-compat: delta with the old sign (negative = improvement).
        "delta_vs_overall_pct": -ari_overall,
        "delta_vs_spec_pct": -ari_spec,
    }


def render(cells: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# LLM vs. fixed baselines — head-to-head (AT)")
    lines.append("")
    lines.append("*Primary metric: AT (mean tardiness over 100 seeds). 동일 인스턴스·동일 disruption · 동일 seed range.*")
    lines.append("")
    lines.append(f"Baseline universe: **{len(ALL_BASELINES)} rules** "
                 f"= 5 B1~B5 (spec §5-1) + 7 literature (WMDD/COVERT/ATC/MDD/MWKR/LWKR/LPT).")
    lines.append("")

    # Headline 1: LLM vs strongest *overall* baseline.
    lines.append("## Headline — LLM vs strongest of ALL baselines (ARI = positive ⇒ LLM better)")
    lines.append("")
    lines.append("| Scen / Var | LLM AT | Best overall | Best-overall AT | ARI |")
    lines.append("|------------|--------|--------------|-----------------|-----|")
    for c in cells:
        if "error" in c:
            continue
        ari = c["ari_vs_overall_pct"]
        sign = "✓ beats" if ari > 0.5 else ("≈ ties" if abs(ari) <= 0.5 else "✗ loses")
        lines.append(
            f"| **{c['scen']} / {c['variant']}** "
            f"| {c['llm_at_mean']:.1f} ± {c['llm_at_std']:.1f} "
            f"| {c['best_overall_name']} "
            f"| {c['best_overall_mean']:.1f} ± {c['best_overall_std']:.1f} "
            f"| **{ari:+.1f}%** ({sign}) |"
        )
    lines.append("")

    # Headline 2: LLM vs spec B1~B5 (paper's framing).
    lines.append("## Headline — LLM vs strongest of B1~B5 (spec §5-1)")
    lines.append("")
    lines.append("| Scen / Var | LLM AT | Best B1~B5 | Best-B1B5 AT | ARI |")
    lines.append("|------------|--------|------------|--------------|-----|")
    for c in cells:
        if "error" in c:
            continue
        ari = c["ari_vs_spec_pct"]
        sign = "✓ beats" if ari > 0.5 else ("≈ ties" if abs(ari) <= 0.5 else "✗ loses")
        lines.append(
            f"| **{c['scen']} / {c['variant']}** "
            f"| {c['llm_at_mean']:.1f} ± {c['llm_at_std']:.1f} "
            f"| {c['best_spec_name']} "
            f"| {c['best_spec_mean']:.1f} ± {c['best_spec_std']:.1f} "
            f"| **{ari:+.1f}%** ({sign}) |"
        )
    lines.append("")

    # Auxiliary metrics MIT and PTJ.
    lines.append("## Auxiliary metrics — MIT (machine idle) and PTJ (%)")
    lines.append("")
    lines.append("| Scen / Var | LLM AT | LLM MIT | LLM PTJ% | Best-baseline MIT | Best-baseline PTJ% |")
    lines.append("|------------|--------|---------|----------|-------------------|-------------------|")
    for c in cells:
        if "error" in c:
            continue
        bb = c["best_overall_name"]
        bb_mit = c["baselines_mit"].get(bb, 0.0)
        bb_ptj = c["baselines_ptj"].get(bb, 0.0)
        lines.append(
            f"| **{c['scen']} / {c['variant']}** "
            f"| {c['llm_at_mean']:.1f} | {c['llm_mit_mean']:.0f} | {c['llm_ptj_pct']:.1f} "
            f"| {bb_mit:.0f} | {bb_ptj:.1f} |"
        )
    lines.append("")

    # Full per-scenario baseline picture, sorted by AT.
    lines.append("## Baseline AT — full ranking per scenario")
    lines.append("")
    for scen in ("S0", "S1", "S2"):
        cell = next((c for c in cells if c.get("scen") == scen and c.get("variant") == "P3"), None)
        if not cell or "error" in cell:
            continue
        lines.append(f"### {scen}")
        lines.append("")
        lines.append("| Rank | Rule | AT | Note |")
        lines.append("|------|------|----|------|")
        sorted_rules = sorted(cell["baselines"].items(), key=lambda kv: kv[1])
        for rank, (name, at) in enumerate(sorted_rules, start=1):
            note_bits: list[str] = []
            if name == cell["best_overall_name"]:
                note_bits.append("★ overall best")
            if name in B1_B5:
                note_bits.append("B1~B5")
            note = "  ".join(note_bits) or ""
            lines.append(f"| {rank} | {name} | {at:.1f} | {note} |")
        lines.append("")

    # P1/P2/P3 ranking per scenario.
    lines.append("## P1 vs P2 vs P3 (same scenario)")
    lines.append("")
    lines.append("| Scen | P1 AT | P2 AT | P3 AT | Best variant |")
    lines.append("|------|-------|-------|-------|--------------|")
    for scen in ("S0", "S1", "S2"):
        row = {v: next((c for c in cells if c["scen"] == scen and c["variant"] == v), None)
               for v in ("P1", "P2", "P3")}
        if any(r is None or "error" in r for r in row.values()):
            continue
        best_v = min(row.items(), key=lambda kv: kv[1]["llm_at_mean"])
        lines.append(
            f"| {scen} | {row['P1']['llm_at_mean']:.1f} | {row['P2']['llm_at_mean']:.1f} "
            f"| {row['P3']['llm_at_mean']:.1f} | **{best_v[0]}** |"
        )
    lines.append("")

    lines.append("## Best LLM expressions")
    lines.append("")
    for c in cells:
        if "error" in c:
            continue
        lines.append(f"### {c['scen']} / {c['variant']}")
        lines.append(
            f"AT = {c['llm_at_mean']:.1f}  "
            f"(vs overall best {c['best_overall_name']} {c['best_overall_mean']:.1f}, "
            f"Δ {c['delta_vs_overall_pct']:+.1f}%  ·  "
            f"vs B1~B5 best {c['best_spec_name']} {c['best_spec_mean']:.1f}, "
            f"Δ {c['delta_vs_spec_pct']:+.1f}%)"
        )
        lines.append("")
        lines.append("```")
        lines.append(c["llm_expr"])
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _load_prior_ats() -> dict[tuple[str, str], float]:
    """Best-effort: load AT means from the prior battery's
    performance_raw.json (composite-or-AT fitness, old prompt). Empty
    dict if unavailable."""
    raw = PRIOR_BATTERY / "performance_raw.json"
    if not raw.exists():
        return {}
    try:
        cells = json.loads(raw.read_text())
    except Exception:
        return {}
    return {(c["scen"], c["variant"]): c["llm_at_mean"]
            for c in cells if "scen" in c and "variant" in c}


def main() -> None:
    combos = [(s, v) for s in ("S0", "S1", "S2") for v in ("P1", "P2", "P3")]
    cells: list[dict] = []
    print(f"Evaluating {len(combos)} cells…")
    for s, v in combos:
        cells.append(evaluate_cell(s, v))

    prior = _load_prior_ats()
    md = render(cells)
    if prior:
        md += "\n## Δ vs prior prompt (runs/p123_battery_fjssp_full)\n\n"
        md += "| Scen / Var | New AT | Prior AT | Δ (new − prior) |\n"
        md += "|-----------|--------|----------|-----------------|\n"
        for c in cells:
            if "error" in c:
                continue
            p = prior.get((c["scen"], c["variant"]))
            if p is None:
                md += f"| **{c['scen']} / {c['variant']}** | {c['llm_at_mean']:.1f} | — | — |\n"
                continue
            d = c["llm_at_mean"] - p
            pct = d / p * 100.0
            md += (f"| **{c['scen']} / {c['variant']}** | {c['llm_at_mean']:.1f} "
                   f"| {p:.1f} | **{d:+.1f}** ({pct:+.1f}%) |\n")

    md_path = BATTERY / "PERFORMANCE.md"
    json_path = BATTERY / "performance_raw.json"
    md_path.write_text(md)
    json_path.write_text(json.dumps(cells, indent=2, ensure_ascii=False))
    print(f"\nReport → {md_path}")


if __name__ == "__main__":
    main()
