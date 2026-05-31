"""Side-by-side dashboard demo: SPT vs LLM-evolved P1/P2/P3.

For each scenario (S0/S1/S2), runs SPT plus the LLM best rule from each
of the three variants (trained on that scenario in
runs/p123_battery_at/) and renders a 3×4 grid of Gantt charts in a
single self-contained HTML file.

Output: runs/dashboard_demo/index.html

Run:
    python3 -m evolution.dashboard_demo
    # open the printed file:// path in a browser
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .baselines import BASELINES


REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY = REPO_ROOT / "runs" / "p123_battery_at"
OUT_DIR = REPO_ROOT / "runs" / "dashboard_demo"

# Scenario knobs match the battery — keep them aligned so the LLM rules
# are evaluated under the conditions they were evolved against.
SCEN_CONFIG = {
    "S0": [],
    "S1": ["--part-delay-ratio", "0.2", "--part-delay-k", "1.0"],
    "S2": ["--urgent-due-ratio", "0.5"],
}

RULE_LABELS = ["SPT", "LLM-P1", "LLM-P2", "LLM-P3"]


def llm_expr(scen: str, variant: str) -> str:
    return json.loads((BATTERY / f"{scen}_{variant}_best.json").read_text())["expr"]


def run_and_dump(scen: str, rule_label: str, expr: str, schedule_path: Path) -> dict:
    argv = [
        str(REPO_ROOT / "target" / "release" / "heuristix"),
        "--scenario", scen,
        "--rule", "expr",
        "--expr", expr,
        "--jobs", "12",
        "--machines", "6",
        "--replications", "1",
        "--seed", "1000",
        "--flexibility", "0.0",
        "--dump-schedule", str(schedule_path),
    ] + SCEN_CONFIG[scen]
    proc = subprocess.run(argv, capture_output=True, text=True, check=True)
    # The stdout line is one JSON per replication.
    rep = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("{")))
    return rep


# ---- SVG rendering --------------------------------------------------------

# 20-color tab20-ish palette; we cycle for jobs beyond 19.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def job_color(job_id: int) -> str:
    return PALETTE[job_id % len(PALETTE)]


def render_gantt_svg(
    snap: dict,
    rep: dict,
    *,
    width: int = 520,
    height: int = 200,
    max_time: float,
) -> str:
    """Return a self-contained <svg>...</svg> string for one schedule."""
    n_machines = snap["n_machines"]
    margin_top = 28
    margin_bottom = 22
    margin_left = 36
    margin_right = 10
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    row_h = plot_h / max(1, n_machines)

    def x(t: float) -> float:
        return margin_left + (t / max_time) * plot_w

    def y(machine: int) -> float:
        return margin_top + machine * row_h

    parts: list[str] = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" '
                 'xmlns="http://www.w3.org/2000/svg" class="gantt">')

    # Background.
    parts.append(f'<rect x="{margin_left}" y="{margin_top}" '
                 f'width="{plot_w}" height="{plot_h}" '
                 'fill="#fafafa" stroke="#ddd"/>')

    # Machine row separators + labels.
    for m in range(n_machines):
        parts.append(f'<line x1="{margin_left}" y1="{y(m)}" '
                     f'x2="{margin_left + plot_w}" y2="{y(m)}" '
                     'stroke="#eee"/>')
        parts.append(f'<text x="{margin_left - 6}" y="{y(m) + row_h/2 + 3}" '
                     'text-anchor="end" font-size="9" fill="#888">'
                     f'M{m}</text>')

    # Op rectangles.
    for op in snap["ops"]:
        if op["start"] is None or op["end"] is None:
            continue
        x0, x1 = x(op["start"]), x(op["end"])
        bar_y = y(op["machine"]) + 2
        bar_h = row_h - 4
        title = (f'Job {op["job"]} op {op["op"]}  '
                 f'machine M{op["machine"]}  '
                 f't={op["start"]:.1f}..{op["end"]:.1f}')
        parts.append(
            f'<rect x="{x0:.1f}" y="{bar_y:.1f}" '
            f'width="{max(1.0, x1-x0):.1f}" height="{bar_h:.1f}" '
            f'fill="{job_color(op["job"])}" opacity="0.85" stroke="#222" '
            f'stroke-width="0.4"><title>{title}</title></rect>'
        )

    # Disruption marker: for S1 the smallest non-zero part_available_time;
    # for S2 the urgent job with release > 0.
    onset = None
    for op in snap["ops"]:
        pat = op.get("part_available_time", 0.0)
        if pat and pat > 0.5:  # ignore numerical 0
            onset = pat if onset is None else min(onset, pat)
    for j in snap.get("jobs", []):
        if j.get("urgent") and j.get("release", 0.0) > 0.5:
            r = j["release"]
            onset = r if onset is None else min(onset, r)
    if onset is not None and onset < max_time:
        parts.append(
            f'<line x1="{x(onset):.1f}" y1="{margin_top}" '
            f'x2="{x(onset):.1f}" y2="{margin_top+plot_h}" '
            'stroke="#d62728" stroke-width="1.4" stroke-dasharray="3,2"/>'
        )
        parts.append(f'<text x="{x(onset):.1f}" y="{margin_top - 4}" '
                     'text-anchor="middle" font-size="9" fill="#d62728">'
                     f'shock t={onset:.0f}</text>')

    # Time axis ticks.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = max_time * frac
        parts.append(f'<line x1="{x(t):.1f}" y1="{margin_top+plot_h}" '
                     f'x2="{x(t):.1f}" y2="{margin_top+plot_h+3}" stroke="#888"/>')
        parts.append(f'<text x="{x(t):.1f}" y="{height-6}" '
                     'text-anchor="middle" font-size="9" fill="#666">'
                     f'{t:.0f}</text>')

    # Header line with metrics.
    m = rep["metrics"]
    label = (f'AT={m["mean_tardiness"]:.1f}   '
             f'mk={m["makespan"]:.0f}   '
             f'PTJ={m["tardy_rate"]*100:.0f}%')
    parts.append(f'<text x="{margin_left}" y="{margin_top - 8}" '
                 'font-size="11" fill="#222" font-family="monospace">'
                 f'{label}</text>')

    parts.append('</svg>')
    return "".join(parts)


def build_html(grid: dict[tuple[str, str], dict], max_time: float) -> str:
    css = """
    body { font-family: -apple-system, system-ui, sans-serif; margin: 16px;
           color: #222; background: #fff; }
    h1 { margin-bottom: 0; }
    .sub { color: #666; margin-top: 4px; }
    table.summary { border-collapse: collapse; margin: 18px 0; }
    table.summary th, table.summary td { border: 1px solid #ddd;
        padding: 6px 12px; text-align: right; font-variant-numeric: tabular-nums; }
    table.summary th { background: #f0f0f0; }
    table.summary td.scen { text-align: left; font-weight: 600; }
    .winner { background: #e8f5e9; font-weight: 600; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
            margin-top: 12px; }
    .cell { border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px;
            background: #fff; }
    .cell .head { font-size: 12px; color: #555; margin-bottom: 2px;
                  display: flex; justify-content: space-between; }
    .cell .head .rule { font-weight: 600; color: #222; }
    .cell .head .scen { color: #888; }
    .gantt { width: 100%; height: auto; display: block; }
    .legend { margin: 16px 0; font-size: 12px; color: #555; }
    .legend .swatch { display: inline-block; width: 14px; height: 14px;
                      border: 1px solid #999; margin-right: 4px;
                      vertical-align: middle; }
    .scen-row { margin-bottom: 20px; }
    .scen-row > h2 { margin: 24px 0 4px 0; font-size: 16px; color: #222; }
    .scen-desc { font-size: 12px; color: #666; margin-bottom: 8px; }
    """

    parts: list[str] = []
    parts.append(f'<!DOCTYPE html><html><head><meta charset="utf-8">'
                 f'<title>heuristiX dashboard demo</title>'
                 f'<style>{css}</style></head><body>')

    parts.append('<h1>heuristiX dashboard demo</h1>')
    parts.append('<p class="sub">SPT vs LLM-evolved P1/P2/P3 across S0/S1/S2 '
                 '(jobs=12 · machines=6 · seed=1000 · single replication for visualization).</p>')

    # Summary table — AT per (scenario, rule).
    parts.append('<h2>AT (mean tardiness)</h2>')
    parts.append('<table class="summary"><tr><th>Scenario</th>')
    for r in RULE_LABELS:
        parts.append(f'<th>{r}</th>')
    parts.append('</tr>')
    for scen in ("S0", "S1", "S2"):
        row_ats = {r: grid[(scen, r)]["rep"]["metrics"]["mean_tardiness"]
                   for r in RULE_LABELS}
        winner = min(row_ats, key=row_ats.get)
        parts.append(f'<tr><td class="scen">{scen}</td>')
        for r in RULE_LABELS:
            cls = ' class="winner"' if r == winner else ''
            parts.append(f'<td{cls}>{row_ats[r]:.1f}</td>')
        parts.append('</tr>')
    parts.append('</table>')

    parts.append('<p class="legend">Red dashed line = external shock onset. '
                 'Each colour = a job; same colour across charts in a row = same job. '
                 'Hover over a bar to see (job, op, machine, time range).</p>')

    # Gantt grid: one block per scenario, with 4 rules side by side.
    scen_descs = {
        "S0": "Normal (no shock).",
        "S1": "Part Delay — head-op part_available_time pushed out for 20% of jobs by 1.0 × mean total processing.",
        "S2": "Urgent Order — one new urgent job arrives mid-simulation; due-ratio 0.5.",
    }
    for scen in ("S0", "S1", "S2"):
        parts.append('<div class="scen-row">')
        parts.append(f'<h2>{scen}</h2>')
        parts.append(f'<div class="scen-desc">{scen_descs[scen]}</div>')
        parts.append('<div class="grid">')
        for r in RULE_LABELS:
            cell = grid[(scen, r)]
            parts.append('<div class="cell">')
            parts.append(f'<div class="head"><span class="rule">{r}</span>'
                         f'<span class="scen">{scen}</span></div>')
            parts.append(cell["svg"])
            parts.append('</div>')
        parts.append('</div></div>')

    parts.append('</body></html>')
    return "".join(parts)


def main() -> None:
    if not BATTERY.exists():
        raise SystemExit(f"missing battery results: {BATTERY}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rule_expr_for = {
        "SPT": BASELINES["SPT"],
        # LLM-* are scenario-specific; resolved per cell below.
        "LLM-P1": None,
        "LLM-P2": None,
        "LLM-P3": None,
    }

    grid: dict[tuple[str, str], dict] = {}
    max_time = 0.0
    snaps_dir = OUT_DIR / "schedules"
    snaps_dir.mkdir(exist_ok=True)

    for scen in ("S0", "S1", "S2"):
        for r in RULE_LABELS:
            if r == "SPT":
                expr = BASELINES["SPT"]
            else:
                variant = r.split("-")[1]
                expr = llm_expr(scen, variant)
            sched_path = snaps_dir / f"{scen}_{r}.json"
            print(f"  running {scen} / {r}…", flush=True)
            rep = run_and_dump(scen, r, expr, sched_path)
            snap = json.loads(sched_path.read_text())
            mk = rep["metrics"]["makespan"]
            max_time = max(max_time, mk)
            grid[(scen, r)] = {"rep": rep, "snap": snap, "expr": expr}

    # Add 5% padding to common max_time.
    common_max = max_time * 1.05

    for (scen, r), cell in grid.items():
        cell["svg"] = render_gantt_svg(cell["snap"], cell["rep"], max_time=common_max)

    html = build_html(grid, common_max)
    out_path = OUT_DIR / "index.html"
    out_path.write_text(html)
    print(f"\nDashboard → file://{out_path}")


if __name__ == "__main__":
    main()
