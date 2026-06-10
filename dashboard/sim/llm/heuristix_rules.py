"""Auto-loader for heuristiX-evolved rules sitting in results/evolution_*.json.

Each evolution result file has the form:
    {
        "method": "heuristiX-P3",
        "instance": "synth12x6_flex05",
        "scenario": "S1",
        "best_rule": {"rule_id": "...", "code": "<DSL expr>", ...},
        "summary": {"mean": 51.4, ...},
        ...
    }

We:
  1. scan results/ for matching files
  2. transpile each `best_rule.code` (DSL → Python)
  3. expose a dict {rule_id: {"callable": fn, "meta": {...}}}

Used by the new /api/heuristix routes in app.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

from .dsl_translator import compile_rule, DSLTranspileError

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class HeuristixRule:
    rule_id:    str
    method:     str
    method_kr:  str
    scenario:   str
    instance:   str
    code_dsl:   str
    fn:         Callable = field(repr=False)
    summary:    dict = field(default_factory=dict)
    convergence: list = field(default_factory=list)
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id":   self.rule_id,
            "method":    self.method,
            "method_kr": self.method_kr,
            "scenario":  self.scenario,
            "instance":  self.instance,
            "code_dsl":  self.code_dsl,
            "summary":   self.summary,
            "convergence_len": len(self.convergence),
            "source":    self.source_file,
        }


def load_all() -> Dict[str, HeuristixRule]:
    """Return {rule_id: HeuristixRule} for every parseable evolution result."""
    out: Dict[str, HeuristixRule] = {}
    if not RESULTS_DIR.exists():
        return out
    for p in sorted(RESULTS_DIR.glob("evolution_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        best = d.get("best_rule", {}) or {}
        dsl  = (best.get("code") or "").strip()
        rid  = best.get("rule_id") or p.stem
        if not dsl:
            continue
        try:
            fn = compile_rule(dsl, rid)
        except DSLTranspileError as e:
            # Log a stub so the UI still sees it but marks as broken.
            print(f"[heuristix_rules] skip {rid}: {e}")
            continue
        out[rid] = HeuristixRule(
            rule_id=rid,
            method=d.get("method", ""),
            method_kr=d.get("method_kr", ""),
            scenario=d.get("scenario", ""),
            instance=d.get("instance", ""),
            code_dsl=dsl,
            fn=fn,
            summary=d.get("summary", {}) or {},
            convergence=d.get("convergence", []) or [],
            source_file=p.name,
        )
    return out


def filter_for(scenario: str = "", instance: str = "") -> List[HeuristixRule]:
    """Return rules optionally filtered by scenario and/or instance."""
    rules = load_all()
    out: List[HeuristixRule] = []
    for r in rules.values():
        if scenario and r.scenario != scenario:
            continue
        if instance and r.instance != instance:
            continue
        out.append(r)
    return out


if __name__ == "__main__":
    rules = load_all()
    print(f"loaded {len(rules)} rules")
    for rid in list(rules)[:5]:
        r = rules[rid]
        print(f"  {rid:50s}  {r.scenario}  AT~{r.summary.get('mean', 0):.1f}")
