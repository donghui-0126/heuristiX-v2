"""Transpile heuristiX evalexpr DSL → simulator-compatible Python callable.

Source DSL (used by heuristiX_v2):
    exp_(-0.125 * max_(0, due_date - (current_time + remaining_pt))) / remaining_pt
    iff(urgent_order_flag, 10, 0) - exp_(-max_(0, machine_available_time - current_time))

Target — Python function with the dashboard signature:
    priority(job, operation, machine, state) -> float

Variable mapping (DSL → Python):
    DSL name                  Python expression
    ────────────────────────  ────────────────────────────────────────
    release_time / release    job["release_time"]
    due_date     / due        job["due_date"]
    urgent_order_flag         (1.0 if job.get("urgent") else 0.0)
    remaining_pt              job.get("remaining_pt", 0.0)
    processing_time / proc    operation["processing_time"]
    part_available_time       operation.get("part_available_time", 0.0)
    machine_id                machine["id"]
    machine_available_time    state.get("now", 0.0)   # idle when scoring
    current_time / now        state.get("now", 0.0)
    slack                     job["due_date"] - state.get("now", 0.0)
    penalty                   job.get("tardiness_penalty", 1.0)
    eligible_machines / n_eligible
                              len(operation.get("eligible_machines", []))
    total_proc                job.get("total_proc", 0.0)
    machine_queue             state.get("n_ready", 0)
    mach_util                 state.get("util", 0.0)
    mach_down                 0.0  # no breakdown in our scenarios

Function mapping:
    exp_(x)                   math.exp(min(50, x))     # cap to avoid overflow
    iff(c, t, e)              (t if c != 0 else e)
    gt/lt/eq                  conditional → 1.0 / 0.0
    max_(a, b) / min_(a, b)   max(a, b) / min(a, b)
    clamp(x, lo, hi)          max(lo, min(hi, x))
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict


# Variable substitutions: DSL name → Python expression that evaluates
# against the dashboard's (job, operation, machine, state) closure.
_VAR_SUBS: Dict[str, str] = {
    "release_time":              'job["release_time"]',
    "release":                   'job["release_time"]',
    "due_date":                  'job["due_date"]',
    "due":                       'job["due_date"]',
    "urgent_order_flag":         '(1.0 if job.get("urgent") else 0.0)',
    "urgent":                    '(1.0 if job.get("urgent") else 0.0)',
    "remaining_pt":              'job.get("remaining_pt", 0.0)',
    "remaining_proc":            'job.get("remaining_pt", 0.0)',
    "processing_time":           'operation["processing_time"]',
    "proc":                      'operation["processing_time"]',
    "part_available_time":       'operation.get("part_available_time", 0.0)',
    "part_avail":                'operation.get("part_available_time", 0.0)',
    "time_to_avail":             'max(0.0, operation.get("part_available_time", 0.0) - state.get("now", 0.0))',
    "machine_id":                'machine.get("id", 0)',
    "machine_available_time":    'state.get("now", 0.0)',
    "current_time":              'state.get("now", 0.0)',
    "now":                       'state.get("now", 0.0)',
    "slack":                     '(job["due_date"] - state.get("now", 0.0))',
    "penalty":                   'job.get("tardiness_penalty", 1.0)',
    "eligible_machines":         'len(operation.get("eligible_machines", []))',
    "n_eligible":                'len(operation.get("eligible_machines", []))',
    "total_proc":                'job.get("total_proc", 0.0)',
    "machine_queue":             'state.get("n_ready", 0)',
    "mach_util":                 'state.get("util", 0.0)',
    "mach_down":                 '0.0',
    "n_ready":                   'state.get("n_ready", 0)',
    "n_running":                 'state.get("n_running", 0)',
    "n_jobs":                    'state.get("n_jobs", 1)',
    "op_idx":                    'operation.get("op_idx", 0)',
    "mat_risk":                  '0.0',
    "inbound_delay":             '0.0',
    "supply_delay_level":        '0.0',
    "urgent_ratio":              'state.get("urgent_ratio", 0.0)',
    "disruption_level":          '0.0',
    "avg_inbound_delay":         '0.0',
}


# Function name substitutions.
_FUNC_SUBS: Dict[str, str] = {
    "exp_":  "_exp",
    "max_":  "max",
    "min_":  "min",
    "iff":   "_iff",
    "gt":    "_gt",
    "lt":    "_lt",
    "eq":    "_eq",
    "clamp": "_clamp",
}


# Identifier pattern — matches DSL variables.
_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


class DSLTranspileError(ValueError):
    pass


def _exp(x: float) -> float:
    # Cap exponent so a rogue rule can't crash with overflow.
    return math.exp(max(-50.0, min(50.0, x)))


def _iff(c: float, t: float, e: float) -> float:
    return t if abs(c) > 1e-12 else e


def _gt(a: float, b: float) -> float:
    return 1.0 if a > b else 0.0


def _lt(a: float, b: float) -> float:
    return 1.0 if a < b else 0.0


def _eq(a: float, b: float) -> float:
    return 1.0 if abs(a - b) < 1e-9 else 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    if math.isnan(x):
        x = 0.0
    return max(lo, min(hi, x))


_HELPER_NS = {
    "_exp": _exp,
    "_iff": _iff,
    "_gt":  _gt,
    "_lt":  _lt,
    "_eq":  _eq,
    "_clamp": _clamp,
    "max": max,
    "min": min,
    "math": math,
}


def transpile(dsl: str) -> str:
    """DSL string → Python expression string referencing job/operation/machine/state."""
    if not dsl or not dsl.strip():
        raise DSLTranspileError("empty expression")

    text = dsl.strip()

    # 1) Substitute known function names (longest first to avoid partial overlaps).
    for fname in sorted(_FUNC_SUBS, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(fname)}\b", _FUNC_SUBS[fname], text)

    # 2) Substitute known DSL variables → Python lookups.
    def replace_ident(m: re.Match) -> str:
        name = m.group(1)
        # Skip our helper functions and language keywords.
        if name in {"max", "min", "math", "True", "False", "None",
                    "_exp", "_iff", "_gt", "_lt", "_eq", "_clamp"}:
            return name
        if name in _VAR_SUBS:
            return _VAR_SUBS[name]
        # numeric literal-ish? leave it alone.
        if name[0].isdigit():
            return name
        # unknown identifier — fall back to 0.0 to keep evaluation safe.
        return "0.0"

    text = _IDENT.sub(replace_ident, text)
    return text


def compile_rule(dsl: str, rule_id: str = "heuristix") -> Callable:
    """Compile a DSL string to a callable matching the dashboard's signature."""
    py_expr = transpile(dsl)
    code_src = (
        "def __rule(job, operation, machine, state):\n"
        f"    try:\n"
        f"        return float({py_expr})\n"
        f"    except Exception:\n"
        f"        return float('-inf')\n"
    )
    local_ns: dict = {}
    try:
        exec(compile(code_src, f"<heuristix:{rule_id}>", "exec"), _HELPER_NS, local_ns)
    except Exception as e:
        raise DSLTranspileError(f"compile failed for {rule_id}: {e}\n  source: {dsl}\n  python: {py_expr}") from e
    fn = local_ns["__rule"]
    fn.__name__ = rule_id
    fn.__doc__ = f"heuristiX rule ({rule_id})\nDSL: {dsl[:120]}{'...' if len(dsl) > 120 else ''}"
    return fn


# Self-test when run directly.
if __name__ == "__main__":
    samples = [
        ("0.0 - proc",                                "B3 SPT 형식"),
        ("0.0 - due",                                 "B2 EDD"),
        ("exp_(-max_(0, due_date - (current_time + remaining_pt))) + iff(urgent_order_flag, 10, 0)",
         "S0/P3 발견된 규칙"),
        ("exp_(-0.125 * max_(0, due_date - (current_time + remaining_pt + part_available_time))) / remaining_pt",
         "S1/P3 발견된 규칙"),
    ]
    job  = {"release_time": 0.0, "due_date": 100.0, "urgent": False,
            "remaining_pt": 30.0, "tardiness_penalty": 1.0}
    op   = {"processing_time": 10.0, "part_available_time": 0.0,
            "eligible_machines": [0, 1, 2], "op_idx": 0}
    mach = {"id": 0}
    st   = {"now": 20.0, "n_ready": 5, "n_running": 1, "n_jobs": 12}
    for dsl, desc in samples:
        rule = compile_rule(dsl, "test")
        print(f"  {desc}")
        print(f"    DSL : {dsl[:80]}{'...' if len(dsl) > 80 else ''}")
        print(f"    py  : {transpile(dsl)[:80]}")
        print(f"    val : {rule(job, op, mach, st):.3f}")
        print()
