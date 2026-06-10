"""Helpers for loading and validating LLM-generated priority functions."""

from __future__ import annotations

import types
from typing import Callable

_ALLOWED_BUILTINS = {
    "abs", "max", "min", "sum", "len", "round", "sorted", "range",
    "int", "float", "bool", "str", "list", "dict", "tuple", "set",
    "True", "False", "None",
}


def load_priority_fn(code: str) -> Callable:
    """Safely exec LLM-generated code and return the `priority` function.

    The code must define a function named `priority`.
    Only a restricted set of builtins is available; no imports are allowed.
    """
    import math as _math

    ns = {
        "__builtins__": {k: v for k, v in vars(__builtins__).items()
                        if k in _ALLOWED_BUILTINS},
        "math": _math,
    }
    exec(compile(code, "<llm_rule>", "exec"), ns)
    fn = ns.get("priority")
    if not callable(fn):
        raise ValueError("Code does not define a callable named `priority`.")
    return fn


def validate_priority_fn(fn: Callable) -> bool:
    """Quick smoke-test: call with dummy dicts and check a float is returned."""
    dummy_job = {
        "release_time": 0.0, "due_date": 10.0, "remaining_pt": 5.0,
        "part_available_time": 0.0, "urgent_order_flag": 0,
    }
    dummy_op = {"eligible_machines": [0], "processing_time": 2.0, "op_idx": 0}
    dummy_mach = {"machine_id": 0, "machine_available_time": 0.0}
    dummy_state = {
        "current_time": 0.0,
        "machine_workloads": {0: 0.0},
        "machine_utilization": {0: 0.0},
        "avg_processing_time": 3.0,
        "n_waiting_jobs": 0,
        "target_utilization": 0.8,
    }
    try:
        result = fn(dummy_job, dummy_op, dummy_mach, dummy_state)
        return isinstance(result, (int, float))
    except Exception:
        return False
