"""Validate an LLM-generated evalexpr expression by running the Rust
simulator once on a tiny instance. If the binary exits non-zero, the
expression is rejected (likely a parse/compile error or unknown variable).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = REPO_ROOT / "target" / "release" / "heuristix"


@dataclass
class ValidationResult:
    ok: bool
    error: str = ""
    sample_at: float | None = None  # AT on the tiny smoke instance


def validate_expression(expr: str, timeout_sec: float = 8.0) -> ValidationResult:
    if not BINARY.exists():
        return ValidationResult(ok=False, error=f"binary not built: {BINARY}")
    argv = [
        str(BINARY),
        "--rule", "expr",
        "--expr", expr,
        "--scenario", "S0",
        "--jobs", "4",
        "--machines", "3",
        "--replications", "1",
        "--seed", "1",
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return ValidationResult(ok=False, error="timeout")
    if proc.returncode != 0:
        return ValidationResult(ok=False, error=(proc.stderr or proc.stdout)[:600])
    # Try to parse the AT from the first JSON line.
    import json
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            return ValidationResult(ok=True, sample_at=float(d["metrics"]["mean_tardiness"]))
        except Exception:
            continue
    return ValidationResult(ok=True)
