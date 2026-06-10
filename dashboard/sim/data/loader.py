"""Load FJSSP benchmark instances and generate TWK due dates.

Supports multiple benchmark families:
  - brandimarte  (Mk01-Mk15, default)
  - hurink
  - barnes
  - kacem
  - fattahi
  - behnke
  - dauzere
  - custom       (user-uploaded, placed in benchmarks/custom/)
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.job import Job, Operation

_ROOT = Path(__file__).parent.parent.parent
INSTANCE_DIR = _ROOT / "fjsp-instances" / "brandimarte"
FAMILIES_DIR = _ROOT / "fjsp-instances"
CUSTOM_DIR = _ROOT / "benchmarks" / "custom"

ALL_INSTANCES = [f"mk{i:02d}" for i in range(1, 11)]

# Canonical family folder names
FAMILY_FOLDERS: Dict[str, Path] = {
    "brandimarte": FAMILIES_DIR / "brandimarte",
    "hurink":      FAMILIES_DIR / "hurink",
    "barnes":      FAMILIES_DIR / "barnes",
    "kacem":       FAMILIES_DIR / "kacem",
    "fattahi":     FAMILIES_DIR / "fattahi",
    "behnke":      FAMILIES_DIR / "behnke",
    "dauzere":     FAMILIES_DIR / "dauzere",
    "custom":      CUSTOM_DIR,
}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_instance(name: str, family: str = "brandimarte") -> Tuple[List[Job], int]:
    """Load an FJSSP instance from a benchmark family.

    Returns
    -------
    jobs       : List[Job]  – release_time=0, due_date=0 (set via generate_due_dates)
    n_machines : int
    """
    folder = FAMILY_FOLDERS.get(family, FAMILIES_DIR / family)
    json_path = folder / f"{name.lower()}.json"
    if json_path.exists():
        return _load_from_path(json_path)
    txt_path = folder / f"{name.lower()}.txt"
    if txt_path.exists():
        return _load_from_txt(txt_path)
    raise FileNotFoundError(f"Instance file not found: {json_path}")


def load_instance_from_path(path: str | Path) -> Tuple[List[Job], int]:
    """Load an FJSSP instance directly from a file path."""
    return _load_from_path(Path(path))


def _load_from_txt(path: Path) -> Tuple[List[Job], int]:
    """Parse standard FJSSP text format (Brandimarte / Hurink style).

    Line 1 : n_jobs  n_machines  [avg_eligible – ignored if present]
    Lines 2+: n_ops  [n_eligible  m1 p1  m2 p2 …]_op1  …
    Machine indices are 1-based in some files, 0-based in others — we
    normalise to 0-based by subtracting 1 when the minimum index > 0.
    """
    with open(path, encoding="utf-8") as fh:
        tokens = fh.read().split()
    pos = 0

    n_jobs    = int(tokens[pos]); pos += 1
    n_machines = int(tokens[pos]); pos += 1
    # some files have a third header token (avg eligible); skip it
    if pos < len(tokens) and not tokens[pos].startswith('-') and int(tokens[pos]) < 100:
        # heuristic: if the token is small it's the avg-eligible header field
        try:
            val = int(tokens[pos])
            if val < n_jobs and val < n_machines:
                pos += 1
        except ValueError:
            pass

    jobs: List[Job] = []
    raw_machine_ids: List[int] = []

    for job_idx in range(n_jobs):
        n_ops = int(tokens[pos]); pos += 1
        operations: List[Operation] = []
        for op_idx in range(n_ops):
            n_el = int(tokens[pos]); pos += 1
            eligible: List[int] = []
            pt_map: Dict[int, float] = {}
            for _ in range(n_el):
                m = int(tokens[pos]);     pos += 1
                p = float(tokens[pos]);   pos += 1
                eligible.append(m)
                pt_map[m] = p
                raw_machine_ids.append(m)
            operations.append(Operation(
                op_idx=op_idx,
                eligible_machines=eligible,
                processing_times=pt_map,
            ))
        jobs.append(Job(job_id=job_idx, release_time=0.0, due_date=0.0, operations=operations))

    # normalise to 0-based if file uses 1-based machine ids
    if raw_machine_ids and min(raw_machine_ids) >= 1:
        for job in jobs:
            for op in job.operations:
                op.eligible_machines = [m - 1 for m in op.eligible_machines]
                op.processing_times  = {m - 1: p for m, p in op.processing_times.items()}

    return jobs, n_machines


def _load_from_path(path: Path) -> Tuple[List[Job], int]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    n_machines: int = data["machines"]
    jobs: List[Job] = []

    for job_idx, job_ops_raw in enumerate(data["jobs"]):
        operations: List[Operation] = []
        for op_idx, op_raw in enumerate(job_ops_raw):
            eligible = [entry["machine"] for entry in op_raw]
            pt_map = {entry["machine"]: float(entry["processing"]) for entry in op_raw}
            operations.append(Operation(
                op_idx=op_idx,
                eligible_machines=eligible,
                processing_times=pt_map,
            ))
        jobs.append(Job(
            job_id=job_idx,
            release_time=0.0,
            due_date=0.0,
            operations=operations,
        ))

    return jobs, n_machines


# ---------------------------------------------------------------------------
# Benchmark discovery
# ---------------------------------------------------------------------------

def list_benchmarks() -> Dict[str, List[str]]:
    """Return {family: [instance_name, ...]} for all available families."""
    result: Dict[str, List[str]] = {}
    for family, folder in FAMILY_FOLDERS.items():
        if folder.exists():
            names_json = {p.stem for p in folder.glob("*.json")}
            names_txt  = {p.stem for p in folder.glob("*.txt")}
            names = sorted(names_json | names_txt)
            if names:
                result[family] = names
    return result


def get_benchmark_stats(name: str, family: str = "brandimarte") -> Dict:
    """Compute statistics for a benchmark instance."""
    jobs, n_machines = load_instance(name, family)
    n_jobs = len(jobs)
    n_ops_total = sum(len(j.operations) for j in jobs)
    n_eligible_total = sum(
        len(op.eligible_machines)
        for j in jobs
        for op in j.operations
    )
    flexibility = n_eligible_total / n_ops_total if n_ops_total else 0.0
    max_ops = max(len(j.operations) for j in jobs)
    min_ops = min(len(j.operations) for j in jobs)

    return {
        "name": name,
        "family": family,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "n_operations": n_ops_total,
        "flexibility": round(flexibility, 2),
        "max_ops_per_job": max_ops,
        "min_ops_per_job": min_ops,
    }


# ---------------------------------------------------------------------------
# Due date generation
# ---------------------------------------------------------------------------

def generate_due_dates(
    jobs: List[Job],
    ddt: float = 1.5,
    seed: int = 42,
) -> List[Job]:
    """Assign due dates using the TWK (Total Work Content) method.

    d_j = release_time_j  +  ddt * sum(min_pt of all ops in job j)
    """
    result = deepcopy(jobs)
    for job in result:
        total_work = job.get_total_min_pt()
        job.due_date = job.release_time + ddt * total_work
    return result


def estimate_makespan(jobs: List[Job], n_machines: int, seed: int = 0) -> float:
    """Quick S0 estimate via FIFO to obtain T_est for scenario timing."""
    from ..rules.baseline import RULES
    from ..core.simulator import FJSSPSimulator

    jjobs = generate_due_dates(deepcopy(jobs), ddt=1.5, seed=seed)
    sim = FJSSPSimulator(jjobs, n_machines, RULES["B1"])
    result = sim.run()
    return result.makespan
