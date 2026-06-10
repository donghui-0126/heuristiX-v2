"""Metric calculations: AT, MIT, PTJ, ARI."""

from __future__ import annotations

from typing import Dict, List


def average_tardiness(completion_times: Dict[int, float], due_dates: Dict[int, float]) -> float:
    n = len(completion_times)
    if n == 0:
        return 0.0
    return sum(max(0.0, completion_times[j] - due_dates[j]) for j in completion_times) / n


def machine_idle_time(machine_mit: Dict[int, float]) -> float:
    return sum(machine_mit.values())


def percent_tardy_jobs(completion_times: Dict[int, float], due_dates: Dict[int, float]) -> float:
    n = len(completion_times)
    if n == 0:
        return 0.0
    tardy = sum(1 for j in completion_times if completion_times[j] > due_dates[j])
    return tardy / n * 100.0


def average_relative_improvement(
    at_baseline: float,
    at_proposed: float,
) -> float:
    """ARI = (AT_baseline - AT_proposed) / AT_baseline × 100 (%).

    Positive value means improvement over baseline.
    """
    if at_baseline <= 0:
        return 0.0
    return (at_baseline - at_proposed) / at_baseline * 100.0


def summarise_runs(at_list: List[float]) -> dict:
    """Compute mean, std, min, max from a list of AT values."""
    import statistics
    if not at_list:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(at_list),
        "std": statistics.stdev(at_list) if len(at_list) > 1 else 0.0,
        "min": min(at_list),
        "max": max(at_list),
        "n": len(at_list),
    }
