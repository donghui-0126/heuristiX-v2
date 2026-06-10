"""Experiment configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    # ---- Instance selection ----
    instances: List[str] = field(
        default_factory=lambda: [f"mk{i:02d}" for i in range(1, 11)]
    )

    # ---- Due-date tightness ----
    ddt: float = 1.5

    # ---- Scenario ----
    scenario: str = "S0"            # "S0" | "S1" | "S2"

    # S1 parameters
    s1_affected_ratio: float = 0.20
    s1_delay_k: float = 1.0

    # S2 parameters
    s2_due_date_factor: float = 0.5

    # ---- Seeds ----
    n_final_seeds: int = 100        # seeds for final performance evaluation
    evolution_eval_seeds: int = 5   # seeds used during EoH fitness evaluation
    seed_offset: int = 0            # base seed (avoids overlap with evolution seeds)

    # ---- Methods to run ----
    run_baselines: bool = True      # B1–B10
    run_p1: bool = False            # LLM without external variables
    run_p2: bool = False            # LLM with all variables
    run_p3: bool = False            # LLM + experience storage

    # ---- EoH settings ----
    n_iter: int = 20
    pool_size: int = 10

    # ---- Output ----
    results_dir: str = "results"
    verbose: bool = True
