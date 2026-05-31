"""LLM-driven heuristic evolution for the heuristiX simulator.

This package implements the offline evolution stage from 창종설 보고서
§4 (LLM dispatching-rule evolution) and §5 (memory bank). It drives the
Rust simulator via subprocess, scores candidate rules across scenarios,
and feeds top-k success/failure memories back into the next prompt.
"""

# Auto-load .env from the repo root so OPENAI_API_KEY etc. are visible to
# the LLM clients without any manual `source` step. Keys already in
# os.environ win (do not overwrite a real shell export).
import os as _os
from pathlib import Path as _Path

def _load_dotenv() -> None:
    p = _Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    try:
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in _os.environ:
                _os.environ[key] = value
    except OSError:
        pass

_load_dotenv()

from .baselines import BASELINES
from .memory import MemoryBank, MemoryItem
from .portfolio import compose_portfolio, evolve_one_scenario
from .simulator import RunResult, ScoreWeights, SCENARIO_WEIGHTS, Simulator, weights_for

__all__ = [
    "BASELINES",
    "MemoryBank", "MemoryItem",
    "RunResult", "ScoreWeights", "SCENARIO_WEIGHTS", "Simulator", "weights_for",
    "compose_portfolio", "evolve_one_scenario",
]
