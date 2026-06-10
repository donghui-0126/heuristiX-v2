"""S0 – normal operation: no disruption events."""

from __future__ import annotations

from typing import List

from .base import BaseScenario, ScenarioEvent


class S0Normal(BaseScenario):

    def build_events(self, jobs, n_machines: int, t_est: float, seed: int = 0) -> List[ScenarioEvent]:
        return []
