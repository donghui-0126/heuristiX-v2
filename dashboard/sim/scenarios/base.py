"""Shared dataclass for scenario events passed to FJSSPSimulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioEvent:
    """A single event placed into the simulator's event queue at construction time."""
    time: float
    event_type: str          # "S1_TRIGGER" | "S2_TRIGGER" | "PART_ARRIVED"
    data: dict = field(default_factory=dict)


class BaseScenario:
    """Interface that all scenario classes implement."""

    def build_events(
        self,
        jobs,
        n_machines: int,
        t_est: float,
        seed: int = 0,
    ):
        """Return a list[ScenarioEvent] to inject into FJSSPSimulator."""
        raise NotImplementedError
