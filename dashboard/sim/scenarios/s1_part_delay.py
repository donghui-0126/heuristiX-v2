"""S1 – 부품 지연 시나리오 (Part Delay).

트리거 시점: Uniform(timing_min, timing_max) × T_est
영향받는 작업의 part_available_time을 trigger_time + delay_amount 로 갱신.

Parameters
----------
affected_ratio : float   영향받는 작업 비율 (0~1, 예: 0.20 = 20%)
delay_k        : float   지연 배수 — delay = avg_total_min_pt × k
timing_min     : float   트리거 시점 하한 (T_est 배수, 기본 0.1)
timing_max     : float   트리거 시점 상한 (T_est 배수, 기본 0.3)
"""

from __future__ import annotations

import random
from typing import List

from .base import BaseScenario, ScenarioEvent


class S1PartDelay(BaseScenario):

    def __init__(
        self,
        affected_ratio: float = 0.20,
        delay_k: float = 1.0,
        timing_min: float = 0.1,
        timing_max: float = 0.3,
    ):
        if not (0.0 < affected_ratio <= 1.0):
            raise ValueError("affected_ratio must be in (0, 1]")
        if delay_k <= 0:
            raise ValueError("delay_k must be positive")
        if not (0.0 <= timing_min < timing_max <= 1.0):
            raise ValueError("timing_min must be < timing_max, both in [0, 1]")

        self.affected_ratio = affected_ratio
        self.delay_k        = delay_k
        self.timing_min     = timing_min
        self.timing_max     = timing_max

    def build_events(self, jobs, n_machines: int, t_est: float, seed: int = 0) -> List[ScenarioEvent]:
        rng = random.Random(seed)

        trigger_time = rng.uniform(self.timing_min, self.timing_max) * t_est
        avg_total_pt = sum(j.get_total_min_pt() for j in jobs) / len(jobs)
        delay_amount = round(avg_total_pt * self.delay_k)

        n_affected = max(1, round(len(jobs) * self.affected_ratio))
        affected_jobs = rng.sample(list(range(len(jobs))), k=n_affected)

        updates = {
            str(jid): trigger_time + delay_amount
            for jid in affected_jobs
        }

        return [ScenarioEvent(
            time=trigger_time,
            event_type="S1_TRIGGER",
            data={"updates": updates},
        )]
