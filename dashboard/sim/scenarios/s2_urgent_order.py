"""S2 – 긴급 주문 시나리오 (Urgent Order).

트리거 시점: Uniform(arrival_min, arrival_max) × T_est 에 긴급 작업 삽입.
납기: trigger_time + due_date_factor × avg_remaining_min_pt

Parameters
----------
due_date_factor : float   납기 여유 계수 (0.1~5.0, 기본 0.5)
arrival_min     : float   도착 시점 하한 (T_est 배수, 기본 0.2)
arrival_max     : float   도착 시점 상한 (T_est 배수, 기본 0.5)
n_urgent_jobs   : int     삽입할 긴급 작업 수 (기본 1)
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List

from .base import BaseScenario, ScenarioEvent


class S2UrgentOrder(BaseScenario):

    def __init__(
        self,
        due_date_factor: float = 0.5,
        arrival_min: float = 0.2,
        arrival_max: float = 0.5,
        n_urgent_jobs: int = 1,
    ):
        if due_date_factor <= 0:
            raise ValueError("due_date_factor must be positive")
        if not (0.0 <= arrival_min < arrival_max <= 1.0):
            raise ValueError("arrival_min must be < arrival_max, both in [0, 1]")
        if n_urgent_jobs < 1:
            raise ValueError("n_urgent_jobs must be >= 1")

        self.due_date_factor = due_date_factor
        self.arrival_min     = arrival_min
        self.arrival_max     = arrival_max
        self.n_urgent_jobs   = n_urgent_jobs

    def build_events(self, jobs, n_machines: int, t_est: float, seed: int = 0) -> List[ScenarioEvent]:
        rng = random.Random(seed)
        events = []
        max_job_id = max(j.job_id for j in jobs)
        avg_remaining = sum(j.get_total_min_pt() for j in jobs) / len(jobs)

        for k in range(self.n_urgent_jobs):
            trigger_time = rng.uniform(self.arrival_min, self.arrival_max) * t_est

            template = deepcopy(rng.choice(jobs))
            for op in template.operations:
                op.status = "PENDING"
                op.assigned_machine = -1
                op.start_time = -1.0
                op.end_time = -1.0

            new_job_id = max_job_id + 1 + k
            template.job_id          = new_job_id
            template.release_time    = trigger_time
            template.urgent_order_flag = 1
            template.completion_time = -1.0
            template.due_date        = trigger_time + self.due_date_factor * avg_remaining

            events.append(ScenarioEvent(
                time=trigger_time,
                event_type="S2_TRIGGER",
                data={"job": template},
            ))

        return events
