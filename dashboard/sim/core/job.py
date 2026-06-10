from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Operation:
    op_idx: int
    eligible_machines: List[int]
    processing_times: Dict[int, float]   # machine_id -> processing_time
    status: str = "PENDING"              # PENDING | READY | PROCESSING | COMPLETED
    assigned_machine: int = -1
    start_time: float = -1.0
    end_time: float = -1.0

    def min_pt(self) -> float:
        return min(self.processing_times.values())

    def pt(self, machine_id: int) -> float:
        return self.processing_times[machine_id]


@dataclass
class Job:
    job_id: int
    release_time: float
    due_date: float
    operations: List[Operation]
    part_available_time: float = 0.0
    urgent_order_flag: int = 0
    completion_time: float = -1.0

    @property
    def is_complete(self) -> bool:
        return all(op.status == "COMPLETED" for op in self.operations)

    @property
    def n_ops(self) -> int:
        return len(self.operations)

    def get_remaining_pt(self, current_time: float = 0.0) -> float:
        """Sum of min processing times for non-COMPLETED ops.
        PROCESSING ops contribute actual remaining wall-clock time.
        """
        total = 0.0
        for op in self.operations:
            if op.status == "COMPLETED":
                continue
            if op.status == "PROCESSING":
                total += max(0.0, op.end_time - current_time)
            else:
                total += op.min_pt()
        return total

    def get_total_min_pt(self) -> float:
        return sum(op.min_pt() for op in self.operations)

    def copy_for_run(self) -> "Job":
        return deepcopy(self)
