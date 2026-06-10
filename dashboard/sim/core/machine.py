from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Machine:
    machine_id: int
    status: str = "IDLE"          # IDLE | BUSY
    available_time: float = 0.0
    busy_time: float = 0.0
    idle_start: float = -1.0      # time when no-work idle began; -1 if not idle-waiting
    total_mit: float = 0.0        # accumulated Machine Idle Time (no work available)
    current_job_id: int = -1
    current_op_idx: int = -1

    def start_busy(self, job_id: int, op_idx: int, finish_time: float, current_time: float):
        if self.idle_start >= 0:
            self.total_mit += current_time - self.idle_start
            self.idle_start = -1.0
        self.status = "BUSY"
        self.current_job_id = job_id
        self.current_op_idx = op_idx
        self.available_time = finish_time

    def finish_job(self, duration: float):
        self.status = "IDLE"
        self.busy_time += duration
        self.current_job_id = -1
        self.current_op_idx = -1

    def start_idle_wait(self, current_time: float):
        if self.idle_start < 0:
            self.idle_start = current_time

    def get_utilization(self, elapsed_time: float) -> float:
        if elapsed_time <= 0:
            return 0.0
        return self.busy_time / elapsed_time
