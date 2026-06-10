"""FJSSP Discrete-Event Simulator.

Event types
-----------
OP_COMPLETE      : operation finished on a machine
S1_TRIGGER       : part-delay scenario fires
S2_TRIGGER       : urgent-order scenario fires
PART_ARRIVED     : wake idle machines after a delayed part arrives
"""

from __future__ import annotations

import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .job import Job, Operation
from .machine import Machine


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Event:
    time: float
    seq: int                                   # tie-break (insertion order)
    event_type: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    completion_times: Dict[int, float]    # job_id -> C_j
    due_dates: Dict[int, float]           # job_id -> d_j
    machine_mit: Dict[int, float]         # machine_id -> idle time (no-work)
    at: float                             # Average Tardiness
    mit_total: float                      # sum of all machine MIT
    ptj: float                            # % Tardy Jobs
    makespan: float


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class FJSSPSimulator:
    """
    Priority function signature
    ---------------------------
    priority(job: dict, operation: dict, machine: dict, state: dict) -> float

    job keys      : release_time, due_date, remaining_pt,
                    part_available_time, urgent_order_flag
    operation keys: eligible_machines, processing_time, op_idx
    machine keys  : machine_id, machine_available_time
    state keys    : current_time, machine_workloads, machine_utilization,
                    avg_processing_time, n_waiting_jobs, target_utilization
    """

    def __init__(
        self,
        jobs: List[Job],
        n_machines: int,
        priority_fn: Callable,
        scenario_events: Optional[List] = None,
    ):
        self.original_jobs = jobs
        self.n_machines = n_machines
        self.priority_fn = priority_fn
        self.scenario_events: List = scenario_events or []

        # Precompute average processing time (static across the run)
        all_pts: List[float] = []
        for j in jobs:
            for op in j.operations:
                all_pts.extend(op.processing_times.values())
        self._avg_pt: float = sum(all_pts) / len(all_pts) if all_pts else 1.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> SimResult:
        # Deep-copy so the simulator is re-entrant
        self.jobs: Dict[int, Job] = {j.job_id: deepcopy(j) for j in self.original_jobs}
        self.machines: Dict[int, Machine] = {i: Machine(i) for i in range(self.n_machines)}
        self._events: list = []
        self._seq: int = 0
        self.current_time: float = 0.0
        self.idle_waiting: set = set()   # machines blocked on empty F_m

        # Mark first operation of every job READY
        for job in self.jobs.values():
            if job.operations:
                job.operations[0].status = "READY"

        # Queue scenario events
        for evt in self.scenario_events:
            self._push(evt.time, evt.event_type, evt.data)

        # Initial assignment pass
        for mid in self.machines:
            self._try_assign(mid)

        # Main event loop
        while not self._all_complete():
            if not self._events:
                # No scheduled events but jobs remain — machines are blocked
                # waiting for delayed parts; advance time to next part arrival.
                t_next = self._next_part_arrival()
                if t_next is None:
                    break   # genuine deadlock; should not happen in valid instances
                self.current_time = t_next
                for mid in list(self.idle_waiting):
                    self._try_assign(mid)
                continue

            evt = heapq.heappop(self._events)
            self.current_time = evt.time

            if evt.event_type == "OP_COMPLETE":
                self._handle_op_complete(evt.data)
            elif evt.event_type == "S1_TRIGGER":
                self._handle_s1(evt.data)
            elif evt.event_type == "S2_TRIGGER":
                self._handle_s2(evt.data)
            elif evt.event_type == "PART_ARRIVED":
                self._handle_part_arrived(evt.data)

        return self._collect_results()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_op_complete(self, data: dict):
        job_id = data["job_id"]
        op_idx = data["op_idx"]
        machine_id = data["machine_id"]
        duration = data["duration"]

        job = self.jobs[job_id]
        op = job.operations[op_idx]
        machine = self.machines[machine_id]

        op.status = "COMPLETED"
        machine.finish_job(duration)

        next_idx = op_idx + 1
        if next_idx < len(job.operations):
            next_op = job.operations[next_idx]
            next_op.status = "READY"
            # Wake any idle machine that can handle the newly ready op
            for mid in next_op.eligible_machines:
                if mid in self.idle_waiting:
                    self._try_assign(mid)
        else:
            job.completion_time = self.current_time

        # Free machine — look for new work
        self._try_assign(machine_id)

    def _handle_s1(self, data: dict):
        """Part-delay event: update part_available_time for affected jobs."""
        for job_id_str, new_pat in data["updates"].items():
            jid = int(job_id_str)
            if jid in self.jobs:
                self.jobs[jid].part_available_time = float(new_pat)
                # Schedule wake-up when the part actually arrives
                self._push(float(new_pat), "PART_ARRIVED", {"job_id": jid})

    def _handle_s2(self, data: dict):
        """Urgent-order event: insert a new job into the running simulation."""
        new_job: Job = data["job"]
        new_job.part_available_time = self.current_time
        if new_job.operations:
            new_job.operations[0].status = "READY"
        self.jobs[new_job.job_id] = new_job

        # Try all currently idle machines
        for mid in list(self.idle_waiting):
            self._try_assign(mid)
        for mid, m in self.machines.items():
            if m.status == "IDLE" and mid not in self.idle_waiting:
                self._try_assign(mid)

    def _handle_part_arrived(self, data: dict):
        job_id = int(data["job_id"])
        if job_id not in self.jobs:
            return
        job = self.jobs[job_id]
        for op in job.operations:
            if op.status == "READY":
                for mid in op.eligible_machines:
                    if mid in self.idle_waiting:
                        self._try_assign(mid)
                break

    # ------------------------------------------------------------------
    # Core scheduling logic
    # ------------------------------------------------------------------

    def _try_assign(self, machine_id: int):
        machine = self.machines[machine_id]
        if machine.status == "BUSY":
            return

        candidates = self._get_candidates(machine_id)

        if not candidates:
            machine.start_idle_wait(self.current_time)
            self.idle_waiting.add(machine_id)
            return

        self.idle_waiting.discard(machine_id)
        state = self._build_state()

        best = max(
            candidates,
            key=lambda c: self._safe_score(c[0], c[1], machine_id, state),
        )
        self._assign(best[0], best[1], machine_id)

    def _get_candidates(self, machine_id: int) -> List[Tuple[int, int]]:
        result = []
        for job in self.jobs.values():
            if job.is_complete:
                continue
            for op in job.operations:
                if op.status != "READY":
                    continue
                if machine_id not in op.eligible_machines:
                    continue
                if job.part_available_time > self.current_time + 1e-9:
                    continue
                result.append((job.job_id, op.op_idx))
        return result

    def _assign(self, job_id: int, op_idx: int, machine_id: int):
        job = self.jobs[job_id]
        op = job.operations[op_idx]
        machine = self.machines[machine_id]

        pt = op.pt(machine_id)
        finish = self.current_time + pt

        op.status = "PROCESSING"
        op.assigned_machine = machine_id
        op.start_time = self.current_time
        op.end_time = finish

        machine.start_busy(job_id, op_idx, finish, self.current_time)

        self._push(finish, "OP_COMPLETE", {
            "job_id": job_id,
            "op_idx": op_idx,
            "machine_id": machine_id,
            "duration": pt,
        })

    # ------------------------------------------------------------------
    # Dict builders for priority function
    # ------------------------------------------------------------------

    def _job_dict(self, job_id: int) -> dict:
        j = self.jobs[job_id]
        return {
            "release_time": j.release_time,
            "due_date": j.due_date,
            "remaining_pt": j.get_remaining_pt(self.current_time),
            "part_available_time": j.part_available_time,
            "urgent_order_flag": j.urgent_order_flag,
        }

    def _op_dict(self, job_id: int, op_idx: int, machine_id: int) -> dict:
        op = self.jobs[job_id].operations[op_idx]
        return {
            "eligible_machines": op.eligible_machines,
            "processing_time": op.pt(machine_id),
            "op_idx": op_idx,
        }

    def _machine_dict(self, machine_id: int) -> dict:
        m = self.machines[machine_id]
        return {
            "machine_id": machine_id,
            "machine_available_time": m.available_time,
        }

    def _build_state(self) -> dict:
        elapsed = max(self.current_time, 1e-9)

        # WINQ proxy: for each machine, sum of processing times of queued READY ops
        workloads: Dict[int, float] = {mid: 0.0 for mid in self.machines}
        for job in self.jobs.values():
            if job.is_complete:
                continue
            for op in job.operations:
                if op.status == "READY" and job.part_available_time <= self.current_time + 1e-9:
                    for mid in op.eligible_machines:
                        workloads[mid] = workloads.get(mid, 0.0) + op.pt(mid)

        utilization: Dict[int, float] = {
            mid: self.machines[mid].get_utilization(elapsed)
            for mid in self.machines
        }

        return {
            "current_time": self.current_time,
            "machine_workloads": workloads,
            "machine_utilization": utilization,
            "avg_processing_time": self._avg_pt,
            "n_waiting_jobs": len(self.idle_waiting),
            "target_utilization": 0.8,
        }

    def _safe_score(self, job_id: int, op_idx: int, machine_id: int, state: dict) -> float:
        try:
            return float(self.priority_fn(
                self._job_dict(job_id),
                self._op_dict(job_id, op_idx, machine_id),
                self._machine_dict(machine_id),
                state,
            ))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push(self, time: float, event_type: str, data: dict = None):
        self._seq += 1
        heapq.heappush(self._events, Event(time, self._seq, event_type, data or {}))

    def _all_complete(self) -> bool:
        return all(j.is_complete for j in self.jobs.values())

    def _next_part_arrival(self) -> Optional[float]:
        times = []
        for job in self.jobs.values():
            if job.is_complete:
                continue
            if job.part_available_time > self.current_time + 1e-9:
                for op in job.operations:
                    if op.status == "READY":
                        times.append(job.part_available_time)
                        break
        return min(times) if times else None

    def _collect_results(self) -> SimResult:
        final_time = max(
            (j.completion_time for j in self.jobs.values() if j.completion_time >= 0),
            default=self.current_time,
        )

        # Finalise MIT for machines still in idle-wait at end of simulation
        for m in self.machines.values():
            if m.idle_start >= 0:
                m.total_mit += final_time - m.idle_start
                m.idle_start = -1.0

        completion_times = {j.job_id: j.completion_time for j in self.jobs.values()}
        due_dates = {j.job_id: j.due_date for j in self.jobs.values()}
        machine_mit = {mid: m.total_mit for mid, m in self.machines.items()}

        n = len(self.jobs)
        at = sum(max(0.0, completion_times[jid] - due_dates[jid]) for jid in completion_times) / n
        tardy = sum(1 for jid in completion_times if completion_times[jid] > due_dates[jid])
        ptj = tardy / n * 100.0
        mit_total = sum(machine_mit.values())

        return SimResult(
            completion_times=completion_times,
            due_dates=due_dates,
            machine_mit=machine_mit,
            at=at,
            mit_total=mit_total,
            ptj=ptj,
            makespan=final_time,
        )
