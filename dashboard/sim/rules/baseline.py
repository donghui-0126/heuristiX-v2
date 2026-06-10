"""Baseline dispatching rules B1 – B10.

All rules share the same signature:
    priority(job, operation, machine, state) -> float

Higher return value = higher priority (selected first).

B1  FIFO         : Arrival order
B2  EDD          : Earliest due date
B3  SPT          : Shortest processing time
B4  CR           : Critical ratio  (time-remaining / remaining-work)
B5  Urgency      : Urgent-order flag bonus

B6  PT+WINQ+SL   : Processing time + Work In Next Queue + Slack
B7  CR+SPT       : Weighted blend of CR and SPT (alpha=0.5)
B8  AT-RPT       : Arrival time + remaining processing time (lower = better)
B9  PDDR         : SPT when machine under-utilised; LPT when over-utilised
B10 ATCS         : Apparent Tardiness Cost with Setup approximation
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# B1 – FIFO
# ---------------------------------------------------------------------------

def b1_fifo(job: dict, operation: dict, machine: dict, state: dict) -> float:
    return -job["release_time"]


# ---------------------------------------------------------------------------
# B2 – EDD
# ---------------------------------------------------------------------------

def b2_edd(job: dict, operation: dict, machine: dict, state: dict) -> float:
    return -job["due_date"]


# ---------------------------------------------------------------------------
# B3 – SPT
# ---------------------------------------------------------------------------

def b3_spt(job: dict, operation: dict, machine: dict, state: dict) -> float:
    return -operation["processing_time"]


# ---------------------------------------------------------------------------
# B4 – CR  (Critical Ratio)
# ---------------------------------------------------------------------------

def b4_cr(job: dict, operation: dict, machine: dict, state: dict) -> float:
    time_remaining = max(job["due_date"] - state["current_time"], 1.0)
    remaining_pt = max(job["remaining_pt"], 1.0)
    cr = time_remaining / remaining_pt
    return -cr     # lower CR = more critical = higher priority


# ---------------------------------------------------------------------------
# B5 – Urgency
# ---------------------------------------------------------------------------

def b5_urgency(job: dict, operation: dict, machine: dict, state: dict) -> float:
    return job["urgent_order_flag"] * 1000.0 - job["due_date"]


# ---------------------------------------------------------------------------
# B6 – PT + WINQ + SL
# ---------------------------------------------------------------------------

def b6_pt_winq_sl(job: dict, operation: dict, machine: dict, state: dict) -> float:
    pt = operation["processing_time"]
    # WINQ: queued work on the current machine (proxy from state)
    winq = state.get("machine_workloads", {}).get(machine["machine_id"], 0.0)
    slack = max(
        0.0,
        job["due_date"] - state["current_time"] - job["remaining_pt"],
    )
    return -(pt + winq + slack)


# ---------------------------------------------------------------------------
# B7 – CR + SPT  (alpha = 0.5)
# ---------------------------------------------------------------------------

def b7_cr_spt(job: dict, operation: dict, machine: dict, state: dict) -> float:
    alpha = 0.5
    time_remaining = max(job["due_date"] - state["current_time"], 1.0)
    remaining_pt = max(job["remaining_pt"], 1.0)
    cr = time_remaining / remaining_pt
    spt_val = operation["processing_time"]
    return -(alpha * cr + (1.0 - alpha) * spt_val)


# ---------------------------------------------------------------------------
# B8 – AT-RPT  (Arrival Time + Remaining Processing Time)
# ---------------------------------------------------------------------------

def b8_at_rpt(job: dict, operation: dict, machine: dict, state: dict) -> float:
    return -(job["release_time"] + job["remaining_pt"])


# ---------------------------------------------------------------------------
# B9 – PDDR  (Priority Dispatching based on Dynamic Resource allocation)
# ---------------------------------------------------------------------------

def b9_pddr(job: dict, operation: dict, machine: dict, state: dict) -> float:
    util = state.get("machine_utilization", {}).get(machine["machine_id"], 0.0)
    target = state.get("target_utilization", 0.8)
    pt = operation["processing_time"]
    if util < target:
        return -pt       # SPT: keep machine busy with short jobs
    else:
        return pt        # LPT: long jobs fill machine time when heavily loaded


# ---------------------------------------------------------------------------
# B10 – ATCS  (Apparent Tardiness Cost with Setup)
# ---------------------------------------------------------------------------
# ATC_score = (1 / pt) * exp(-max(0, slack) / (k1 * avg_pt))
# No explicit setup times → setup_penalty = 1.0

def b10_atcs(job: dict, operation: dict, machine: dict, state: dict) -> float:
    k1 = 2.0
    avg_pt = max(state.get("avg_processing_time", 1.0), 1e-9)
    pt = max(operation["processing_time"], 1e-9)
    remaining_pt = max(job["remaining_pt"], 1e-9)
    slack = max(0.0, job["due_date"] - state["current_time"] - remaining_pt)
    atc_score = (1.0 / pt) * math.exp(-slack / (k1 * avg_pt))
    return atc_score     # setup_penalty = 1.0 (no setup data in standard FJSSP)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RULES: dict = {
    "B1":  b1_fifo,
    "B2":  b2_edd,
    "B3":  b3_spt,
    "B4":  b4_cr,
    "B5":  b5_urgency,
    "B6":  b6_pt_winq_sl,
    "B7":  b7_cr_spt,
    "B8":  b8_at_rpt,
    "B9":  b9_pddr,
    "B10": b10_atcs,
}

BASELINE_IDS = list(RULES.keys())
