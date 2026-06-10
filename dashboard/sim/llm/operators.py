"""EoH evolution operators: E1 (mutation), E2 (crossover), M1 (reflection), M2 (simplification).

Each operator builds a prompt for LLM-A.
All prompts follow the PDF-specified structure:
  Scenario / Baseline Rules / Variables / Previous Performance / Memory / Task
"""

from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared context blocks
# ---------------------------------------------------------------------------

_BASELINES_BLOCK = """\
=== Reference Baseline Rules ===
B1  FIFO    : earliest release_time first
B2  EDD     : earliest due_date first
B3  SPT     : shortest processing_time first
B4  CR      : (due_date - current_time) / max(remaining_pt, 1e-9)
B5  Urgency : urgent_order_flag (1=urgent, 0=normal)
B6  PT+WINQ+SL: processing_time + work-in-next-queue + slack composite
B7  CR+SPT  : CR ratio with SPT as tiebreaker
B8  Composite: release_time + remaining_pt balance
B9  Util    : machine utilization balancing
B10 ATCS    : urgency × exp(-slack / (k × avg_processing_time))"""

_VARIABLES_BLOCK = """\
=== Available Variables (use EXACT key names) ===
  job  : release_time, due_date, remaining_pt, part_available_time, urgent_order_flag
  op   : processing_time, op_idx, eligible_machines (list[int])
  mach : machine_id, machine_available_time
  state: current_time, machine_workloads (dict[int,float]),
         machine_utilization (dict[int,float]), avg_processing_time,
         n_waiting_jobs, target_utilization"""

_FORMAT_BLOCK = """\
=== Output Format (REQUIRED — no markdown, no extra text) ===
Thought: <1-2 sentences on the heuristic idea>
Code:
def priority(job: dict, operation: dict, machine: dict, state: dict) -> float:
    <implementation using only: abs, max, min, sum, len, round, int, float, bool, math>
    return <float>"""


def _elite_block(top_rules: List[tuple]) -> str:
    """top_rules: [(rule_id, code, avg_at), ...]"""
    if not top_rules:
        return ""
    lines = ["=== Previous Elite Performance (lower AT = better) ==="]
    for i, (rid, code, at) in enumerate(top_rules[:3], 1):
        snippet = "\n".join(("  " + l) for l in code.splitlines()[:6])
        lines.append(f"#{i} AT={at:.4f} | {rid}\n{snippet}")
    return "\n".join(lines)


def _memory_block(experience_block: Optional[str]) -> str:
    if not experience_block or "(No prior" in experience_block:
        return ""
    return f"=== Memory (relevant lessons from prior iterations) ===\n{experience_block}"


# ---------------------------------------------------------------------------
# E1 – Structure mutation  (탐색)
# ---------------------------------------------------------------------------

def e1_mutation(
    rule_id: str,
    code: str,
    at: float,
    scenario_desc: str = "",
    top_rules: Optional[List[tuple]] = None,
) -> str:
    elite = _elite_block(top_rules or [(rule_id, code, at)])
    scen  = f"=== Scenario: {scenario_desc} ===" if scenario_desc else ""
    return "\n\n".join(filter(None, [
        scen,
        _BASELINES_BLOCK,
        elite,
        _VARIABLES_BLOCK,
        f"""\
=== Task: E1 — Structure Mutation (탐색) ===
The rule below achieved AT = {at:.4f}. Create a STRUCTURALLY DIFFERENT variant
that may perform better. Possible changes: alter variable combinations,
add/remove a condition, change the weighting strategy, introduce a new term.

Current rule ({rule_id}):
{code}""",
        _FORMAT_BLOCK,
    ]))


# ---------------------------------------------------------------------------
# E2 – Crossover  (교차)
# ---------------------------------------------------------------------------

def e2_crossover(
    rule_a_id: str, code_a: str, at_a: float,
    rule_b_id: str, code_b: str, at_b: float,
    scenario_desc: str = "",
) -> str:
    scen = f"=== Scenario: {scenario_desc} ===" if scenario_desc else ""
    return "\n\n".join(filter(None, [
        scen,
        _BASELINES_BLOCK,
        f"""\
=== Previous Elite Performance ===
#1 AT={at_a:.4f} | {rule_a_id}
{chr(10).join('  '+l for l in code_a.splitlines()[:6])}

#2 AT={at_b:.4f} | {rule_b_id}
{chr(10).join('  '+l for l in code_b.splitlines()[:6])}""",
        _VARIABLES_BLOCK,
        f"""\
=== Task: E2 — Crossover (교차) ===
Combine the strengths of the two elite rules above into ONE new rule.
Keep what makes each effective; discard what hurts performance.""",
        _FORMAT_BLOCK,
    ]))


# ---------------------------------------------------------------------------
# M1 – Reflection (반성적 개선)
# ---------------------------------------------------------------------------

def m1_reflection(
    rule_id: str,
    code: str,
    at: float,
    best_baseline_at: float,
    scenario_description: str,
    experience_block: Optional[str] = None,
) -> str:
    gap     = at - best_baseline_at
    gap_pct = gap / max(best_baseline_at, 1e-9) * 100
    memory  = _memory_block(experience_block)

    return "\n\n".join(filter(None, [
        f"=== Scenario: {scenario_description} ===",
        _BASELINES_BLOCK,
        f"""\
=== Performance Feedback ===
Best baseline AT : {best_baseline_at:.4f}
This rule's AT   : {at:.4f}  (gap = {gap:+.4f}, {gap_pct:+.1f}%)

Current rule ({rule_id}):
{code}""",
        memory,
        _VARIABLES_BLOCK,
        f"""\
=== Task: M1 — Reflection (반성적 개선) ===
Identify WHY this rule underperforms in this scenario. Then REWRITE
the priority function to address those weaknesses. Focus on:
- How the rule handles part-delayed jobs (part_available_time)
- How it balances urgency (urgent_order_flag) vs. throughput
- Whether it exploits machine flexibility (len(eligible_machines))""",
        _FORMAT_BLOCK,
    ]))


# ---------------------------------------------------------------------------
# M2 – Simplification  (단순화)
# ---------------------------------------------------------------------------

def m2_simplification(
    rule_id: str,
    code: str,
    at: float,
    scenario_desc: str = "",
) -> str:
    scen = f"=== Scenario: {scenario_desc} ===" if scenario_desc else ""
    return "\n\n".join(filter(None, [
        scen,
        f"""\
=== Task: M2 — Simplification (단순화) ===
The rule below has AT = {at:.4f}.

{rule_id}:
{code}

Remove unnecessary conditions, redundant variables, or over-engineered logic
that adds noise without improving Average Tardiness. The simplified rule must
be at least as good and easier to understand.""",
        _VARIABLES_BLOCK,
        _FORMAT_BLOCK,
    ]))


# ---------------------------------------------------------------------------
# Initial seed generation
# ---------------------------------------------------------------------------

def initial_generation_prompt(scenario_description: str) -> str:
    """Prompt for generating the 5 LLM-seeded initial rules."""
    return "\n\n".join([
        f"=== Scenario: {scenario_description} ===",
        _BASELINES_BLOCK,
        _VARIABLES_BLOCK,
        """\
=== Task: Initial Rule Generation ===
Design a NOVEL FJSSP dispatching rule that minimises Average Tardiness (AT)
in this scenario. Consider combining multiple variables creatively:
- Weight urgency more strongly when part_available_time indicates recent delays
- Use machine flexibility (len(eligible_machines)) to defer flexible jobs
- Adapt weights dynamically based on machine_utilization or n_waiting_jobs
- Combine slack (due_date - current_time) with processing_time for priority""",
        _FORMAT_BLOCK,
    ])
