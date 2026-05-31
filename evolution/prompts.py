"""Prompt templates for the LLM-A (rule generator) and LLM-S (reflector)
agents.

Templates follow 실험설계서_수정 §5-2 "PIPELINE 적용용 최소 프롬프트" — a
deliberately minimal frame that pins (a) the machine-idle event context,
(b) the priority(job, op, machine, state) signature, (c) AT as the sole
objective, and (d) a strictly bounded variable / function vocabulary.

Variants:
  P1 — disruption variables (`urgent_order_flag`, `part_available_time`)
       are hidden from the allowed-variables list.
  P2 — full spec variable set; no memory.
  P3 — full spec variable set; memory bank top-K lessons injected.
"""

from __future__ import annotations

from .memory import MemoryBank
from .simulator import RunResult


VARIANTS = ("P1", "P2", "P3")

# Variables hidden from P1's "Allowed variables only:" block.
_P1_HIDDEN_VARS = {
    "urgent_order_flag",     # S2 marker
    "part_available_time",   # S1 marker
}

# Full Spec §3-2 variable set (canonical names; the engine has the short
# aliases for backward compatibility).
_SPEC_VARS = [
    "release_time", "due_date", "remaining_pt", "part_available_time", "urgent_order_flag",
    "eligible_machines", "processing_time",
    "machine_id", "machine_available_time",
    "current_time",
]

_ALLOWED_FUNCTIONS = "iff, gt, lt, eq, max_, min_, clamp, exp_"


def _allowed_vars_block(variant: str) -> str:
    """Render the 'Allowed variables only:' block, hiding disruption
    variables for P1."""
    keep = [v for v in _SPEC_VARS if not (variant == "P1" and v in _P1_HIDDEN_VARS)]
    # Re-split into 3 lines to match the spec layout.
    job_vars = [v for v in ["release_time", "due_date", "remaining_pt",
                            "part_available_time", "urgent_order_flag"] if v in keep]
    op_vars = [v for v in ["eligible_machines", "processing_time"] if v in keep]
    mach_vars = [v for v in ["machine_id", "machine_available_time"] if v in keep]
    state_vars = [v for v in ["current_time"] if v in keep]
    parts: list[str] = ["Allowed variables only:"]
    if job_vars:   parts.append(", ".join(job_vars) + ",")
    if op_vars:    parts.append(", ".join(op_vars) + ",")
    if mach_vars:  parts.append(", ".join(mach_vars) + ",")
    if state_vars: parts.append(", ".join(state_vars))
    # Trim the trailing comma if the very last non-empty line ends with one.
    if parts[-1].endswith(","):
        parts[-1] = parts[-1].rstrip(",")
    return "\n".join(parts)


# ---- LLM-A (rule generator) -------------------------------------------- #

LLM_A_SYSTEM = (
    "You are LLM-A for dynamic FJSSP dispatching in a manufacturing setting "
    "under external disruption.\n"
    "At each machine-idle event, the simulator gives feasible (job, operation) "
    "candidates for that machine.\n"
    "Your job is to design a single priority scoring expression for "
    "`priority(job, operation, machine, state)`.\n"
    "The simulator dispatches the candidate with the highest score.\n"
    "Feasibility is enforced by the simulator (predecessor done, "
    "part_available_time <= current_time, machine_id in eligible_machines).\n"
    "Your objective is to minimize Average Tardiness "
    "(AT): AT = (1/n) * sum_j max(0, C_j - d_j).\n"
    "Use only allowed variables/functions and output exactly Thought and Code."
)


# Task directives — fed into {TASK_DETAIL}.
def _task_detail(operation: str, elite: list[RunResult]) -> tuple[str, str]:
    """Return (TASK_TYPE, TASK_DETAIL) for the current operation."""
    if operation == "crossover" and len(elite) >= 2:
        return (
            "crossover",
            "Combine the strengths of these two top expressions into a new one.\n"
            f"  A: {elite[0].expr}\n"
            f"  B: {elite[1].expr}",
        )
    if operation == "modify" and elite:
        return (
            "modify",
            "Tune weights or thresholds of this expression; keep its structure.\n"
            f"  base: {elite[0].expr}",
        )
    if operation == "simplify" and elite:
        return (
            "simplify",
            "Remove terms that do not pull their weight; produce a simpler "
            "expression with similar performance.\n"
            f"  base: {elite[0].expr}",
        )
    # Default: explore.
    if elite:
        elite_summary = "\n".join(
            f"  - AT={r.mean_tardiness:.1f}  expr: {r.expr}"
            for r in elite[:3]
        )
        return (
            "explore",
            "Generate a NEW priority expression structurally distinct from "
            "the elite below.\n" + elite_summary,
        )
    return (
        "explore",
        "Generate a priority expression. No prior elite — start from "
        "standard dispatching intuition.",
    )


def _format_memory_lessons(
    bank: MemoryBank,
    scenario: str,
    k: int = 3,
    *,
    retrieval_mode: str = "keyword",
    current_state=None,
    query_embedding=None,
) -> str:
    """Render TOP_K lessons in a compact form for {TOP_K_LESSONS}."""
    items = bank.retrieve(
        scenario, top_k=k,
        mode=retrieval_mode,
        current_state=current_state,
        query_embedding=query_embedding,
    )
    if not items:
        return "(none)"
    lines: list[str] = []
    for m in items:
        sign = "+" if m.perf_delta >= 0 else ""
        lines.append(
            f"- [{m.type} | Δ={sign}{m.perf_delta:.1f}%] {m.title}: {m.content}"
        )
    return "\n".join(lines)


def build_generation_prompt(
    scenario: str,
    elite: list[RunResult],
    memory: MemoryBank,
    operation: str = "explore",
    *,
    retrieval_mode: str = "keyword",
    current_state=None,
    query_embedding=None,
    variant: str = "P3",
) -> str:
    """LLM-A user prompt."""
    task_type, task_detail = _task_detail(operation, elite)
    memory_block = _format_memory_lessons(
        memory, scenario,
        retrieval_mode=retrieval_mode,
        current_state=current_state,
        query_embedding=query_embedding,
    )
    return f"""Scenario: {scenario}

{_allowed_vars_block(variant)}

Allowed functions only:
{_ALLOWED_FUNCTIONS}

Rules:
- Use only allowed variables/functions.
- No extra text, no markdown, no multiline code, no custom functions.
- Return only:
Thought: <1-2 sentences>
Code: <single evalexpr float expression; higher score = higher priority>

Memory:
{memory_block}

Task:
{task_type} - {task_detail}
"""


# ---- LLM-S (reflector) ------------------------------------------------- #

LLM_S_SYSTEM = (
    "You are LLM-S for the same dynamic FJSSP dispatching experiment.\n"
    "The policy being evolved is a priority scoring rule used at machine-idle "
    "events to pick one feasible (job, operation) candidate.\n"
    "Feasibility is enforced by the simulator; reflection must focus on "
    "priority-design effects, not feasibility logic.\n"
    "Your objective is to extract transferable lessons that improve AT "
    "minimization under disruption.\n"
    "Output only LESSON blocks. No prose outside blocks."
)


def build_reflection_prompt(
    scenario: str,
    successes: list[RunResult],
    failures: list[RunResult],
    *,
    variant: str = "P3",
) -> str:
    """LLM-S user prompt."""

    def _bullets(rs: list[RunResult]) -> str:
        if not rs:
            return "(none)"
        return "\n".join(
            f"- AT={r.mean_tardiness:.1f}  expr: {r.expr}"
            for r in rs
        )

    return f"""Scenario: {scenario}

SUCCESS rules:
{_bullets(successes)}

FAILURE rules:
{_bullets(failures)}

Write 2 to 4 lessons in this exact format:
LESSON:
type: success | failure | strategy
title: <short imperative phrase>
description: <when it applies>
content: <actionable rule pattern or pitfall>
perf_delta: <signed percent vs best fixed baseline>
END

Constraints:
- Focus on AT reduction logic under disruption.
- Prefer variable weighting / threshold insights.
- Do not copy full expressions verbatim.
"""


# ---- Backward-compatible exports -------------------------------------- #

# Kept for any older importer; rendering of the bindings block now lives
# inline in build_generation_prompt.
def bindings_for(variant: str) -> str:
    return _allowed_vars_block(variant)


def scenario_description_for(variant: str, scenario: str) -> str:
    # Kept for backward compat; the new prompt embeds Scenario: <id> only.
    return scenario


SCENARIO_DESCRIPTIONS = {
    "S0": "Normal: no external shocks.",
    "S1": "Part Delay.",
    "S2": "Urgent Order.",
}

BINDINGS = _allowed_vars_block("P3")
