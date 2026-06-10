"""OpenAI GPT wrapper — LLM-A (generator) + LLM-S (reflector)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
except Exception as _e:
    _client = None
    MODEL = "gpt-4o-mini"
    print(f"[LLM] OpenAI client init warning: {_e}")


# ---------------------------------------------------------------------------
# LLM-A : 생성자 (Heuristic Generator)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are LLM-A, a heuristic-evolution agent for FJSSP (Flexible Job Shop Scheduling Problem).

Your task: write a Python `priority` function that scores each (job, operation, machine) triplet.
The simulator calls this when a machine becomes idle and assigns the HIGHEST-scoring candidate.

You MUST respond in EXACTLY this two-section format — no markdown fences, no extra text:
Thought: <1-2 sentences explaining the heuristic idea and which variables drive it>
Code:
def priority(job: dict, operation: dict, machine: dict, state: dict) -> float:
    <implementation>
    return <float>

Allowed builtins: abs, max, min, sum, len, round, int, float, bool, math module.
No imports inside the function. Handle division by zero (use max(..., 1e-9) or .get()).

Available keys:
  job  : release_time, due_date, remaining_pt, part_available_time, urgent_order_flag
  op   : processing_time, op_idx, eligible_machines (list[int])
  mach : machine_id, machine_available_time
  state: current_time, machine_workloads (dict[int,float]),
         machine_utilization (dict[int,float]), avg_processing_time,
         n_waiting_jobs, target_utilization
"""


def _extract_code(raw: str) -> str:
    """Extract the Python function from a Thought/Code response."""
    # Split on 'Code:' marker (case-insensitive)
    m = re.split(r'(?i)^code\s*:', raw, maxsplit=1, flags=re.MULTILINE)
    if len(m) == 2:
        raw = m[1].strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    return raw


def generate_rule(prompt: str, temperature: float = 0.8) -> Optional[str]:
    """Call LLM-A and return the Python priority function code.

    Parses the Thought/Code response format; returns None on error.
    """
    if _client is None:
        return None
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        return _extract_code(raw)
    except Exception as exc:
        print(f"[LLM-A] generate_rule error: {exc}")
        return None


# ---------------------------------------------------------------------------
# LLM-S : 반성가 (Reflector)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_REFLECTOR = """\
You are LLM-S, a reflector in a heuristic-evolution loop for FJSSP scheduling.
Output zero or more LESSON blocks in EXACTLY this format.
No prose outside the blocks.

LESSON:
type: success | failure | strategy
title: <short imperative phrase>
description: <when it applies — scenario, conditions>
content: <specific variable combination, weight choice, or pitfall>
perf_delta: <signed percent vs best baseline; e.g. +12.5 or -7.0>
END
"""


def _parse_lessons(raw: str) -> List[dict]:
    """Extract all LESSON...END blocks from LLM-S output."""
    lessons = []
    blocks = re.findall(r'LESSON:(.*?)END', raw, re.DOTALL)
    for block in blocks:
        lesson: dict = {}
        for line in block.strip().splitlines():
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                lesson[key.strip()] = val.strip()
        if 'title' in lesson:
            lessons.append(lesson)
    return lessons


def generate_lessons(
    scenario_desc: str,
    success_rules: List[tuple],   # [(rule_id, code, at_score, best_baseline_at)]
    failure_rules: List[tuple],
) -> List[dict]:
    """Call LLM-S to extract transferable lessons from this generation's results.

    Returns a list of parsed lesson dicts.
    """
    if _client is None:
        return []
    if not success_rules and not failure_rules:
        return []

    def _rule_lines(rules, label):
        lines = [f"=== {label} ==="]
        for rid, code, at, best in rules[:5]:
            pct = (best - at) / max(best, 1e-9) * 100
            snippet = "\n".join(("  " + l) for l in code.splitlines()[:6])
            lines.append(f"- {rid} | AT={at:.3f} | {pct:+.1f}% vs baseline\n{snippet}")
        return "\n".join(lines)

    prompt = (
        f"Scenario: {scenario_desc}\n\n"
        + _rule_lines(success_rules, "SUCCESS rules (beat best baseline by ≥5%)")
        + "\n\n"
        + _rule_lines(failure_rules, "FAILURE rules (lost to best baseline by ≥5%)")
        + "\n\nExtract 2-4 transferable lessons. "
          "Focus on WHICH variables mattered, which weights were too aggressive, "
          "which conditional structures worked. Avoid restating rule code verbatim."
    )

    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_REFLECTOR},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        return _parse_lessons(raw)
    except Exception as exc:
        print(f"[LLM-S] generate_lessons error: {exc}")
        return []
