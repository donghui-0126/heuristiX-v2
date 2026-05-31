"""NL → evalexpr DSL generator using gpt-4o-mini.

This is the *constrained* code-modification layer: the LLM only produces
expressions in our existing dispatching-rule DSL, which the Rust simulator
already validates via `ExprRule::new` (compile-check). Generated code that
fails the compile-check is rejected without touching the workspace.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from openai import OpenAI


_SYSTEM = """\
You are a heuristic dispatching-rule generator for a job-shop scheduling
simulator. You output a SINGLE evalexpr expression.

Allowed variables (use these exact names — others are rejected):
  Job:        release, due, slack, urgent, penalty,
              total_proc, remaining_proc,
              part_avail, time_to_avail
  Operation:  proc, op_idx
  Machine:    machine_id, machine_queue, mach_util
  State:      now, n_ready, n_running, n_jobs

Also accepted (aliases): release_time, due_date, remaining_pt,
processing_time, urgent_order_flag, part_available_time,
machine_available_time, current_time.

Allowed functions only:
  iff(cond, then, else), gt(a,b), lt(a,b), eq(a,b),
  max_(a,b), min_(a,b), clamp(x,lo,hi), exp_(x)

Output rules:
  - One single expression returning a Float. Higher score = higher priority.
  - No Python if/else, no comments, no markdown, no multi-line.
  - To prefer "smaller X", use `-X`.
  - To gate on a boolean, use `iff(...)`.

Return exactly:
Thought: <one line>
Code: <expression>
"""


_PATTERN = re.compile(r"Thought:\s*(?P<t>.+?)\s*Code:\s*(?P<c>.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class GenResult:
    thought: str
    code: str
    raw: str


def _strip(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip().strip("`").strip('"').strip("'").rstrip(";.")


def generate_baseline(nl_request: str, model: str = "gpt-4o-mini") -> GenResult:
    """Translate an NL request into a single evalexpr expression.

    Raises RuntimeError if the model output cannot be parsed.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=400,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": nl_request},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    m = _PATTERN.search(text)
    if not m:
        # Fallback: treat whole reply as code.
        return GenResult(thought="(no thought parsed)", code=_strip(text), raw=text)
    return GenResult(thought=m["t"].strip(), code=_strip(m["c"]), raw=text)
