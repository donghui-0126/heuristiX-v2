"""LLM client interface for the evolution loop.

Provides:
  - `LLMClient` protocol with `generate_rule()` and `reflect()`
  - `MockLLM` — does not call any API; performs syntactic mutations of
    elite rules so the loop is end-to-end runnable offline
  - `parse_rule_response()` / `parse_reflection_response()` — extract the
    structured fields from a free-form model reply

Plug in a real LLM by implementing the `LLMClient` protocol — see the
docstring of `AnthropicLLM` (a stub) for the wire format expected.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .baselines import BASELINES
from .memory import MemoryBank, MemoryItem
from .prompts import (
    LLM_A_SYSTEM, LLM_S_SYSTEM,
    build_generation_prompt, build_reflection_prompt,
)
from .simulator import RunResult


# --- response parsing ---------------------------------------------------

_RULE_PATTERN = re.compile(
    r"Thought:\s*(?P<thought>.+?)\s*Code:\s*(?P<code>.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_rule_response(text: str) -> tuple[str, str]:
    """Extract (thought, code) from an LLM reply. Strips fences and trailing
    punctuation so the code is directly evalexpr-feedable."""
    m = _RULE_PATTERN.search(text)
    if not m:
        # Treat the whole reply as code as a last resort.
        return ("", _strip_code(text))
    return (m["thought"].strip(), _strip_code(m["code"]))


def _strip_code(s: str) -> str:
    s = s.strip()
    # Remove ```...``` fences if present.
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Some LLMs wrap the expression in quotes or backticks.
    s = s.strip().strip("`").strip('"').strip("'")
    # Strip trailing semicolons/periods.
    s = s.rstrip(";.")
    return s


_LESSON_BLOCK = re.compile(
    r"LESSON:\s*(.*?)\s*END", re.IGNORECASE | re.DOTALL,
)
_FIELD = re.compile(r"^(type|title|description|content|perf_delta)\s*:\s*(.+?)$",
                    re.IGNORECASE | re.MULTILINE)


def parse_reflection_response(text: str, scenario: str) -> list[MemoryItem]:
    items: list[MemoryItem] = []
    for blk in _LESSON_BLOCK.finditer(text):
        body = blk.group(1)
        fields = {k.lower(): v.strip() for k, v in _FIELD.findall(body)}
        try:
            perf = float(fields.get("perf_delta", "0").rstrip("%"))
        except ValueError:
            perf = 0.0
        kind_raw = fields.get("type", "strategy").lower()
        kind = kind_raw if kind_raw in ("success", "failure", "strategy") else "strategy"
        items.append(MemoryItem(
            title=fields.get("title", "(untitled)"),
            description=fields.get("description", ""),
            content=fields.get("content", ""),
            scenario=scenario,
            perf_delta=perf,
            type=kind,  # type: ignore[arg-type]
        ))
    return items


# --- client protocol ----------------------------------------------------

class LLMClient(Protocol):
    """Dual interface: LLM-A generates rules; LLM-S reflects on labelled
    results. May be the same client or two separate clients (창종설 §4-4
    EvoDR-style dual-expert)."""

    def generate_rule(
        self,
        scenario: str,
        elite: list[RunResult],
        memory: MemoryBank,
        operation: str = "explore",
        *,
        retrieval_mode: str = "keyword",
        current_state=None,
        query_embedding=None,
        variant: str = "P3",
    ) -> tuple[str, str]:  # (thought, code)
        ...

    def reflect(
        self,
        scenario: str,
        successes: list[RunResult],
        failures: list[RunResult],
        *,
        variant: str = "P3",
    ) -> list[MemoryItem]:
        ...


# --- mock client (no API) ----------------------------------------------

@dataclass
class MockLLM:
    """Offline stand-in. Generates mutations of the elite expressions so
    the evolution loop runs end-to-end without an LLM. Useful for
    integration testing the simulator + memory pipeline."""

    seed: int = 0
    _rng: random.Random = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # --- generation ------------------------------------------------------

    def generate_rule(
        self,
        scenario: str,
        elite: list[RunResult],
        memory: MemoryBank,
        operation: str = "explore",
        *,
        retrieval_mode: str = "keyword",
        current_state=None,
        query_embedding=None,
        variant: str = "P3",
    ) -> tuple[str, str]:
        # Render the prompt anyway — it acts as a self-test that all the
        # template paths work. We discard the rendered text in mock mode.
        _ = build_generation_prompt(
            scenario, elite, memory, operation=operation,
            retrieval_mode=retrieval_mode, current_state=current_state,
            query_embedding=query_embedding, variant=variant,
        )

        if not elite:
            seed = self._rng.choice(list(BASELINES.values()))
        else:
            seed = self._rng.choice([r.expr for r in elite[:max(1, len(elite) // 2)]])

        if operation == "crossover" and len(elite) >= 2:
            a, b = elite[0].expr, elite[1].expr
            w = round(self._rng.uniform(0.3, 0.7), 2)
            return (
                f"crossover of top-2 with weight {w}",
                f"({w}) * ({a}) + ({1 - w:.2f}) * ({b})",
            )
        if operation == "modify":
            scaled = self._scale_constants(seed, factor=self._rng.uniform(0.5, 1.5))
            return ("scale numeric constants", scaled)
        if operation == "simplify":
            return ("drop the secondary term (mock)", seed)
        # explore: append a synthetic term. P1 must not reference disruption
        # variables; P2/P3 may.
        if variant == "P1":
            extra = self._rng.choice([
                "0.5 * remaining_proc",
                "0.0 - 0.1 * proc",
                "0.0 - 0.2 * slack",
            ])
        else:
            extra = self._rng.choice([
                "5.0 * mat_risk",
                "iff(gt(time_to_avail, 0.0), 0.0 - 0.5 * time_to_avail, 0.0)",
                "3.0 * urgent",
                "2.0 * disruption_level",
                "0.0 - 0.05 * inbound_delay",
            ])
        return (
            f"add extra term: {extra}",
            f"({seed}) + {extra}",
        )

    # --- reflection ------------------------------------------------------

    def reflect(
        self,
        scenario: str,
        successes: list[RunResult],
        failures: list[RunResult],
        *,
        variant: str = "P3",
    ) -> list[MemoryItem]:
        # Successes/failures arrive pre-classified by the LLM-as-judge
        # step in evolve.py (창종설 §5-3 5% threshold). MockLLM extracts
        # one success + one failure + one strategy as placeholders.
        _ = build_reflection_prompt(scenario, successes, failures, variant=variant)

        items: list[MemoryItem] = []
        if successes:
            top = successes[0]
            items.append(MemoryItem(
                title=f"winning structure held up under {scenario}",
                description=f"top success rule on {scenario}; obj={top.primary_objective:.1f}",
                content=top.expr,
                scenario=scenario,
                perf_delta=-(top.gap_ratio or 0.0) if top.gap_ratio is not None else 0.0,
                type="success",
            ))
        if failures:
            worst = failures[-1]
            items.append(MemoryItem(
                title=f"this structure underperformed in {scenario}",
                description=f"worst failure on {scenario}; obj={worst.primary_objective:.1f}",
                content=worst.expr,
                scenario=scenario,
                perf_delta=(worst.gap_ratio or 5.0) if worst.gap_ratio is not None else 5.0,
                type="failure",
            ))
        items.append(MemoryItem(
            title="combine slack-pressure with supply-chain term",
            description="any scenario with disruption_level > 0",
            content="mix CR-style slack term with mat_risk and time_to_avail features",
            scenario=scenario,
            perf_delta=0.0,
            type="strategy",
        ))
        return items

    # --- helpers ---------------------------------------------------------

    def _scale_constants(self, expr: str, factor: float) -> str:
        # Scale standalone numeric literals (does not touch operators).
        def repl(m: re.Match) -> str:
            v = float(m.group(0))
            return f"{v * factor:.3f}"
        return re.sub(r"\b\d+(?:\.\d+)?\b", repl, expr)


# --- real LLM stub ------------------------------------------------------

_SYSTEM_GENERATION = LLM_A_SYSTEM
_SYSTEM_REFLECTION = LLM_S_SYSTEM


class AnthropicLLM:
    """Real Claude client via the Anthropic SDK.

    Wire format expected from the model:
      - generate_rule: free-form reply with `Thought:` then `Code:`
      - reflect: zero-or-more `LESSON: ... END` blocks

    Setup:
        pip install anthropic
        export ANTHROPIC_API_KEY=...
    """

    def __init__(self, model: str = "claude-opus-4-7", api_key: Optional[str] = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "AnthropicLLM requires `pip install anthropic`. "
                "For offline runs use MockLLM instead."
            ) from e
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_rule(
        self,
        scenario: str,
        elite: list[RunResult],
        memory: MemoryBank,
        operation: str = "explore",
        *,
        retrieval_mode: str = "keyword",
        current_state=None,
        query_embedding=None,
        variant: str = "P3",
    ) -> tuple[str, str]:
        prompt = build_generation_prompt(
            scenario, elite, memory, operation=operation,
            retrieval_mode=retrieval_mode, current_state=current_state,
            query_embedding=query_embedding, variant=variant,
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM_GENERATION,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return parse_rule_response(text)

    def reflect(
        self,
        scenario: str,
        successes: list[RunResult],
        failures: list[RunResult],
        *,
        variant: str = "P3",
    ) -> list[MemoryItem]:
        prompt = build_reflection_prompt(scenario, successes, failures, variant=variant)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_REFLECTION,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return parse_reflection_response(text, scenario=scenario)


class OpenAILLM:
    """Real GPT client via the OpenAI Python SDK.

    Setup:
        pip install openai
        export OPENAI_API_KEY=...

    Defaults to `gpt-5` — override with --model. Works with any chat-
    completion-compatible model the account has access to (gpt-5, gpt-4o,
    gpt-4.1, etc.). Reasoning models are also supported via the same
    `chat.completions.create` endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAILLM requires `pip install openai`. "
                "For offline runs use MockLLM instead."
            ) from e
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        # `max_completion_tokens` is the modern parameter name; some older
        # SDK versions still expect `max_tokens`. We try the new one first
        # and fall back if the SDK rejects it.
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_tokens,
            )
        except TypeError:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()

    def generate_rule(
        self,
        scenario: str,
        elite: list[RunResult],
        memory: MemoryBank,
        operation: str = "explore",
        *,
        retrieval_mode: str = "keyword",
        current_state=None,
        query_embedding=None,
        variant: str = "P3",
    ) -> tuple[str, str]:
        prompt = build_generation_prompt(
            scenario, elite, memory, operation=operation,
            retrieval_mode=retrieval_mode, current_state=current_state,
            query_embedding=query_embedding, variant=variant,
        )
        text = self._chat(_SYSTEM_GENERATION, prompt, max_tokens=512)
        return parse_rule_response(text)

    def reflect(
        self,
        scenario: str,
        successes: list[RunResult],
        failures: list[RunResult],
        *,
        variant: str = "P3",
    ) -> list[MemoryItem]:
        prompt = build_reflection_prompt(scenario, successes, failures, variant=variant)
        text = self._chat(_SYSTEM_REFLECTION, prompt, max_tokens=1024)
        return parse_reflection_response(text, scenario=scenario)


def build_llm(provider: str, model: Optional[str] = None) -> LLMClient:
    """Factory: return a configured LLM client for the given provider.

    Providers: "mock", "anthropic", "openai". Used by `evolve.main()`.
    """
    p = provider.lower()
    if p == "mock":
        return MockLLM()
    if p == "anthropic":
        return AnthropicLLM(model=model or "claude-opus-4-7")
    if p == "openai":
        return OpenAILLM(model=model or "gpt-5")
    raise ValueError(f"unknown provider: {provider} (use mock / anthropic / openai)")
