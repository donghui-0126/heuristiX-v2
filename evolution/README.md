# LLM-driven heuristic evolution

Python driver implementing the offline evolution stage from 창종설 보고서
§4–5. Wraps the Rust simulator via subprocess; mutates, scores, and
reflects on dispatching-rule candidates expressed in the simulator's
evalexpr DSL.

## Layout

| File | Role |
| --- | --- |
| `simulator.py` | Subprocess wrapper + `ScoreWeights` per scenario (창종설 §6-2). |
| `baselines.py` | The five fixed rules (FIFO/EDD/SPT/CR/Urgency) as DSL strings. |
| `prompts.py` | LLM-A (rule generator) and LLM-S (reflector) prompt templates. |
| `memory.py` | ReasoningBank-style `MemoryBank` with `success`/`failure`/`strategy` items. |
| `llm.py` | `LLMClient` protocol + `MockLLM` + `AnthropicLLM` + `OpenAILLM`. |
| `evolve.py` | Main loop: evaluate → LLM-as-judge classify → LLM-S reflect → LLM-A generate → reseed. |
| `robustness.py` | Cross-scenario evaluator: runs one rule across S0–S5 and reports score stddev. |
| `portfolio.py` | DSevolve-style: evolve a specialist per scenario, compose nested-iff portfolio, evaluate. |

## Quick start (offline, no API key)

```bash
# Build the simulator binary once; the driver re-uses it across calls.
cargo build --release

# 5 iterations on the combined-shock scenario, mock LLM:
python -m evolution.evolve --scenario S5 --iterations 5 --replications 5

# Save best rule and memory bank:
python -m evolution.evolve --scenario S5 --iterations 10 \
  --memory-out runs/s5_memory.json --best-out runs/s5_best.json
```

The mock LLM performs syntactic mutations of elite expressions
(constant scaling, crossover blends, supply-chain term grafting). It
exercises every code path of the loop end-to-end without an API call.

## Switching to a real LLM

### OpenAI / GPT

```bash
pip install openai
export OPENAI_API_KEY=...
python -m evolution.evolve --scenario S5 --iterations 10 \
  --provider openai --model gpt-5
# Other models: --model gpt-4o, --model gpt-4.1, --model o3-mini, etc.
```

### Anthropic / Claude

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python -m evolution.evolve --scenario S5 --iterations 10 \
  --provider anthropic --model claude-opus-4-7
```

Wire format expected from the model:

- **Generation** (LLM-A): a free-form reply containing
  ```
  Thought: <rationale>
  Code: <a single evalexpr expression>
  ```
- **Reflection** (LLM-S): zero or more blocks of the form
  ```
  LESSON:
  type: success | failure | strategy
  title: ...
  description: ...
  content: ...
  perf_delta: <signed percent>
  END
  ```

Both are parsed by `llm.parse_rule_response()` and
`llm.parse_reflection_response()`.

## Cross-scenario robustness (창종설 §6-1)

```bash
# Evaluate any rule across S0–S5; reports per-scenario score + stddev.
python -m evolution.robustness --rule-file runs/s5_best.json --replications 10

# Compare a baseline directly:
python -m evolution.robustness --baseline FIFO

# Custom expression:
python -m evolution.robustness --expr "iff(urgent, 1.0, 0.0)" --label Urgency

# Save full report as JSON:
python -m evolution.robustness --baseline EDD --out runs/edd_robustness.json
```

The `score_stddev` is the report's "robustness" metric — lower is more
consistent across shocks.

## Scenario-specific objective weights

`RunResult.score(weights)` and `RunResult.primary_objective` use the
weights table in `simulator.py`:

| Scenario | tardiness | makespan | urgent | idle |
| --- | --- | --- | --- | --- |
| S0 / default | 1.0 | 0.5 | 5.0 | 0.1 |
| S1 (part delay) | 1.0 | 0.5 | 2.0 | **0.5** |
| S2 (mat shortage) | 1.0 | 0.5 | 2.0 | 0.3 |
| S3 (urgent surge) | 1.0 | 0.3 | **8.0** | 0.05 |
| S4 (DDT shock) | **2.0** | 0.3 | 4.0 | 0.05 |
| S5 (combined) | 1.0 | 0.5 | 5.0 | 0.2 |

These weights drive the evolution loop's elite selection — running
`evolve --scenario S3` automatically prioritises urgent-job tardiness
without any extra flag.

## Dual-expert mode (EvoDR §4-4)

Use a strong model for **LLM-A** (rule generation) and a cheaper one for
**LLM-S** (reflection):

```bash
python -m evolution.evolve --scenario S5 --iterations 10 \
  --gen-provider openai   --gen-model gpt-5 \
  --reflect-provider openai --reflect-model gpt-4o-mini
```

If only `--provider`/`--model` is given, both A and S share one client.

## LLM-as-judge memory (창종설 §5-3)

Before each reflection, `evolve.py` re-evaluates every baseline (cached)
to find the strongest fixed rule for the active scenario, then labels
each population member as:

- **success** if `obj ≤ baseline_obj × (1 − τ)`
- **failure** if `obj ≥ baseline_obj × (1 + τ)`
- **neutral** otherwise (not stored in memory)

The threshold τ defaults to 0.05 and is configurable via
`--success-threshold`. Only labelled rules enter the reflection prompt,
keeping memory items high-signal.

## Heuristic portfolio (DSevolve)

```bash
# Evolve specialists per S0..S5 (≈30s total with mock), compose,
# evaluate composite vs ATC across all scenarios:
python -m evolution.portfolio --iterations 3 --replications 5 \
  --out runs/portfolio_v1.json

# Use externally-evolved rules (e.g., from real-LLM runs):
python -m evolution.portfolio --rules-file runs/per_scenario_best.json
```

The composed expression is a single nested `iff()` that switches
between specialist rules based on `disruption_level`,
`supply_delay_level`, `urgent_ratio`, `mat_risk`, and slack tightness.
It can be passed straight back to the Rust simulator via
`--rule expr --expr "$(jq -r .portfolio_expr runs/portfolio_v1.json)"`.

## Adding a different LLM provider

Implement the `LLMClient` protocol from `llm.py` (two methods:
`generate_rule` and `reflect`). Add a branch in `llm.build_llm()` and
the new provider name will be available via `--provider`.

## What this driver does NOT do

- Plot results — emit JSON via `--best-out` / `robustness --out` and
  post-process externally.
- Persist intermediate populations — only `--memory-out` and `--best-out`.
