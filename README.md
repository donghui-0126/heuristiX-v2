# heuristiX

**LLM-based dispatching-rule evolution for dynamic FJSSP under external
supply-chain disruption.**

Pairs a fast Rust discrete-event simulator with a Python LLM evolution
loop (gpt-4o-mini) and a self-service web platform that lets team
members describe new rules in natural language, validate them against
the simulator, and run experiments — all from one dashboard.

```
[ user via web ]
       │  "Critical ratio weighted by penalty"
       ▼
[ hub/api  ─ gpt-4o-mini ─ DSL ]
       │  evalexpr expression
       ▼
[ Rust validator (ExprRule compile-check) ]   ← safety boundary
       │  parse OK + smoke run
       ▼
[ workspace/<user>/baselines.json ]
       │
       ▼
[ /api/experiments  →  evolution.evolve ]
       │  workspace baseline joins B1~B5 in initial population
       ▼
[ best.json + convergence + log ]  ───▶  History tab in web UI
```

## Repository structure

| Path | What |
|---|---|
| `src/` | Rust discrete-event simulator (FJSSP, S0/S1/S2 disruptions, 12 baselines, evalexpr DSL, Brandimarte loader) |
| `evolution/` | Python LLM evolution loop (LLM-A generator + LLM-S reflector, EoH 4 operators, MemoryBank) |
| `hub/` | FastAPI + Streamlit web platform (this is the team-collaboration layer) |
| `data/brandimarte/` | FJSSP benchmark format loader + sample |
| `docs/` | PIPELINE, PROMPTS reference docs |

## Quickstart — local

```bash
# 1. Build the Rust simulator
cargo build --release

# 2. Install Python deps
pip install -r hub/requirements.txt
pip install -r evolution/requirements.txt 2>/dev/null || true

# 3. Set your OpenAI key
export OPENAI_API_KEY=sk-...

# 4. Run a baseline experiment from the CLI
python3 -m evolution.evolve \
    --scenario S1 --variant P3 \
    --iterations 5 --replications 20 \
    --provider openai --model gpt-4o-mini

# 5. Or launch the web platform
PYTHONPATH=. uvicorn hub.api.main:app --host 127.0.0.1 --port 8000 &
streamlit run hub/ui/app.py --server.port 8501
# open http://localhost:8501
```

## Quickstart — Docker

```bash
cp hub/.env.example .env   # add your OPENAI_API_KEY
docker compose up --build
# open http://localhost:8501
```

## What gets measured

Primary metric: **AT** (Average Tardiness)
$$\mathrm{AT} = \tfrac{1}{n}\sum_j \max(0, C_j - d_j)$$

Auxiliary: MIT (machine idle), PTJ (tardy rate), ARI (vs best baseline).

## Scenarios (실험설계서_수정 §4)

- **S0** Normal — no external shock
- **S1** Part Delay — `part_available_time` for a fraction of jobs is shifted into the future
- **S2** Urgent Order — one new urgent job is inserted mid-simulation

## P1 / P2 / P3 variants (§5-2)

- **P1** — LLM only sees normal scheduling variables
- **P2** — LLM also sees disruption variables (`urgent_order_flag`, `part_available_time`)
- **P3** — P2 + memory bank of past success/failure lessons

## Status

Active research repository. Results from the latest batteries:
`runs/p123_battery_v3_{strict,fjssp,full}/` (gitignored — regenerable).

See `hub/README.md` for platform details, `docs/PIPELINE.md` for the
end-to-end evolution loop.
