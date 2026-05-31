# heuristiX hub — end-to-end research platform

LLM 기반 dispatching rule 진화 실험을 팀이 공유하는 self-service 웹 환경.
자연어로 새 규칙을 만들고, 진화 실험을 launch하고, 결과를 비교한다.

```
[ NL ]
   │  "Critical ratio weighted by penalty"
   ▼
[ FastAPI /api/baselines/generate ]
   │   gpt-4o-mini → evalexpr DSL
   ▼
[ Rust ExprRule compile-check ]   ← 안전 boundary
   │   parse OK + tiny smoke run
   ▼
[ workspace.baselines.json ]      ← per-user 상태
   │
   ▼
[ /api/experiments  →  evolution.evolve ]
   │
   ▼
[ /api/experiments/{id}  →  best AT + convergence ]
```

## Quickstart

### A. Local (Python venv)

```bash
# 1. build Rust simulator once
cargo build --release

# 2. install Python deps
pip install -r hub/requirements.txt

# 3. set your key
export OPENAI_API_KEY=sk-...

# 4. launch backend
PYTHONPATH=. uvicorn hub.api.main:app --host 127.0.0.1 --port 8000 &

# 5. launch UI
HX_API_URL=http://127.0.0.1:8000 \
  streamlit run hub/ui/app.py --server.port 8501
```

Open <http://127.0.0.1:8501>.

### B. Docker compose

```bash
cp hub/.env.example .env
# edit .env to add OPENAI_API_KEY

docker compose up --build
```

Open <http://localhost:8501>.

## Endpoints

| Method | Path | What |
|---|---|---|
| GET | `/api/health` | liveness + canonical baseline count |
| POST | `/api/baselines/generate` | NL → evalexpr DSL → validate → save |
| GET | `/api/baselines?user=...` | canonical + user-added baselines |
| DELETE | `/api/baselines/{name}?user=...` | remove from workspace |
| POST | `/api/experiments` | launch p123 evolution loop |
| GET | `/api/experiments?user=...` | list user's experiments |
| GET | `/api/experiments/{id}?user=...` | status + best + log tail |

### Example — generate a baseline via curl

```bash
curl -X POST http://localhost:8000/api/baselines/generate \
  -H 'content-type: application/json' \
  -d '{
    "user": "alice",
    "name": "WeightedCR",
    "nl": "Critical ratio but multiplied by tardiness penalty"
  }'
```

Returns:
```json
{
  "name": "WeightedCR",
  "expr": "iff(remaining_proc > 0, -((slack / remaining_proc) * penalty), 0)",
  "thought": "Higher penalty jobs should be prioritized via slack-based CR.",
  "sample_at": 33.79,
  "saved": true
}
```

## Safety model

The platform is *not* "let the LLM modify your codebase". The LLM only emits
expressions in the existing evalexpr DSL, and **every generated expression
runs through `ExprRule::new` on the Rust side before being saved** — invalid
syntax, unknown variables, or div-by-zero get rejected without touching disk.

What the LLM *cannot* do:
- modify any `.rs` or `.py` source file
- add new variable names
- escape the evalexpr sandbox

What the LLM *can* do:
- compose any combination of allowed variables + helper functions
- propose new dispatching rules
- propose modifications to user-added rules

This is the "L1+L2" safety tier from the design discussion; broader code
modification (L3+) would require a sandboxed Claude Code/Cursor backend.

## Architecture

```
hub/
├── api/
│   ├── main.py        # FastAPI app + routes
│   ├── llm.py         # gpt-4o-mini → DSL
│   ├── validator.py   # Rust binary as safety check
│   └── workspace.py   # per-user JSON state
├── ui/
│   └── app.py         # Streamlit UI (Baselines/Run/History tabs)
├── workspace/<user>/  # per-user JSON state (volume-mounted)
├── Dockerfile         # multi-stage: Rust build + Python runtime
├── entrypoint.sh      # launches FastAPI + Streamlit
└── requirements.txt
```

The platform never writes to `evolution/baselines.py` or other source files.
User additions live entirely in `hub/workspace/<user>/baselines.json`, which
the experiment runner could compose with canonical baselines at launch time
(future work: wire this into `evolution/evolve.py` so user-added rules join
the initial population).
