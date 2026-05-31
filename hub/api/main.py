"""heuristiX research platform — FastAPI app.

Endpoints
  GET  /api/health
  POST /api/baselines/generate     — NL → evalexpr; validate; save
  GET  /api/baselines              — list all (BASELINES + workspace)
  DELETE /api/baselines/{name}     — remove from workspace
  POST /api/experiments            — launch p123_battery against workspace
  GET  /api/experiments            — list experiments
  GET  /api/experiments/{id}       — get experiment status + results
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hub.api.llm import generate_baseline
from hub.api.validator import validate_expression
from hub.api.workspace import REPO_ROOT, get_workspace

# Pull canonical BASELINES dict at import time (one source of truth).
sys.path.insert(0, str(REPO_ROOT))
from evolution.baselines import BASELINES  # noqa: E402


_EXPERIMENT_REGISTRY: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"[platform] REPO_ROOT={REPO_ROOT}")
    print(f"[platform] {len(BASELINES)} canonical baselines available")
    yield


app = FastAPI(title="heuristiX research platform", version="0.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---- schemas ------------------------------------------------------------

class GenerateBaselineReq(BaseModel):
    user: str = Field(..., examples=["alice"])
    name: str = Field(..., examples=["WeightedCR"])
    nl: str = Field(..., examples=["Critical ratio but weighted by tardiness penalty"])


class GenerateBaselineResp(BaseModel):
    name: str
    expr: str
    thought: str
    sample_at: float | None
    saved: bool


class LaunchExperimentReq(BaseModel):
    user: str
    scenario: str = "S1"
    variant: str = "P3"
    iterations: int = 5
    replications: int = 20
    jobs: int = 12
    machines: int = 6
    flexibility: float = 0.5
    part_delay_ratio: float = 0.2
    part_delay_k: float = 1.0
    urgent_due_ratio: float = 0.3


# ---- routes -------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "n_canonical_baselines": len(BASELINES)}


@app.post("/api/baselines/generate", response_model=GenerateBaselineResp)
def generate_baseline_endpoint(req: GenerateBaselineReq):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY not set on server")
    ws = get_workspace(req.user)
    if req.name in BASELINES:
        raise HTTPException(400, f"name '{req.name}' collides with canonical baseline")

    # 1. LLM generates DSL expression.
    try:
        result = generate_baseline(req.nl)
    except Exception as e:
        raise HTTPException(502, f"LLM generation failed: {e}")

    # 2. Validate via the Rust simulator (compile-check + tiny smoke run).
    v = validate_expression(result.code)
    if not v.ok:
        return GenerateBaselineResp(
            name=req.name, expr=result.code, thought=result.thought,
            sample_at=None, saved=False,
        )

    # 3. Save to workspace.
    ws.add_baseline(req.name, result.code, description=req.nl)
    return GenerateBaselineResp(
        name=req.name, expr=result.code, thought=result.thought,
        sample_at=v.sample_at, saved=True,
    )


@app.get("/api/baselines")
def list_baselines(user: str = Query(...)):
    ws = get_workspace(user)
    user_bls = ws.load_baselines()
    return {
        "canonical": BASELINES,
        "user": user_bls,
        "effective": {**BASELINES, **user_bls},
    }


@app.delete("/api/baselines/{name}")
def delete_baseline(name: str, user: str = Query(...)):
    ws = get_workspace(user)
    removed = ws.remove_baseline(name)
    if not removed:
        raise HTTPException(404, f"no user baseline named {name}")
    return {"removed": name}


# ---- experiments --------------------------------------------------------

@app.post("/api/experiments")
def launch_experiment(req: LaunchExperimentReq):
    ws = get_workspace(req.user)
    exp_id = uuid.uuid4().hex[:8]
    out_dir = ws.experiments_dir / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best_out = out_dir / "best.json"
    log_path = out_dir / "evolve.log"

    binary = REPO_ROOT / "target" / "release" / "heuristix"
    if not binary.exists():
        raise HTTPException(500, "Rust binary not built (run cargo build --release)")

    argv = [
        sys.executable, "-m", "evolution.evolve",
        "--scenario", req.scenario,
        "--variant", req.variant,
        "--iterations", str(req.iterations),
        "--replications", str(req.replications),
        "--jobs", str(req.jobs),
        "--machines", str(req.machines),
        "--seed", "1000",
        "--flexibility", str(req.flexibility),
        "--part-delay-ratio", str(req.part_delay_ratio),
        "--part-delay-k", str(req.part_delay_k),
        "--urgent-due-ratio", str(req.urgent_due_ratio),
        "--provider", "openai",
        "--model", "gpt-4o-mini",
        "--best-out", str(best_out),
    ]
    # Wire workspace baselines into the initial population so user-added
    # rules actually compete in the evolution loop.
    if ws.baselines_path.exists() and ws.load_baselines():
        argv += ["--extra-baselines-json", str(ws.baselines_path)]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT)

    _EXPERIMENT_REGISTRY[exp_id] = {
        "id": exp_id, "user": req.user, "pid": proc.pid,
        "started": time.time(), "config": req.model_dump(),
        "best_out": str(best_out), "log": str(log_path),
        "proc": proc, "log_fh": log_fh,
    }
    return {"experiment_id": exp_id, "pid": proc.pid, "log": str(log_path)}


@app.get("/api/experiments")
def list_experiments(user: str = Query(...)):
    out = []
    for eid, rec in list(_EXPERIMENT_REGISTRY.items()):
        if rec["user"] != user:
            continue
        proc = rec["proc"]
        status = "running" if proc.poll() is None else (
            "completed" if proc.returncode == 0 else f"failed({proc.returncode})"
        )
        out.append({
            "id": eid,
            "status": status,
            "started": rec["started"],
            "config": rec["config"],
            "log": rec["log"],
            "best_out": rec["best_out"],
        })
    return {"experiments": sorted(out, key=lambda x: -x["started"])}


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str, user: str = Query(...)):
    rec = _EXPERIMENT_REGISTRY.get(exp_id)
    if rec is None or rec["user"] != user:
        raise HTTPException(404, "experiment not found")
    proc = rec["proc"]
    is_done = proc.poll() is not None
    status = "running" if not is_done else (
        "completed" if proc.returncode == 0 else f"failed({proc.returncode})"
    )

    best: Optional[dict] = None
    if is_done and Path(rec["best_out"]).exists():
        try:
            best = json.loads(Path(rec["best_out"]).read_text())
        except Exception:
            best = None

    # Tail last 80 lines of log for live preview.
    tail = ""
    try:
        with open(rec["log"]) as f:
            lines = f.readlines()
        tail = "".join(lines[-80:])
    except Exception:
        pass

    return {
        "id": exp_id,
        "status": status,
        "config": rec["config"],
        "started": rec["started"],
        "elapsed": time.time() - rec["started"],
        "best": best,
        "log_tail": tail,
    }
