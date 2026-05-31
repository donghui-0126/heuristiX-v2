"""heuristiX research platform — Streamlit UI.

Pages
  Baselines : NL → DSL generation + workspace management
  Run       : Launch an evolution experiment
  History   : View past experiment results

Connects to the FastAPI backend at HX_API_URL (default localhost:8000).
"""

from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st


API_URL = os.environ.get("HX_API_URL", "http://127.0.0.1:8000")


# ---- helpers ------------------------------------------------------------

def api_get(path: str, **params):
    r = httpx.get(f"{API_URL}{path}", params=params, timeout=30.0)
    r.raise_for_status()
    return r.json()


def api_post(path: str, **body):
    r = httpx.post(f"{API_URL}{path}", json=body, timeout=120.0)
    r.raise_for_status()
    return r.json()


def api_delete(path: str, **params):
    r = httpx.delete(f"{API_URL}{path}", params=params, timeout=10.0)
    r.raise_for_status()
    return r.json()


# ---- layout -------------------------------------------------------------

st.set_page_config(page_title="heuristiX hub", layout="wide")

with st.sidebar:
    st.markdown("## heuristiX hub")
    st.caption("LLM 기반 dispatching rule 진화 research platform")
    user = st.text_input("User ID", value="alice",
                         help="A-Z, a-z, 0-9, _ and - only, max 32 chars")
    st.divider()
    try:
        health = api_get("/api/health")
        st.success(f"API ok · {health['n_canonical_baselines']} canonical rules")
    except Exception as e:
        st.error(f"API offline: {e}")
        st.stop()

    bls = api_get("/api/baselines", user=user)
    st.markdown("### Canonical baselines")
    st.markdown(", ".join(f"`{n}`" for n in bls["canonical"]))
    st.markdown("### Your baselines")
    if not bls["user"]:
        st.caption("_(none yet — add one in the Baselines tab)_")
    else:
        for name, expr in bls["user"].items():
            with st.expander(f"`{name}`"):
                st.code(expr, language="text")
                if st.button("Remove", key=f"rm-{name}"):
                    api_delete(f"/api/baselines/{name}", user=user)
                    st.rerun()


tab_b, tab_r, tab_h = st.tabs(["🧬 Baselines", "🚀 Run", "📜 History"])


# ---- Baselines tab ------------------------------------------------------

with tab_b:
    st.subheader("Generate a new dispatching rule via NL")
    st.caption(
        "Describe the rule in natural language. gpt-4o-mini will translate it "
        "into our evalexpr DSL, the Rust simulator will compile-check it, and "
        "if it passes, it's added to your workspace."
    )

    cols = st.columns([1, 2])
    with cols[0]:
        rule_name = st.text_input("Rule name", value="MyRule")
    with cols[1]:
        nl = st.text_area(
            "Describe in natural language",
            value="Critical ratio but weighted by tardiness penalty — urgent jobs should get a boost",
            height=80,
        )

    if st.button("Generate", type="primary"):
        with st.spinner("Generating + validating…"):
            try:
                resp = api_post("/api/baselines/generate", user=user, name=rule_name, nl=nl)
            except httpx.HTTPStatusError as e:
                st.error(f"API error: {e.response.text}")
                resp = None

        if resp:
            cols = st.columns([1, 2])
            with cols[0]:
                if resp["saved"]:
                    st.success(f"Saved as `{resp['name']}`")
                    if resp["sample_at"] is not None:
                        st.metric("Smoke test AT", f"{resp['sample_at']:.1f}")
                else:
                    st.warning("Generated but failed validation — not saved")
            with cols[1]:
                st.markdown(f"**Thought:** {resp['thought']}")
                st.code(resp["expr"], language="text")
            st.rerun()


# ---- Run tab -------------------------------------------------------------

with tab_r:
    st.subheader("Launch an evolution experiment")
    st.caption("Use canonical + your baselines as seed population, evolve via LLM-A/LLM-S.")
    cols = st.columns(3)
    with cols[0]:
        scen = st.selectbox("Scenario", ["S0", "S1", "S2"], index=1)
        variant = st.selectbox("Variant", ["P1", "P2", "P3"], index=2)
    with cols[1]:
        flexibility = st.slider("Routing flexibility", 0.0, 1.0, 0.5, 0.1)
        iterations = st.slider("Evolution iterations", 1, 20, 5)
        replications = st.slider("Replications per evaluation", 5, 100, 20)
    with cols[2]:
        jobs = st.number_input("Jobs", 4, 50, 12)
        machines = st.number_input("Machines", 2, 20, 6)
        if scen == "S1":
            pdr = st.slider("S1 part-delay-ratio", 0.0, 0.5, 0.2, 0.05)
            pdk = st.slider("S1 part-delay-k", 0.3, 3.0, 1.0, 0.1)
            udr = 0.5
        elif scen == "S2":
            udr = st.slider("S2 urgent-due-ratio", 0.2, 1.5, 0.3, 0.1)
            pdr, pdk = 0.2, 1.0
        else:
            pdr, pdk, udr = 0.2, 1.0, 0.5

    if st.button("Launch experiment", type="primary"):
        resp = api_post(
            "/api/experiments",
            user=user, scenario=scen, variant=variant,
            iterations=int(iterations), replications=int(replications),
            jobs=int(jobs), machines=int(machines), flexibility=float(flexibility),
            part_delay_ratio=float(pdr), part_delay_k=float(pdk),
            urgent_due_ratio=float(udr),
        )
        st.success(f"Launched experiment `{resp['experiment_id']}` (pid {resp['pid']})")
        st.session_state["watch_exp"] = resp["experiment_id"]
        time.sleep(0.5)
        st.rerun()


# ---- History tab --------------------------------------------------------

with tab_h:
    st.subheader("Experiment history")
    exps = api_get("/api/experiments", user=user)["experiments"]
    if not exps:
        st.caption("_No experiments launched yet._")
    else:
        cols = st.columns([1, 1, 1, 2])
        cols[0].markdown("**ID**"); cols[1].markdown("**Status**")
        cols[2].markdown("**Elapsed**"); cols[3].markdown("**Config**")
        for e in exps:
            cols = st.columns([1, 1, 1, 2])
            cols[0].code(e["id"])
            cols[1].write(e["status"])
            cols[2].write(f"{time.time() - e['started']:.0f}s")
            cfg = e["config"]
            cols[3].write(f"{cfg['scenario']}/{cfg['variant']} flex={cfg['flexibility']} iter={cfg['iterations']}")

        watch_id = st.selectbox("Inspect", [e["id"] for e in exps],
                                 index=0 if "watch_exp" not in st.session_state else
                                 max(0, [e["id"] for e in exps].index(st.session_state.get("watch_exp"))
                                     if st.session_state.get("watch_exp") in [e["id"] for e in exps] else 0))
        if watch_id:
            detail = api_get(f"/api/experiments/{watch_id}", user=user)
            st.markdown(f"### {watch_id} — {detail['status']}")
            if detail.get("best"):
                b = detail["best"]
                cols = st.columns(3)
                cols[0].metric("AT (best)", f"{b.get('primary_objective', 0):.1f}")
                cols[1].metric("gap vs FIFO", f"{b.get('gap_ratio', 0):+.1f}%" if b.get("gap_ratio") is not None else "—")
                cols[2].metric("feasibility", f"{b.get('feasible_job_ratio', 0):.2f}")
                st.markdown("**Best expression:**")
                st.code(b["expr"], language="text")
                if b.get("convergence"):
                    import altair as alt
                    import pandas as pd
                    df = pd.DataFrame({"iter": range(1, len(b["convergence"]) + 1),
                                       "best AT": b["convergence"]})
                    chart = alt.Chart(df).mark_line(point=True).encode(
                        x="iter:O", y=alt.Y("best AT:Q", scale=alt.Scale(zero=False)),
                    ).properties(height=240)
                    st.altair_chart(chart, use_container_width=True)

            with st.expander("Log (last 80 lines)"):
                st.code(detail.get("log_tail", "(empty)"))

            if detail["status"] == "running":
                if st.button("Refresh"):
                    st.rerun()
