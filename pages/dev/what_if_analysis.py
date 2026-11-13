import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_state as S
from src import config as C
from src import ui_components as U
from src import what_if_engine as W

p = S.setup_page(need_ai=True)
name, df, _ = S.dataset()
thr = S.thresholds()
U.page_header("What-if Analysis", "Explore how the MODEL's estimate responds to hypothetical edits of the engine's latest sensor readings. "
              "Observed data are never changed.", "SCENARIO EXPLORER")
U.note(W.SCENARIO_WARNING, warn=True)

ids = sorted(df[C.ID_COL].unique())
c1, c2 = st.columns([1, 3])
eng = c1.selectbox("Engine", ids, index=ids.index(st.session_state.get("selected_engine")) if st.session_state.get("selected_engine") in ids else 0,
                   key="wi_eng")
g = df[df[C.ID_COL] == eng].sort_values(C.CYCLE_COL)
last = g.iloc[-1]
bounds = W.sensor_bounds(p)
chosen = c2.multiselect("Sensors to modify (only these will change)", p.sensors, format_func=C.sensor_label, key="wi_sens",
                        default=[p.sensors[2]] if p.sensors else [])
with st.expander("Baseline sensor values at the last observed cycle", expanded=False):
    base = pd.DataFrame({"Sensor": [C.sensor_label(s) for s in p.sensors], "Baseline value": [float(last[s]) for s in p.sensors],
                         "Training min": [bounds[s]["min"] for s in p.sensors], "Training max": [bounds[s]["max"] for s in p.sensors]})
    st.dataframe(base, hide_index=True, width="stretch")

changes = {}
cols = st.columns(3)
for i, s in enumerate(chosen):
    b = bounds[s]
    v0 = float(last[s])
    step = (b["max"] - b["min"]) / 200 or 0.01
    with cols[i % 3]:
        st.markdown(f"**{C.sensor_label(s)}** · baseline {v0:.5g}")
        # number input has NO silent clamping: invalid values are reported by the validator, not auto-fixed
        changes[s] = st.number_input("Hypothetical value", value=v0, step=float(step), format="%.5f", key=f"wi_val_{eng}_{s}",
                                     help=f"Training range [{b['min']:.5g}, {b['max']:.5g}]")
        pct = 100 * (changes[s] - v0) / (b["max"] - b["min"]) if b["max"] > b["min"] else 0
        st.caption(f"change: {changes[s]-v0:+.5g} ({pct:+.1f}% of training range)")

if chosen:
    prev = pd.DataFrame([{"Sensor": C.sensor_label(s), "Baseline": float(last[s]), "Hypothetical": v, "Change": v - float(last[s])}
                         for s, v in changes.items()])
    st.markdown("**Preview of modified values**")
    st.dataframe(prev, hide_index=True, width="stretch")
run = st.button("Analyze scenario", type="primary", disabled=not chosen)
if run:
    r = W.run_scenario(p, g, changes, thr)
    r["label"] = f"Engine {eng} · " + ", ".join(f"{C.sensor_label(s)}={v:.4g}" for s, v in changes.items())
    if not r["errors"]:
        st.session_state.setdefault("scenarios", []).append(r)
    st.session_state["last_scenario"] = r

r = st.session_state.get("last_scenario")
if r:
    if r["errors"]:
        for e in r["errors"]:
            st.error(e)
    else:
        for w in r["warnings"][:-1]:
            U.note(w, warn=True)
        k = st.columns(4)
        k[0].markdown(U.kpi("Baseline RUL", f"{r['baseline_rul']:.1f}", "cycles", cls=r["baseline_state"]), unsafe_allow_html=True)
        k[1].markdown(U.kpi("Scenario RUL", f"{r['scenario_rul']:.1f}", "cycles", cls=r["scenario_state"]), unsafe_allow_html=True)
        k[2].markdown(U.kpi("Absolute change", f"{r['delta']:+.2f}", "cycles", "scenario − baseline"), unsafe_allow_html=True)
        k[3].markdown(f"<div class='kpi'><div class='l'>State: baseline → scenario</div><div class='v' style='font-size:1rem;padding-top:8px'>"
                      f"{U.badge(r['baseline_state'])} → {U.badge(r['scenario_state'])}</div></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = go.Figure(go.Bar(x=["Baseline", "Scenario"], y=[r["baseline_rul"], r["scenario_rul"]],
                                   marker_color=[C.STATE_COLORS[r["baseline_state"]], C.STATE_COLORS[r["scenario_state"]]],
                                   text=[f"{r['baseline_rul']:.1f}", f"{r['scenario_rul']:.1f}"], textposition="outside", width=.45,
                                   hovertemplate="%{x}: %{y:.1f} cycles<extra></extra>"))
            U.add_thresholds(fig, thr)
            fig.update_yaxes(title="Estimated RUL (cycles)", range=[0, p.cap * 1.12])
            st.plotly_chart(U.style_fig(fig, f"Engine {r['engine_id']} at cycle {r['cycle']}: baseline vs. hypothetical", 360, False), width="stretch")
        with c2:
            st.markdown("**Values changed**")
            st.dataframe(r["changed"], hide_index=True, width="stretch")
            if r["scenario_at_cap"] or r["baseline_at_cap"]:
                U.note(f"One estimate sits at the {p.cap:.0f}-cycle cap; differences near the cap are compressed.", warn=True)
            U.note(r["warnings"][-1], warn=True)
sc = st.session_state.get("scenarios", [])
if sc:
    rows = [{"scenario": x["label"], "engine": x["engine_id"], "cycle": x["cycle"], "baseline_rul": x["baseline_rul"],
             "scenario_rul": x["scenario_rul"], "delta_cycles": x["delta"], "baseline_state": x["baseline_state"],
             "scenario_state": x["scenario_state"]} for x in sc]
    st.markdown("##### Scenario log (this session)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button("Download scenario comparisons (CSV)", pd.DataFrame(rows).to_csv(index=False).encode(), "whatif_scenarios.csv", "text/csv")
