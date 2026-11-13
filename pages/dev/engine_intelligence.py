import time

import plotly.graph_objects as go
import streamlit as st

from src import app_state as S
from src import config as C
from src import recommendation_engine as RE
from src import ui_components as U
from src.engine_analysis import analyze_engine

p = S.setup_page(need_ai=True)
name, df, rul_true = S.dataset()
thr = S.thresholds()
U.page_header("Engine Intelligence", "Select an engine, run the analysis and review the staged results: validation → health → RUL → "
              "sensors → explanation → actions.", "ENGINE INTELLIGENCE")

ids = sorted(df[C.ID_COL].unique().tolist())
sel_default = st.session_state.get("selected_engine")
c1, c2, c3 = st.columns([1.2, 2.2, 1.1])
eng = c1.selectbox("Engine", ids, index=ids.index(sel_default) if sel_default in ids else 0)
st.session_state["selected_engine"] = eng
g = df[df[C.ID_COL] == eng].sort_values(C.CYCLE_COL)
cmin, cmax = int(g[C.CYCLE_COL].min()), int(g[C.CYCLE_COL].max())
as_of = c2.slider("Analyse as of cycle (uses only data up to this cycle)", cmin, cmax, cmax) if cmax > cmin else cmax
run = c3.button("Analyze Engine", type="primary", width="stretch")
key = (name, int(eng), int(as_of), thr["warning"], thr["critical"], st.session_state["sigma_trigger"])

if run:
    with st.spinner("Computing model estimate, SHAP explanation, anomaly analysis and rules…"):
        res = analyze_engine(p, S.get_anomaly_ref(), S.get_explainer(), g[g[C.CYCLE_COL] <= as_of], thr,
                             st.session_state["sigma_trigger"])
    st.session_state["analysis"] = {"key": key, "res": res, "played": False}

A = st.session_state.get("analysis")
if A is None or A["key"] != key:
    U.note("Choose an engine and press 'Analyze Engine'. Results are computed from the saved model; nothing is pre-filled.")
    if A is not None and A["key"] != key:
        U.note("Selection or thresholds changed since the last analysis — press 'Analyze Engine' to refresh.", warn=True)
    st.stop()

res = A["res"]
critical = (res.get("state") == "Critical")
animate = (not A["played"]) and not critical  # critical status is never delayed
pause = (lambda s=.35: time.sleep(s)) if animate else (lambda s=.0: None)

# ---- Stage 1: validation
U.stage("Stage 1 · Input validation")
dq = res["dq"]
if not dq.ok:
    st.error("Validation failed: " + "; ".join(dq.errors))
    st.stop()
v1, v2, v3, v4 = st.columns(4)
v1.markdown(U.kpi("Engine", str(res["engine_id"]), note=name), unsafe_allow_html=True)
v2.markdown(U.kpi("Last cycle analysed", str(res["last_cycle"]), note=f"{res['n_cycles']} cycles of history"), unsafe_allow_html=True)
v3.markdown(U.kpi("Validation", dq.status.title(), note=f"{len(dq.warnings)} warning(s)"), unsafe_allow_html=True)
o = res["ood"]
v4.markdown(U.kpi("Out-of-range features", f"{o['n_outside']}" if o.get("available") else "n/a",
                  note="vs. training min/max" + (" · FLAG" if o.get("flag") else "")), unsafe_allow_html=True)
if dq.warnings:
    with st.expander("Validation warnings"):
        for w in dq.warnings:
            st.write("• " + w)
pause()

# ---- Stage 2: health
U.stage("Stage 2 · Health assessment")
state = res["state"]
why = {"Critical": f"predicted RUL {res['rul']:.1f} < critical threshold {thr['critical']:.0f}",
       "Warning": f"predicted RUL {res['rul']:.1f} within [{thr['critical']:.0f}, {thr['warning']:.0f}]",
       "Healthy": f"predicted RUL {res['rul']:.1f} > warning threshold {thr['warning']:.0f}"}[state]
st.markdown(f"<div class='banner {state} {'reveal' if animate else ''}'>{U.badge(state)} &nbsp; Derived state because {why} cycles. "
            f"<span style='color:#8da0bd'>(Decision-support category, configurable in Settings.)</span></div>", unsafe_allow_html=True)
pause()

# ---- Stage 3: RUL
U.stage("Stage 3 · RUL prediction")
r1, r2 = st.columns([1, 2.4])
with r1:
    U.count_up(res["rul"], "Estimated remaining useful life", C.STATE_COLORS[state], animate=animate)
    if res["at_cap"]:
        U.note(f"Estimate is at the {p.cap:.0f}-cycle modelling cap: true RUL may be higher.", warn=True)
    if rul_true is not None and name.startswith("Official") and int(as_of) == cmax:
        st.caption(f"Official ground-truth RUL (evaluation only): **{rul_true.loc[eng]:.0f}** cycles "
                   f"(capped: {min(rul_true.loc[eng], p.cap):.0f}).")
    else:
        st.caption("No ground-truth RUL is available for this selection; the estimate cannot be directly verified.")
with r2:
    h = res["rul_history"]
    fig = go.Figure(go.Scatter(x=h["cycle"], y=h["rul_pred"], mode="lines", line=dict(color=U.SERIES[0], width=2.2),
                               name="Model RUL estimate", hovertemplate="cycle %{x}<br>RUL %{y:.1f}<extra></extra>"))
    U.add_thresholds(fig, thr)
    fig.update_xaxes(title="Operating cycle"); fig.update_yaxes(title="Estimated RUL (cycles)", range=[0, p.cap * 1.05])
    st.plotly_chart(U.style_fig(fig, "Estimated RUL history (each point uses only data up to that cycle)", 300, False), width="stretch")
pause()

# ---- Stage 4: sensors
U.stage("Stage 4 · Sensor intelligence")
d = res["engine_df"]
shifts = res["shifts"]
sl, sr = st.columns([1, 3])
with sl:
    default_s = [s["sensor"] for s in shifts[:2]] or p.sensors[:2]
    sensors = st.multiselect("Sensors", p.sensors, default=default_s, format_func=C.sensor_label, key="ei_sensors")
    cyc = st.slider("Cycle range", int(d[C.CYCLE_COL].min()), int(d[C.CYCLE_COL].max()),
                    (int(d[C.CYCLE_COL].min()), int(d[C.CYCLE_COL].max())), key="ei_cycles") if len(d) > 1 else (1, 1)
    show_roll = st.checkbox("Show 10-cycle rolling mean", True)
with sr:
    dd = d[d[C.CYCLE_COL].between(*cyc)]
    fig = go.Figure()
    for i, s in enumerate(sensors):
        col = U.SERIES[i % 8]
        fig.add_trace(go.Scatter(x=dd[C.CYCLE_COL], y=dd[s], mode="lines", name=C.sensor_label(s),
                                 line=dict(color=col, width=1), opacity=.55,
                                 hovertemplate=f"{C.sensor_label(s)}<br>cycle %{{x}}<br>%{{y:.4g}}<extra></extra>"))
        if show_roll:
            fig.add_trace(go.Scatter(x=dd[C.CYCLE_COL], y=dd[s].rolling(10, min_periods=1).mean(), mode="lines",
                                     name=f"{C.sensor_label(s)} (10-cycle mean)", line=dict(color=col, width=2.4)))
    fig.update_xaxes(title="Operating cycle"); fig.update_yaxes(title="Sensor value")
    st.plotly_chart(U.style_fig(fig, "Sensor trajectories (separate scales: select few sensors for readability)", 340), width="stretch")
if shifts:
    with st.expander("Sensors ranked by shift vs. the engine's early baseline"):
        import pandas as pd
        t = pd.DataFrame(shifts)[["label", "baseline_mean", "recent_mean", "shift_sigma", "trend_rho"]]
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "label": "Sensor", "baseline_mean": "Baseline mean", "recent_mean": "Recent mean (last 10)",
            "shift_sigma": st.column_config.NumberColumn("Shift (σ of early noise)", format="%+.1f"),
            "trend_rho": st.column_config.NumberColumn("Trend ρ (Spearman)", format="%+.2f")})
a = res.get("anomaly")
if a:
    st.caption(f"Anomaly detector (separate from the RUL model): {a['recent_flag_fraction']*100:.0f}% of the last {a['recent_n']} cycles flagged; "
               "an anomaly is unusual behaviour relative to early-life reference, not a confirmed failure.")
pause()

# ---- Stage 5: explainability
U.stage("Stage 5 · Explainability (SHAP)")
loc = res["shap_local"]
if loc:
    t = loc["by_sensor"].head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["shap"], y=t["label"], orientation="h",
                           marker_color=[U.SERIES[1] if v < 0 else U.SERIES[0] for v in t["shap"]],
                           hovertemplate="%{y}: %{x:+.2f} cycles<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#8da0bd", width=1))
    fig.update_xaxes(title="SHAP contribution to RUL estimate (cycles)")
    st.plotly_chart(U.style_fig(fig, "Local explanation — top sensors (blue raises, orange lowers the estimate)", 360, False), width="stretch")
    st.markdown(res["shap_text"])
    with st.expander("Strongest individual engineered features"):
        st.dataframe(loc["table"].head(10)[["label", "value", "shap"]], hide_index=True, width="stretch",
                     column_config={"label": "Feature", "value": st.column_config.NumberColumn("Value", format="%.4g"),
                                    "shap": st.column_config.NumberColumn("SHAP (cycles)", format="%+.2f")})
    st.caption("Explanation computed for the latest analysed cycle only. SHAP describes model attribution, not physical causation.")
else:
    U.note("SHAP explanation unavailable: " + res["shap_reason"], warn=True)
pause()

# ---- Stage 6: recommendations
U.stage("Stage 6 · Maintenance actions")
for r in res["recs"]:
    status = st.session_state["rec_status"].get(r["id"], r["status"])
    icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "Info": "🔵"}[r["priority"]]
    with st.expander(f"{icon} {r['priority']} · {r['title']}  —  [{status}]", expanded=(r["priority"] in ("Critical",))):
        st.markdown(f"**Recommended action:** {r['action']}")
        st.markdown(f"**Trigger:** {r['trigger']}  \n**Evidence:** {r['evidence']}")
        st.markdown(f"**Evidence strength:** {r['evidence_strength']} — {RE.EVIDENCE_DEFINITIONS[r['evidence_strength']]}  \n"
                    f"**Finding type:** {r['finding_type']}  \n**Source:** {r['source']}")
        st.markdown(f"**Limitations:** {r['limitation']}  \n**Suggested verification:** {r['verification']}")
        new = st.selectbox("Action status", RE.STATUSES, index=RE.STATUSES.index(status), key=f"st_{r['id']}")
        st.session_state["rec_status"][r["id"]] = new
st.markdown("<div class='stage'>Analysis complete</div>", unsafe_allow_html=True)
b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Re-run analysis"):
    st.session_state["analysis"]["played"] = False
    st.rerun()
if b2.button("Reset"):
    st.session_state.pop("analysis", None)
    st.rerun()
A["played"] = True
