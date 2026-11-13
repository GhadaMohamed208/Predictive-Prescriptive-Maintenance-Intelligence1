import plotly.graph_objects as go
import streamlit as st

from src import analysis_service as AS
from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import ui_components as U
from src.dashboard_panels import registry_panel

p = S.setup_page()
rec = S.active()
df = rec["df"]
V = S.view(rec)
tbl = V["table"]
ai = rec["capability"] == DS.CAP_AI
U.page_header("Fleet Overview", f"{rec['source_file'] or rec['label']} · {DS.source_text(rec)} · historical data. "
              "Condition is evaluated at each asset's last record.", "FLEET")
order, colr = AS.status_order(rec), AS.status_colors(rec)
cnt = AS.status_counts(rec, tbl)
k = st.columns(len(order) + 1)
k[0].markdown(U.kpi("Assets", str(len(tbl))), unsafe_allow_html=True)
for c, s in zip(k[1:], order):
    c.markdown(U.kpi(s, str(cnt[s]), cls=s if s in C.STATES else ""), unsafe_allow_html=True)
st.write("")

f1, f2, f3, f4 = st.columns([1.6, 1.4, 1.2, 1.2])
states = f1.multiselect("Condition", order, default=order, key="fleet_states")
prios = f2.multiselect("Priority", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"], key="fleet_prio")
q = f3.text_input("Search asset", "", key="fleet_q")
flt = tbl[tbl["state"].isin(states) & tbl["priority"].isin(prios)]
if q.strip():
    flt = flt[flt["name"].astype(str).str.contains(q.strip(), case=False, regex=False)]
if ai:
    rmax = float(p.cap)
    rr = f4.slider("Remaining life (cycles)", 0.0, rmax, (0.0, rmax), key="fleet_rul")
    flt = flt[flt["rul_pred"].between(*rr)]

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Bar(x=order, y=[cnt[s] for s in order], marker_color=[colr[s] for s in order],
                           text=[cnt[s] for s in order], textposition="outside", width=.5))
    fig.update_yaxes(title="Assets")
    st.plotly_chart(U.style_fig(fig, "Condition distribution", 310, False), width="stretch")
with c2:
    pc = tbl["priority"].value_counts()
    pr = ["Critical", "High", "Medium", "Low"]
    fig = go.Figure(go.Bar(x=pr, y=[int(pc.get(x, 0)) for x in pr], marker_color=[U.PRIORITY_COLORS[x] for x in pr],
                           text=[int(pc.get(x, 0)) for x in pr], textposition="outside", width=.5))
    fig.update_yaxes(title="Assets")
    st.plotly_chart(U.style_fig(fig, "Maintenance priority distribution", 310, False), width="stretch")

st.markdown(f"##### Assets ({len(flt)} of {len(tbl)} shown)")
cols = ["attention_rank", "name", "last_cycle", "n_cycles", "state", "priority", "n_findings"] + (["rul_pred"] if ai else ["max_shift_sigma", "top_sensor"])
if ai and "rul_true_official" in flt:
    cols.append("rul_true_official")
st.dataframe(flt[cols], hide_index=True, width="stretch", height=360, column_config={
    "attention_rank": "Rank", "name": "Asset", "last_cycle": "Last cycle", "n_cycles": "Records", "state": "Condition",
    "priority": "Priority", "n_findings": "Open findings", "top_sensor": "Largest drift sensor",
    "max_shift_sigma": st.column_config.NumberColumn("Largest drift (x normal noise)", format="%.1f"),
    "rul_pred": st.column_config.ProgressColumn("Estimated remaining life", min_value=0, max_value=float(p.cap), format="%.0f"),
    "rul_true_official": st.column_config.NumberColumn("Reference RUL (provided with data)", format="%.0f")})
st.download_button("Export shown assets (CSV)", flt[cols].to_csv(index=False).encode(), "fleet_assets.csv", "text/csv")

st.markdown("##### Sensor and degradation trends")
sel = st.multiselect("Compare assets (up to 6)", flt[C.ID_COL].tolist(), default=flt[C.ID_COL].head(3).tolist(),
                     format_func=lambda u: DS.asset_name(rec, u), max_selections=6, key="fleet_sel")
if sel:
    sensors = rec["sensors"]
    sens = st.selectbox("Sensor", sensors, format_func=lambda s: DS.sensor_label(rec, s), key="fleet_sensor")
    fig = go.Figure()
    for i, e in enumerate(sel):
        g = df[df[C.ID_COL] == e].sort_values(C.CYCLE_COL)
        fig.add_trace(go.Scatter(x=g[C.CYCLE_COL], y=g[sens], mode="lines", name=DS.asset_name(rec, e),
                                 line=dict(color=U.SERIES[i % 8], width=1.6)))
    fig.update_xaxes(title="Cycle / record"); fig.update_yaxes(title=DS.sensor_label(rec, sens))
    st.plotly_chart(U.style_fig(fig, f"{DS.sensor_label(rec, sens)} · selected assets", 380), width="stretch")
    if ai:
        fig = go.Figure()
        for i, e in enumerate(sel):
            h = p.predict_rows(df[df[C.ID_COL] == e])
            fig.add_trace(go.Scatter(x=h[C.CYCLE_COL], y=h["rul_pred"], mode="lines", name=DS.asset_name(rec, e),
                                     line=dict(color=U.SERIES[i % 8], width=2)))
        U.add_thresholds(fig, S.thresholds())
        fig.update_xaxes(title="Cycle"); fig.update_yaxes(title="Estimated remaining life (cycles)")
        st.plotly_chart(U.style_fig(fig, "Estimated remaining life over time (each point uses only data up to that cycle)", 380), width="stretch")
    else:
        U.note("Remaining-life curves are not available for this dataset (no supported remaining-life model).")
c1, c2 = st.columns([3, 1])
pick = c1.selectbox("Open asset", flt[C.ID_COL].tolist() or tbl[C.ID_COL].tolist(), format_func=lambda u: DS.asset_name(rec, u), key="fleet_pick")
if c2.button("Open asset", type="primary"):
    S.goto_engine(pick)

st.write("")
registry_panel()
