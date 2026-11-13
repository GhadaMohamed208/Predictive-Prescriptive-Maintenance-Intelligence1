import plotly.graph_objects as go
import streamlit as st

from src import analysis_service as AS
from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import ui_components as U
from src.dashboard_panels import registry_panel

p = S.setup_page(need_dataset=False)
rec = S.active()
if rec is None:
    stats = [("Dataset", "None loaded"), ("Built-in demo", "NASA C-MAPSS"), ("AI-supported equipment", "Turbofan engines"),
             ("Other data", "Sensor analysis")]
    st.markdown(U.hero_html(stats), unsafe_allow_html=True)
    st.write("")
    c1, c2, _ = st.columns([1.4, 1.4, 3])
    if c1.button("Load built-in NASA demo dataset", type="primary", width="stretch"):
        S.activate_builtin()
        st.rerun()
    if c2.button("Upload your own data", width="stretch"):
        st.switch_page("pages/data_source.py")
    st.caption("No data is shown until you choose a source. The turbofan graphic is decorative, not live telemetry.")
    st.write("")
    registry_panel()
    st.stop()

V = S.view(rec)
tbl, F = V["table"], V["findings"]
rs = st.session_state["rec_status"]
K = AS.kpis(rec, V, rs)
ai = rec["capability"] == DS.CAP_AI
U.page_header("Control Room", f"{rec['source_file'] or rec['label']} · {DS.source_text(rec)} · "
              "historical data (not live telemetry)", "CONTROL ROOM")
if not ai:
    U.note("This dataset is analysed with sensor statistics only: no remaining-life or health-class estimates are available for it, so "
           "those indicators are not shown.")
cols = st.columns(6)
cols[0].markdown(U.kpi("Monitored assets", str(K["assets"]), note="in the active dataset"), unsafe_allow_html=True)
cols[1].markdown(U.kpi("Healthy" if ai else "Stable", str(K["healthy"]), cls="Healthy"), unsafe_allow_html=True)
cols[2].markdown(U.kpi("Needing attention", str(K["attention"]), cls="Warning"), unsafe_allow_html=True)
if ai:
    cols[3].markdown(U.kpi("Critical", str(K["critical"]), cls="Critical"), unsafe_allow_html=True)
else:
    cols[3].markdown(U.kpi("Critical", "n/a", note="needs a remaining-life model"), unsafe_allow_html=True)
cols[4].markdown(U.kpi("Active warnings", str(K["warnings"]), note="findings of medium or higher priority"), unsafe_allow_html=True)
cols[5].markdown(U.kpi("High-priority actions", str(K["high_actions"]), note="open, critical or high"), unsafe_allow_html=True)
st.write("")

left, right = st.columns([1.5, 1])
with left:
    st.markdown("<div class='panel'><h4>Assets needing attention first</h4></div>", unsafe_allow_html=True)
    top = tbl.head(10).copy()
    show = ["attention_rank", "name", "last_cycle", "state", "priority"] + (["rul_pred"] if ai else ["top_sensor"])
    cfg = {"attention_rank": st.column_config.NumberColumn("Rank", width="small"), "name": "Asset",
           "last_cycle": "Last cycle", "state": "Condition", "priority": "Priority", "top_sensor": "Largest sensor drift",
           "rul_pred": st.column_config.ProgressColumn("Estimated remaining life (cycles)", min_value=0, max_value=float(p.cap), format="%.0f")}
    st.dataframe(top[show], hide_index=True, width="stretch", column_config=cfg)
    c1, c2 = st.columns([1, 3])
    pick = c2.selectbox("Open asset", tbl[C.ID_COL].tolist(), format_func=lambda u: DS.asset_name(rec, u),
                        label_visibility="collapsed", key="home_pick")
    if c1.button("Open asset", type="primary"):
        S.goto_engine(pick)
with right:
    order = AS.status_order(rec)
    cnt = AS.status_counts(rec, tbl)
    colr = AS.status_colors(rec)
    fig = go.Figure(go.Bar(x=order, y=[cnt[s] for s in order], marker_color=[colr[s] for s in order],
                           text=[cnt[s] for s in order], textposition="outside", width=.5))
    fig.update_yaxes(title="Assets")
    st.plotly_chart(U.style_fig(fig, "Asset condition", 300, False), width="stretch")

l2, r2 = st.columns([1.5, 1])
with l2:
    st.markdown("<div class='panel'><h4>Latest findings (as of each asset's last record)</h4></div>", unsafe_allow_html=True)
    if len(F) and (F["priority"] != "Low").any():
        ev = F[~F["priority"].isin(["Low"])].head(8)[["priority", "name", "condition", "last_cycle"]]
        st.dataframe(ev, hide_index=True, width="stretch", column_config={
            "priority": "Priority", "name": "Asset", "condition": "Finding", "last_cycle": "At cycle"})
    else:
        U.note("No findings were raised for this dataset.")
with r2:
    pc = tbl["priority"].value_counts()
    pr = ["Critical", "High", "Medium", "Low"]
    fig = go.Figure(go.Bar(x=pr, y=[int(pc.get(x, 0)) for x in pr], marker_color=[U.PRIORITY_COLORS[x] for x in pr],
                           text=[int(pc.get(x, 0)) for x in pr], textposition="outside", width=.5))
    fig.update_yaxes(title="Assets")
    st.plotly_chart(U.style_fig(fig, "Assets by maintenance priority", 300, False), width="stretch")
if ai:
    h1, h2 = st.columns(2)
    h1.markdown(U.kpi("Median estimated remaining life", f"{tbl['rul_pred'].median():.0f}", "cycles",
                      "model estimate, not a failure date"), unsafe_allow_html=True)
    h2.markdown(U.kpi("Healthy share", f"{K['healthy']/max(K['assets'],1)*100:.0f}%", note="of monitored assets"), unsafe_allow_html=True)
else:
    h1, h2 = st.columns(2)
    h1.markdown(U.kpi("Assets with sensor drift", str(K["attention"])), unsafe_allow_html=True)
    h2.markdown(U.kpi("Sensors analysed", str(len(rec["sensors"]))), unsafe_allow_html=True)
st.write("")
registry_panel()
st.write("")
U.note("Conditions and priorities are decision-support categories computed from the active dataset with configurable settings; they are "
       "not industry standards. " + C.DISCLAIMER)
