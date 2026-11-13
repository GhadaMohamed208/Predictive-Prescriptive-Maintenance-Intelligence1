import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analysis_service as AS
from src import anomaly_detection as AD
from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import generic_analysis as GA
from src import ui_components as U

p = S.setup_page()
rec = S.active()
df, sensors = rec["df"], rec["sensors"]
ai = rec["capability"] == DS.CAP_AI
thr, sigma = S.thresholds(), float(st.session_state["sigma_trigger"])
lab = lambda s: DS.sensor_label(rec, s)
U.page_header("Sensor & Degradation Analysis", f"All analyses use the active dataset: {rec['source_file'] or rec['label']} "
              f"({DS.source_text(rec)}, historical data).", "DEGRADATION ANALYSIS")
tab_o, tab_d, tab_a = st.tabs(["Dataset overview", "Asset degradation", "Unusual behaviour & drift"])

with tab_o:
    lens = df.groupby(C.ID_COL)[C.CYCLE_COL].count()
    k = st.columns(4)
    k[0].markdown(U.kpi("Assets", str(df[C.ID_COL].nunique())), unsafe_allow_html=True)
    k[1].markdown(U.kpi("Records", f"{len(df):,}"), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Records per asset", f"{int(lens.min())}–{int(lens.max())}", note=f"median {lens.median():.0f}"), unsafe_allow_html=True)
    k[3].markdown(U.kpi("Sensors analysed", str(len(sensors))), unsafe_allow_html=True)
    st.write("")
    summ = pd.DataFrame({"Sensor": [lab(s) for s in sensors], "Min": [df[s].min() for s in sensors], "Mean": [df[s].mean() for s in sensors],
                         "Max": [df[s].max() for s in sensors], "Std. dev.": [df[s].std() for s in sensors],
                         "Missing": [int(df[s].isna().sum()) for s in sensors]})
    st.dataframe(summ, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.4g") for c in ("Min", "Mean", "Max", "Std. dev.")})
    c1, c2 = st.columns(2)
    sd = c1.selectbox("Sensor distribution", sensors, format_func=lab, key="eda_sd")
    with c1:
        fig = go.Figure(go.Histogram(x=df[sd], nbinsx=40, marker_color=U.SERIES[0], marker_line=dict(color="#0d1626", width=1)))
        fig.update_xaxes(title=lab(sd)); fig.update_yaxes(title="Records")
        st.plotly_chart(U.style_fig(fig, f"{lab(sd)} · distribution in the active dataset", 320, False), width="stretch")
    with c2:
        if len(sensors) >= 2:
            corr = df[sensors].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=[lab(s) for s in corr.columns], y=[lab(s) for s in corr.index], zmin=-1, zmax=1,
                                       colorscale=[[0, "#d95926"], [.5, "#1a2234"], [1, "#3987e5"]], colorbar=dict(title="Correlation")))
            st.plotly_chart(U.style_fig(fig, "Sensor correlation in the active dataset", 320, False), width="stretch")
        else:
            U.note("Correlation view unavailable: it needs at least two measurement columns.")

ids = sorted(df[C.ID_COL].unique().tolist())
cur = st.session_state.get("selected_asset")
with tab_d:
    c1, c2 = st.columns([1, 3])
    e = c1.selectbox("Asset", ids, index=ids.index(cur) if cur in ids else 0, format_func=lambda u: DS.asset_name(rec, u), key="deg_asset")
    st.session_state["selected_asset"] = e
    g = df[df[C.ID_COL] == e].sort_values(C.CYCLE_COL).reset_index(drop=True)
    sel = c2.multiselect("Sensors", sensors, default=sensors[:2], format_func=lab, key="deg_s")
    if len(g) < GA.MIN_RECORDS:
        U.note(f"Degradation trends are unavailable for this asset: it has {len(g)} records, at least {GA.MIN_RECORDS} are needed to "
               "compare recent behaviour with the asset's own early baseline.", warn=True)
    elif not sel:
        U.note("Select at least one sensor.")
    else:
        k = GA.window(len(g))
        fig = go.Figure()
        for i, s in enumerate(sel):
            col = U.SERIES[i % 8]
            base = g[s].iloc[:k].mean()
            fig.add_trace(go.Scatter(x=g[C.CYCLE_COL], y=g[s] - base, name=f"{lab(s)} change", mode="lines", line=dict(color=col, width=1), opacity=.5))
            fig.add_trace(go.Scatter(x=g[C.CYCLE_COL], y=(g[s] - base).rolling(10, min_periods=1).mean(), name=f"{lab(s)} (10-record average)",
                                     mode="lines", line=dict(color=col, width=2.5)))
        fig.add_hline(y=0, line=dict(color="#8da0bd", width=1))
        fig.update_xaxes(title="Cycle / record"); fig.update_yaxes(title="Change from baseline")
        st.plotly_chart(U.style_fig(fig, f"{DS.asset_name(rec, e)}: change from baseline (first {k} records)", 380), width="stretch")
        s1 = st.selectbox("Compare one sensor with the other assets of this dataset", sensors, format_func=lab, key="deg_ref")
        if len(ids) >= 3:
            d2 = df.copy()
            d2["pos"] = d2.groupby(C.ID_COL).cumcount() + 1
            q = d2.groupby("pos")[s1].quantile([.1, .5, .9]).unstack()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q.index, y=q[.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=q.index, y=q[.1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(57,135,229,.2)", name="Other assets 10–90%"))
            fig.add_trace(go.Scatter(x=q.index, y=q[.5], mode="lines", name="Dataset median", line=dict(color=U.SERIES[0], width=2)))
            fig.add_trace(go.Scatter(x=np.arange(1, len(g) + 1), y=g[s1], mode="lines", name=DS.asset_name(rec, e), line=dict(color=U.SERIES[1], width=2)))
            fig.update_xaxes(title="Record number (from the start of each asset's history)"); fig.update_yaxes(title=lab(s1))
            st.plotly_chart(U.style_fig(fig, f"{lab(s1)}: selected asset vs. the other assets in the active dataset", 340), width="stretch")
        else:
            U.note("Peer comparison unavailable: it needs at least 3 assets in the dataset.")
    if ai:
        st.markdown("##### Estimated remaining life")
        h = p.predict_rows(g)
        res = p.predict_latest(g, thr)[0].iloc[0]
        a, b = st.columns([1, 3])
        a.markdown(U.kpi("Estimated remaining life", f"{res['rul_pred']:.0f}", "cycles", "model estimate, not a failure date", res["state"]), unsafe_allow_html=True)
        a.markdown(U.badge(res["state"]), unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(x=h[C.CYCLE_COL], y=h["rul_pred"], mode="lines", line=dict(color=U.SERIES[0], width=2.2)))
        U.add_thresholds(fig, thr)
        fig.update_xaxes(title="Cycle"); fig.update_yaxes(title="Estimated remaining life (cycles)", range=[0, p.cap * 1.05])
        with b:
            st.plotly_chart(U.style_fig(fig, "Estimated remaining life over time (each point uses only data up to that cycle)", 300, False), width="stretch")
    else:
        U.note("Remaining-life estimation is unavailable for this dataset: the implemented model supports only NASA C-MAPSS-compatible "
               "turbofan engine data. Trends and drift analyses above are available.")

with tab_a:
    e2 = st.session_state.get("selected_asset", ids[0])
    e2 = st.selectbox("Asset", ids, index=ids.index(e2) if e2 in ids else 0, format_func=lambda u: DS.asset_name(rec, u), key="deg_asset2")
    st.session_state["selected_asset"] = e2
    g2 = df[df[C.ID_COL] == e2].sort_values(C.CYCLE_COL)
    if len(g2) < GA.MIN_RECORDS:
        U.note(f"Unavailable: this asset has {len(g2)} records; at least {GA.MIN_RECORDS} are needed.", warn=True)
        st.stop()
    if ai:
        ref = S.get_anomaly_ref()
        sc = AD.score(ref, g2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc[C.CYCLE_COL], y=sc["anomaly_score"], mode="lines", name="Unusual-behaviour score", line=dict(color=U.SERIES[0], width=2)))
        fl = sc[sc["anomaly_flag"]]
        fig.add_trace(go.Scatter(x=fl[C.CYCLE_COL], y=fl["anomaly_score"], mode="markers", name="Flagged as unusual", marker=dict(color="#ef4444", size=7, symbol="diamond")))
        fig.add_hline(y=ref["threshold"], line=dict(color="#f59e0b", dash="dash"), annotation_text="flag level")
        fig.update_xaxes(title="Cycle"); fig.update_yaxes(title="Unusual-behaviour score (higher = more unusual)")
        st.plotly_chart(U.style_fig(fig, f"{DS.asset_name(rec, e2)}: unusual behaviour compared with the early-life pattern", 340), width="stretch")
        sm = AD.engine_summary(sc)
        k = st.columns(3)
        k[0].markdown(U.kpi("Recent records flagged", f"{sm['recent_flag_fraction']*100:.0f}%", note=f"last {sm['recent_n']} records"), unsafe_allow_html=True)
        k[1].markdown(U.kpi("First flagged at cycle", str(sm["first_flag_cycle"] or "—")), unsafe_allow_html=True)
        k[2].markdown(U.kpi("Total flagged records", str(sm["n_flagged_total"])), unsafe_allow_html=True)
        sh = AD.sensor_shifts(g2, ref)
    else:
        sh = GA.shift_table(g2, sensors, GA.noise_reference(df, sensors), rec.get("sensor_labels"))
    st.markdown("##### Sensor drift vs. the asset's own early readings")
    t = sh.head(10).copy()
    t["Direction"] = np.where(t["shift_sigma"] > 0, "increased", "decreased")
    t["Flag"] = np.where(t["abs_sigma"] >= sigma, "Notable drift", "Within normal range")
    t["label"] = t["sensor"].map(lab)
    st.dataframe(t[["label", "baseline_mean", "recent_mean", "shift_sigma", "Direction", "Flag"]], hide_index=True, width="stretch", column_config={
        "label": "Sensor", "baseline_mean": st.column_config.NumberColumn("Early average", format="%.4g"),
        "recent_mean": st.column_config.NumberColumn("Recent average", format="%.4g"),
        "shift_sigma": st.column_config.NumberColumn("Change (x normal noise)", format="%+.1f")})
    st.caption(f"'Notable drift' = change larger than {sigma:.0f} times the sensor's normal early noise (adjustable in Settings). "
               "A drift or an unusual-behaviour flag is an observation: it does not prove a physical fault.")
