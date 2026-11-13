import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_state as S
from src import config as C
from src import explainability as EX
from src import ui_components as U

p = S.setup_page(need_ai=True)
name, df, _ = S.dataset()
thr = S.thresholds()
U.page_header("Explainable AI", "SHAP attributions for the selected model: which engineered features pushed the RUL estimate up or down. "
              "Attribution is statistical, not a physical diagnosis.", "EXPLAINABILITY")
g = S.get_global_shap()
expl = S.get_explainer()
if g is None or expl is None:
    U.note("SHAP is unavailable for the selected model type (the app never fabricates explanations). "
           "Retrain with a tree/linear model supported by SHAP.", warn=True)
    st.stop()

tab_g, tab_l = st.tabs(["Global importance & summary plot", "Local explanation"])
with tab_g:
    c1, c2 = st.columns([1, 1.4])
    mode = c1.radio("Aggregate by", ["Sensor (sum over its features)", "Individual engineered feature"], horizontal=True)
    by_sensor = mode.startswith("Sensor")
    imp = EX.global_importance(g, by_sensor=by_sensor, window=p.cfg.window)
    top = imp.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["label"], orientation="h", marker_color=U.SERIES[0],
                           hovertemplate="%{y}<br>mean |SHAP| = %{x:.2f} cycles<extra></extra>"))
    fig.update_xaxes(title="Mean |SHAP| (cycles)")
    st.plotly_chart(U.style_fig(fig, "Global feature importance", 460, False), width="stretch")
    st.caption(f"Computed on a documented random sample of {g['n']:,} training rows (seed {g['seed']}) — {g['kind']} explainer.")

    # SHAP summary (beeswarm-style) for the top features
    vals, X = g["values"], g["X"]
    order = np.argsort(-np.abs(vals).mean(axis=0))[:12][::-1]
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for rank, j in enumerate(order):
        f = X.columns[j]
        xv = X[f].to_numpy()
        lo, hi = np.percentile(xv, [2, 98])
        cn = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
        fig.add_trace(go.Scattergl(x=vals[:, j], y=rank + rng.uniform(-.28, .28, len(xv)), mode="markers",
                                   marker=dict(size=4, color=cn, colorscale=[[0, "#3987e5"], [.5, "#8c7fb8"], [1, "#e66767"]],
                                               cmin=0, cmax=1, showscale=(rank == 0),
                                               colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["low", "high"])),
                                   customdata=np.c_[xv], showlegend=False,
                                   hovertemplate=f"{EX.feature_label(f, p.cfg.window)}<br>value %{{customdata[0]:.4g}}<br>SHAP %{{x:+.2f}}<extra></extra>"))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[EX.feature_label(X.columns[j], p.cfg.window) for j in order])
    fig.update_xaxes(title="SHAP value (impact on RUL estimate, cycles)")
    fig.add_vline(x=0, line=dict(color="#8da0bd", width=1))
    st.plotly_chart(U.style_fig(fig, "SHAP summary plot (top 12 features; colour = feature value)", 520, False), width="stretch")
    U.note("SHAP describes how the model uses its inputs; it does not prove that a sensor reflects a physical failure mechanism. "
           "No component diagnosis can be derived from these charts.")

with tab_l:
    ids = sorted(df[C.ID_COL].unique())
    c1, c2 = st.columns([1, 3])
    e = c1.selectbox("Engine", ids, key="xai_eng", index=ids.index(st.session_state.get("selected_engine")) if st.session_state.get("selected_engine") in ids else 0)
    gdf = df[df[C.ID_COL] == e].sort_values(C.CYCLE_COL)
    cmin, cmax = int(gdf[C.CYCLE_COL].min()), int(gdf[C.CYCLE_COL].max())
    as_of = c2.slider("Explain the prediction as of cycle", cmin, cmax, cmax, key="xai_cyc") if cmax > cmin else cmax
    sub = gdf[gdf[C.CYCLE_COL] <= as_of]
    res, feat = p.predict_latest(sub, thr)
    loc = EX.local_explanation(p.model, expl, feat[p.feature_names], window=p.cfg.window)
    k = st.columns(3)
    r0 = res.iloc[0]
    k[0].markdown(U.kpi("Predicted RUL", f"{r0['rul_pred']:.1f}", "cycles", f"cycle {int(r0['last_cycle'])}", r0["state"]), unsafe_allow_html=True)
    k[1].markdown(U.kpi("Average training prediction", f"{loc['base_value']:.1f}", "cycles", "SHAP base value"), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Sum of contributions", f"{loc['model_output']-loc['base_value']:+.1f}", "cycles", f"raw model output {loc['model_output']:.1f} (clipped to [0, {p.cap:.0f}])"), unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        t = loc["by_sensor"].head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["shap"], y=t["label"], orientation="h",
                               marker_color=[U.SERIES[1] if v < 0 else U.SERIES[0] for v in t["shap"]],
                               hovertemplate="%{y}: %{x:+.2f} cycles<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#8da0bd", width=1)); fig.update_xaxes(title="Contribution (cycles)")
        st.plotly_chart(U.style_fig(fig, "By sensor (grouped)", 420, False), width="stretch")
    with cc2:
        t = loc["table"].head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["shap"], y=t["label"], orientation="h",
                               marker_color=[U.SERIES[1] if v < 0 else U.SERIES[0] for v in t["shap"]], customdata=t["value"],
                               hovertemplate="%{y}<br>value %{customdata:.4g}<br>SHAP %{x:+.2f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#8da0bd", width=1)); fig.update_xaxes(title="Contribution (cycles)")
        st.plotly_chart(U.style_fig(fig, "By engineered feature", 420, False), width="stretch")
    st.markdown(EX.explanation_text(loc))
    with st.expander("Ranked contributions and the values used"):
        st.dataframe(loc["table"][["label", "value", "shap"]].head(25), hide_index=True, width="stretch",
                     column_config={"label": "Feature", "value": st.column_config.NumberColumn("Value used", format="%.4g"),
                                    "shap": st.column_config.NumberColumn("SHAP (cycles)", format="%+.2f")})
    st.caption(f"Explained observation: engine {e}, cycle {int(r0['last_cycle'])}; features computed only from cycles ≤ {int(r0['last_cycle'])}.")
