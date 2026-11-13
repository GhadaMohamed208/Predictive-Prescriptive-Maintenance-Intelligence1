import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_state as S
from src import config as C
from src import ui_components as U

p = S.setup_page()
art = S.get_artifacts()
m, comp, oof, tp, abl = art["metrics"], art["comparison"], art["oof"], art["test_pred"], art["ablation"]
U.page_header("Model Performance", "Measured results from the reproducible training run. Errors are in RUL cycles. Nothing on this page is hard-coded.",
              "EVALUATION")
if m is None or comp is None:
    U.note("Metrics files are missing; run python -m src.train_models.", warn=True); st.stop()

st.markdown(f"**Selected model:** {p.metadata['model_name']} · trained {p.metadata['trained_at_utc']} · seed {p.metadata['random_seed']}")
st.caption(p.metadata["selection_rationale"])
cv, tc, tr = m["internal_validation_cv_selected"], m["official_test_last_cycle_vs_capped_truth"], m["official_test_last_cycle_vs_raw_truth"]
k = st.columns(4)
k[0].markdown(U.kpi("Internal CV MAE", f"{cv['mae']:.2f}", "cycles", "engine-grouped 5-fold, all cycles"), unsafe_allow_html=True)
k[1].markdown(U.kpi("Internal CV RMSE", f"{cv['rmse']:.2f}", "cycles", f"R² {cv['r2']:.3f}"), unsafe_allow_html=True)
k[2].markdown(U.kpi("Official test MAE", f"{tc['mae']:.2f}", "cycles", f"final cycles, n={tc['n']}, capped truth"), unsafe_allow_html=True)
k[3].markdown(U.kpi("Official test RMSE", f"{tc['rmse']:.2f}", "cycles", f"R² {tc['r2']:.3f} · NASA score {tc['nasa_score']:.0f}"), unsafe_allow_html=True)
st.write("")
tabs = st.tabs(["Model comparison", "Actual vs. predicted", "Residual analysis", "Method & definitions"])
with tabs[0]:
    c = comp.sort_values("cv_rmse")
    fig = go.Figure(go.Bar(x=c["cv_rmse"], y=c["model"], orientation="h", marker_color=[U.SERIES[0] if mm == p.metadata["model_name"] else "#4a5d80" for mm in c["model"]],
                           error_x=dict(type="data", array=c["cv_rmse_fold_std"], color="#8da0bd"), text=c["cv_rmse"].round(2), textposition="outside",
                           hovertemplate="%{y}<br>CV RMSE %{x:.2f} cycles<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title="Grouped-CV RMSE (cycles) ± std across folds")
    st.plotly_chart(U.style_fig(fig, "Candidate comparison (selected model highlighted)", 340, False), width="stretch")
    cols = ["model", "cv_mae", "cv_rmse", "cv_r2", "cv_rmse_fold_std", "cv_rmse_late_life_rul<=60", "cv_rmse_truncation_emulation", "train_time_s", "predict_ms_per_row", "params"]
    st.dataframe(comp[cols], hide_index=True, width="stretch", column_config={
        "cv_mae": st.column_config.NumberColumn("CV MAE", format="%.2f"), "cv_rmse": st.column_config.NumberColumn("CV RMSE", format="%.2f"),
        "cv_r2": st.column_config.NumberColumn("CV R²", format="%.3f"), "cv_rmse_fold_std": st.column_config.NumberColumn("Fold std", format="%.2f"),
        "cv_rmse_late_life_rul<=60": st.column_config.NumberColumn("RMSE (true RUL ≤ 60)", format="%.2f"),
        "cv_rmse_truncation_emulation": st.column_config.NumberColumn("RMSE (random cut-off per engine)", format="%.2f"),
        "train_time_s": st.column_config.NumberColumn("Train time (s)", format="%.2f"), "predict_ms_per_row": st.column_config.NumberColumn("ms / row", format="%.3f")})
    if abl is not None:
        st.markdown("**Feature-set ablation** (single reference model, same engine-grouped CV)")
        st.dataframe(abl, hide_index=True, width="stretch")
        st.caption(f"Chosen: {p.metadata['feature_set']}. Cycle-count features were evaluated but not used so that the pipeline depends only on sensor behaviour.")
with tabs[1]:
    c1, c2 = st.columns(2)
    if tp is not None:
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tp["rul_true"], y=tp["rul_pred"], mode="markers", name="Test engines",
                                     marker=dict(color=U.SERIES[0], size=8, line=dict(color="#0d1626", width=.8)),
                                     customdata=tp[C.ID_COL], hovertemplate="engine %{customdata}<br>true %{x:.0f}<br>pred %{y:.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, p.cap], y=[0, p.cap], mode="lines", name="Perfect", line=dict(color="#8da0bd", dash="dash")))
            fig.update_xaxes(title="Official true RUL (capped, cycles)"); fig.update_yaxes(title="Predicted RUL (cycles)")
            st.plotly_chart(U.style_fig(fig, f"Official test set — final cycle per engine (n={len(tp)})", 400), width="stretch")
    if oof is not None:
        with c2:
            col = f"pred_{p.metadata['model_name']}"
            s = oof.sample(min(4000, len(oof)), random_state=1)
            fig = go.Figure(go.Scattergl(x=s["rul"], y=s[col], mode="markers", marker=dict(color=U.SERIES[2], size=4, opacity=.45), name="OOF rows"))
            fig.add_trace(go.Scatter(x=[0, p.cap], y=[0, p.cap], mode="lines", name="Perfect", line=dict(color="#8da0bd", dash="dash")))
            fig.update_xaxes(title="True RUL (capped, cycles)"); fig.update_yaxes(title="Out-of-fold prediction (cycles)")
            st.plotly_chart(U.style_fig(fig, "Internal validation — out-of-fold predictions (4,000-row sample)", 400), width="stretch")
with tabs[2]:
    c1, c2 = st.columns(2)
    if tp is not None:
        with c1:
            fig = go.Figure(go.Histogram(x=tp["error"], nbinsx=20, marker_color=U.SERIES[0], marker_line=dict(color="#0d1626", width=1)))
            fig.add_vline(x=0, line=dict(color="#8da0bd", dash="dash"))
            fig.update_xaxes(title="Prediction error = predicted − true (cycles); >0 means late/optimistic"); fig.update_yaxes(title="Engines")
            st.plotly_chart(U.style_fig(fig, "Test error distribution", 360, False), width="stretch")
        with c2:
            fig = go.Figure(go.Scatter(x=tp["rul_true"], y=tp["error"], mode="markers", marker=dict(color=U.SERIES[3], size=8),
                                       customdata=tp[C.ID_COL], hovertemplate="engine %{customdata}<br>true %{x:.0f}<br>error %{y:+.1f}<extra></extra>"))
            fig.add_hline(y=0, line=dict(color="#8da0bd", dash="dash"))
            fig.update_xaxes(title="Official true RUL (capped, cycles)"); fig.update_yaxes(title="Error (cycles)")
            st.plotly_chart(U.style_fig(fig, "Residuals vs. true RUL", 360, False), width="stretch")
        worst = tp.reindex(tp["error"].abs().sort_values(ascending=False).index).head(8)
        st.markdown("**Largest test errors**")
        st.dataframe(worst[[C.ID_COL, C.CYCLE_COL, "rul_true", "rul_true_raw", "rul_pred", "error"]], hide_index=True, width="stretch")
with tabs[3]:
    st.markdown(f"""
**Targets.** Train RUL = final cycle − current cycle; capped at **{p.cap:.0f}** cycles (a modelling convention, not a NASA definition). The same cap is applied to the official
test RUL values for the headline test metrics so both sides share a scale; metrics against the *uncapped* official values are also reported below.

**Validation.** Engine-grouped 5-fold CV (GroupKFold) for feature selection, tuning and model choice. The official test set was evaluated **once**, after selection, at each engine's
last observed cycle (the only point with ground truth).

**Metrics.** MAE, RMSE (cycles), R², and the NASA asymmetric score (Saxena et al. 2008; lower is better; late predictions penalised more).

**Derived states.** Healthy/Warning/Critical come from configurable thresholds on predicted RUL; they have no independent labels, so no classification accuracy is reported.
""")
    st.dataframe(pd.DataFrame({"Metric": ["MAE", "RMSE", "R²", "NASA score"],
                               "Test vs capped truth": [tc["mae"], tc["rmse"], tc["r2"], tc["nasa_score"]],
                               "Test vs raw (uncapped) truth": [tr["mae"], tr["rmse"], tr["r2"], tr["nasa_score"]]}), hide_index=True, width="stretch")
    st.caption(m["notes"]["raw"])
