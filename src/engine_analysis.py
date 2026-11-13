"""UI-independent full analysis of one engine trajectory (used by the app, reports, tests)."""
from __future__ import annotations

import pandas as pd

from . import anomaly_detection as AD
from . import config as C
from . import explainability as EX
from . import recommendation_engine as RE
from .data_validation import validate_trajectories
from .inference import Predictor


def analyze_engine(predictor: Predictor, anomaly_ref: dict | None, explainer, df_engine: pd.DataFrame,
                   thresholds: dict, sigma_trigger: float = 3.0, extra_dq_warnings: list[str] | None = None) -> dict:
    d = df_engine.sort_values(C.CYCLE_COL).reset_index(drop=True)
    eid = int(d[C.ID_COL].iloc[0])
    out: dict = {"engine_id": eid, "engine_df": d, "thresholds": thresholds, "cap": predictor.cap}

    vr = validate_trajectories(d, sensors=predictor.sensors)
    out["dq"] = vr
    if not vr.ok:
        out["error"] = "; ".join(vr.errors)
        return out

    res, last_feat = predictor.predict_latest(d, thresholds)
    r = res.iloc[0]
    out.update({"rul": float(r["rul_pred"]), "state": r["state"], "last_cycle": int(r["last_cycle"]),
                "at_cap": bool(r["at_cap"]), "n_cycles": int(len(d)), "feature_row": last_feat})
    out["rul_history"] = predictor.predict_rows(d).rename(columns={C.CYCLE_COL: "cycle"})
    out["ood"] = predictor.ood_report(last_feat)

    out["anomaly"], out["anomaly_scores"], out["shifts"] = None, None, []
    if anomaly_ref is not None:
        sc = AD.score(anomaly_ref, d)
        out["anomaly_scores"] = sc
        out["anomaly"] = AD.engine_summary(sc)
        out["anomaly_threshold"] = anomaly_ref["threshold"]
        out["shifts"] = AD.sensor_shifts(d, anomaly_ref).to_dict("records")

    out["shap_local"], out["shap_text"], out["shap_reason"] = None, "", ""
    if explainer is not None:
        try:
            loc = EX.local_explanation(predictor.model, explainer, last_feat[predictor.feature_names],
                                       window=predictor.cfg.window)
            out["shap_local"] = loc
            out["shap_text"] = EX.explanation_text(loc)
        except Exception as exc:  # graceful fallback; never fabricate
            out["shap_reason"] = f"SHAP computation failed: {exc}"
    else:
        out["shap_reason"] = "No SHAP explainer is available for the selected model type."

    ctx = RE.Context(
        engine_id=eid, last_cycle=out["last_cycle"], pred_rul=out["rul"], state=out["state"],
        thresholds=thresholds, cap=predictor.cap, n_cycles=out["n_cycles"], at_cap=out["at_cap"],
        anomaly=out["anomaly"], shifts=out["shifts"],
        shap_top=[] if not out["shap_local"] else out["shap_local"]["by_sensor"].head(3).to_dict("records"),
        ood=out["ood"], dq_warnings=list(vr.warnings) + list(extra_dq_warnings or []),
        sigma_trigger=sigma_trigger)
    out["recs"] = RE.evaluate(ctx)
    return out


def build_report_info(res: dict, predictor: Predictor, dataset_name: str, metrics: dict | None,
                      rul_true: float | None = None, input_mode: str = "Compatible engine trajectory") -> dict:
    """Assemble the dict consumed by report_generator.engine_report_pdf (only computed values)."""
    meta = dict(predictor.metadata)
    meta["feature_names"] = predictor.feature_names
    return {
        "dataset_name": dataset_name, "engine_id": res["engine_id"], "last_cycle": res["last_cycle"], "rul": res["rul"],
        "state": res["state"], "at_cap": res["at_cap"], "cap": predictor.cap, "thresholds": res["thresholds"],
        "dq": res["dq"].to_dict(), "n_cycles": res["n_cycles"], "ood": res.get("ood"), "model_meta": meta,
        "metrics": metrics, "rul_history": res["rul_history"], "engine_df": res["engine_df"], "shifts": res["shifts"],
        "shap_local": res["shap_local"], "shap_text": res["shap_text"], "shap_reason": res["shap_reason"],
        "anomaly": res["anomaly"], "anomaly_scores": res["anomaly_scores"],
        "anomaly_threshold": res.get("anomaly_threshold"), "recs": res["recs"], "rul_true": rul_true,
        "input_mode": input_mode,
    }
