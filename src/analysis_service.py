"""UI-independent analysis of the ACTIVE dataset (fleet table + findings + single-asset detail).

Everything here is a pure function of (dataset record, thresholds, sigma trigger) so that the same numbers feed the
Control Room, Fleet Overview, Recommendations, Sensor page and the PDF reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import anomaly_detection as AD
from . import config as C
from . import dataset_state as DS
from . import fleet_analysis as FA
from . import generic_analysis as GA
from . import recommendation_engine as RE

PRIORITIES = ["Critical", "High", "Medium", "Low"]          # asset-level priority labels (Info is not a priority)
OPEN_STATUSES = ("Pending", "In Progress")


def _priority_of(recs: list[dict]) -> str:
    ps = [r["priority"] for r in recs if r["priority"] in PRIORITIES]
    return min(ps, key=PRIORITIES.index) if ps else "Low"


def fleet_view(record: dict, predictor, anomaly_ref, thr: dict, sigma: float) -> dict:
    """Return {'table', 'findings', 'shifts'} for the active dataset.

    table columns (all datasets): unit, name, last_cycle, n_cycles, state, priority, n_findings, attention_rank
      AI datasets add: rul_pred, at_cap, anomaly_recent_fraction, anomaly_persistent, ood_flag, rul_true_official
      sensor-only add: max_shift_sigma, top_sensor
    findings: one row per triggered recommendation (+ unit, name, last_cycle, state, rul)."""
    df = record["df"]
    findings: list[dict] = []
    if record["capability"] == DS.CAP_AI:
        tbl = FA.fleet_table(predictor, df, thr, anomaly_ref, record.get("rul_true"))
        tbl["ood_flag"] = tbl["ood_fraction"] > 0.05
        shifts = {}
        for uid, g in df.groupby(C.ID_COL):
            shifts[int(uid)] = AD.sensor_shifts(g, anomaly_ref) if anomaly_ref is not None else pd.DataFrame()
        by_unit = tbl.set_index(C.ID_COL)
        for uid, g in df.groupby(C.ID_COL):
            r = by_unit.loc[uid]
            an = None
            if anomaly_ref is not None and not pd.isna(r["anomaly_recent_fraction"]):
                an = {"recent_flag_fraction": float(r["anomaly_recent_fraction"]), "recent_n": 10, "last_score": float("nan"),
                      "first_flag_cycle": None, "last_flag": False}
            ctx = RE.Context(
                engine_id=int(uid), last_cycle=int(r["last_cycle"]), pred_rul=float(r["rul_pred"]), state=r["state"],
                thresholds=thr, cap=float(predictor.cap), n_cycles=int(r["n_cycles"]), at_cap=bool(r["at_cap"]),
                anomaly=an, shifts=shifts[int(uid)].to_dict("records") if len(shifts[int(uid)]) else [],
                ood={"flag": bool(r["ood_flag"]), "n_outside": 0, "fraction": float(r["ood_fraction"])},
                sigma_trigger=sigma)
            for rec in RE.evaluate(ctx):
                findings.append({**rec, "last_cycle": int(r["last_cycle"]), "state": r["state"], "rul": float(r["rul_pred"])})
    else:
        sensors = record["sensors"]
        tbl, shifts = GA.asset_table(df, sensors, sigma, record.get("sensor_labels"))
        for _, r in tbl.iterrows():
            uid = int(r[C.ID_COL])
            if r["state"] == "Insufficient history":
                ctx = RE.Context(engine_id=uid, last_cycle=int(r["last_cycle"]), pred_rul=None, state=None, thresholds=thr,
                                 cap=0.0, n_cycles=int(r["n_cycles"]), sigma_trigger=sigma)
            else:
                ctx = RE.Context(engine_id=uid, last_cycle=int(r["last_cycle"]), pred_rul=None, state=None, thresholds=thr,
                                 cap=0.0, n_cycles=int(r["n_cycles"]), shifts=shifts[uid].to_dict("records"),
                                 sigma_trigger=sigma)
            for rec in RE.evaluate(ctx):
                findings.append({**rec, "last_cycle": int(r["last_cycle"]), "state": r["state"], "rul": float("nan")})
        tbl = tbl.copy()
    F = pd.DataFrame(findings)
    if len(F):
        F["name"] = F["engine_id"].map(lambda u: DS.asset_name(record, u))
        F["_o"] = F["priority"].map(RE.PRIORITY_ORDER)
        F = F.sort_values(["_o", "engine_id"]).drop(columns="_o").reset_index(drop=True)
    pr = {u: _priority_of([f for f in findings if f["engine_id"] == u]) for u in tbl[C.ID_COL]}
    nf = {u: sum(1 for f in findings if f["engine_id"] == u and f["priority"] in ("Critical", "High", "Medium")) for u in tbl[C.ID_COL]}
    tbl["priority"] = tbl[C.ID_COL].map(pr)
    tbl["n_findings"] = tbl[C.ID_COL].map(nf)
    tbl["name"] = tbl[C.ID_COL].map(lambda u: DS.asset_name(record, u))
    if record["capability"] == DS.CAP_SENSOR:
        order = {"Drift detected": 0, "Insufficient history": 1, "Stable": 2}
        tbl["_s"] = tbl["state"].map(order)
        tbl = tbl.sort_values(["_s", "max_shift_sigma"], ascending=[True, False], na_position="last").drop(columns="_s")
        tbl["attention_rank"] = range(1, len(tbl) + 1)
        tbl = tbl.reset_index(drop=True)
    return {"table": tbl, "findings": F, "shifts": shifts}


def status_order(record: dict) -> list[str]:
    return C.STATES if record["capability"] == DS.CAP_AI else GA.SENSOR_STATUSES


def status_counts(record: dict, tbl: pd.DataFrame) -> dict:
    vc = tbl["state"].value_counts()
    return {s: int(vc.get(s, 0)) for s in status_order(record)}


def status_colors(record: dict) -> dict:
    if record["capability"] == DS.CAP_AI:
        return C.STATE_COLORS
    return {"Drift detected": C.STATE_COLORS["Warning"], "Stable": C.STATE_COLORS["Healthy"], "Insufficient history": "#8da0bd"}


def kpis(record: dict, view: dict, rec_status: dict | None = None) -> dict:
    """KPI values for the Control Room (only what the dataset supports)."""
    t, F = view["table"], view["findings"]
    rec_status = rec_status or {}
    out = {"assets": int(len(t))}
    if record["capability"] == DS.CAP_AI:
        out.update({"healthy": int((t["state"] == "Healthy").sum()),
                    "attention": int(t["state"].isin(["Warning", "Critical"]).sum()),
                    "critical": int((t["state"] == "Critical").sum())})
    else:
        out.update({"healthy": int((t["state"] == "Stable").sum()),
                    "attention": int((t["state"] == "Drift detected").sum()),
                    "critical": None})
    if len(F):
        actionable = F[F["priority"].isin(["Critical", "High", "Medium"])]
        out["warnings"] = int(len(actionable))
        open_hi = F[F["priority"].isin(["Critical", "High"]) &
                    ~F["id"].map(lambda i: rec_status.get(i, "Pending")).isin(["Completed", "Dismissed"])]
        out["high_actions"] = int(len(open_hi))
    else:
        out["warnings"] = out["high_actions"] = 0
    return out


# ------------------------------------------------------------------ single asset
def asset_detail(record: dict, uid: int, predictor, anomaly_ref, thr: dict, sigma: float, as_of: int | None = None,
                 explainer=None) -> dict:
    """Detailed analysis of one asset, using only data up to ``as_of`` (default: all)."""
    df = record["df"]
    g = df[df[C.ID_COL] == uid].sort_values(C.CYCLE_COL)
    if as_of is not None:
        g = g[g[C.CYCLE_COL] <= as_of]
    if record["capability"] == DS.CAP_AI:
        from .engine_analysis import analyze_engine
        res = analyze_engine(predictor, anomaly_ref, explainer, g, thr, sigma)
        res["capability"] = DS.CAP_AI
        return res
    sensors = record["sensors"]
    nref = GA.noise_reference(df, sensors)
    out = {"engine_id": int(uid), "engine_df": g.reset_index(drop=True), "thresholds": thr, "capability": DS.CAP_SENSOR,
           "n_cycles": int(len(g)), "last_cycle": int(g[C.CYCLE_COL].iloc[-1]) if len(g) else 0, "rul": None, "state": None}
    if len(g) < GA.MIN_RECORDS:
        out["state"] = "Insufficient history"
        out["shifts"] = []
    else:
        t = GA.shift_table(g, sensors, nref, record.get("sensor_labels"))
        out["shifts"] = t.to_dict("records")
        mx = float(t["abs_sigma"].iloc[0]) if len(t) and np.isfinite(t["abs_sigma"].iloc[0]) else float("nan")
        out["state"] = "Drift detected" if (np.isfinite(mx) and mx >= sigma) else "Stable"
    ctx = RE.Context(engine_id=int(uid), last_cycle=out["last_cycle"], pred_rul=None, state=None, thresholds=thr, cap=0.0,
                     n_cycles=out["n_cycles"], shifts=out["shifts"], sigma_trigger=sigma)
    out["recs"] = RE.evaluate(ctx)
    out["anomaly"], out["anomaly_scores"], out["dq"] = None, None, None
    return out
