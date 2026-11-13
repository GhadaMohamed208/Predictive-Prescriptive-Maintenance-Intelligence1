"""Separate, unsupervised anomaly / degradation analysis (NOT a failure detector).

Method: Isolation Forest fitted on early-life behaviour of the training engines
(cycles ANOMALY_REF_CYCLES). The input per cycle is the 10-cycle rolling mean of every
modelling sensor minus the engine's own early baseline (removes between-engine initial
wear). A cycle is "anomalous" if its isolation score exceeds the 99th percentile of the
reference scores. An anomaly means: "unusual relative to early-life reference behaviour".
It may reflect progressive degradation, an unusual operating condition or a sensor problem.
No fault labels exist for this detector, so it is not evaluated as a fault classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C
from .feature_engineering import FeatureConfig, build_features


def _anomaly_inputs(df: pd.DataFrame, sensors: list[str], window: int = C.ROLL_WINDOW,
                    baseline_cycles: int = C.BASELINE_CYCLES) -> pd.DataFrame:
    cfg = FeatureConfig(sensors=sensors, window=window, baseline_cycles=baseline_cycles,
                        use_current=True, use_rolling_mean=True, use_rolling_std=False,
                        use_slope=False, use_baseline_dev=True)
    f = build_features(df, cfg)
    out = f[[C.ID_COL, C.CYCLE_COL]].copy()
    for s in sensors:
        base = f[f"{s}__cur"] - f[f"{s}__dev0"]
        out[s] = f[f"{s}__rmean"] - base
    return out


def fit_reference(train: pd.DataFrame, sensors: list[str], seed: int = C.RANDOM_SEED) -> dict:
    X = _anomaly_inputs(train, sensors)
    lo, hi = C.ANOMALY_REF_CYCLES
    ref = X[(X[C.CYCLE_COL] >= lo) & (X[C.CYCLE_COL] <= hi)]
    iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=1)
    iso.fit(ref[sensors].to_numpy())
    ref_scores = -iso.score_samples(ref[sensors].to_numpy())  # higher = more anomalous
    thr = float(np.quantile(ref_scores, C.ANOMALY_THRESHOLD_QUANTILE))
    # robust per-sensor reference noise (MAD-based sigma) of the shift features
    med = ref[sensors].median()
    sigma = (1.4826 * (ref[sensors] - med).abs().median()).replace(0, np.nan)
    return {"model": iso, "threshold": thr, "sensors": list(sensors),
            "ref_sigma": sigma.fillna(ref[sensors].std()).to_dict(),
            "ref_rows": int(len(ref)), "ref_cycles": list(C.ANOMALY_REF_CYCLES),
            "quantile": C.ANOMALY_THRESHOLD_QUANTILE}


def score(ref: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Per (engine, cycle) anomaly score and flag, using only past data of each engine."""
    X = _anomaly_inputs(df, ref["sensors"])
    sc = -ref["model"].score_samples(X[ref["sensors"]].to_numpy())
    out = X[[C.ID_COL, C.CYCLE_COL]].copy()
    out["anomaly_score"] = sc
    out["anomaly_flag"] = sc > ref["threshold"]
    return out


def engine_summary(scores: pd.DataFrame, last_n: int = 10) -> dict:
    s = scores.sort_values(C.CYCLE_COL)
    tail = s.tail(last_n)
    return {
        "last_score": float(s["anomaly_score"].iloc[-1]),
        "last_flag": bool(s["anomaly_flag"].iloc[-1]),
        "recent_flag_fraction": float(tail["anomaly_flag"].mean()),
        "recent_n": int(len(tail)),
        "n_flagged_total": int(s["anomaly_flag"].sum()),
        "first_flag_cycle": (int(s.loc[s["anomaly_flag"], C.CYCLE_COL].iloc[0])
                             if s["anomaly_flag"].any() else None),
    }


def sensor_shifts(df_engine: pd.DataFrame, ref: dict, recent_n: int = 10,
                  baseline_n: int = C.BASELINE_CYCLES) -> pd.DataFrame:
    """Robust shift of each sensor: mean(last recent_n) - mean(first baseline_n), expressed in
    units of the early-life reference noise (MAD sigma of the 10-cycle-mean shift feature).
    Also returns the monotonic trend strength (Spearman rho of sensor vs cycle)."""
    d = df_engine.sort_values(C.CYCLE_COL)
    rows = []
    for s in ref["sensors"]:
        base = d[s].iloc[:baseline_n].mean()
        recent = d[s].iloc[-recent_n:].mean()
        sig = ref["ref_sigma"].get(s) or np.nan
        z = (recent - base) / sig if sig and np.isfinite(sig) and sig > 0 else np.nan
        rho = d[[C.CYCLE_COL, s]].corr(method="spearman").iloc[0, 1] if len(d) > 3 else np.nan
        rows.append({"sensor": s, "label": C.sensor_label(s), "baseline_mean": base,
                     "recent_mean": recent, "shift": recent - base, "shift_sigma": z,
                     "trend_rho": rho})
    out = pd.DataFrame(rows)
    out["abs_sigma"] = out["shift_sigma"].abs()
    return out.sort_values("abs_sigma", ascending=False).reset_index(drop=True)
