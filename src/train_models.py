"""Reproducible training / evaluation script.

    python -m src.train_models            # full run (a few minutes on a laptop)
    python -m src.train_models --quick    # reduced grids (smoke run; NOT for reporting)

Protocol
  * Train file -> RUL targets (raw + capped). Engine-grouped 5-fold CV (GroupKFold) is the only
    tool used for feature-set choice, tuning and model selection.
  * The official test set is touched exactly ONCE, after the final model has been selected, and
    only at each engine's last observed cycle (the only point with a ground-truth RUL).
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import anomaly_detection as AD
from . import config as C
from . import explainability as EX
from .data_loader import load_fd001, verify_subset_identity
from .data_validation import validate_trajectories
from .evaluate_models import regression_metrics
from .feature_engineering import FeatureConfig, build_features, feature_names
from .rul_target import add_rul_train, final_cycle_targets

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

SEED = C.RANDOM_SEED
CAP = C.RUL_CAP


def make_model(kind: str, params: dict):
    if kind == "Dummy (mean)":
        return DummyRegressor(strategy="mean")
    if kind == "Ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(**params))])
    if kind == "Random Forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=2, **params)
    if kind == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=SEED, **params)
    if kind == "XGBoost":
        return XGBRegressor(random_state=SEED, n_jobs=2, tree_method="hist", **params)
    raise ValueError(kind)


def grids(quick: bool) -> dict:
    g = {
        "Dummy (mean)": [{}],
        "Ridge": [{"alpha": a} for a in ([10.0] if quick else [1.0, 10.0, 100.0])],
        "Random Forest": [dict(n_estimators=120 if not quick else 40, max_depth=d, min_samples_leaf=l,
                               max_features=0.4) for d, l in
                          ([(12, 10)] if quick else [(10, 10), (14, 5), (None, 20)])],
        "HistGradientBoosting": [dict(learning_rate=lr, max_leaf_nodes=nl, max_iter=300 if not quick else 80,
                                      l2_regularization=1.0, min_samples_leaf=40) for lr, nl in
                                 ([(0.05, 15)] if quick else [(0.05, 15), (0.05, 31), (0.1, 15)])],
    }
    if HAS_XGB:
        g["XGBoost"] = [dict(n_estimators=400 if not quick else 80, learning_rate=0.05, max_depth=d,
                             min_child_weight=m, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0)
                        for d, m in ([(4, 10)] if quick else [(3, 10), (5, 10), (5, 30)])]
    return g


def clipped(p):
    return np.clip(p, 0, CAP)


def truncation_emulation_rmse(oof: pd.DataFrame, pred_col: str, reps: int = 100, seed: int = SEED,
                              min_len: int = 31) -> float:
    """Mean RMSE when each engine is observed at one random cut-off cycle (>= min_len), emulating
    the test-set situation (one prediction per engine at its last observed cycle). Uses OOF
    predictions so it is leakage-free."""
    rng = np.random.default_rng(seed)
    groups = {u: g.reset_index(drop=True) for u, g in oof.groupby(C.ID_COL)}
    out = []
    for _ in range(reps):
        err = []
        for g in groups.values():
            k = rng.integers(min(min_len, len(g)) - 1, len(g))
            err.append(g.loc[k, pred_col] - g.loc[k, "rul"])
        out.append(np.sqrt(np.mean(np.square(err))))
    return float(np.mean(out))


def run(quick: bool = False) -> None:
    t0 = time.time()
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    train, test, rul_true = load_fd001()
    ident = verify_subset_identity(train, test, rul_true)
    print("Subset identity:", ident["matches_FD001"], ident["checks"])
    if not ident["matches_FD001"]:
        print("WARNING: files do not match all documented FD001 properties.")
    vr_train = validate_trajectories(train, sensors=C.SENSOR_COLS, require_contiguous=True)
    vr_test = validate_trajectories(test, sensors=C.SENSOR_COLS)
    C.VALIDATION_LOG_PATH.write_text(json.dumps({
        "subset_identity": ident, "train": vr_train.to_dict(), "test": vr_test.to_dict(),
        "notes": ["Extra file 'x.txt' in the supplied archive is not part of the standard C-MAPSS "
                  "distribution (218 single-column rows, matches no subset) and was ignored.",
                  "FD002-FD004 were present in the archive but are out of scope (multiple operating "
                  "conditions need condition-aware modelling)."]}, indent=2, default=str))
    assert vr_train.ok and vr_test.ok, "Data validation failed"

    # sensors with <=2 distinct training values are uninformative -> excluded (derived from data)
    uninformative = [s for s in C.SENSOR_COLS if train[s].nunique() <= 2]
    sensors = [s for s in C.SENSOR_COLS if s not in uninformative]
    print("Excluded sensors:", uninformative, "| modelling sensors:", len(sensors))

    tr = add_rul_train(train, CAP).sort_values([C.ID_COL, C.CYCLE_COL]).reset_index(drop=True)
    y = tr["rul"].to_numpy()
    groups = tr[C.ID_COL].to_numpy()
    cv = GroupKFold(n_splits=5)

    # ---------------- feature-set ablation (single reference model, grouped CV) ---------------
    variants = {
        "A: current sensors": dict(use_rolling_mean=False, use_rolling_std=False, use_slope=False, use_baseline_dev=False),
        "B: A + cycle": dict(use_rolling_mean=False, use_rolling_std=False, use_slope=False, use_baseline_dev=False, use_cycle=True),
        "C: A + rolling mean/std/slope": dict(use_baseline_dev=False),
        "D: C + shift vs engine baseline": dict(),
        "E: D + cycle": dict(use_cycle=True),
    }
    abl_rows, feats_cache = [], {}
    for name, kw in variants.items():
        cfg = FeatureConfig(sensors=sensors, **kw)
        F = build_features(tr, cfg)
        X = F[feature_names(cfg)]
        feats_cache[name] = (cfg, X)
        m = HistGradientBoostingRegressor(random_state=SEED, learning_rate=0.05, max_iter=200 if not quick else 60,
                                          max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=40)
        p = clipped(cross_val_predict(m, X, y, groups=groups, cv=cv))
        met = regression_metrics(y, p)
        abl_rows.append({"feature_set": name, "n_features": X.shape[1], **{k: met[k] for k in ("mae", "rmse", "r2")}})
        print(f"  ablation {name:34s} RMSE={met['rmse']:.2f} MAE={met['mae']:.2f}")
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(C.ABLATION_PATH, index=False)
    # choose the smallest feature set within 1% of the best RMSE, excluding cycle-based sets
    # (age as an input would make a what-if/upload workflow depend on cycle bookkeeping)
    elig = abl[~abl["feature_set"].str.contains("cycle")]
    best_rmse = elig["rmse"].min()
    chosen_fs = elig[elig["rmse"] <= best_rmse * 1.01].sort_values("n_features").iloc[0]["feature_set"]
    cfg, X = feats_cache[chosen_fs]
    fnames = feature_names(cfg)
    print("Chosen feature set:", chosen_fs, X.shape)

    # ---------------- model candidates: tune with grouped CV ------------------------------
    results, oof_cols, best_params = [], {}, {}
    oof = tr[[C.ID_COL, C.CYCLE_COL, "rul", "rul_raw"]].copy()
    for kind, plist in grids(quick).items():
        scored = []
        for params in plist:
            ts = time.time()
            p = clipped(cross_val_predict(make_model(kind, params), X, y, groups=groups, cv=cv))
            scored.append((np.sqrt(np.mean((p - y) ** 2)), params, p, time.time() - ts))
            print(f"  {kind:22s} {params} -> CV RMSE {scored[-1][0]:.3f} ({scored[-1][3]:.0f}s)")
        rm, params, p, _ = min(scored, key=lambda t: t[0])
        best_params[kind] = params
        oof[f"pred_{kind}"] = p
        met = regression_metrics(y, p)
        late = y <= 60
        met_late = regression_metrics(y[late], p[late])
        # fold-level spread of RMSE
        fold_rmse = [float(np.sqrt(np.mean((p[te] - y[te]) ** 2))) for _, te in cv.split(X, y, groups)]
        ts = time.time()
        m = make_model(kind, params).fit(X, y)
        fit_time = time.time() - ts
        ts = time.time(); m.predict(X.iloc[:2000]); pred_ms = (time.time() - ts) / 2000 * 1000
        results.append({
            "model": kind, "params": json.dumps(params), "cv_mae": met["mae"], "cv_rmse": met["rmse"],
            "cv_r2": met["r2"], "cv_rmse_fold_std": float(np.std(fold_rmse)),
            "cv_nasa_score_rows": met["nasa_score"],
            "cv_rmse_late_life_rul<=60": met_late["rmse"], "cv_mae_late_life_rul<=60": met_late["mae"],
            "cv_rmse_truncation_emulation": truncation_emulation_rmse(oof, f"pred_{kind}"),
            "train_time_s": fit_time, "predict_ms_per_row": pred_ms,
            "shap_supported": EX.explainer_kind(m) is not None and type(m).__name__ != "HistGradientBoostingRegressor",
        })
    comp = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    comp.to_csv(C.COMPARISON_PATH, index=False)
    oof.to_csv(C.OOF_PATH, index=False)
    print(comp[["model", "cv_mae", "cv_rmse", "cv_r2", "train_time_s"]])

    # ---------------- selection ------------------------------------------------------------
    cand = comp[comp["model"] != "Dummy (mean)"]
    best = cand.iloc[0]
    near = cand[cand["cv_rmse"] <= best["cv_rmse"] * 1.02]
    near = near.sort_values(["shap_supported", "cv_rmse"], ascending=[False, True])
    sel = near.iloc[0]
    sel_name = sel["model"]
    rationale = (f"Selected '{sel_name}' (grouped 5-fold CV RMSE {sel['cv_rmse']:.2f} cycles, MAE "
                 f"{sel['cv_mae']:.2f}). Rule: among non-baseline candidates within 2% of the best CV RMSE "
                 f"('{best['model']}', {best['cv_rmse']:.2f}), prefer one with a supported SHAP explainer, "
                 f"then the lowest CV RMSE. The Dummy baseline reaches CV RMSE "
                 f"{float(comp.loc[comp['model'] == 'Dummy (mean)', 'cv_rmse'].iloc[0]):.2f}.")
    print(rationale)

    # ---------------- final fit on ALL training rows --------------------------------------
    final = make_model(sel_name, best_params[sel_name])
    ts = time.time(); final.fit(X, y); final_time = time.time() - ts

    # ---------------- official test: evaluated ONCE, last cycle only ------------------------
    Ft = build_features(test, cfg)
    last = Ft.sort_values([C.ID_COL, C.CYCLE_COL]).groupby(C.ID_COL).tail(1).reset_index(drop=True)
    tgt = final_cycle_targets(test, rul_true, CAP)
    assert (last[C.ID_COL].to_numpy() == tgt[C.ID_COL].to_numpy()).all()
    assert (last[C.CYCLE_COL].to_numpy() == tgt[C.CYCLE_COL].to_numpy()).all()
    pt = clipped(final.predict(last[fnames]))
    tp = tgt.copy(); tp["rul_pred"] = pt
    tp["error"] = tp["rul_pred"] - tp["rul_true"]
    tp.to_csv(C.TEST_PRED_PATH, index=False)
    m_cap = regression_metrics(tp["rul_true"], pt)
    m_raw = regression_metrics(tp["rul_true_raw"], pt)
    print("TEST (capped truth):", m_cap, "\nTEST (raw truth):", m_raw)

    # ---------------- SHAP global ------------------------------------------------------------
    g = EX.compute_global(final, X)
    shap_note = "available"
    if g is not None:
        EX.save_global(C.SHAP_GLOBAL_PATH, g, fnames)
    else:
        shap_note = "unavailable for the selected model type"
    print("SHAP:", shap_note)

    # ---------------- anomaly detector -----------------------------------------------------
    ref = AD.fit_reference(tr, sensors)
    joblib.dump(ref, C.ANOMALY_PATH)
    asc = AD.score(ref, tr)
    merged = asc.merge(tr[[C.ID_COL, C.CYCLE_COL, "rul_raw"]], on=[C.ID_COL, C.CYCLE_COL])
    from scipy.stats import spearmanr
    rho = float(spearmanr(merged["anomaly_score"], -merged["rul_raw"])[0])
    flag_by_phase = {
        "flag_rate_rul>125": float(merged.loc[merged.rul_raw > 125, "anomaly_flag"].mean()),
        "flag_rate_rul_60_125": float(merged.loc[(merged.rul_raw <= 125) & (merged.rul_raw > 60), "anomaly_flag"].mean()),
        "flag_rate_rul<=60": float(merged.loc[merged.rul_raw <= 60, "anomaly_flag"].mean()),
    }

    # ---------------- reference statistics -----------------------------------------------------
    ref_stats = {
        "sensors": {s: {"min": float(train[s].min()), "max": float(train[s].max()),
                        "p01": float(train[s].quantile(.01)), "p99": float(train[s].quantile(.99)),
                        "mean": float(train[s].mean()), "std": float(train[s].std())} for s in C.SENSOR_COLS},
        "features": {f: {"min": float(X[f].min()), "max": float(X[f].max())} for f in fnames},
        "excluded_sensors": uninformative,
    }
    C.REFERENCE_STATS_PATH.write_text(json.dumps(ref_stats, indent=2))

    bundle = {"model": final, "feature_config": cfg.to_dict(), "feature_names": fnames,
              "rul_cap": CAP, "model_name": sel_name, "params": best_params[sel_name],
              "sensors": sensors}
    joblib.dump(bundle, C.MODEL_PATH)
    meta = {
        "product": C.PRODUCT_NAME, "app_version": C.APP_VERSION,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "NASA C-MAPSS FD001", "model_name": sel_name, "params": best_params[sel_name],
        "rul_cap": CAP, "feature_set": chosen_fs, "n_features": len(fnames), "feature_names": fnames,
        "feature_config": cfg.to_dict(), "modelling_sensors": sensors, "excluded_sensors": uninformative,
        "selection_rationale": rationale, "random_seed": SEED, "cv": "GroupKFold(5) grouped by engine",
        "final_train_time_s": final_time, "n_train_rows": int(len(X)), "n_train_engines": int(tr[C.ID_COL].nunique()),
        "explainer": shap_note, "shap_sample_rows": None if g is None else g["n"],
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__,
                             **({"xgboost": __import__("xgboost").__version__} if HAS_XGB else {})},
        "xgboost_available": HAS_XGB, "quick_run": quick,
    }
    C.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    C.METRICS_PATH.write_text(json.dumps({
        "selected_model": sel_name,
        "internal_validation_cv_selected": {"mae": float(sel["cv_mae"]), "rmse": float(sel["cv_rmse"]),
                                             "r2": float(sel["cv_r2"])},
        "official_test_last_cycle_vs_capped_truth": m_cap,
        "official_test_last_cycle_vs_raw_truth": m_raw,
        "notes": {
            "capped": f"Predictions and training targets are capped at {CAP}; 'capped truth' applies the same "
                      f"cap to the official RUL values so both sides share a scale.",
            "raw": "Raw-truth metrics compare capped predictions with uncapped official RULs; they are "
                   "reported for transparency (engines with true RUL > cap are under-predicted by construction).",
            "test_usage": "Official test set evaluated once, after model selection, at the final cycle of each engine."},
        "anomaly_detector_sanity": {"spearman_score_vs_elapsed_life_train": rho, **flag_by_phase,
                                     "threshold": ref["threshold"]},
        "elapsed_s": time.time() - t0}, indent=2))
    print(f"Done in {time.time() - t0:.0f}s. Artifacts written to {C.MODELS_DIR} and {C.ARTIFACTS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
