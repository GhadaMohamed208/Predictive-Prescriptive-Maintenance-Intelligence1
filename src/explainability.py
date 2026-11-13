"""SHAP explanations. SHAP values describe MODEL ATTRIBUTION, not physical causation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .feature_engineering import feature_label, feature_sensor

TREE_NAMES = ("RandomForestRegressor", "HistGradientBoostingRegressor", "XGBRegressor")


def _final_estimator(pipe):
    return pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe


def explainer_kind(pipe) -> str | None:
    name = type(_final_estimator(pipe)).__name__
    if name in TREE_NAMES:
        return "tree"
    if name == "Ridge":
        return "linear"
    return None


def _transform(pipe, X: pd.DataFrame):
    if hasattr(pipe, "steps") and len(pipe.steps) > 1:
        return pipe[:-1].transform(X)
    return X.to_numpy()


def make_explainer(pipe, background: pd.DataFrame | None = None):
    """Return (explainer, kind). Tree models use TreeExplainer (tree_path_dependent, no
    background needed); Ridge uses LinearExplainer with a documented background sample."""
    import shap

    kind = explainer_kind(pipe)
    est = _final_estimator(pipe)
    if kind == "tree":
        name = type(est).__name__
        if name == "HistGradientBoostingRegressor":
            # shap does not support HGB natively: use the model-agnostic Permutation-free path
            return None, "unsupported"
        return shap.TreeExplainer(est), kind
    if kind == "linear":
        if background is None:
            raise ValueError("Background sample required for linear explainer.")
        return shap.LinearExplainer(est, _transform(pipe, background)), kind
    return None, "unsupported"


def shap_values(pipe, explainer, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(explainer.shap_values(_transform(pipe, X)))


def expected_value(explainer) -> float:
    ev = explainer.expected_value
    return float(np.ravel(ev)[0])


def compute_global(pipe, X_train: pd.DataFrame, n: int = 1500, seed: int = C.RANDOM_SEED):
    """SHAP on a random, documented sample of training rows. Returns dict or None."""
    sample = X_train.sample(min(n, len(X_train)), random_state=seed)
    try:
        expl, kind = make_explainer(pipe, background=sample.sample(min(200, len(sample)),
                                                                   random_state=seed))
    except Exception:
        return None
    if expl is None:
        return None
    vals = shap_values(pipe, expl, sample)
    return {"values": vals, "X": sample, "expected": expected_value(expl), "kind": kind,
            "n": int(len(sample)), "seed": seed}


def save_global(path, g: dict, feature_names: list[str]) -> None:
    np.savez_compressed(path, values=g["values"], X=g["X"].to_numpy(),
                        expected=g["expected"], names=np.array(feature_names),
                        n=g["n"], seed=g["seed"], kind=g["kind"])


def load_global(path):
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        return None
    names = list(z["names"])
    return {"values": z["values"], "X": pd.DataFrame(z["X"], columns=names),
            "expected": float(z["expected"]), "n": int(z["n"]), "seed": int(z["seed"]),
            "kind": str(z["kind"]), "names": names}


def global_importance(g: dict, by_sensor: bool = False, window: int = C.ROLL_WINDOW) -> pd.DataFrame:
    names = list(g["X"].columns)
    imp = np.abs(g["values"]).mean(axis=0)
    df = pd.DataFrame({"feature": names, "mean_abs_shap": imp})
    df["label"] = [feature_label(n, window) for n in names]
    df["sensor"] = [feature_sensor(n) for n in names]
    if by_sensor:
        df = (df.dropna(subset=["sensor"]).groupby("sensor", as_index=False)["mean_abs_shap"].sum())
        df["label"] = df["sensor"].map(C.sensor_label)
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(pipe, explainer, x_row: pd.DataFrame, window: int = C.ROLL_WINDOW) -> dict:
    """Local SHAP for one feature row (1-row DataFrame). Contributions are in RUL cycles
    (model output scale, before clipping to [0, cap])."""
    v = shap_values(pipe, explainer, x_row)[0]
    names = list(x_row.columns)
    df = pd.DataFrame({"feature": names, "label": [feature_label(n, window) for n in names],
                       "sensor": [feature_sensor(n) for n in names],
                       "value": x_row.iloc[0].to_numpy(), "shap": v})
    df["abs"] = df["shap"].abs()
    df = df.sort_values("abs", ascending=False).reset_index(drop=True)
    by_sensor = (df.dropna(subset=["sensor"]).groupby("sensor", as_index=False)["shap"].sum())
    by_sensor["abs"] = by_sensor["shap"].abs()
    by_sensor["label"] = by_sensor["sensor"].map(C.sensor_label)
    by_sensor = by_sensor.sort_values("abs", ascending=False).reset_index(drop=True)
    base = expected_value(explainer)
    return {"table": df, "by_sensor": by_sensor, "base_value": base,
            "model_output": float(base + v.sum())}


def explanation_text(loc: dict, top_k: int = 3) -> str:
    """Deterministic sentence grounded in the computed values."""
    bs = loc["by_sensor"].head(top_k)
    parts = []
    for _, r in bs.iterrows():
        direction = "lowering" if r["shap"] < 0 else "raising"
        parts.append(f"{r['label']} ({direction} the estimate by {abs(r['shap']):.1f} cycles)")
    return ("Largest grouped contributions to this model's RUL estimate relative to the average "
            f"training prediction ({loc['base_value']:.1f} cycles): " + "; ".join(parts) +
            ". These are statistical attributions of the model, not proof of a physical cause.")
