"""Regression metrics (cycles) and health-state helper."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric scoring function of Saxena et al., PHM08 (Damage Propagation Modeling).

    d = y_pred - y_true ; s = sum(exp(-d/13) - 1) for d < 0 (early) and sum(exp(d/10) - 1)
    for d >= 0 (late). Lower is better; late predictions are penalised more heavily."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


def risk_state(rul: float, thresholds: dict) -> str:
    """Healthy: rul > warning; Warning: critical <= rul <= warning; Critical: rul < critical."""
    if rul < thresholds["critical"]:
        return "Critical"
    if rul <= thresholds["warning"]:
        return "Warning"
    return "Healthy"
