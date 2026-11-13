"""Descriptive (non-AI) sensor analysis for any validated trajectory data.

Used when the remaining-life model does not apply. Everything is computed from the active dataset only:
  * per-sensor drift = (mean of the latest records - mean of the asset's own early records) measured in units of
    the typical early-life noise of that sensor across the dataset;
  * a simple status per asset (Stable / Drift detected / Insufficient history).
No remaining-life, no health model, no invented thresholds beyond the user-configurable drift trigger."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

MIN_RECORDS = C.MIN_HISTORY_CYCLES
SENSOR_STATUSES = ["Drift detected", "Stable", "Insufficient history"]


def window(n: int) -> int:
    """Records used for the early baseline and for the latest window (same size for both)."""
    return int(max(5, min(20, n // 5)))


def noise_reference(df: pd.DataFrame, sensors: list[str]) -> dict:
    """Typical early-life noise per sensor: median over assets of the std of each asset's baseline window."""
    stds: dict[str, list[float]] = {s: [] for s in sensors}
    for _, g in df.sort_values([C.ID_COL, C.CYCLE_COL]).groupby(C.ID_COL):
        if len(g) < MIN_RECORDS:
            continue
        k = window(len(g))
        for s in sensors:
            v = g[s].iloc[:k].to_numpy(dtype=float)
            if len(v) > 1 and np.isfinite(v).all():
                stds[s].append(float(np.std(v, ddof=1)))
    ref = {}
    for s, v in stds.items():
        v = [x for x in v if x > 0]
        ref[s] = float(np.median(v)) if v else float("nan")
    return ref


def shift_table(g: pd.DataFrame, sensors: list[str], noise_ref: dict, labels: dict | None = None) -> pd.DataFrame:
    """Per-sensor drift for one asset (same columns as anomaly_detection.sensor_shifts)."""
    g = g.sort_values(C.CYCLE_COL)
    n = len(g)
    k = window(n)
    rows = []
    for s in sensors:
        base = float(g[s].iloc[:k].mean())
        recent = float(g[s].iloc[-k:].mean())
        sig = noise_ref.get(s, float("nan"))
        z = (recent - base) / sig if (np.isfinite(sig) and sig > 0) else float("nan")
        rho = g[[C.CYCLE_COL, s]].corr(method="spearman").iloc[0, 1] if n > 3 and g[s].nunique() > 1 else float("nan")
        rows.append({"sensor": s, "label": (labels or {}).get(s, str(s)), "baseline_mean": base, "recent_mean": recent,
                     "shift": recent - base, "shift_sigma": z, "trend_rho": rho})
    out = pd.DataFrame(rows)
    out["abs_sigma"] = out["shift_sigma"].abs()
    return out.sort_values("abs_sigma", ascending=False, na_position="last").reset_index(drop=True)


def asset_table(df: pd.DataFrame, sensors: list[str], sigma_trigger: float, labels: dict | None = None):
    """One row per asset + the per-asset shift tables. Returns (table, {unit: shift_df})."""
    nref = noise_reference(df, sensors)
    rows, shifts = [], {}
    for uid, g in df.sort_values([C.ID_COL, C.CYCLE_COL]).groupby(C.ID_COL):
        n = len(g)
        row = {C.ID_COL: int(uid), "last_cycle": int(g[C.CYCLE_COL].iloc[-1]), "n_cycles": int(n)}
        if n < MIN_RECORDS:
            row.update({"state": "Insufficient history", "max_shift_sigma": float("nan"), "top_sensor": "—"})
            shifts[int(uid)] = pd.DataFrame()
        else:
            t = shift_table(g, sensors, nref, labels)
            shifts[int(uid)] = t
            top = t.iloc[0] if len(t) and np.isfinite(t["abs_sigma"].iloc[0]) else None
            mx = float(top["abs_sigma"]) if top is not None else float("nan")
            row.update({"state": "Drift detected" if (top is not None and mx >= sigma_trigger) else "Stable",
                        "max_shift_sigma": mx, "top_sensor": top["label"] if top is not None else "—"})
        rows.append(row)
    return pd.DataFrame(rows), shifts
