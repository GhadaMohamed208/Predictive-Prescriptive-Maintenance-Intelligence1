"""Leakage-safe, per-engine, causal feature engineering.

Every feature at cycle t is computed ONLY from observations with cycle <= t of the same
engine (rolling windows look backward; the baseline uses the engine's first N cycles but
is expanding for t < N, so no future row is used).

Feature naming: <sensor>__<kind>, kind in {cur, rmean, rstd, slope, dev0}.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class FeatureConfig:
    sensors: list[str] = field(default_factory=lambda: list(C.MODEL_SENSORS))
    window: int = C.ROLL_WINDOW
    baseline_cycles: int = C.BASELINE_CYCLES
    use_current: bool = True
    use_rolling_mean: bool = True
    use_rolling_std: bool = True
    use_slope: bool = True
    use_baseline_dev: bool = True
    use_cycle: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**d)


def feature_names(cfg: FeatureConfig) -> list[str]:
    names: list[str] = []
    if cfg.use_cycle:
        names.append("age_cycles")
    for s in cfg.sensors:
        if cfg.use_current:
            names.append(f"{s}__cur")
        if cfg.use_rolling_mean:
            names.append(f"{s}__rmean")
        if cfg.use_rolling_std:
            names.append(f"{s}__rstd")
        if cfg.use_slope:
            names.append(f"{s}__slope")
        if cfg.use_baseline_dev:
            names.append(f"{s}__dev0")
    return names


def feature_label(name: str, window: int = C.ROLL_WINDOW) -> str:
    if name == "age_cycles":
        return "Operating cycle"
    s, kind = name.split("__")
    lab = C.sensor_label(s)
    return {
        "cur": f"{lab} (current)",
        "rmean": f"{lab} (rolling mean, {window})",
        "rstd": f"{lab} (rolling std, {window})",
        "slope": f"{lab} (rolling slope, {window})",
        "dev0": f"{lab} (shift vs. engine baseline)",
    }[kind]


def feature_sensor(name: str) -> str | None:
    return None if name == "age_cycles" else name.split("__")[0]


def _window_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing window sum with window truncated at the series start (axis 0)."""
    cs = np.cumsum(a, axis=0)
    out = cs.copy()
    if len(a) > w:
        out[w:] = cs[w:] - cs[:-w]
    return out


def _engine_features(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    X = g[cfg.sensors].to_numpy(dtype=float)
    t = g[C.CYCLE_COL].to_numpy(dtype=float)
    n = len(g)
    w = cfg.window
    cnt = np.minimum(np.arange(1, n + 1), w).astype(float)[:, None]
    sx = _window_sum(X, w)
    sxx = _window_sum(X * X, w)
    # centre time to keep the slope numerically stable
    tc = (t - t[0])[:, None]
    st = _window_sum(tc, w)
    stt = _window_sum(tc * tc, w)
    sxt = _window_sum(X * tc, w)
    mean = sx / cnt
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (sxx - cnt * mean ** 2) / (cnt - 1)
        var = np.where(cnt > 1, np.maximum(var, 0.0), 0.0)  # warm-up (1 obs): std = 0
        den = cnt * stt - st ** 2
        slope = np.where((cnt > 1) & (den > 0), (cnt * sxt - st * sx) / den, 0.0)  # warm-up: 0
    std = np.sqrt(var)
    # expanding baseline over the first `baseline_cycles` rows (past-only)
    b = cfg.baseline_cycles
    cum_mean = np.cumsum(X, axis=0) / np.arange(1, n + 1)[:, None]
    base = cum_mean.copy()
    if n > b:
        base[b:] = cum_mean[b - 1]
    cols: dict[str, np.ndarray] = {}
    if cfg.use_cycle:
        cols["age_cycles"] = t
    for j, s in enumerate(cfg.sensors):
        if cfg.use_current:
            cols[f"{s}__cur"] = X[:, j]
        if cfg.use_rolling_mean:
            cols[f"{s}__rmean"] = mean[:, j]
        if cfg.use_rolling_std:
            cols[f"{s}__rstd"] = std[:, j]
        if cfg.use_slope:
            cols[f"{s}__slope"] = slope[:, j]
        if cfg.use_baseline_dev:
            cols[f"{s}__dev0"] = X[:, j] - base[:, j]
    return pd.DataFrame(cols, index=g.index)


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a frame [unit, cycle, <features...>] sorted by (unit, cycle)."""
    cfg = cfg or FeatureConfig()
    missing = [c for c in [C.ID_COL, C.CYCLE_COL] + cfg.sensors if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    d = df.sort_values([C.ID_COL, C.CYCLE_COL], kind="mergesort")
    parts = [_engine_features(g, cfg) for _, g in d.groupby(C.ID_COL, sort=True)]
    feats = pd.concat(parts) if parts else pd.DataFrame(columns=feature_names(cfg))
    feats = feats[feature_names(cfg)]  # enforce training feature order
    out = pd.concat([d[[C.ID_COL, C.CYCLE_COL]], feats], axis=1)
    return out
