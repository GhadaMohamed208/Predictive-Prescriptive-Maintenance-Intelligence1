"""RUL target construction.

Train: RUL = final_cycle(engine) - cycle  (>= 0).
Test : only one ground-truth RUL per engine, valid at the engine's last observed cycle.
The optional cap (default 125) is a modelling convention; raw and capped targets are
always kept in separate columns so they cannot be mixed silently."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def add_rul_train(df: pd.DataFrame, cap: float | None = C.RUL_CAP) -> pd.DataFrame:
    out = df.copy()
    last = out.groupby(C.ID_COL)[C.CYCLE_COL].transform("max")
    out["rul_raw"] = (last - out[C.CYCLE_COL]).astype(float)
    if (out["rul_raw"] < 0).any():  # pragma: no cover - guarded by construction
        raise ValueError("Negative RUL encountered.")
    out["rul"] = out["rul_raw"] if cap is None else out["rul_raw"].clip(upper=cap)
    return out


def final_cycle_targets(test: pd.DataFrame, rul_true: pd.Series,
                            cap: float | None = C.RUL_CAP) -> pd.DataFrame:
    """One row per test engine: last observed cycle with the official true RUL attached."""
    last = test.sort_values([C.ID_COL, C.CYCLE_COL]).groupby(C.ID_COL).tail(1).set_index(C.ID_COL)
    ids = last.index.to_numpy()
    if set(ids) != set(rul_true.index):
        raise ValueError("Test engine ids do not align with RUL file entries.")
    res = pd.DataFrame({
        C.CYCLE_COL: last[C.CYCLE_COL].to_numpy(),
        "rul_true_raw": rul_true.loc[ids].to_numpy(),
    }, index=ids)
    res["rul_true"] = res["rul_true_raw"] if cap is None else res["rul_true_raw"].clip(upper=cap)
    res.index.name = C.ID_COL
    return res.reset_index()
