"""Fleet-level table and attention ranking for the engines in the loaded dataset.

For FD001 the "fleet" is the set of simulated engines in the supplied files; it is not a
real operating fleet. Ranking criteria (documented, no invented costs):
  1. Derived state (Critical > Warning > Healthy)
  2. Lower predicted RUL first
  3. Persistent anomaly flag (recent flag fraction >= 0.5) breaks ties upward
  4. Data-quality / OOD issues are shown as a column (they reduce trust, they do not change rank)."""
from __future__ import annotations

import pandas as pd

from . import anomaly_detection as AD
from . import config as C
from .inference import Predictor

STATE_RANK = {"Critical": 0, "Warning": 1, "Healthy": 2}


def fleet_table(predictor: Predictor, df: pd.DataFrame, thresholds: dict, anomaly_ref: dict | None = None,
                rul_true: pd.Series | None = None) -> pd.DataFrame:
    res, last_feat = predictor.predict_latest(df, thresholds)
    res = res.copy()
    cyc = df.groupby(C.ID_COL)[C.CYCLE_COL].agg(["count"]).rename(columns={"count": "n_cycles"})
    res = res.merge(cyc, left_on=C.ID_COL, right_index=True, how="left")
    # OOD fraction per engine
    ood = []
    for i in range(len(last_feat)):
        o = predictor.ood_report(last_feat.iloc[[i]])
        ood.append(o.get("fraction", float("nan")))
    res["ood_fraction"] = ood
    if anomaly_ref is not None:
        sc = AD.score(anomaly_ref, df)
        summ = {u: AD.engine_summary(g) for u, g in sc.groupby(C.ID_COL)}
        res["anomaly_recent_fraction"] = res[C.ID_COL].map(lambda u: summ[u]["recent_flag_fraction"])
        res["anomaly_persistent"] = res["anomaly_recent_fraction"] >= 0.5
    else:
        res["anomaly_recent_fraction"] = float("nan")
        res["anomaly_persistent"] = False
    if rul_true is not None:
        res["rul_true_official"] = res[C.ID_COL].map(rul_true)
    res["_rank_state"] = res["state"].map(STATE_RANK)
    res = res.sort_values(["_rank_state", "rul_pred", "anomaly_persistent"], ascending=[True, True, False])
    res["attention_rank"] = range(1, len(res) + 1)
    return res.drop(columns="_rank_state").reset_index(drop=True)


def state_counts(table: pd.DataFrame) -> dict:
    vc = table["state"].value_counts()
    return {s: int(vc.get(s, 0)) for s in C.STATES}
