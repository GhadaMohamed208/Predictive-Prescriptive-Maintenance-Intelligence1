"""What-if analysis: hypothetical edits to the LATEST observed cycle of one engine.

Original data is never modified (works on a copy). Only the sensors the user explicitly
selects are changed. The rolling/baseline features are then recomputed causally, so the
edited last row also influences that engine's rolling statistics. A changed prediction
describes the MODEL's response, not a physical or causal effect on a real machine."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .evaluate_models import risk_state
from .inference import Predictor

SCENARIO_WARNING = (
    "Hypothetical scenario: editing a sensor value is not the same as physically changing a machine "
    "component. The result shows how this statistical model responds to the edited input; it is not "
    "evidence that an intervention would extend real-world life, and it does not establish causation.")


def sensor_bounds(predictor: Predictor) -> dict:
    """Hard bounds = min/max of the training data; 'typical' = 1st-99th percentile."""
    ref = predictor.reference["sensors"]
    return {s: {"min": ref[s]["min"], "max": ref[s]["max"], "p01": ref[s]["p01"], "p99": ref[s]["p99"]}
            for s in predictor.sensors}


def validate_changes(changes: dict, bounds: dict, baseline: dict) -> dict:
    """Return {'errors': [...], 'warnings': [...]}. Errors block the run."""
    errors, warnings = [], []
    if not changes:
        errors.append("No sensor changes selected.")
    for s, v in changes.items():
        if s not in bounds:
            errors.append(f"{C.sensor_label(s)} is not a supported modelling sensor.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{C.sensor_label(s)}: value is not a number.")
            continue
        if v != v or v in (float("inf"), float("-inf")):
            errors.append(f"{C.sensor_label(s)}: value must be finite.")
            continue
        b = bounds[s]
        if v < b["min"] or v > b["max"]:
            errors.append(f"{C.sensor_label(s)}: {v:g} is outside the range observed in training data "
                          f"[{b['min']:g}, {b['max']:g}]; the model cannot be trusted there.")
        elif v < b["p01"] or v > b["p99"]:
            warnings.append(f"{C.sensor_label(s)}: {v:g} is in the extreme tail of the training data "
                            f"(outside the 1st–99th percentile [{b['p01']:g}, {b['p99']:g}]).")
        if s in baseline and baseline[s] != 0 and abs(v - baseline[s]) > 0.5 * (b["max"] - b["min"]):
            warnings.append(f"{C.sensor_label(s)}: change exceeds half of the training range in one step; "
                            "such a jump is physically unlikely between consecutive cycles.")
    return {"errors": errors, "warnings": warnings}


def run_scenario(predictor: Predictor, engine_df: pd.DataFrame, changes: dict, thresholds: dict) -> dict:
    d = engine_df.sort_values(C.CYCLE_COL).copy()  # copy: the observed data stay untouched
    base_row = d.iloc[-1]
    baseline_vals = {s: float(base_row[s]) for s in predictor.sensors}
    check = validate_changes(changes, sensor_bounds(predictor), baseline_vals)
    res = {"errors": check["errors"], "warnings": check["warnings"] + [SCENARIO_WARNING]}
    if check["errors"]:
        return res
    sim = d.copy()
    for s, v in changes.items():
        sim.loc[sim.index[-1], s] = float(v)
    base_pred, base_feat = predictor.predict_latest(d, thresholds)
    sc_pred, sc_feat = predictor.predict_latest(sim, thresholds)
    ood = predictor.ood_report(sc_feat)
    if ood.get("flag"):
        res["warnings"].append(f"Scenario features fall outside the training range ({ood['n_outside']} features).")
    b = float(base_pred["rul_pred"].iloc[0]); s_ = float(sc_pred["rul_pred"].iloc[0])
    res.update({
        "baseline_rul": b, "scenario_rul": s_, "delta": s_ - b,
        "baseline_state": risk_state(b, thresholds), "scenario_state": risk_state(s_, thresholds),
        "changed": pd.DataFrame([{"sensor": C.sensor_label(s), "baseline": baseline_vals[s],
                                  "scenario": float(v), "change": float(v) - baseline_vals[s]}
                                 for s, v in changes.items()]),
        "engine_id": int(base_row[C.ID_COL]), "cycle": int(base_row[C.CYCLE_COL]),
        "scenario_at_cap": bool(sc_pred["at_cap"].iloc[0]), "baseline_at_cap": bool(base_pred["at_cap"].iloc[0]),
    })
    return res
