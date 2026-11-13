"""Column inference, mapping and compatibility classification for uploaded data."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .data_validation import ValidationReport, validate_trajectories

ID_NAMES = {"unit", "unitnumber", "unitnr", "unitid", "engine", "engineid", "enginenumber", "enginenr", "id", "machine", "machineid", "asset", "assetid", "assetname", "equipment", "equipmentid", "equipmentname", "device", "deviceid", "pump", "motor", "tag"}
CYCLE_NAMES = {"cycle", "cycles", "time", "timecycles", "timeincycles", "timestep", "t", "opcycle", "operatingcycle", "timestamp", "datetime", "date", "recordtime", "record", "recordid", "readingtime"}
_SENSOR_RE = re.compile(r"^(?:s|sensor|sensormeasurement|sm|meas|measurement)0*(\d{1,2})$")


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def infer_columns(df: pd.DataFrame) -> dict:
    """Best-effort auto-mapping. Returns {'unit': col|None, 'cycle': col|None, 'sensors': {sK: col|None}}"""
    out = {"unit": None, "cycle": None, "sensors": {s: None for s in C.MODEL_SENSORS}}
    for c in df.columns:
        n = _norm(c)
        if out["unit"] is None and n in ID_NAMES:
            out["unit"] = c
        elif out["cycle"] is None and n in CYCLE_NAMES:
            out["cycle"] = c
        else:
            m = _SENSOR_RE.match(n)
            if m:
                key = f"s{int(m.group(1))}"
                if key in out["sensors"] and out["sensors"][key] is None:
                    out["sensors"][key] = c
    return out


def apply_mapping(df: pd.DataFrame, unit: str, cycle: str, sensor_map: dict[str, str]) -> pd.DataFrame:
    """Return a NEW frame using the model's standard column names (original df untouched)."""
    cols = {unit: C.ID_COL, cycle: C.CYCLE_COL}
    cols.update({v: k for k, v in sensor_map.items() if v})
    if len(set(cols)) != len(cols):
        raise ValueError("A source column was mapped to more than one target.")
    out = df[list(cols)].rename(columns=cols).copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce") if out[c].dtype == object else out[c]
    return out


def compatibility(df_mapped: pd.DataFrame, reference: dict, sensors: list[str]) -> dict:
    """Mode classification. Fraction of values outside the training min/max per sensor tells whether
    units/equipment are plausibly the same. Does not repair anything."""
    frac = {}
    for s in sensors:
        r = reference["sensors"][s]
        v = df_mapped[s].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        frac[s] = float(np.mean((v < r["min"]) | (v > r["max"]))) if len(v) else 1.0
    med = float(np.median(list(frac.values()))) if frac else 1.0
    worst = sorted(frac.items(), key=lambda kv: -kv[1])[:5]
    unsupported = med > 0.5
    return {"fraction_outside": frac, "median_fraction_outside": med, "unsupported": unsupported,
            "warn_sensors": [s for s, f in frac.items() if f > 0.2], "worst": worst}


MODE_C_HELP = (
    "This data cannot be scored reliably by the C-MAPSS FD001 model. To analyse a different machine type, sensor "
    "configuration, unit convention or sampling interval you would need: (1) a documented mapping of your sensors to "
    "the model's 14 input sensors, or (2) domain-specific training data with run-to-failure (or labelled degradation) "
    "histories, and (3) re-running the training pipeline (python -m src.train_models) with your data and RUL targets.")


def classify_mode(report: ValidationReport, comp: dict | None, missing_sensors: list[str], has_truth: bool) -> tuple[str, str]:
    if missing_sensors:
        return "C", ("Unsupported schema: required model sensors are not mapped: "
                     + ", ".join(C.sensor_label(s) for s in missing_sensors) + ". " + MODE_C_HELP)
    if not report.ok:
        return "invalid", "Validation errors must be fixed before inference."
    if comp and comp["unsupported"]:
        return "C", (f"Unsupported equipment/units: the median sensor has {comp['median_fraction_outside']*100:.0f}% of values "
                     "outside the range of the training data, so this is probably a different machine, scaling or unit system. "
                     + MODE_C_HELP)
    return ("B" if has_truth else "A"), ""


def example_template(test_df: pd.DataFrame, engines=(1, 2), max_rows=60) -> pd.DataFrame:
    parts = [test_df[test_df[C.ID_COL] == e].sort_values(C.CYCLE_COL).tail(max_rows) for e in engines]
    cols = [C.ID_COL, C.CYCLE_COL] + C.SENSOR_COLS
    return pd.concat(parts)[cols].reset_index(drop=True)
