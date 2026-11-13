"""Upload assessment: turns an arbitrary tabular file into a validated, clearly-classified dataset.

No Streamlit here (unit-testable). The outcome is explicit:

* ``ok=False``  -> the file is rejected with specific reasons; nothing becomes active.
* capability ``ai``          -> the data match the equipment/sensor schema the implemented model supports.
* capability ``sensor_only`` -> valid trajectories, analysed with descriptive statistics only; the reasons why
  the remaining-life model does not apply are listed. Nothing is ever scored with the wrong model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import dataset_state as DS
from . import upload_mapping as UM
from .data_validation import ValidationReport, validate_trajectories

SUPPORTED_EXTENSIONS = (".csv", ".txt", ".tsv", ".xlsx")
RESERVED = {C.ID_COL, C.CYCLE_COL, "timestamp"}


def check_extension(filename: str) -> str | None:
    """Return an error message when the file type is not supported."""
    low = (filename or "").lower()
    if not low.endswith(SUPPORTED_EXTENSIONS):
        return (f"'{filename}' is not a supported file type. Supported formats: CSV, TSV, whitespace-separated TXT "
                "(including the NASA C-MAPSS layout) and Excel (.xlsx).")
    return None


def candidate_sensor_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Columns that look like numeric measurements (>= 80 % of their values parse as numbers)."""
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if len(df) and v.notna().mean() >= 0.8:
            out.append(c)
    return out


def prepare_generic(original: pd.DataFrame, unit_col: str, time_col: str, sensor_cols: list[str]):
    """Return (df, asset_names, notes, has_timestamp) with the internal column names (unit, cycle, sensors...)."""
    original = original.reset_index(drop=True)
    notes: list[str] = []
    bad = [c for c in sensor_cols if c in RESERVED]
    if bad:
        raise ValueError("Measurement column(s) use reserved names (" + ", ".join(bad) + "). Rename them in the file.")
    if unit_col in sensor_cols or time_col in sensor_cols or unit_col == time_col:
        raise ValueError("The asset-id column, the time/cycle column and the measurements must be different columns.")
    if not sensor_cols:
        raise ValueError("Select at least one measurement column.")
    out = pd.DataFrame(index=original.index)
    # asset identifier: whole numbers are kept, anything else (e.g. 'PUMP-01') is mapped to codes
    raw_id = original[unit_col]
    num_id = pd.to_numeric(raw_id, errors="coerce")
    asset_names: dict[int, str] = {}
    if raw_id.isna().any():
        raise ValueError(f"The asset-id column '{unit_col}' contains {int(raw_id.isna().sum())} empty value(s).")
    if num_id.notna().all() and (num_id % 1 == 0).all() and (num_id >= 0).all():
        out[C.ID_COL] = num_id.astype(int)
    else:
        names = sorted(raw_id.astype(str).str.strip().unique())
        code = {n: i + 1 for i, n in enumerate(names)}
        out[C.ID_COL] = raw_id.astype(str).str.strip().map(code).astype(int)
        asset_names = {i: n for n, i in code.items()}
        notes.append(f"Asset identifiers are text; {len(names)} assets were numbered internally and keep their original names in the app.")
    # time axis: numeric cycle/time, or timestamps converted to a per-asset sequence
    t_raw = original[time_col]
    is_dt = pd.api.types.is_datetime64_any_dtype(t_raw)
    t_num = pd.to_numeric(t_raw, errors="coerce") if not is_dt else pd.Series(np.nan, index=t_raw.index)
    has_ts = False
    if t_num.notna().all():
        out[C.CYCLE_COL] = t_num
    else:
        ts = pd.to_datetime(t_raw, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"The time column '{time_col}' is neither numeric nor a recognisable timestamp "
                             f"({int(ts.isna().sum())} unreadable value(s)).")
        has_ts = True
        out["timestamp"] = ts
        dup = int(pd.DataFrame({"i": out[C.ID_COL], "t": ts}).duplicated().sum())
        if dup:
            raise ValueError(f"{dup} duplicate record(s): the same asset has more than one row for the same timestamp.")
        out[C.CYCLE_COL] = out.assign(_id=out[C.ID_COL]).groupby("_id")["timestamp"].rank(method="first").astype(int)
        notes.append("Timestamps were converted to a per-asset record sequence (1, 2, 3, ...); original timestamps are kept.")
    n_coerced = 0
    for c in sensor_cols:
        v = pd.to_numeric(original[c], errors="coerce")
        n_coerced += int(v.isna().sum() - original[c].isna().sum())
        out[c] = v
    if n_coerced:
        notes.append(f"{n_coerced} non-numeric measurement value(s) could not be read and are treated as missing (reported below).")
    return out, asset_names, notes, has_ts


def assess_upload(original: pd.DataFrame, filename: str, unit_col: str, time_col: str, sensor_cols: list[str],
                  model_sensor_map: dict[str, str | None], reference: dict | None, has_truth: bool = False,
                  model_sensors: list[str] | None = None) -> dict:
    """Validate and classify an upload. Never raises for bad data: reasons are returned."""
    model_sensors = model_sensors or C.MODEL_SENSORS
    res = {"ok": False, "errors": [], "warnings": [], "notes": [], "capability": None, "mode": None, "df": None,
           "asset_names": {}, "report": None, "why_not_ai": [], "sensors": [], "sensor_labels": {},
           "unused_columns": [], "has_timestamp": False, "comp": None}
    try:
        gdf, names, notes, has_ts = prepare_generic(original, unit_col, time_col, sensor_cols)
    except ValueError as e:
        res["errors"].append(str(e))
        return res
    res["notes"] += notes
    report = validate_trajectories(gdf, sensors=list(sensor_cols))
    res["report"] = report
    if not report.ok:
        res["errors"] += report.errors
        return res
    res["warnings"] += report.warnings
    used = {unit_col, time_col, *sensor_cols}
    res["unused_columns"] = [c for c in original.columns if c not in used]

    # ---- can the supported (C-MAPSS turbofan) model be applied?
    missing = [s for s in model_sensors if not model_sensor_map.get(s)]
    if missing:
        res["why_not_ai"].append("The remaining-life model needs the 14 aircraft-engine sensors (Sensor 2, 3, 4, 7, 8, 9, 11, 12, 13, "
                                 "14, 15, 17, 20, 21); not found: " + ", ".join(C.sensor_label(s) for s in missing) + ".")
    elif has_ts or names:
        res["why_not_ai"].append("The remaining-life model needs numeric engine numbers and numeric operating cycles.")
    else:
        try:
            mdf = UM.apply_mapping(original, unit_col, time_col, {k: v for k, v in model_sensor_map.items() if v})
            mrep = validate_trajectories(mdf, sensors=list(model_sensors))
            if not mrep.ok:
                res["why_not_ai"] += ["Model-sensor validation failed: " + "; ".join(mrep.errors)]
            else:
                comp = UM.compatibility(mdf, reference, list(model_sensors)) if reference else None
                res["comp"] = comp
                if comp and comp["unsupported"]:
                    res["why_not_ai"].append(
                        f"The values are far outside the range of the data the model supports (median sensor: "
                        f"{comp['median_fraction_outside']*100:.0f}% outside). This is probably a different machine type, scaling or units.")
                else:
                    res.update({"capability": DS.CAP_AI, "mode": "B" if has_truth else "A", "df": mdf,
                                "sensors": list(model_sensors), "report": mrep, "ok": True})
                    res["warnings"] = list(mrep.warnings)
                    if comp and comp["warn_sensors"]:
                        res["warnings"].append("More than 20% of the values are outside the supported range for: "
                                               + ", ".join(C.sensor_label(s) for s in comp["warn_sensors"]) + ". Check units/scaling.")
                    return res
        except ValueError as e:
            res["why_not_ai"].append(str(e))
    # ---- sensor-only analysis
    res.update({"capability": DS.CAP_SENSOR, "mode": "S", "df": gdf, "asset_names": names, "sensors": list(sensor_cols),
                "sensor_labels": {c: str(c) for c in sensor_cols}, "has_timestamp": has_ts, "ok": True})
    return res
