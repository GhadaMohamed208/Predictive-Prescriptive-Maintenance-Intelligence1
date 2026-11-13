"""Data-quality validation for trajectories (training files and uploads).

Validation never silently repairs data: missing/infinite values and duplicate engine-cycle
records are reported as errors; nothing is imputed with zero."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        return "FAILED" if self.errors else ("PASSED WITH WARNINGS" if self.warnings else "PASSED")

    def to_frame(self) -> pd.DataFrame:
        rows = [("Error", m) for m in self.errors] + [("Warning", m) for m in self.warnings] \
            + [("Info", m) for m in self.info]
        return pd.DataFrame(rows, columns=["Severity", "Message"])

    def to_dict(self) -> dict:
        return {"status": self.status, "errors": self.errors, "warnings": self.warnings,
                "info": self.info, "stats": self.stats}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


def validate_trajectories(df: pd.DataFrame, sensors: list[str] | None = None,
                          id_col: str = C.ID_COL, cycle_col: str = C.CYCLE_COL,
                          min_history: int = C.MIN_HISTORY_CYCLES,
                          require_contiguous: bool = False, expected_columns: list[str] | None = None
                          ) -> ValidationReport:
    sensors = list(sensors if sensors is not None else C.MODEL_SENSORS)
    r = ValidationReport()
    if df is None or len(df) == 0:
        r.errors.append("The dataset contains no rows.")
        return r
    miss = [c for c in [id_col, cycle_col] + sensors if c not in df.columns]
    if miss:
        r.errors.append("Missing required column(s): " + ", ".join(miss) + ".")
        return r
    if expected_columns is not None:
        extra = [c for c in df.columns if c not in expected_columns]
        if extra:
            r.warnings.append("Unexpected column(s) present (reported, not silently ignored): "
                              + ", ".join(map(str, extra)))
    # numeric types
    for c in [id_col, cycle_col] + sensors:
        if not pd.api.types.is_numeric_dtype(df[c]):
            r.errors.append(f"Column '{c}' is not numeric (found {df[c].dtype}).")
    if r.errors:
        return r
    # missing / infinite
    sub = df[[id_col, cycle_col] + sensors]
    n_na = int(sub.isna().sum().sum())
    if n_na:
        bad = sub.isna().sum()
        bad = bad[bad > 0]
        r.errors.append(f"{n_na} missing value(s) in: " + ", ".join(f"{k} ({v})" for k, v in bad.items())
                        + ". Missing measurements are never replaced by zero; fix or remove them.")
    n_inf = int(np.isinf(sub.to_numpy(dtype=float)).sum())
    if n_inf:
        r.errors.append(f"{n_inf} infinite value(s) found in required columns.")
    if r.errors:
        r.stats["rows"] = int(len(df))
        return r
    # identifiers & cycles
    if (df[id_col] % 1 != 0).any() or (df[id_col] < 0).any():
        r.errors.append("Engine identifiers must be non-negative whole numbers.")
    if (df[cycle_col] % 1 != 0).any() or (df[cycle_col] < 1).any():
        r.errors.append("Cycle numbers must be whole numbers >= 1.")
    dup = int(df.duplicated(subset=[id_col, cycle_col]).sum())
    if dup:
        r.errors.append(f"{dup} duplicate engine-cycle record(s) found.")
    full_dup = int(df.duplicated().sum())
    if full_dup and not dup:
        r.warnings.append(f"{full_dup} fully duplicated row(s).")
    if r.errors:
        r.stats["rows"] = int(len(df))
        return r
    # order & contiguity
    d = df.sort_values([id_col, cycle_col], kind="mergesort")
    unordered = int((df.reset_index(drop=True)[[id_col, cycle_col]]
                     .ne(d.reset_index(drop=True)[[id_col, cycle_col]]).any(axis=1)).sum())
    if unordered:
        r.warnings.append("Rows were not sorted by (engine, cycle); they will be sorted for analysis.")
    g = d.groupby(id_col)[cycle_col]
    gaps = (g.diff().dropna() > 1).groupby(d.loc[g.diff().dropna().index, id_col]).any()
    n_gap = int(gaps.sum())
    if n_gap:
        msg = f"{n_gap} engine(s) have gaps in their cycle sequence."
        (r.errors if require_contiguous else r.warnings).append(msg)
    starts = g.min()
    if (starts > 1).any():
        r.warnings.append(f"{int((starts > 1).sum())} engine(s) do not start at cycle 1; rolling "
                          "baselines then refer to the first supplied cycles.")
    lens = g.count()
    short = lens[lens < min_history]
    if len(short):
        r.warnings.append(f"{len(short)} engine(s) have fewer than {min_history} cycles of history; "
                          "rolling features use a documented warm-up and predictions are less reliable.")
    const = [s for s in sensors if d[s].nunique() <= 1]
    if const:
        r.warnings.append("Constant sensor(s) in this file: " + ", ".join(const))
    r.stats.update({
        "rows": int(len(df)), "engines": int(d[id_col].nunique()),
        "cycles_per_engine_min": int(lens.min()), "cycles_per_engine_median": float(lens.median()),
        "cycles_per_engine_max": int(lens.max()), "duplicates": dup, "missing_values": n_na,
        "infinite_values": n_inf, "engines_with_short_history": int(len(short)),
        "engines_with_cycle_gaps": n_gap,
    })
    r.info.append(f"{r.stats['rows']:,} rows, {r.stats['engines']} engine(s); cycles per engine "
                  f"{r.stats['cycles_per_engine_min']}–{r.stats['cycles_per_engine_max']}.")
    return r


def validate_rul_alignment(train_rul: pd.DataFrame | None = None, test_ids=None, rul=None) -> list[str]:
    msgs: list[str] = []
    if train_rul is not None and (train_rul["rul_raw"] < 0).any():
        msgs.append("Negative RUL in training targets.")
    if test_ids is not None and rul is not None and set(test_ids) != set(rul.index):
        msgs.append("Test engine ids do not align with RUL entries.")
    return msgs
