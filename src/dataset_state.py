"""Central dataset / session management (single source of truth).

Design rules
------------
* Exactly ONE dataset is "active" for the whole application. It lives in a plain (non-widget)
  session key, so Streamlit's widget-state clean-up on page change can never drop it.
* "No dataset" is an explicit state (``get_active() is None``). Nothing ever falls back to the
  built-in NASA data silently; pages must ask the user to choose a source.
* Every active dataset carries a content-based ``token``. All caches, generated reports and
  stored results are keyed by that token, so a dataset change invalidates every dependent result.
* All functions work on any ``MutableMapping`` (``st.session_state`` in the app, a ``dict`` in tests).
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import MutableMapping

import pandas as pd

from . import config as C

KEY_ACTIVE = "ds_active"
KEY_NONCE = "ds_nonce"          # bumps on every clear -> fresh file-uploader widget keys

KIND_BUILTIN = "builtin_nasa"
KIND_BUILTIN_TRAIN = "builtin_nasa_train"   # development mode only
KIND_UPLOAD = "upload"

CAP_AI = "ai"                   # equipment type supported by the implemented, validated model
CAP_SENSOR = "sensor_only"      # valid trajectories, analysed statistically (no RUL / health model)

# session keys that hold results derived from the active dataset
DEPENDENT_KEYS = ("analysis", "scenarios", "last_scenario", "rec_status", "pdf_asset", "pdf_fleet",
                  "analysis_done", "selected_engine", "selected_asset", "action_log")
# widget-key prefixes of dataset-dependent controls (their options change with the dataset)
DEPENDENT_PREFIXES = ("deg_", "ei_", "fleet_", "rep_", "rec_", "home_", "sa_", "wi_", "xai_", "eda_", "st_")


def dev_mode() -> bool:
    """Internal model-development pages are visible only when MAINT_DEV_MODE=1."""
    return os.environ.get("MAINT_DEV_MODE", "").strip() in ("1", "true", "True", "yes")


# ------------------------------------------------------------------ records
def _token(kind: str, name: str, df: pd.DataFrame, nonce: int | str = "") -> str:
    h = int(pd.util.hash_pandas_object(df, index=False).sum() % (2 ** 61))
    return hashlib.sha1(f"{kind}|{name}|{len(df)}|{df.shape[1]}|{h}|{nonce}".encode()).hexdigest()[:12]


def make_record(kind: str, label: str, df: pd.DataFrame, *, capability: str, sensors: list[str],
                source_file: str | None = None, rul_true: pd.Series | None = None,
                input_mode: str = "", report: dict | None = None, asset_names: dict | None = None,
                sensor_labels: dict | None = None, notes: list[str] | None = None,
                has_timestamp: bool = False) -> dict:
    """Build the immutable description of a dataset that is about to become active."""
    if kind not in (KIND_BUILTIN, KIND_BUILTIN_TRAIN, KIND_UPLOAD):
        raise ValueError(f"unknown dataset kind {kind!r}")
    if capability not in (CAP_AI, CAP_SENSOR):
        raise ValueError(f"unknown capability {capability!r}")
    df = df.sort_values([C.ID_COL, C.CYCLE_COL]).reset_index(drop=True)
    return {
        "token": _token(kind, label, df), "kind": kind, "label": label, "source_file": source_file,
        "df": df, "rul_true": rul_true, "capability": capability, "sensors": list(sensors),
        "input_mode": input_mode, "report": report or {}, "asset_names": asset_names or {},
        "sensor_labels": sensor_labels or {}, "notes": notes or [], "has_timestamp": has_timestamp,
        "n_assets": int(df[C.ID_COL].nunique()), "n_rows": int(len(df)),
    }


# ------------------------------------------------------------------ state operations
def get_active(ss: MutableMapping) -> dict | None:
    return ss.get(KEY_ACTIVE)


def active_token(ss: MutableMapping) -> str | None:
    a = get_active(ss)
    return a["token"] if a else None


def invalidate_dependents(ss: MutableMapping) -> list[str]:
    """Remove every result/selection that was derived from a previous dataset."""
    gone = []
    for k in list(ss.keys()):
        if k in DEPENDENT_KEYS or any(str(k).startswith(p) for p in DEPENDENT_PREFIXES):
            del ss[k]
            gone.append(k)
    return gone


def activate(ss: MutableMapping, record: dict) -> None:
    """Make ``record`` the dataset of the whole app and discard everything derived from the old one."""
    invalidate_dependents(ss)
    ss[KEY_ACTIVE] = record
    ss["rec_status"] = {}


def uploader_key(ss: MutableMapping, base: str) -> str:
    return f"{base}_{ss.get(KEY_NONCE, 0)}"


def clear_session(ss: MutableMapping) -> int:
    """Reset the application to its clean initial state (no dataset, default settings).

    Returns the number of keys removed. File-uploader widgets are re-keyed via a nonce because a
    Streamlit uploader cannot be emptied programmatically."""
    nonce = int(ss.get(KEY_NONCE, 0)) + 1
    keys = list(ss.keys())
    for k in keys:
        del ss[k]
    ss[KEY_NONCE] = nonce
    return len(keys)


# ------------------------------------------------------------------ helpers shared by pages
def source_badge(record: dict | None) -> str:
    if record is None:
        return "NO DATASET"
    return {KIND_BUILTIN: "BUILT-IN NASA DEMO", KIND_BUILTIN_TRAIN: "BUILT-IN NASA (TRAINING)",
            KIND_UPLOAD: "UPLOADED DATA"}[record["kind"]]


def source_text(record: dict | None) -> str:
    if record is None:
        return "No dataset"
    return {KIND_BUILTIN: "Built-in NASA demo data", KIND_BUILTIN_TRAIN: "Built-in NASA training data", KIND_UPLOAD: "Uploaded data"}[record["kind"]]


def input_mode_label(record: dict) -> str:
    """Human-readable input mode used in the UI and in reports (never a hard-coded default)."""
    if record["kind"] == KIND_BUILTIN:
        return "Built-in NASA C-MAPSS FD001 demo data (aircraft turbofan engines, AI-supported)"
    if record["kind"] == KIND_BUILTIN_TRAIN:
        return "Built-in NASA C-MAPSS FD001 training data (run-to-failure, development use)"
    base = "Uploaded file"
    if record["capability"] == CAP_AI:
        return f"{base}: C-MAPSS-compatible engine trajectories (AI-supported" + \
               (", with reference RUL values)" if record.get("rul_true") is not None else ")")
    return f"{base}: sensor measurements per asset (statistical analysis only; no RUL model applies)"


def capability_text(record: dict) -> str:
    if record["capability"] == CAP_AI:
        return "AI-supported: remaining-life estimate, anomaly detection and recommendations"
    return "Sensor analysis only: trends, statistics and sensor-drift findings (no remaining-life model for this data)"


def asset_name(record: dict | None, uid) -> str:
    """Original asset identifier as supplied by the user (falls back to the numeric id)."""
    if record:
        n = record.get("asset_names", {}).get(int(uid))
        if n is not None:
            return str(n)
    return str(int(uid))


def sensor_label(record: dict | None, col: str) -> str:
    if record:
        lab = record.get("sensor_labels", {}).get(col)
        if lab:
            return lab
    return C.sensor_label(col)
