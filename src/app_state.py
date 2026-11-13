"""Streamlit state & cached resources shared by all pages (models are loaded once)."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from . import config as C
from . import explainability as EX
from . import fleet_analysis as FA
from . import dataset_state as DS
from .data_loader import load_fd001
from .inference import ArtifactsMissingError, Predictor, artifacts_available

SRC_TEST = "Official test set (FD001)"
SRC_TRAIN = "Training set (run-to-failure)"
SRC_UPLOAD = "Uploaded data"


@st.cache_resource(show_spinner=False)
def get_predictor() -> Predictor:
    return Predictor.load()


@st.cache_resource(show_spinner=False)
def get_anomaly_ref():
    return joblib.load(C.ANOMALY_PATH) if C.ANOMALY_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def get_explainer():
    p = get_predictor()
    g = EX.load_global(C.SHAP_GLOBAL_PATH)
    try:
        bg = g["X"].sample(min(200, len(g["X"])), random_state=C.RANDOM_SEED) if g else None
        expl, kind = EX.make_explainer(p.model, background=bg)
        return expl
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_global_shap():
    return EX.load_global(C.SHAP_GLOBAL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    return load_fd001()


@st.cache_data(show_spinner=False)
def get_artifacts() -> dict:
    out: dict = {}
    out["metrics"] = json.loads(C.METRICS_PATH.read_text()) if C.METRICS_PATH.exists() else None
    out["comparison"] = pd.read_csv(C.COMPARISON_PATH) if C.COMPARISON_PATH.exists() else None
    out["ablation"] = pd.read_csv(C.ABLATION_PATH) if C.ABLATION_PATH.exists() else None
    out["oof"] = pd.read_csv(C.OOF_PATH) if C.OOF_PATH.exists() else None
    out["test_pred"] = pd.read_csv(C.TEST_PRED_PATH) if C.TEST_PRED_PATH.exists() else None
    out["validation_log"] = json.loads(C.VALIDATION_LOG_PATH.read_text()) if C.VALIDATION_LOG_PATH.exists() else None
    return out


def require_artifacts() -> bool:
    """Render a setup message and return False when the model has not been trained."""
    if artifacts_available():
        return True
    st.error("Trained model artifacts were not found.")
    st.markdown("Train and evaluate the models first (takes a few minutes):")
    st.code("python -m src.train_models", language="bash")
    st.caption("No placeholder model or metrics are shipped; the app only shows results from real runs.")
    return False


# ----------------------------------------------------------------- settings / state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("thr_warning", C.DEFAULT_THRESHOLDS["warning"])
    ss.setdefault("thr_critical", C.DEFAULT_THRESHOLDS["critical"])
    ss.setdefault("sigma_trigger", 3.0)
    ss.setdefault("rec_status", {})        # recommendation id -> status (cleared whenever the dataset changes)


def thresholds() -> dict:
    return {"warning": float(st.session_state["thr_warning"]), "critical": float(st.session_state["thr_critical"])}


# ----------------------------------------------------------------- active dataset (see dataset_state.py)
def active() -> dict | None:
    return DS.get_active(st.session_state)


def activate_builtin(kind: str = DS.KIND_BUILTIN) -> dict:
    """Load a built-in NASA dataset on explicit user request and make it the active dataset."""
    train, test, rul = load_data()
    p = get_predictor()
    if kind == DS.KIND_BUILTIN_TRAIN:
        rec = DS.make_record(kind, "NASA C-MAPSS FD001 - training set", train, capability=DS.CAP_AI, sensors=p.sensors)
    else:
        rec = DS.make_record(kind, "NASA C-MAPSS FD001 - demo fleet", test, capability=DS.CAP_AI, sensors=p.sensors,
                             rul_true=rul)
    DS.activate(st.session_state, rec)
    return rec


def activate_upload(assessment: dict, filename: str, rul_true=None) -> dict:
    sensors = assessment["sensors"]
    rec = DS.make_record(DS.KIND_UPLOAD, f"Uploaded: {filename}", assessment["df"], capability=assessment["capability"],
                         sensors=sensors, source_file=filename, rul_true=rul_true,
                         input_mode=assessment["mode"], report=assessment["report"].to_dict() if assessment["report"] else {},
                         asset_names=assessment["asset_names"], sensor_labels=assessment["sensor_labels"],
                         notes=assessment["notes"], has_timestamp=assessment["has_timestamp"])
    DS.activate(st.session_state, rec)
    return rec


def clear_session() -> None:
    DS.clear_session(st.session_state)
    _view_cached.clear()


def require_dataset(need_ai: bool = False) -> dict:
    """Return the active dataset record or render the 'choose a data source' panel and stop the page.

    There is deliberately NO fallback to the built-in data."""
    rec = active()
    if rec is None:
        from . import ui_components as U
        U.page_header("No dataset loaded", "Choose a data source to begin. Nothing is loaded automatically.", "DATA SOURCE")
        st.markdown("<div class='panel'><h4>Select a data source</h4>The built-in NASA C-MAPSS dataset is a simulated fleet of aircraft "
                    "turbofan engines for demonstration. Alternatively upload your own sensor data.</div>", unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.4, 1.4, 3])
        if c1.button("Load built-in NASA demo dataset", type="primary", width="stretch", key="rq_builtin"):
            activate_builtin()
            st.rerun()
        if c2.button("Upload your own data", width="stretch", key="rq_upload"):
            st.switch_page("pages/data_source.py")
        st.stop()
    if need_ai and rec["capability"] != DS.CAP_AI:
        from . import ui_components as U
        U.page_header("Not available for this dataset", "", "UNAVAILABLE")
        U.note("This view needs remaining-life estimates, which are only available for datasets supported by the implemented "
               "model (NASA C-MAPSS-compatible turbofan engine trajectories). The active dataset is analysed with sensor "
               "statistics only. Open Sensor & Degradation Analysis for the analyses that are available.", warn=True)
        st.stop()
    return rec


def dataset(source: str | None = None):
    """Compatibility accessor: (label, df, rul_true) of the ACTIVE dataset (stops the page if none)."""
    rec = require_dataset()
    return rec["label"], rec["df"], rec.get("rul_true")


@st.cache_data(show_spinner=False)
def _view_cached(token: str, warn: float, crit: float, sigma: float, _record: dict):
    from . import analysis_service as AS
    p = get_predictor() if _record["capability"] == DS.CAP_AI else None
    return AS.fleet_view(_record, p, get_anomaly_ref() if p else None, {"warning": warn, "critical": crit}, sigma)


def view(record: dict | None = None) -> dict:
    """Fleet table + findings of the active dataset (cached per dataset token and settings)."""
    record = record or require_dataset()
    t = thresholds()
    return _view_cached(record["token"], t["warning"], t["critical"], float(st.session_state["sigma_trigger"]), record)


def fleet(source: str | None = None) -> pd.DataFrame:
    return view()["table"]


def current_rec_status(rec_id: str, default: str = "Pending") -> str:
    return st.session_state["rec_status"].get(rec_id, default)


def render_sidebar() -> None:
    """Sidebar identity + the ACTIVE DATASET card (read-only view of the central state) + Clear session."""
    rec = active()
    with st.sidebar:
        st.markdown("<div style='font:700 .95rem ui-monospace,monospace;letter-spacing:1.5px;color:#22d3ee'>"
                    "◈ MAINTENANCE INTELLIGENCE</div><div style='font-size:.72rem;color:#8da0bd;margin:2px 0 6px'>"
                    "Condition monitoring &amp; maintenance decision support</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("ACTIVE DATASET")
        if rec is None:
            st.markdown("<div class='note'>No dataset loaded. Open <b>Data Source</b> to load the built-in NASA demo or upload your own data.</div>",
                        unsafe_allow_html=True)
        else:
            import html
            cls = "ds-up" if rec["kind"] == DS.KIND_UPLOAD else "ds-nasa"
            st.markdown(
                f"<div class='dscard {cls}'><div class='b'>{DS.source_badge(rec)}</div>"
                f"<div class='n'>{html.escape(rec['source_file'] or rec['label'])}</div>"
                f"<div class='m'>{rec['n_assets']} assets · {rec['n_rows']:,} records · historical data</div>"
                f"<div class='m'>{'AI-supported analysis' if rec['capability'] == DS.CAP_AI else 'Sensor analysis only'}</div></div>",
                unsafe_allow_html=True)
        if st.button("Clear session", key="sb_clear", width="stretch",
                     help="Removes the loaded dataset, results, action statuses and settings. Stored equipment and maintenance records are kept."):
            clear_session()
            st.rerun()


def setup_page(need_dataset: bool = True, need_ai: bool = False):
    """Common prelude for every page. Returns Predictor (model artifacts are required by the app)."""
    init_state()
    if not require_artifacts():
        st.stop()
    if need_dataset:
        require_dataset(need_ai)
    return get_predictor()


def goto_engine(engine_id: int, page: str = "pages/sensor_analysis.py") -> None:
    st.session_state["selected_asset"] = int(engine_id)
    st.session_state["selected_engine"] = int(engine_id)
    st.switch_page(page)
