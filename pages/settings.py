import json

import streamlit as st

from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import ui_components as U

p = S.setup_page(need_dataset=False)
U.page_header("Settings", "Alert settings, session control and limitations.", "SETTINGS")
st.markdown("#### Condition levels and sensor-drift trigger")
c1, c2, c3 = st.columns(3)
w = c1.number_input("Warning level (cycles of remaining life)", 1.0, float(p.cap), float(st.session_state["thr_warning"]), 1.0)
cr = c2.number_input("Critical level (cycles of remaining life)", 0.0, float(p.cap), float(st.session_state["thr_critical"]), 1.0)
sg = c3.number_input("Sensor-drift trigger (x normal noise)", 1.0, 10.0, float(st.session_state["sigma_trigger"]), 0.5)
if cr >= w:
    st.error("The critical level must be lower than the warning level.")
else:
    st.session_state.update({"thr_warning": w, "thr_critical": cr, "sigma_trigger": sg})
    st.caption(f"Healthy: remaining life > {w:.0f} · Warning: {cr:.0f}–{w:.0f} · Critical: < {cr:.0f} cycles. Remaining-life levels apply only to "
               "AI-supported datasets. These are configurable rules, not industry standards.")
if st.button("Reset to defaults"):
    st.session_state.update({"thr_warning": C.DEFAULT_THRESHOLDS["warning"], "thr_critical": C.DEFAULT_THRESHOLDS["critical"], "sigma_trigger": 3.0})
    st.rerun()

st.markdown("#### Session")
rec = S.active()
st.caption("Active dataset: " + (f"{rec['label']} ({DS.source_text(rec)})" if rec else "none"))
if st.button("Clear session", type="primary", key="set_clear"):
    S.clear_session()
    st.rerun()
st.caption("Clearing removes the loaded dataset and file information, cached analyses, recommendations and action statuses, generated reports and settings. Stored equipment and maintenance records are NOT removed.")

st.markdown("#### Limitations")
for t in ("AI-supported analysis exists only for NASA C-MAPSS-compatible aircraft turbofan engine data (simulated). Other equipment types may be registered "
          "and analysed with sensor statistics, but no remaining-life prediction is made for them.",
          "Remaining-life estimates carry error and have an upper limit; they are not failure dates.",
          "A sensor drift or unusual-behaviour flag is an observation. It does not prove a physical fault or identify a component.",
          "Uploaded and built-in data are historical records, not live telemetry.",
          "Action statuses are kept for this browser session only (export CSV to keep them)."):
    st.markdown("- " + t)
U.note(C.DISCLAIMER)
if DS.dev_mode():
    st.markdown("#### Developer information (MAINT_DEV_MODE)")
    st.json({k: v for k, v in p.metadata.items() if k not in ("feature_names", "feature_config")}, expanded=False)
