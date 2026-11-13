"""Industrial Maintenance Intelligence Platform - Streamlit entry point.

Run:  streamlit run app.py        (set MAINT_DEV_MODE=1 to expose internal model-development pages)
"""
import streamlit as st

from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import ui_components as U

st.set_page_config(page_title=C.PRODUCT_SHORT, page_icon=":material/precision_manufacturing:",
                   layout="wide", initial_sidebar_state="expanded")
S.init_state()
U.inject_css()

PAGES = [
    st.Page("pages/home.py", title="Control Room", icon=":material/dashboard:", default=True),
    st.Page("pages/fleet_overview.py", title="Fleet Overview", icon=":material/groups:"),
    st.Page("pages/equipment.py", title="Equipment Management", icon=":material/precision_manufacturing:"),
    st.Page("pages/maintenance.py", title="Maintenance Management", icon=":material/engineering:"),
    st.Page("pages/sensor_analysis.py", title="Sensor & Degradation Analysis", icon=":material/show_chart:"),
    st.Page("pages/recommendations.py", title="Maintenance Recommendations", icon=":material/build:"),
    st.Page("pages/reports.py", title="Reports & Export", icon=":material/description:"),
    st.Page("pages/data_source.py", title="Data Source", icon=":material/upload_file:"),
    st.Page("pages/settings.py", title="Settings", icon=":material/settings:"),
]
if DS.dev_mode():
    PAGES += [st.Page("pages/dev/engine_intelligence.py", title="[Dev] Engine Intelligence", icon=":material/monitor_heart:"),
              st.Page("pages/dev/explainable_ai.py", title="[Dev] Explainable AI", icon=":material/psychology:"),
              st.Page("pages/dev/what_if_analysis.py", title="[Dev] What-if", icon=":material/tune:"),
              st.Page("pages/dev/model_performance.py", title="[Dev] Model Performance", icon=":material/analytics:")]
S.render_sidebar()
st.navigation(PAGES).run()
