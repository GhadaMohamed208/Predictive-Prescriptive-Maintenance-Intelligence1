"""Dashboard panel for STORED records (equipment registry + maintenance). Kept visually and semantically separate from the
dataset-derived KPIs: registered equipment is not the same thing as assets monitored in the active dataset."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from . import config as C
from . import storage as DB
from . import ui_components as U

STATUS_COLORS = {"Operational": "#34d399", "Standby": "#3987e5", "Degraded": "#f59e0b", "Under Maintenance": "#9085e9", "Out of Service": "#ef4444"}


def registry_panel() -> None:
    try:
        z = DB.summary()
    except Exception as e:  # database unavailable: say so, never show invented numbers
        U.note(f"Stored equipment and maintenance records are unavailable: {e}", warn=True)
        return
    st.markdown("<div class='panel'><h4>Equipment registry &amp; maintenance (stored records)</h4>"
                "<span style='color:#8da0bd;font-size:.82rem'>Counts come from the registry and maintenance database. They are separate from the "
                "assets monitored in the active dataset above.</span></div>", unsafe_allow_html=True)
    if z["equipment_total"] == 0 and z["maintenance_total"] == 0:
        U.note("No equipment has been registered yet. Open Equipment Management to register equipment manually or import a CSV/Excel file.")
        if st.button("Open Equipment Management", key="home_goto_eq"):
            st.switch_page("pages/equipment.py")
        return
    m = z["maintenance_by_status"]
    k = st.columns(6)
    k[0].markdown(U.kpi("Registered equipment", str(z["equipment_total"])), unsafe_allow_html=True)
    k[1].markdown(U.kpi("Requiring attention", str(z["equipment_attention"]), note="degraded / under maintenance / out of service, or open high-priority work", cls="Warning"), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Pending", str(m["Pending"]), note="maintenance records"), unsafe_allow_html=True)
    k[3].markdown(U.kpi("In progress", str(m["In Progress"]), note="maintenance records"), unsafe_allow_html=True)
    k[4].markdown(U.kpi("Completed", str(m["Completed"]), note="maintenance records", cls="Healthy"), unsafe_allow_html=True)
    k[5].markdown(U.kpi("Critical open actions", str(z["critical_open"]), note="critical, pending or in progress", cls="Critical"), unsafe_allow_html=True)
    ec = z["equipment_by_status"]
    fig = go.Figure(go.Bar(x=list(ec), y=list(ec.values()), marker_color=[STATUS_COLORS[s] for s in ec], text=list(ec.values()), textposition="outside", width=.5))
    fig.update_yaxes(title="Equipment")
    st.plotly_chart(U.style_fig(fig, "Registered equipment by operational status", 260, False), width="stretch")
