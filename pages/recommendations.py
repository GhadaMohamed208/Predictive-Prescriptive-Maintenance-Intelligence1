import html

import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import recommendation_engine as RE
from src import storage as DB
from src import ui_components as U

p = S.setup_page()
rec = S.active()
V = S.view(rec)
F = V["findings"].copy()
rs = st.session_state["rec_status"]
U.page_header("Maintenance Recommendations", f"Findings and suggested actions for {rec['source_file'] or rec['label']} "
              f"({DS.source_text(rec)}). Each finding states what was observed, why, and what to verify.", "RECOMMENDATIONS")
if rec["capability"] != DS.CAP_AI:
    U.note("Findings for this dataset come from sensor drift only. Remaining-life based findings are not available because no "
           "supported remaining-life model applies to this data.")
if F.empty:
    U.note("No findings were raised for this dataset.")
    st.stop()
# ---- optional: turn a finding into a maintenance record for registered equipment
mk = st.session_state.get("rec_make")
if mk is not None and mk in set(F["id"]):
    r = F[F["id"] == mk].iloc[0]
    st.markdown(f"##### Create maintenance record from finding {mk}")
    eqs = DB.list_equipment()
    if eqs.empty:
        U.note("No equipment is registered. Register the equipment first; the record must belong to a registered asset.", warn=True)
        if st.button("Open Equipment Management"):
            st.switch_page("pages/equipment.py")
    else:
        ids_ = eqs["asset_id"].tolist()
        guess = DB.find_equipment_for_dataset_asset(rec["token"], r["name"])
        dup = DB.find_record_by_source(mk, rec["token"])
        if dup:
            st.warning(f"A maintenance record for this finding already exists: {DB.mr_label(dup['id'])} ({dup['status']}).")
        with st.form("rec_make_form"):
            c1, c2, c3 = st.columns(3)
            a = c1.selectbox("Registered equipment *", ids_, index=ids_.index(guess) if guess in ids_ else 0, key="rec_mk_asset",
                             format_func=lambda x: f"{x} · {eqs.set_index('asset_id').loc[x, 'name']}")
            pr = c2.selectbox("Priority", C.MAINT_PRIORITIES, index=C.MAINT_PRIORITIES.index(r["priority"] if r["priority"] in C.MAINT_PRIORITIES else "Low"), key="rec_mk_prio")
            tech = c3.text_input("Assigned technician", key="rec_mk_tech")
            note_ = st.text_area("Additional notes", key="rec_mk_notes", height=60)
            st.caption(f"Dataset asset '{r['name']}' will be linked to the chosen equipment. Issue, evidence and recommended action are copied from the finding.")
            ok = st.form_submit_button("Create maintenance record", type="primary", disabled=bool(dup))
        if ok:
            try:
                rid = DB.create_maintenance({"asset_id": a, "type": "Predictive", "priority": pr, "reported_issue": r["condition"],
                                             "description": r["evidence"], "recommended_action": r["action"], "technician": tech, "notes": note_,
                                             "source": "Recommendation", "source_ref": mk, "dataset_ref": rec["token"]})
                DB.link_dataset(a, rec, r["name"])
                st.session_state["rec_status"][mk] = "In Progress"
                st.session_state.pop("rec_make", None)
                st.session_state["rec_flash"] = f"Maintenance record {DB.mr_label(rid)} created for {a} from finding {mk}."
                st.rerun()
            except DB.StorageError as ex:
                st.error(str(ex))
        if st.button("Cancel"):
            st.session_state.pop("rec_make", None)
            st.rerun()
    st.markdown("---")
if st.session_state.get("rec_flash"):
    st.success(st.session_state.pop("rec_flash"))
F["status"] = F["id"].map(lambda i: rs.get(i, "Pending"))
F["Asset"] = F["name"]

c1, c2, c3 = st.columns([1.4, 1.4, 1.6])
prio = c1.multiselect("Priority", ["Critical", "High", "Medium", "Low", "Info"], default=["Critical", "High", "Medium"], key="rec_prio")
stat = c2.multiselect("Action status", RE.STATUSES, default=["Pending", "In Progress"], key="rec_stat")
q = c3.text_input("Search asset", "", key="rec_q")
V2 = F[F["priority"].isin(prio) & F["status"].isin(stat)]
if q.strip():
    V2 = V2[V2["Asset"].astype(str).str.contains(q.strip(), case=False, regex=False)]
cnt = F["priority"].value_counts()
k = st.columns(5)
for c, pr in zip(k, ["Critical", "High", "Medium", "Low", "Info"]):
    c.markdown(U.kpi(pr, str(int(cnt.get(pr, 0))), "findings"), unsafe_allow_html=True)
st.write("")
if V2.empty:
    U.note("No findings match the current filters.")
    st.stop()

st.markdown(f"##### Action register ({len(V2)} shown) — update the status inline")
view = V2[["id", "Asset", "priority", "condition", "finding_class", "status"]].copy()
edited = st.data_editor(view, hide_index=True, width="stretch", height=340, key="rec_editor",
                        disabled=[c for c in view.columns if c != "status"],
                        column_config={"id": "ID", "priority": "Priority", "condition": "Detected condition", "finding_class": "Finding type",
                                       "status": st.column_config.SelectboxColumn("Action status", options=RE.STATUSES, required=True)})
for _, r in edited.iterrows():
    rs[r["id"]] = r["status"]
exp = V2.copy()
exp["status"] = exp["id"].map(lambda i: rs.get(i, "Pending"))
exp["sensors"] = exp["sensors"].map("; ".join)
st.download_button("Export action register (CSV)", exp[["id", "Asset", "priority", "condition", "finding_class", "evidence", "action",
                                                         "verification", "status", "sensors"]].to_csv(index=False).encode(),
                   "maintenance_recommendations.csv", "text/csv")

st.markdown("##### Finding details")
for _, r in V2.head(40).iterrows():
    stt = rs.get(r["id"], "Pending")
    with st.expander(f"{r['priority']} · {r['Asset']} · {r['condition']}  [{stt}]", expanded=False):
        pc = U.PRIORITY_COLORS[r["priority"]]
        ev = ""
        if r["sensor_evidence"]:
            ev = "<br>".join(f"• {html.escape(str(x['label']))} {x['direction']} by {abs(x['shift_sigma']):.1f}× its normal early noise"
                             + (f" (from {x['baseline']:.4g} to {x['recent']:.4g})" if x.get("baseline") is not None else "")
                             for x in r["sensor_evidence"])
        else:
            ev = "No individual sensor is singled out for this finding."
        st.markdown(
            "<div class='fc'>"
            f"{U.chip(r['priority'] + ' priority', pc)}{U.chip(r['finding_class'])}"
            f"<div class='k'>Detected condition</div><div class='x'>{html.escape(r['condition'])}</div>"
            f"<div class='k'>Why it was raised</div><div class='x'>{html.escape(r['evidence'])}</div>"
            f"<div class='k'>Sensor evidence</div><div class='x'>{ev}</div>"
            f"<div class='k'>Recommended action</div><div class='x'>{html.escape(r['action'])}</div>"
            f"<div class='k'>How to verify</div><div class='x'>{html.escape(r['verification'])}</div>"
            f"<div class='k'>Confirmed fault?</div><div class='x'>{html.escape(r['confirmation'])}</div>"
            f"<div class='k'>Limitation</div><div class='x'>{html.escape(r['limitation'])}</div>"
            f"<div class='k'>Current action status</div><div class='x'>{html.escape(stt)}</div></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1.4, 3])
        if b1.button("Open asset trends", key=f"rec_open_{r['id']}"):
            S.goto_engine(r["engine_id"])
        if r["priority"] != "Info" and b2.button("Create maintenance record", key=f"rec_mkbtn_{r['id']}"):
            st.session_state["rec_make"] = r["id"]
            st.rerun()
with st.expander("What the finding types mean"):
    for k_, v_ in RE.FINDING_CLASS_HELP.items():
        st.markdown(f"**{k_}** — {v_}")
    U.note("Only a physical inspection can confirm a fault; this system never does.")
