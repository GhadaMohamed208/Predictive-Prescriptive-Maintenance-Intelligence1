import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import storage as DB
from src import ui_components as U

S.setup_page(need_dataset=False)
ss = st.session_state
U.page_header("Maintenance Management", "Create and follow maintenance records for registered equipment. Records are stored permanently; "
              "completed records are retained.", "MAINTENANCE")
eq = DB.list_equipment()
if eq.empty:
    U.note("No equipment is registered yet. Maintenance records must belong to registered equipment.")
    if st.button("Open Equipment Management", type="primary"):
        st.switch_page("pages/equipment.py")
    st.stop()
asset_opts = eq["asset_id"].tolist()
label = lambda a: f"{a} · {eq.set_index('asset_id').loc[a, 'name']}"
if ss.get("mm_flash"):
    st.success(ss.pop("mm_flash"))

with st.expander("Create maintenance record", expanded=False):
    with st.form("mm_new"):
        c1, c2, c3 = st.columns(3)
        a = c1.selectbox("Related equipment (Asset ID) *", asset_opts, format_func=label, key="mmn_asset")
        typ = c2.selectbox("Maintenance type *", C.MAINT_TYPES, key="mmn_type")
        pr = c3.selectbox("Priority *", C.MAINT_PRIORITIES, index=1, key="mmn_prio")
        issue = st.text_input("Reported issue *", key="mmn_issue")
        desc = st.text_area("Description", key="mmn_desc", height=70)
        act = st.text_area("Recommended action", key="mmn_act", height=70)
        c1, c2 = st.columns(2)
        tech = c1.text_input("Assigned technician", key="mmn_tech")
        plan = c2.date_input("Planned date (optional)", value=None, key="mmn_plan")
        notes = st.text_area("Additional notes", key="mmn_notes", height=60)
        ok = st.form_submit_button("Create record", type="primary")
    if ok:
        try:
            rid = DB.create_maintenance({"asset_id": a, "type": typ, "priority": pr, "reported_issue": issue, "description": desc, "recommended_action": act,
                                         "technician": tech, "planned_date": plan.isoformat() if plan else None, "notes": notes})
            ss["mm_flash"] = f"Maintenance record {DB.mr_label(rid)} created for {a}."
            st.rerun()
        except DB.StorageError as ex:
            st.error(str(ex))

cur = ss.get("mm_asset")
f1, f2, f3, f4 = st.columns([1.6, 1.4, 1.4, 1.4])
fa = f1.selectbox("Equipment", ["All equipment"] + asset_opts, index=(asset_opts.index(cur) + 1) if cur in asset_opts else 0,
                  format_func=lambda x: x if x == "All equipment" else label(x), key="mm_fa")
ss["mm_asset"] = None if fa == "All equipment" else fa
fs = f2.multiselect("Status", C.MAINT_STATUSES, default=C.MAINT_STATUSES, key="mm_fs")
fp = f3.multiselect("Priority", C.MAINT_PRIORITIES, default=C.MAINT_PRIORITIES, key="mm_fp")
q = f4.text_input("Search issue / technician", "", key="mm_q")
mr = DB.list_maintenance(None if fa == "All equipment" else fa)
if mr.empty:
    U.note("No maintenance records" + ("" if fa == "All equipment" else f" for {fa}") + " yet.")
    st.stop()
flt = mr[mr["status"].isin(fs) & mr["priority"].isin(fp)]
if q.strip():
    m = flt["reported_issue"].str.contains(q.strip(), case=False, regex=False) | flt["technician"].fillna("").str.contains(q.strip(), case=False, regex=False)
    flt = flt[m]
cnt = mr["status"].value_counts()
k = st.columns(4)
for c, s in zip(k, C.MAINT_STATUSES):
    c.markdown(U.kpi(s, str(int(cnt.get(s, 0))), "records"), unsafe_allow_html=True)
st.write("")
st.caption(f"{len(flt)} of {len(mr)} records")
cols = ["record", "asset_id", "asset_name", "type", "reported_issue", "priority", "status", "technician", "planned_date", "completion_date", "source"]
ev = st.dataframe(flt[cols], hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row", key="mm_table")
st.download_button("Export shown records (CSV)", flt.drop(columns=["id"]).to_csv(index=False).encode(), "maintenance_records.csv", "text/csv")
if flt.empty:
    st.stop()
rids = flt["id"].tolist()
rows = ev.selection.rows if ev is not None else []
if rows:
    ss["mm_selected"] = int(rids[rows[0]])
cur_r = ss.get("mm_selected")
rid = st.selectbox("Open record", rids, index=rids.index(cur_r) if cur_r in rids else 0, format_func=DB.mr_label, key="mm_open")
ss["mm_selected"] = int(rid)
r = DB.get_maintenance(rid)
st.markdown(f"### {DB.mr_label(rid)} · {r['asset_id']}")
st.markdown(U.chip(r["status"], "#22d3ee") + U.chip(r["priority"] + " priority", U.PRIORITY_COLORS[r["priority"]]) + U.chip(r["type"]) + U.chip("From recommendation" if r["source"] == "Recommendation" else "Manual"), unsafe_allow_html=True)
a, b = st.columns(2)
a.markdown(f"**Reported issue:** {r['reported_issue']}  \n**Description:** {r['description'] or '—'}  \n**Recommended action:** {r['recommended_action'] or '—'}")
b.markdown(f"**Created:** {r['created_date']}  \n**Planned:** {r['planned_date'] or '—'}  \n**Completed:** {r['completion_date'] or '—'}  \n**Technician:** {r['technician'] or '—'}")
if r["source"] == "Recommendation":
    st.caption(f"Created from finding {r['source_ref']} (dataset {r['dataset_ref']}).")
st.markdown(f"**Work performed:** {r['work_performed'] or '—'}  \n**Notes:** {r['notes'] or '—'}")
with st.form(f"mm_upd_{rid}"):
    st.markdown("**Update record**")
    c1, c2, c3 = st.columns(3)
    trans = DB.allowed_transitions(r["status"])
    ns = c1.selectbox("Status", trans, key=f"mmu_status_{rid}")
    tech = c2.text_input("Assigned technician", r["technician"] or "", key=f"mmu_tech_{rid}")
    cd = c3.date_input("Completion date (when completing)", value=None, key=f"mmu_cd_{rid}")
    wp = st.text_area("Work performed", r["work_performed"] or "", key=f"mmu_wp_{rid}", height=70)
    nt = st.text_area("Notes", r["notes"] or "", key=f"mmu_nt_{rid}", height=70)
    save = st.form_submit_button("Save update", type="primary")
if save:
    try:
        ch = {"status": ns, "technician": tech, "work_performed": wp, "notes": nt}
        if ns == "Completed" and cd:
            ch["completion_date"] = cd.isoformat()
        DB.update_maintenance(rid, ch)
        ss["mm_flash"] = f"{DB.mr_label(rid)} updated."
        st.rerun()
    except DB.StorageError as ex:
        st.error(str(ex))
ev_t = DB.maintenance_events(rid)
with st.expander("Record history"):
    st.dataframe(ev_t, hide_index=True, width="stretch")
