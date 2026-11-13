import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import storage as DB
from src import ui_components as U

S.setup_page(need_dataset=False)
ss = st.session_state
U.page_header("Equipment Management", "Register equipment, import lists, and keep one record per asset with its sensor data, analyses, reports "
              "and maintenance history.", "EQUIPMENT")
U.note(f"Records are stored in a database file ({DB.db_path().name}) and survive app restarts. Registering equipment does not mean an AI "
       "model exists for it: AI support is shown per record. On Streamlit Community Cloud the file system is temporary across redeploys, so "
       "export your lists regularly.")
tab_r, tab_n, tab_i = st.tabs(["Registry", "Register equipment", "Import CSV / Excel"])


def eq_form(prefix: str, e: dict | None = None):
    e = e or {}
    c1, c2, c3 = st.columns(3)
    asset_id = c1.text_input("Asset ID *", e.get("asset_id") or "", key=f"{prefix}_id", disabled=bool(e))
    name = c2.text_input("Equipment name *", e.get("name") or "", key=f"{prefix}_name")
    typ = c3.selectbox("Equipment type *", C.EQUIPMENT_TYPES, index=C.EQUIPMENT_TYPES.index(e["type"]) if e.get("type") in C.EQUIPMENT_TYPES else 0, key=f"{prefix}_type")
    c1, c2, c3 = st.columns(3)
    man = c1.text_input("Manufacturer", e.get("manufacturer") or "", key=f"{prefix}_man")
    mod = c2.text_input("Model", e.get("model") or "", key=f"{prefix}_mod")
    ser = c3.text_input("Serial number (optional)", e.get("serial_number") or "", key=f"{prefix}_ser")
    c1, c2, c3 = st.columns(3)
    has_d = bool(e.get("installation_date"))
    inst = c1.date_input("Installation date", value=pd.to_datetime(e["installation_date"]).date() if has_d else None, key=f"{prefix}_inst",
                         min_value=pd.Timestamp("1900-01-01").date())
    loc = c2.text_input("Location / facility", e.get("location") or "", key=f"{prefix}_loc")
    dep = c3.text_input("Department / production area (optional)", e.get("department") or "", key=f"{prefix}_dep")
    c1, c2 = st.columns([1, 2])
    stt = c1.selectbox("Operational status *", C.EQUIPMENT_STATUSES, index=C.EQUIPMENT_STATUSES.index(e.get("status", "Operational")), key=f"{prefix}_stat")
    notes = c2.text_area("Notes", e.get("notes") or "", key=f"{prefix}_notes", height=80)
    return {"asset_id": asset_id, "name": name, "type": typ, "manufacturer": man, "model": mod, "serial_number": ser,
            "installation_date": inst.isoformat() if inst else None, "location": loc, "department": dep, "status": stt, "notes": notes}


# ------------------------------------------------------------------ register
with tab_n:
    with st.form("eq_new_form", clear_on_submit=False):
        data = eq_form("eqn")
        go_ = st.form_submit_button("Register equipment", type="primary")
    if go_:
        try:
            e = DB.add_equipment(data)
            ss["eq_flash"] = f"Equipment {e['asset_id']} registered."
            ss["eq_selected"] = e["asset_id"]
            st.rerun()
        except DB.StorageError as ex:
            st.error(str(ex))
if ss.get("eq_flash"):
    st.success(ss.pop("eq_flash"))

# ------------------------------------------------------------------ import
with tab_i:
    st.markdown("Upload a CSV or Excel (.xlsx) list. Every row is checked; invalid rows are listed with the reason and **are not imported**. "
                "Required columns: Asset ID, Equipment Name, Type. Allowed types: " + ", ".join(C.EQUIPMENT_TYPES) + ".")
    tpl = DB.import_template()
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Template (CSV)", tpl.to_csv(index=False).encode(), "equipment_template.csv", "text/csv")
    import io
    buf = io.BytesIO(); tpl.to_excel(buf, index=False)
    c2.download_button("Template (Excel)", buf.getvalue(), "equipment_template.xlsx")
    f = st.file_uploader("Equipment list", type=["csv", "xlsx"], key=DS.uploader_key(ss, "eq_import"))
    if f is not None:
        try:
            raw = f.getvalue()
            if len(raw) > C.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"File exceeds {C.MAX_UPLOAD_MB} MB.")
            df = pd.read_excel(io.BytesIO(raw)) if f.name.lower().endswith(".xlsx") else pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
            if df.empty:
                raise ValueError("The file has no data rows.")
        except Exception as ex:
            st.error(f"Could not read '{f.name}': {ex}")
            st.stop()
        res = DB.parse_equipment_table(df)
        if res["missing_columns"]:
            st.error("Required column(s) not found: " + ", ".join(res["missing_columns"]) + ". Nothing can be imported from this file.")
            st.caption("Columns found: " + ", ".join(map(str, df.columns)))
        else:
            k = st.columns(3)
            k[0].markdown(U.kpi("Rows in file", str(len(df))), unsafe_allow_html=True)
            k[1].markdown(U.kpi("Valid", str(len(res["valid"])), cls="Healthy"), unsafe_allow_html=True)
            k[2].markdown(U.kpi("Rejected", str(len(res["rejected"])), cls="Critical" if len(res["rejected"]) else ""), unsafe_allow_html=True)
            if res["unmapped"]:
                st.caption("Columns ignored (not recognised): " + ", ".join(map(str, res["unmapped"])))
            if len(res["rejected"]):
                st.markdown("**Rejected rows** (row number counts the header as row 1)")
                st.dataframe(res["rejected"], hide_index=True, width="stretch")
                st.download_button("Download rejected rows (CSV)", res["rejected"].to_csv(index=False).encode(), "rejected_rows.csv", "text/csv")
            if res["valid"]:
                st.markdown("**Rows that will be imported**")
                st.dataframe(pd.DataFrame(res["valid"])[["asset_id", "name", "type", "location", "status"]], hide_index=True, width="stretch")
                if st.button(f"Import {len(res['valid'])} valid row(s)", type="primary"):
                    try:
                        n = DB.import_equipment(res["valid"])
                        ss["eq_flash"] = f"Imported {n} equipment record(s)" + (f"; {len(res['rejected'])} row(s) were rejected." if len(res["rejected"]) else ".")
                        st.rerun()
                    except DB.StorageError as ex:
                        st.error(str(ex))
            else:
                st.warning("No valid rows to import.")

# ------------------------------------------------------------------ registry
with tab_r:
    eq = DB.list_equipment()
    if eq.empty:
        U.note("No equipment registered yet. Use 'Register equipment' or 'Import CSV / Excel'.")
        st.stop()
    f1, f2, f3, f4 = st.columns([1.5, 1.5, 1.5, 1.5])
    q = f1.text_input("Search Asset ID or name", "", key="eq_q")
    ft = f2.multiselect("Type", C.EQUIPMENT_TYPES, key="eq_ft")
    locs = sorted([x for x in eq["location"].dropna().unique()])
    fl = f3.multiselect("Location", locs, key="eq_fl")
    fs = f4.multiselect("Status", C.EQUIPMENT_STATUSES, key="eq_fs")
    flt = eq
    if q.strip():
        m = flt["asset_id"].str.contains(q.strip(), case=False, regex=False) | flt["name"].str.contains(q.strip(), case=False, regex=False)
        flt = flt[m]
    if ft:
        flt = flt[flt["type"].isin(ft)]
    if fl:
        flt = flt[flt["location"].isin(fl)]
    if fs:
        flt = flt[flt["status"].isin(fs)]
    st.caption(f"{len(flt)} of {len(eq)} equipment records")
    show = flt[["asset_id", "name", "type", "location", "status", "manufacturer", "model"]]
    ev = st.dataframe(show, hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row", key="eq_table")
    st.download_button("Export shown list (CSV)", flt.to_csv(index=False).encode(), "equipment.csv", "text/csv")
    ids = flt["asset_id"].tolist()
    if not ids:
        st.stop()
    rows = ev.selection.rows if ev is not None else []
    if rows:
        ss["eq_selected"] = ids[rows[0]]
    cur = ss.get("eq_selected")
    sel = st.selectbox("Open equipment record", ids, index=ids.index(cur) if cur in ids else 0, key="eq_open")
    ss["eq_selected"] = sel
    e = DB.get_equipment(sel)
    links = DB.list_links(sel)
    lvl, ltxt = DB.ai_support_level(e, links)
    mh = DB.list_maintenance(sel)
    open_n = int(mh["status"].isin(DB.OPEN).sum()) if len(mh) else 0
    st.markdown(f"### {e['asset_id']} · {e['name']}")
    st.markdown(U.chip(e["status"], "#22d3ee") + U.chip(e["type"]) + U.chip(f"AI level {lvl}/3", "#34d399" if lvl == 3 else "#8da0bd"), unsafe_allow_html=True)
    st.caption(ltxt)
    t_ov, t_data, t_rep, t_mnt, t_edit = st.tabs(["Overview", "Sensor data & analyses", "Reports", f"Maintenance ({open_n} open)", "Edit / delete"])
    with t_ov:
        a, b = st.columns(2)
        a.markdown(f"**Identification**  \nAsset ID: {e['asset_id']}  \nName: {e['name']}  \nType: {e['type']}  \nSerial number: {e['serial_number'] or '—'}")
        b.markdown(f"**Specification & location**  \nManufacturer: {e['manufacturer'] or '—'}  \nModel: {e['model'] or '—'}  \nInstalled: {e['installation_date'] or '—'}  \n"
                   f"Location: {e['location'] or '—'}  \nDepartment: {e['department'] or '—'}")
        st.markdown(f"**Operational status:** {e['status']}  \n**Notes:** {e['notes'] or '—'}")
        st.caption(f"Created {e['created_at']} · last updated {e['updated_at']}")
    with t_data:
        rec = S.active()
        st.markdown("**Linked sensor datasets**")
        st.dataframe(links[["dataset_label", "dataset_asset", "capability", "linked_at"]] if len(links) else pd.DataFrame(columns=["dataset_label", "dataset_asset", "capability", "linked_at"]),
                     hide_index=True, width="stretch")
        if rec is None:
            U.note("Load a dataset (Data Source) to link this equipment to an asset in it.")
        else:
            V = S.view(rec)
            tbl = V["table"]
            names = tbl["name"].astype(str).tolist()
            guess = DB.find_equipment_for_dataset_asset(rec["token"], "")  # noqa: F841
            pick = st.selectbox(f"Asset in the active dataset ({rec['source_file'] or rec['label']})", names, key="eq_dsasset")
            c1, c2 = st.columns(2)
            if c1.button("Link this equipment to the selected dataset asset"):
                DB.link_dataset(sel, rec, pick)
                st.success("Linked."); st.rerun()
            linked_here = len(links) and ((links["dataset_token"] == rec["token"]) & (links["dataset_asset"] == pick)).any()
            if linked_here:
                row = tbl[tbl["name"].astype(str) == pick].iloc[0]
                if c2.button("Save current analysis to this record"):
                    rul = float(row["rul_pred"]) if "rul_pred" in row and not pd.isna(row.get("rul_pred")) else None
                    DB.save_analysis(sel, rec, pick, {"state": row["state"], "rul": rul, "priority": row["priority"], "n_findings": int(row["n_findings"]),
                                                      "last_cycle": int(row["last_cycle"]), "summary": f"{DS.source_text(rec)}: condition {row['state']}, priority {row['priority']}"})
                    st.success("Analysis saved to the equipment record."); st.rerun()
            else:
                c2.caption("Link first to save an analysis for this equipment.")
        st.markdown("**Previous analyses**")
        an = DB.list_analyses(sel)
        if an.empty:
            st.caption("No analyses saved for this equipment yet.")
        else:
            st.dataframe(an[["analyzed_at", "dataset_label", "dataset_asset", "state", "rul", "priority", "n_findings", "last_cycle"]], hide_index=True, width="stretch")
    with t_rep:
        rp = DB.list_reports(sel)
        if rp.empty:
            st.caption("No reports attached. Generate an asset report in Reports & Export and attach it to this record.")
        else:
            for _, r in rp.iterrows():
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{r['filename']}** · {r['report_type']} · {r['dataset_label']} · {r['generated_at']}")
                c2.download_button("Download", DB.get_report_content(int(r["id"])) or b"", r["filename"], "application/pdf", key=f"eq_rp_{r['id']}")
    with t_mnt:
        if mh.empty:
            st.caption("No maintenance records for this equipment.")
        else:
            st.dataframe(mh[["record", "type", "reported_issue", "priority", "status", "technician", "planned_date", "completion_date"]], hide_index=True, width="stretch")
        if st.button("Open full maintenance history"):
            ss["mm_asset"] = sel
            st.switch_page("pages/maintenance.py")
    with t_edit:
        with st.form("eq_edit_form"):
            data = eq_form("eqe", e)
            save = st.form_submit_button("Save changes", type="primary")
        if save:
            try:
                DB.update_equipment(sel, data)
                st.success("Saved."); st.rerun()
            except DB.StorageError as ex:
                st.error(str(ex))
        st.markdown("**Delete equipment**")
        sure = st.checkbox(f"I understand that {sel} will be permanently deleted.", key=f"eq_del_ok_{sel}")
        if st.button("Delete equipment", disabled=not sure):
            try:
                DB.delete_equipment(sel)
                ss.pop("eq_selected", None)
                ss.pop(f"eq_del_ok_{sel}", None)
                ss["eq_flash"] = f"Equipment {sel} deleted."
                st.rerun()
            except DB.StorageError as ex:
                st.error(str(ex))
