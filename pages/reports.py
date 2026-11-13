import pandas as pd
import streamlit as st

from src import analysis_service as AS
from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import report_generator as RG
from src import storage as DB
from src import ui_components as U

p = S.setup_page()
rec = S.active()
tok = rec["token"]
thr, sigma = S.thresholds(), float(st.session_state["sigma_trigger"])
V = S.view(rec)
tbl, F = V["table"], V["findings"]
rs = st.session_state["rec_status"]
ai = rec["capability"] == DS.CAP_AI
U.page_header("Reports & Export", f"Reports use only the active dataset: {rec['source_file'] or rec['label']} ({DS.source_text(rec)}). "
              "Each report is stamped with its data source and is discarded when the dataset changes.", "REPORTING")


def stamp() -> dict:
    return {"dataset_name": rec["source_file"] or rec["label"], "source_badge": DS.source_badge(rec), "input_mode": DS.input_mode_label(rec),
            "capability_text": DS.capability_text(rec), "dataset_token": rec["token"]}


def valid(key):
    v = st.session_state.get(key)
    if v and v[0] == tok:
        return v
    return None


t1, t2, t3 = st.tabs(["Asset report (PDF)", "Fleet summary (PDF)", "Data exports (CSV)"])
with t1:
    ids = sorted(rec["df"][C.ID_COL].unique())
    cur = st.session_state.get("selected_asset")
    e = st.selectbox("Asset", ids, index=ids.index(cur) if cur in ids else 0, format_func=lambda u: DS.asset_name(rec, u), key="rep_asset")
    st.session_state["selected_asset"] = e
    if st.button("Generate asset report", type="primary"):
        with st.spinner("Analysing asset and rendering PDF…"):
            res = AS.asset_detail(rec, int(e), p if ai else None, S.get_anomaly_ref() if ai else None, thr, sigma)
            if "error" in res:
                st.error(res["error"])
            else:
                recs = [dict(r, status=rs.get(r["id"], r["status"])) for r in res["recs"]]
                truth = None
                if rec.get("rul_true") is not None and int(e) in rec["rul_true"].index:
                    truth = float(rec["rul_true"].loc[int(e)])
                info = {**stamp(), "asset_name": DS.asset_name(rec, e), "n_cycles": res["n_cycles"], "last_cycle": res["last_cycle"],
                        "state": res["state"], "rul": res.get("rul"), "at_cap": res.get("at_cap", False), "thresholds": thr,
                        "dq": res["dq"].to_dict() if res.get("dq") is not None else None, "rul_history": res.get("rul_history"),
                        "engine_df": res["engine_df"], "shifts": res["shifts"], "recs": recs, "anomaly": res.get("anomaly"),
                        "sensor_labels": rec.get("sensor_labels", {}), "rul_true": truth}
                fn = f"asset_{DS.asset_name(rec, e)}_report.pdf".replace(" ", "_").replace("/", "-")
                st.session_state["pdf_asset"] = (tok, fn, RG.asset_report_pdf(info))
    v = valid("pdf_asset")
    if v:
        st.success(f"Report ready for the active dataset: {v[1]} ({len(v[2])/1024:.0f} KB)")
        st.download_button("Download asset report (PDF)", v[2], v[1], "application/pdf", type="primary")
        eqs = DB.list_equipment()
        if len(eqs):
            ids_ = eqs["asset_id"].tolist()
            g = DB.find_equipment_for_dataset_asset(tok, DS.asset_name(rec, e))
            c1, c2 = st.columns([2, 1])
            tgt = c1.selectbox("Attach this report to equipment record", ids_, index=ids_.index(g) if g in ids_ else 0, key="rep_attach")
            if c2.button("Attach to equipment record"):
                DB.save_report(tgt, "Asset report", v[1], rec, v[2])
                DB.link_dataset(tgt, rec, DS.asset_name(rec, e))
                st.success(f"Report attached to {tgt}.")
with t2:
    st.markdown(f"Fleet summary for **{rec['source_file'] or rec['label']}** ({len(tbl)} assets).")
    if st.button("Generate fleet report", type="primary"):
        with st.spinner("Rendering PDF…"):
            info = {**stamp(), "table": tbl, "findings": F, "status_order": AS.status_order(rec), "status_colors": AS.status_colors(rec),
                    "counts": AS.status_counts(rec, tbl), "ai": ai, "kpis": AS.kpis(rec, V, rs)}
            st.session_state["pdf_fleet"] = (tok, "fleet_summary_report.pdf", RG.summary_report_pdf(info))
    v = valid("pdf_fleet")
    if v:
        st.success(f"Report ready for the active dataset ({len(v[2])/1024:.0f} KB)")
        st.download_button("Download fleet report (PDF)", v[2], v[1], "application/pdf", type="primary")
with t3:
    cols = [c for c in tbl.columns if c not in ("ood_fraction", "ood_flag", "at_cap")]
    st.download_button("Asset table (CSV)", tbl[cols].to_csv(index=False).encode(), "assets.csv", "text/csv")
    if len(F):
        X = F.copy()
        X["status"] = X["id"].map(lambda i: rs.get(i, "Pending"))
        X["sensors"] = X["sensors"].map("; ".join)
        st.download_button("Recommendations and statuses (CSV)", X[["id", "name", "priority", "condition", "finding_class", "evidence", "action",
                                                                    "verification", "status", "sensors"]].to_csv(index=False).encode(),
                           "recommendations.csv", "text/csv")
    else:
        st.caption("No recommendations to export.")
