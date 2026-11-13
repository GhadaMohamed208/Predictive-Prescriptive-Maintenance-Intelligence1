import io

import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import dataset_state as DS
from src import ui_components as U
from src import upload_flow as UF
from src import upload_mapping as UM
from src.data_loader import DataFormatError, read_rul_file, read_uploaded_bytes

p = S.setup_page(need_dataset=False)
ss = st.session_state
U.page_header("Data Source", "Choose the dataset the whole application works with. It stays active on every page until you replace it "
              "or clear the session.", "DATA SOURCE")

rec = S.active()
st.markdown("##### Active dataset")
if rec is None:
    U.note("No dataset is active. Load the built-in NASA demo or upload a file below.")
else:
    k = st.columns(4)
    k[0].markdown(U.kpi("Source", DS.source_text(rec), note=rec["source_file"] or rec["label"]), unsafe_allow_html=True)
    k[1].markdown(U.kpi("Assets", str(rec["n_assets"])), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Records", f"{rec['n_rows']:,}", note="historical data"), unsafe_allow_html=True)
    k[3].markdown(U.kpi("Analysis", "AI-supported" if rec["capability"] == DS.CAP_AI else "Sensor only"), unsafe_allow_html=True)
    st.caption(DS.input_mode_label(rec) + ". " + DS.capability_text(rec) + ".")
    for n in rec.get("notes", []):
        st.caption("• " + n)
    c1, c2, c3, _ = st.columns([1.2, 1.2, 1.2, 3])
    if c1.button("Open Control Room", type="primary", width="stretch"):
        st.switch_page("pages/home.py")
    if c2.button("Open Fleet Overview", width="stretch"):
        st.switch_page("pages/fleet_overview.py")
    if c3.button("Clear session", width="stretch", key="ds_clear"):
        S.clear_session()
        st.rerun()

st.markdown("---")
tab_b, tab_u = st.tabs(["Built-in NASA demo dataset", "Upload your own data"])
with tab_b:
    st.markdown("**NASA C-MAPSS FD001**: simulated run-to-failure data of 100 aircraft turbofan engines (test portion). "
                "It is demonstration data, not a real operating fleet. It is the only equipment type with an implemented AI model.")
    if st.button("Load built-in NASA demo dataset", type="primary"):
        S.activate_builtin()
        st.success("Built-in NASA demo dataset is now active for the whole application.")
        st.rerun()
    if DS.dev_mode() and st.button("Load NASA training set (development only)"):
        S.activate_builtin(DS.KIND_BUILTIN_TRAIN)
        st.rerun()

with tab_u:
    with st.expander("Which files can I upload?", expanded=False):
        st.markdown(
            "**Formats:** CSV, TSV, whitespace-separated TXT (also the NASA C-MAPSS layout) and Excel (.xlsx), up to "
            f"{C.MAX_UPLOAD_MB} MB.  \n**Required:** an asset identifier column, a cycle / time column (number or timestamp) and at least one "
            "numeric measurement column. Rows must be one record per asset and time step.  \n"
            "**What you get:** if the file contains the 14 aircraft-engine sensors supported by the implemented model and the values are "
            "in the supported range, you get AI-supported analysis (remaining life, anomalies, recommendations). Any other valid sensor data "
            "is analysed statistically (trends and sensor drift) and is clearly labelled as such. Nothing is ever scored with a model "
            "that does not fit the data.")
    st.download_button("Download example CSV (2 NASA engines, 60 records each)",
                       UM.example_template(S.load_data()[1]).to_csv(index=False).encode(), "example_engines.csv", "text/csv")
    f = st.file_uploader(f"1 · Select a file (max {C.MAX_UPLOAD_MB} MB)", type=["csv", "txt", "tsv", "xlsx"],
                         key=DS.uploader_key(ss, "upl_file"))
    if f is None:
        st.stop()
    err = UF.check_extension(f.name)
    if err:
        st.error(err)
        st.stop()
    try:
        raw = f.getvalue()
        if f.name.lower().endswith(".xlsx"):
            if len(raw) > C.MAX_UPLOAD_MB * 1024 * 1024:
                raise DataFormatError(f"File exceeds the {C.MAX_UPLOAD_MB} MB limit.")
            original = pd.read_excel(io.BytesIO(raw))
            original.columns = [str(c).strip() for c in original.columns]
            if len(original) == 0 or original.shape[1] < 3:
                raise DataFormatError("The sheet needs at least 3 columns and one data row.")
        else:
            original = read_uploaded_bytes(raw)
    except Exception as e:  # unreadable / corrupt / wrong type: clear message, active dataset untouched
        st.error(f"Could not read '{f.name}': {e}")
        if rec:
            st.info("The active dataset has not been changed.")
        st.stop()
    st.caption(f"File: **{f.name}** · {f.size/1024:.1f} KB · {len(original):,} rows × {original.shape[1]} columns")
    st.dataframe(original.head(8), hide_index=True, width="stretch")

    st.markdown("##### 2 · Column mapping")
    guess = UM.infer_columns(original)
    cols = list(original.columns)
    NONE = "— select —"
    opts = [NONE] + cols

    def _ix(v):
        return opts.index(v) if v in opts else 0
    m1, m2 = st.columns(2)
    unit = m1.selectbox("Asset / equipment identifier", opts, index=_ix(guess["unit"]), key="up_unit")
    tcol = m2.selectbox("Cycle or time column", opts, index=_ix(guess["cycle"]), key="up_time")
    if NONE in (unit, tcol):
        st.error("Select the asset identifier and the cycle/time column to continue.")
        st.stop()
    cands = UF.candidate_sensor_columns(original, [unit, tcol])
    default = [c for c in cands if not str(c).lower().startswith(("op", "setting"))] or cands
    sens = st.multiselect("Measurement (sensor) columns", cands, default=default, key="up_sens")
    smap14 = dict(guess["sensors"])
    with st.expander("Advanced: map columns to the 14 aircraft-engine sensors (only for NASA-style data)"):
        cs = st.columns(4)
        for i, s in enumerate(C.MODEL_SENSORS):
            ch = cs[i % 4].selectbox(C.sensor_label(s), opts, index=_ix(smap14[s]), key=f"up_m_{s}")
            smap14[s] = None if ch == NONE else ch
    has_truth = False
    f_rul = None
    if all(smap14.values()):
        f_rul = st.file_uploader("Optional · reference RUL file (one value per engine, ordered by engine number)", type=["txt", "csv"],
                                 key=DS.uploader_key(ss, "upl_rul"))
        has_truth = f_rul is not None

    A = UF.assess_upload(original, f.name, unit, tcol, sens, smap14, S.get_predictor().reference, has_truth)
    st.markdown("##### 3 · Validation result")
    rep = A["report"]
    k = st.columns(4)
    k[0].markdown(U.kpi("Validation", rep.status.title() if rep else "Failed"), unsafe_allow_html=True)
    k[1].markdown(U.kpi("Assets", str(rep.stats.get("engines", "—")) if rep else "—"), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Records", f"{rep.stats.get('rows', len(original)):,}" if rep else "—"), unsafe_allow_html=True)
    k[3].markdown(U.kpi("Analysis level", {"ai": "AI-supported", "sensor_only": "Sensor only", None: "Rejected"}[A["capability"]]),
                  unsafe_allow_html=True)
    for e in A["errors"]:
        st.error(e)
    if not A["ok"]:
        st.error("This file cannot be used. Fix the problems above and upload it again.")
        if rec:
            st.info(f"The active dataset ({rec['label']}) has not been changed.")
        st.stop()
    for n in A["notes"]:
        st.info(n)
    for w in A["warnings"]:
        st.warning(w)
    if A["unused_columns"]:
        st.caption("Columns present in the file but not used: " + ", ".join(map(str, A["unused_columns"])))
    if A["capability"] == DS.CAP_AI:
        st.success("The file matches the aircraft-engine sensor schema supported by the implemented model: AI-supported analysis is available.")
    else:
        st.warning("Remaining-life estimates are NOT available for this file:")
        for w in A["why_not_ai"]:
            st.markdown("- " + w)
        st.markdown("Available: sensor statistics, degradation trends, sensor-drift findings, recommendations and reports. "
                    "To get remaining-life estimates for another equipment type, a validated model for that type would be required.")

    rul_true = None
    if A["capability"] == DS.CAP_AI and has_truth:
        try:
            rul_true = read_rul_file(io.BytesIO(f_rul.getvalue()))
            ids = sorted(A["df"][C.ID_COL].unique())
            if len(rul_true) != len(ids):
                st.error(f"The RUL file has {len(rul_true)} values but the data has {len(ids)} engines: one value per engine is required.")
                st.stop()
            rul_true.index = ids
            st.caption("Reference values are matched to engines in ascending engine-number order: "
                       + ", ".join(f"engine {i} → {rul_true.loc[i]:.0f}" for i in ids[:3]) + (", …" if len(ids) > 3 else ""))
        except Exception as e:
            st.error(f"Could not read the RUL file: {e}")
            st.stop()

    st.markdown("##### 4 · Activate")
    ok = st.checkbox("I confirm the mapping and validation results are correct.", key="up_confirm")
    if st.button("Make this the active dataset", type="primary", disabled=not ok):
        S.activate_upload(A, f.name, rul_true)
        st.session_state["flash"] = f"'{f.name}' is now the active dataset for the whole application."
        st.rerun()
if st.session_state.get("flash"):
    st.success(st.session_state.pop("flash"))
