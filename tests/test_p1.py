"""Priority-1 tests: central dataset state, upload validation, sensor-only analysis, reports, no hidden NASA fallback."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import analysis_service as AS  # noqa: E402
from src import config as C  # noqa: E402
from src import dataset_state as DS  # noqa: E402
from src import report_generator as RG  # noqa: E402
from src import upload_flow as UF  # noqa: E402
from src import upload_mapping as UM  # noqa: E402
from src.data_loader import load_fd001, read_uploaded_bytes  # noqa: E402
from src.inference import Predictor  # noqa: E402

THR = C.DEFAULT_THRESHOLDS


@pytest.fixture(scope="module")
def nasa():
    return load_fd001()


@pytest.fixture(scope="module")
def predictor():
    return Predictor.load()


def pump_df(n_assets=4, n=60, seed=1, ids=("PUMP-A", "PUMP-B", "PUMP-C", "PUMP-D")):
    rng = np.random.default_rng(seed)
    rows = []
    for i, a in enumerate(ids[:n_assets]):
        for t in range(n):
            drift = (t / n) * 3.0 if i == 0 else 0.0   # only PUMP-A drifts
            rows.append({"Asset": a, "Timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=t),
                         "vibration_mm_s": 2.0 + drift + rng.normal(0, .15), "temp_C": 60 + rng.normal(0, .5)})
    return pd.DataFrame(rows)


def record_from(df, name="pumps.csv"):
    A = UF.assess_upload(df, name, "Asset", "Timestamp", ["vibration_mm_s", "temp_C"], {s: None for s in C.MODEL_SENSORS}, None)
    assert A["ok"], A["errors"]
    return DS.make_record(DS.KIND_UPLOAD, f"Uploaded: {name}", A["df"], capability=A["capability"], sensors=A["sensors"],
                          source_file=name, asset_names=A["asset_names"], sensor_labels=A["sensor_labels"])


# ---- 1. central state -----------------------------------------------------------------------------------
def test_initial_state_has_no_dataset():
    ss = {}
    assert DS.get_active(ss) is None          # explicit empty state, never NASA


def test_activate_persists_and_replaces(nasa):
    ss = {}
    a = DS.make_record(DS.KIND_BUILTIN, "NASA", nasa[1], capability=DS.CAP_AI, sensors=C.MODEL_SENSORS, rul_true=nasa[2])
    DS.activate(ss, a)
    assert DS.active_token(ss) == a["token"]
    b = record_from(pump_df())
    DS.activate(ss, b)
    assert DS.get_active(ss)["kind"] == DS.KIND_UPLOAD and DS.active_token(ss) == b["token"] != a["token"]


def test_dataset_change_invalidates_dependents(nasa):
    ss = {"pdf_asset": ("x", "f", b""), "pdf_fleet": 1, "rec_status": {"R-1": "Completed"}, "selected_asset": 7, "deg_asset": 7,
          "fleet_q": "9", "rec_editor": {}, "thr_warning": 55.0}
    DS.activate(ss, record_from(pump_df()))
    for k in ("pdf_asset", "pdf_fleet", "selected_asset", "deg_asset", "fleet_q", "rec_editor"):
        assert k not in ss
    assert ss["rec_status"] == {}
    assert ss["thr_warning"] == 55.0          # user settings are not dataset results


def test_clear_session_wipes_everything_and_rekeys_uploader():
    ss = {}
    DS.activate(ss, record_from(pump_df()))
    ss.update({"thr_warning": 40.0, "pdf_asset": 1, "anything": 1})
    k1 = DS.uploader_key(ss, "upl_file")
    DS.clear_session(ss)
    assert DS.get_active(ss) is None
    assert set(ss) == {DS.KEY_NONCE}
    assert DS.uploader_key(ss, "upl_file") != k1


def test_token_depends_on_content_not_name():
    a = record_from(pump_df(seed=1), "same.csv")
    b = record_from(pump_df(seed=2), "same.csv")
    assert a["token"] != b["token"]


def test_labels_distinguish_sources(nasa):
    n = DS.make_record(DS.KIND_BUILTIN, "NASA", nasa[1], capability=DS.CAP_AI, sensors=C.MODEL_SENSORS)
    u = record_from(pump_df())
    assert "BUILT-IN" in DS.source_badge(n) and "UPLOADED" in DS.source_badge(u)
    assert "NASA" in DS.input_mode_label(n) and "NASA" not in DS.input_mode_label(u)
    assert "Sensor" in DS.capability_text(u)


def test_dev_mode_flag(monkeypatch):
    monkeypatch.delenv("MAINT_DEV_MODE", raising=False)
    assert not DS.dev_mode()
    monkeypatch.setenv("MAINT_DEV_MODE", "1")
    assert DS.dev_mode()


# ---- 2. upload validation -------------------------------------------------------------------------------
def test_supported_extensions():
    assert UF.check_extension("a.csv") is None and UF.check_extension("a.XLSX") is None
    assert "not a supported" in UF.check_extension("a.pdf")


def test_sensor_only_upload_with_text_ids_and_timestamps():
    df = pump_df()
    A = UF.assess_upload(df, "p.csv", "Asset", "Timestamp", ["vibration_mm_s", "temp_C"], {s: None for s in C.MODEL_SENSORS}, None)
    assert A["ok"] and A["capability"] == DS.CAP_SENSOR and A["has_timestamp"]
    assert A["asset_names"] == {1: "PUMP-A", 2: "PUMP-B", 3: "PUMP-C", 4: "PUMP-D"}
    assert A["why_not_ai"] and "14" in A["why_not_ai"][0]       # explains why no RUL
    assert "timestamp" in A["df"].columns and A["df"][C.CYCLE_COL].min() == 1


def test_missing_value_rejected_not_filled():
    df = pump_df()
    df.loc[3, "temp_C"] = np.nan
    A = UF.assess_upload(df, "p.csv", "Asset", "Timestamp", ["vibration_mm_s", "temp_C"], {s: None for s in C.MODEL_SENSORS}, None)
    assert not A["ok"] and any("missing" in e.lower() for e in A["errors"])


def test_missing_required_column_choice_rejected():
    A = UF.assess_upload(pump_df(), "p.csv", "Asset", "Timestamp", [], {s: None for s in C.MODEL_SENSORS}, None)
    assert not A["ok"] and "measurement" in A["errors"][0].lower()


def test_non_numeric_measurements_and_bad_time_rejected():
    df = pump_df()
    df["vibration_mm_s"] = df["vibration_mm_s"].astype(str)
    df.loc[0, "vibration_mm_s"] = "high"
    A = UF.assess_upload(df, "p.csv", "Asset", "Timestamp", ["vibration_mm_s", "temp_C"], {s: None for s in C.MODEL_SENSORS}, None)
    assert not A["ok"]
    df2 = pump_df(); df2["Timestamp"] = "not a date"
    A2 = UF.assess_upload(df2, "p.csv", "Asset", "Timestamp", ["vibration_mm_s"], {s: None for s in C.MODEL_SENSORS}, None)
    assert not A2["ok"] and "time column" in A2["errors"][0]


def test_duplicate_records_rejected():
    df = pd.concat([pump_df(), pump_df().head(3)])
    A = UF.assess_upload(df, "p.csv", "Asset", "Timestamp", ["vibration_mm_s"], {s: None for s in C.MODEL_SENSORS}, None)
    assert not A["ok"] and any("duplicate" in e.lower() for e in A["errors"])


def test_nasa_template_is_ai_and_wrong_units_are_not(nasa, predictor):
    tpl = UM.example_template(nasa[1])
    g = UM.infer_columns(tpl)
    A = UF.assess_upload(tpl, "t.csv", g["unit"], g["cycle"], [c for c in tpl.columns if c not in (g["unit"], g["cycle"])], g["sensors"],
                         predictor.reference)
    assert A["ok"] and A["capability"] == DS.CAP_AI and A["mode"] == "A"
    bad = tpl.copy()
    for s in C.MODEL_SENSORS:
        bad[s] = bad[s] * 1000 + 5e4                   # different units
    B = UF.assess_upload(bad, "b.csv", g["unit"], g["cycle"], C.MODEL_SENSORS, g["sensors"], predictor.reference)
    assert B["ok"] and B["capability"] == DS.CAP_SENSOR            # never scored by the wrong model
    assert any("outside the range" in w for w in B["why_not_ai"])
    miss = tpl.drop(columns=["s7"])
    M = UF.assess_upload(miss, "m.csv", g["unit"], g["cycle"], [s for s in C.MODEL_SENSORS if s != "s7"], {**g["sensors"], "s7": None},
                         predictor.reference)
    assert M["capability"] == DS.CAP_SENSOR and "Sensor 7" in M["why_not_ai"][0]


def test_garbage_bytes_give_clear_error():
    from src.data_loader import DataFormatError
    with pytest.raises(DataFormatError):
        read_uploaded_bytes(b"")
    with pytest.raises(DataFormatError):
        read_uploaded_bytes(b"onlyonecolumn\n1\n2")


# ---- 3. analysis follows the active dataset -------------------------------------------------------------
def test_sensor_only_view_uses_only_uploaded_data():
    rec = record_from(pump_df())
    V = AS.fleet_view(rec, None, None, THR, 3.0)
    t = V["table"]
    assert len(t) == 4 and "rul_pred" not in t.columns          # no RUL invented
    assert set(t["name"]) == {"PUMP-A", "PUMP-B", "PUMP-C", "PUMP-D"}
    assert t.loc[t["name"] == "PUMP-A", "state"].iloc[0] == "Drift detected"
    assert (t.loc[t["name"] != "PUMP-A", "state"] == "Stable").all()
    F = V["findings"]
    assert set(F["rule_id"]) <= {"R-SENSOR-SHIFT", "R-DQ-SHORT", "R-DQ-GENERAL"}
    assert (F["name"] == "PUMP-A").all()
    assert all("Detected anomaly" == f for f in F["finding_class"])
    k = AS.kpis(rec, V)
    assert k["assets"] == 4 and k["attention"] == 1 and k["critical"] is None


def test_ai_view_matches_between_calls_and_differs_per_dataset(nasa, predictor):
    import joblib
    ref = joblib.load(C.ANOMALY_PATH)
    sub = nasa[1][nasa[1].unit <= 12]
    rec = DS.make_record(DS.KIND_BUILTIN, "n", sub, capability=DS.CAP_AI, sensors=predictor.sensors, rul_true=nasa[2].loc[1:12])
    V = AS.fleet_view(rec, predictor, ref, THR, 3.0)
    assert len(V["table"]) == 12 and {"rul_pred", "priority", "state"} <= set(V["table"].columns)
    assert set(V["findings"]["finding_class"]) <= {RE_C for RE_C in __import__("src.recommendation_engine", fromlist=["x"]).FINDING_CLASS_HELP}
    other = DS.make_record(DS.KIND_UPLOAD, "u", nasa[1][nasa[1].unit.between(20, 25)], capability=DS.CAP_AI, sensors=predictor.sensors)
    V2 = AS.fleet_view(other, predictor, ref, THR, 3.0)
    assert set(V2["table"]["unit"]) == set(range(20, 26)) and not set(V["table"]["unit"]) & set(V2["table"]["unit"])


def test_recommendations_never_claim_confirmed_fault(nasa, predictor):
    rec = record_from(pump_df())
    F = AS.fleet_view(rec, None, None, THR, 3.0)["findings"]
    assert F["confirmation"].str.contains("No fault is confirmed").all()
    assert not F["finding_class"].str.contains("Confirmed", case=False).any()
    for col in ("condition", "evidence", "action", "verification", "sensor_evidence"):
        assert col in F.columns


# ---- 4. reports -----------------------------------------------------------------------------------------
def test_reports_for_sensor_only_dataset_carry_dataset_stamp():
    rec = record_from(pump_df())
    V = AS.fleet_view(rec, None, None, THR, 3.0)
    uid = int(V["table"].loc[V["table"]["name"] == "PUMP-A", C.ID_COL].iloc[0])
    res = AS.asset_detail(rec, uid, None, None, THR, 3.0)
    stamp = {"dataset_name": "pumps.csv", "source_badge": DS.source_badge(rec), "input_mode": DS.input_mode_label(rec),
             "capability_text": DS.capability_text(rec), "dataset_token": rec["token"]}
    info = {**stamp, "asset_name": "PUMP-A", "n_cycles": res["n_cycles"], "last_cycle": res["last_cycle"], "state": res["state"], "rul": None,
            "thresholds": THR, "dq": None, "engine_df": res["engine_df"], "shifts": res["shifts"], "recs": res["recs"],
            "sensor_labels": rec["sensor_labels"]}
    pdf = RG.asset_report_pdf(info)
    assert pdf[:5] == b"%PDF-"
    from pypdf import PdfReader
    txt = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "UPLOADED DATA" in txt and "pumps.csv" in txt and "PUMP-A" in txt and rec["token"] in txt
    assert "Compatible engine trajectory" not in txt and "RMSE" not in txt and "NASA C-MAPSS FD001" not in txt
    assert "Remaining life not available" in txt
    fl = RG.summary_report_pdf({**stamp, "table": V["table"], "findings": V["findings"], "status_order": AS.status_order(rec),
                                "status_colors": AS.status_colors(rec), "counts": AS.status_counts(rec, V["table"]), "ai": False,
                                "kpis": AS.kpis(rec, V)})
    txt2 = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(fl)).pages)
    assert "PUMP-A" in txt2 and "RMSE" not in txt2 and "1 Drift detected" in txt2


def test_ai_asset_report_has_correct_mode_and_no_model_internals(nasa, predictor):
    import joblib
    ref = joblib.load(C.ANOMALY_PATH)
    sub = nasa[1][nasa[1].unit <= 5]
    rec = DS.make_record(DS.KIND_BUILTIN, "NASA C-MAPSS FD001 - demo fleet", sub, capability=DS.CAP_AI, sensors=predictor.sensors)
    res = AS.asset_detail(rec, 3, predictor, ref, THR, 3.0)
    stamp = {"dataset_name": rec["label"], "source_badge": DS.source_badge(rec), "input_mode": DS.input_mode_label(rec),
             "capability_text": DS.capability_text(rec), "dataset_token": rec["token"]}
    info = {**stamp, "asset_name": "3", "n_cycles": res["n_cycles"], "last_cycle": res["last_cycle"], "state": res["state"], "rul": res["rul"],
            "at_cap": res["at_cap"], "thresholds": THR, "dq": res["dq"].to_dict(), "rul_history": res["rul_history"], "engine_df": res["engine_df"],
            "shifts": res["shifts"], "recs": res["recs"], "anomaly": res["anomaly"], "sensor_labels": {}}
    from pypdf import PdfReader
    txt = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(RG.asset_report_pdf(info))).pages)
    assert "BUILT-IN NASA DEMO" in txt and "Built-in NASA C-MAPSS FD001 demo data" in txt
    for banned in ("RMSE", "SHAP", "XGBoost", "Isolation Forest", "cross-validation", "R²"):
        assert banned not in txt, banned
