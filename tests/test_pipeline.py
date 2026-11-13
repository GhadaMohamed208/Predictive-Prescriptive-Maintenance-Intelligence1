"""Automated tests. Run:  pytest -q   (requires the trained artifacts: python -m src.train_models)"""
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import anomaly_detection as AD  # noqa: E402
from src import config as C  # noqa: E402
from src import recommendation_engine as RE  # noqa: E402
from src import upload_mapping as UM  # noqa: E402
from src import what_if_engine as W  # noqa: E402
from src.data_loader import DataFormatError, load_fd001, read_uploaded_bytes, verify_subset_identity  # noqa: E402
from src.data_validation import validate_trajectories  # noqa: E402
from src.engine_analysis import analyze_engine, build_report_info  # noqa: E402
from src.evaluate_models import nasa_score, regression_metrics, risk_state  # noqa: E402
from src.feature_engineering import FeatureConfig, build_features, feature_names  # noqa: E402
from src.inference import ArtifactsMissingError, Predictor  # noqa: E402
from src.rul_target import add_rul_train, final_cycle_targets  # noqa: E402

THR = C.DEFAULT_THRESHOLDS


@pytest.fixture(scope="module")
def data():
    return load_fd001()


@pytest.fixture(scope="module")
def predictor():
    return Predictor.load()


# 1-2 ---------------------------------------------------------------- loading & schema
def test_nasa_file_loading_shapes(data):
    tr, te, rul = data
    assert tr.shape == (20631, 26) and te.shape == (13096, 26) and len(rul) == 100
    assert list(tr.columns) == C.RAW_COLUMNS


def test_subset_identity_is_fd001(data):
    assert verify_subset_identity(*data)["matches_FD001"]


def test_wrong_column_count_rejected(tmp_path):
    from src.data_loader import read_cmapss
    f = tmp_path / "bad.txt"
    f.write_text("1 1 0.1 0.2\n1 2 0.1 0.2\n")
    with pytest.raises(DataFormatError):
        read_cmapss(f)


# 3-4 ---------------------------------------------------------------- targets
def test_rul_target_generation(data):
    tr = add_rul_train(data[0], cap=None)
    assert (tr["rul_raw"] >= 0).all()
    last = tr.groupby("unit").tail(1)
    assert (last["rul_raw"] == 0).all()
    first = tr.groupby("unit").head(1)
    assert (first["rul_raw"] == tr.groupby("unit")["cycle"].max().to_numpy() - 1).all()
    capped = add_rul_train(data[0], cap=125)
    assert capped["rul"].max() == 125 and capped["rul_raw"].max() > 125  # raw & capped kept separate


def test_train_test_target_alignment(data):
    _, te, rul = data
    t = final_cycle_targets(te, rul, cap=125)
    assert len(t) == 100 and (t["unit"].to_numpy() == np.arange(1, 101)).all()
    assert (t["cycle"].to_numpy() == te.groupby("unit")["cycle"].max().to_numpy()).all()
    assert (t["rul_true"] <= 125).all() and (t["rul_true_raw"].to_numpy() == rul.to_numpy()).all()
    with pytest.raises(ValueError):
        final_cycle_targets(te[te.unit != 5], rul)


# 5-6 ---------------------------------------------------------------- validation
def _small(n=30, units=(1,)):
    rng = np.random.default_rng(0)
    rows = []
    for u in units:
        for c in range(1, n + 1):
            rows.append({"unit": u, "cycle": c, **{s: rng.normal(size=1)[0] + c * .01 for s in C.MODEL_SENSORS}})
    return pd.DataFrame(rows)


def test_validation_detects_problems():
    d = _small()
    assert validate_trajectories(d).ok
    d2 = d.copy(); d2.loc[3, "s2"] = np.nan
    assert any("missing" in e for e in validate_trajectories(d2).errors)
    d3 = d.copy(); d3.loc[4, "s3"] = np.inf
    assert any("infinite" in e for e in validate_trajectories(d3).errors)
    d4 = pd.concat([d, d.iloc[[5]]])
    assert any("duplicate" in e for e in validate_trajectories(d4).errors)
    assert any("Missing required" in e for e in validate_trajectories(d.drop(columns=["s7"])).errors)
    d5 = d.copy(); d5["s4"] = d5["s4"].astype(str) + "x"
    assert any("not numeric" in e for e in validate_trajectories(d5).errors)
    assert not validate_trajectories(d.iloc[0:0]).ok


def test_cycle_gaps_and_short_history_warn():
    d = _small(n=8)
    assert any("fewer than" in w for w in validate_trajectories(d).warnings)
    g = _small(n=30)
    g = g[g.cycle != 10]
    assert any("gaps" in w for w in validate_trajectories(g).warnings)
    assert any("gaps" in e for e in validate_trajectories(g, require_contiguous=True).errors)


def test_unexpected_columns_reported_not_ignored():
    d = _small(); d["mystery"] = 1.0
    r = validate_trajectories(d, expected_columns=["unit", "cycle"] + C.MODEL_SENSORS)
    assert any("mystery" in w for w in r.warnings)


# 7 ordering / 8 leakage -------------------------------------------------------
def test_engine_cycle_ordering_is_enforced():
    d = _small(units=(1, 2))
    shuffled = d.sample(frac=1.0, random_state=1)
    a = build_features(d); b = build_features(shuffled)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert (b.groupby("unit")["cycle"].diff().dropna() > 0).all()


def test_no_future_leakage_in_features(data):
    tr = data[0]
    eng = tr[tr.unit == 3]
    full = build_features(eng)
    for t in (1, 2, 11, 57, 120):
        part = build_features(eng[eng.cycle <= t])
        np.testing.assert_allclose(part.iloc[-1][feature_names(FeatureConfig())].to_numpy(dtype=float),
                                   full[full.cycle == t].iloc[0][feature_names(FeatureConfig())].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)
    # perturbing a FUTURE row must not change earlier features
    mod = eng.copy(); mod.loc[mod.index[-1], "s11"] += 100
    f2 = build_features(mod)
    np.testing.assert_allclose(f2.iloc[:-1].drop(columns=["unit", "cycle"]).to_numpy(), full.iloc[:-1].drop(columns=["unit", "cycle"]).to_numpy())


def test_rolling_slope_matches_polyfit(data):
    eng = data[0][data[0].unit == 1].iloc[:40]
    f = build_features(eng)
    w = eng.iloc[30 - 10:30]
    expected = np.polyfit(w["cycle"], w["s11"], 1)[0]
    assert f.iloc[29]["s11__slope"] == pytest.approx(expected, rel=1e-6, abs=1e-9)


# 9 feature order / model reload / predictions ---------------------------------
def test_consistent_feature_order_at_inference(predictor, data):
    eng = data[1][data[1].unit == 7]
    shuffled_cols = eng[list(reversed(eng.columns))]
    a = predictor.predict_latest(eng)[0]["rul_pred"].iloc[0]
    b = predictor.predict_latest(shuffled_cols)[0]["rul_pred"].iloc[0]
    assert a == pytest.approx(b)
    assert list(predictor.features(eng).columns[2:]) == predictor.feature_names


def test_model_save_and_reload(predictor, data, tmp_path):
    eng = data[1][data[1].unit == 2]
    before = predictor.predict_rows(eng)["rul_pred"].to_numpy()
    path = tmp_path / "m.joblib"
    joblib.dump(predictor.bundle, path)
    p2 = Predictor.load(path, C.METADATA_PATH)
    np.testing.assert_allclose(before, p2.predict_rows(eng)["rul_pred"].to_numpy())


def test_missing_artifacts_message(tmp_path):
    with pytest.raises(ArtifactsMissingError):
        Predictor.load(tmp_path / "none.joblib", tmp_path / "none.json")


def test_prediction_output_validity(predictor, data):
    res, _ = predictor.predict_latest(data[1])
    assert len(res) == 100 and np.isfinite(res["rul_pred"]).all()
    assert res["rul_pred"].between(0, predictor.cap).all()
    assert set(res["state"]) <= set(C.STATES)


def test_saved_test_metrics_reproduce(predictor, data):
    _, te, rul = data
    res, _ = predictor.predict_latest(te)
    t = final_cycle_targets(te, rul, predictor.cap)
    m = regression_metrics(t["rul_true"], res["rul_pred"])
    import json
    saved = json.loads(C.METRICS_PATH.read_text())["official_test_last_cycle_vs_capped_truth"]
    assert m["rmse"] == pytest.approx(saved["rmse"], rel=1e-6)


# 10-11 thresholds -----------------------------------------------------------
@pytest.mark.parametrize("rul,exp", [(61, "Healthy"), (60.01, "Healthy"), (60, "Warning"), (30, "Warning"), (29.99, "Critical"), (0, "Critical")])
def test_risk_state_boundaries(rul, exp):
    assert risk_state(rul, THR) == exp


def test_nasa_score_known_values():
    assert nasa_score([100], [100]) == 0
    assert nasa_score([100], [110]) == pytest.approx(np.exp(1) - 1)       # late: d=10
    assert nasa_score([100], [87]) == pytest.approx(np.exp(1) - 1)        # early: d=-13


# 12 recommendations ---------------------------------------------------------
def _ctx(rul, **kw):
    base = dict(engine_id=1, last_cycle=100, pred_rul=rul, state=risk_state(rul, THR), thresholds=THR, cap=125, n_cycles=100)
    base.update(kw)
    return RE.Context(**base)


def test_recommendation_rule_triggers():
    ids = lambda ctx: {r["rule_id"] for r in RE.evaluate(ctx)}
    assert ids(_ctx(10)) >= {"R-RUL-CRIT"} and "R-RUL-WARN" not in ids(_ctx(10))
    assert "R-RUL-WARN" in ids(_ctx(45)) and "R-RUL-CRIT" not in ids(_ctx(45))
    assert ids(_ctx(100)) == {"R-RUL-OK"}
    assert "R-DQ-SHORT" in ids(_ctx(100, n_cycles=5))
    an = {"recent_flag_fraction": .8, "recent_n": 10, "last_score": .6, "first_flag_cycle": 90, "last_flag": True}
    assert "R-ANOM-PERSIST" in ids(_ctx(100, anomaly=an))
    sh = [{"label": "Sensor 11", "shift_sigma": 6.0, "trend_rho": .9, "abs_sigma": 6.0}]
    r = [x for x in RE.evaluate(_ctx(20, shifts=sh)) if x["rule_id"] == "R-RUL-CRIT"][0]
    assert r["evidence_strength"] == "Strong"
    required = {"id", "action", "trigger", "evidence", "priority", "evidence_strength", "source", "limitation", "verification", "status"}
    assert required <= set(RE.evaluate(_ctx(10))[0])
    assert all(x["status"] == "Pending" for x in RE.evaluate(_ctx(10)))
    # no rule asserts a physical component diagnosis
    banned = ("bearing", "lubrication", "cooling", "replace the")
    assert not any(b in (x.action + x.limitation).lower() for x in RE.RULES for b in banned)


# 13 what-if -----------------------------------------------------------------
def test_what_if_validation_and_run(predictor, data):
    eng = data[1][data[1].unit == 4]
    last_val = float(eng.iloc[-1]["s11"])
    b = W.sensor_bounds(predictor)["s11"]
    bad = W.run_scenario(predictor, eng, {"s11": b["max"] + 5}, THR)
    assert bad["errors"] and "baseline_rul" not in bad
    assert W.run_scenario(predictor, eng, {}, THR)["errors"]
    assert W.run_scenario(predictor, eng, {"s11": float("nan")}, THR)["errors"]
    assert W.run_scenario(predictor, eng, {"s1": 1.0}, THR)["errors"]
    original = eng.copy()
    ok = W.run_scenario(predictor, eng, {"s11": last_val}, THR)   # no actual change
    assert not ok["errors"] and ok["delta"] == pytest.approx(0, abs=1e-9)
    pd.testing.assert_frame_equal(eng, original)                   # observed data untouched
    mid = (b["min"] + b["max"]) / 2
    ch = W.run_scenario(predictor, eng, {"s11": b["min"]}, THR)
    assert not ch["errors"] and list(ch["changed"]["sensor"]) == ["Sensor 11"] and W.SCENARIO_WARNING in ch["warnings"]


# 14 pdf ---------------------------------------------------------------------
def test_pdf_generation(predictor, data):
    from src import report_generator as RG
    import json
    eng = data[1][data[1].unit == 9]
    expl = None
    try:
        from src import explainability as EX
        g = EX.load_global(C.SHAP_GLOBAL_PATH)
        expl, _ = EX.make_explainer(predictor.model, background=g["X"].head(100))
    except Exception:
        pass
    ref = joblib.load(C.ANOMALY_PATH)
    res = analyze_engine(predictor, ref, expl, eng, THR)
    info = build_report_info(res, predictor, "Official test set (FD001)", json.loads(C.METRICS_PATH.read_text()), 42.0)
    pdf = RG.engine_report_pdf(info)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 20_000
    # report still renders (sections flagged unavailable) with no explainer / anomaly detector
    res2 = analyze_engine(predictor, None, None, eng, THR)
    pdf2 = RG.engine_report_pdf(build_report_info(res2, predictor, "x", None))
    assert pdf2[:5] == b"%PDF-"
    from src import fleet_analysis as FA
    tbl = FA.fleet_table(predictor, data[1][data[1].unit <= 15], THR, ref)
    fp = RG.fleet_report_pdf({"dataset_name": "t", "table": tbl, "counts": FA.state_counts(tbl), "thresholds": THR,
                              "model_meta": {**predictor.metadata, "feature_names": predictor.feature_names}, "metrics": None})
    assert fp[:5] == b"%PDF-"


# 15-16 uploads / unsupported ------------------------------------------------
def test_csv_upload_parsing_and_mapping():
    df = UM.example_template(load_fd001()[1])
    df2 = df.rename(columns={"unit": "Engine ID", "cycle": "Time (cycles)", "s2": "Sensor 2", "s3": "sensor_3", "s4": "SM4"})
    g = UM.infer_columns(df2)
    assert g["unit"] == "Engine ID" and g["cycle"] == "Time (cycles)"
    assert g["sensors"]["s2"] == "Sensor 2" and g["sensors"]["s3"] == "sensor_3" and g["sensors"]["s4"] == "SM4"
    raw = df.to_csv(index=False).encode()
    back = read_uploaded_bytes(raw)
    assert back.shape == df.shape
    mapped = UM.apply_mapping(back, "unit", "cycle", {s: s for s in C.MODEL_SENSORS})
    assert validate_trajectories(mapped).ok


def test_malformed_and_oversized_uploads_rejected():
    for bad in (b"", b"   \n  ", b"a\n1\n"):
        with pytest.raises(DataFormatError):
            read_uploaded_bytes(bad)
    with pytest.raises(DataFormatError):
        read_uploaded_bytes(b"x" * (C.MAX_UPLOAD_MB * 1024 * 1024 + 1))
    with pytest.raises(DataFormatError):
        read_uploaded_bytes(b"a,b,c\n" + b"1,2,3\n" * 20, max_rows=5)
    # binary garbage never executes anything; it just fails or parses as data
    try:
        read_uploaded_bytes(bytes(range(256)) * 10)
    except DataFormatError:
        pass


def test_unsupported_schema_and_units_classified_mode_c(predictor):
    d = _small(units=(1, 2))
    rep = validate_trajectories(d, sensors=C.MODEL_SENSORS[:5])
    mode, why = UM.classify_mode(rep, None, C.MODEL_SENSORS[5:], False)
    assert mode == "C" and "not mapped" in why and "retrain" in why.lower() or "training data" in why
    scaled = d.copy()
    for s in C.MODEL_SENSORS:
        scaled[s] = scaled[s] * 1e6 + 1e9
    comp = UM.compatibility(scaled, predictor.reference, C.MODEL_SENSORS)
    mode2, why2 = UM.classify_mode(validate_trajectories(scaled), comp, [], False)
    assert comp["unsupported"] and mode2 == "C"
    comp_ok = UM.compatibility(UM.example_template(load_fd001()[1]), predictor.reference, C.MODEL_SENSORS)
    assert not comp_ok["unsupported"]
    assert UM.classify_mode(validate_trajectories(d), comp_ok, [], True)[0] == "B"
    assert UM.classify_mode(validate_trajectories(d), comp_ok, [], False)[0] == "A"


# edge cases -----------------------------------------------------------------
def test_edge_single_engine_and_short_trajectory(predictor):
    one = _small(n=1)
    r, _ = predictor.predict_latest(one)
    assert len(r) == 1 and np.isfinite(r["rul_pred"]).all()
    short = _small(n=3)
    rows = predictor.predict_rows(short)
    assert len(rows) == 3 and np.isfinite(rows["rul_pred"]).all()


def test_missing_sensor_raises_clear_error(predictor):
    d = _small().drop(columns=["s11"])
    with pytest.raises(KeyError):
        predictor.predict_latest(d)
    assert not analyze_engine(predictor, None, None, d, THR).get("dq").ok


def test_anomaly_detector_flags_late_life_more(data):
    ref = joblib.load(C.ANOMALY_PATH)
    tr = add_rul_train(data[0], None)
    sc = AD.score(ref, tr).merge(tr[["unit", "cycle", "rul_raw"]], on=["unit", "cycle"])
    assert sc.loc[sc.rul_raw <= 30, "anomaly_flag"].mean() > sc.loc[sc.rul_raw > 150, "anomaly_flag"].mean() + .3
