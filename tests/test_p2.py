"""Priority-2 tests: equipment registry, imports, maintenance management, persistence, integrity, dashboard numbers."""
import io
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import dataset_state as DS  # noqa: E402
from src import storage as S  # noqa: E402

EQ = {"asset_id": "PMP-001", "name": "Cooling pump 1", "type": "Pumps", "manufacturer": "ExampleCo", "model": "CW-200", "installation_date": "2021-03-15",
      "location": "Plant A", "status": "Operational"}


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setenv(C.DB_PATH_ENV, str(tmp_path / "t.db"))
    return tmp_path / "t.db"


def test_register_and_unique_ids_case_insensitive():
    S.add_equipment(EQ)
    with pytest.raises(S.StorageError, match="already exists"):
        S.add_equipment({**EQ, "asset_id": "pmp-001"})
    assert len(S.list_equipment()) == 1


@pytest.mark.parametrize("bad,msg", [({"asset_id": ""}, "Asset ID is required"), ({"name": ""}, "name is required"), ({"type": "Toaster"}, "not allowed"),
                                     ({"status": "Flying"}, "not allowed"), ({"installation_date": "yesterday-ish"}, "valid date"),
                                     ({"asset_id": "bad;drop"}, "invalid")])
def test_equipment_validation(bad, msg):
    with pytest.raises(S.StorageError, match=msg):
        S.add_equipment({**EQ, **bad})


def test_all_required_types_accepted():
    for i, t in enumerate(C.EQUIPMENT_TYPES):
        S.add_equipment({**EQ, "asset_id": f"A{i}", "type": t})
    assert set(S.list_equipment()["type"]) == set(C.EQUIPMENT_TYPES)


def test_edit_keeps_id_and_updates_fields():
    S.add_equipment(EQ)
    S.update_equipment("PMP-001", {**EQ, "name": "Renamed", "status": "Degraded", "asset_id": "HACK"})
    e = S.get_equipment("PMP-001")
    assert e["name"] == "Renamed" and e["status"] == "Degraded" and S.get_equipment("HACK") is None


def test_import_valid_and_rejected_rows_reported():
    df = pd.DataFrame([
        {"Asset ID": "M-1", "Equipment Name": "Motor 1", "Type": "Electric Motors", "Installation Date": "2020-01-01"},
        {"Asset ID": "M-2", "Equipment Name": "Motor 2", "Type": "electric motors"},
        {"Asset ID": "M-1", "Equipment Name": "Dup in file", "Type": "Pumps"},
        {"Asset ID": "", "Equipment Name": "No id", "Type": "Pumps"},
        {"Asset ID": "M-5", "Equipment Name": "Bad type", "Type": "Spaceship"},
        {"Asset ID": "M-6", "Equipment Name": "Bad date", "Type": "Fans", "Installation Date": "31/31/2020"},
        {"Asset ID": "M-7", "Equipment Name": "Bad status", "Type": "Fans", "Operational Status": "weird"}])
    S.add_equipment({**EQ, "asset_id": "M-2"})     # already registered -> reject
    r = S.parse_equipment_table(df)
    ids = [v["asset_id"] for v in r["valid"]]
    assert ids == ["M-1"]
    rej = r["rejected"]
    assert list(rej["row"]) == [3, 4, 5, 6, 7, 8]
    txt = " | ".join(rej["reason"])
    for k in ("already exists in the registry", "duplicated in the file", "required", "not allowed", "valid date"):
        assert k in txt
    assert S.import_equipment(r["valid"]) == 1 and len(S.list_equipment()) == 2


def test_import_missing_required_columns_and_garbage():
    r = S.parse_equipment_table(pd.DataFrame({"Foo": [1], "Bar": [2]}))
    assert r["missing_columns"] == ["asset_id", "name", "type"] and not r["valid"]
    r2 = S.parse_equipment_table(pd.DataFrame({"Asset ID": [None, float("nan")], "Name": ["a", "b"], "Type": ["Pumps", "Pumps"]}))
    assert not r2["valid"] and len(r2["rejected"]) == 2


def test_import_is_atomic():
    S.add_equipment({**EQ, "asset_id": "X"})
    with pytest.raises(S.StorageError):
        S.import_equipment([{**S.normalise_equipment({**EQ, "asset_id": "NEW"})}, {**S.normalise_equipment({**EQ, "asset_id": "X"})}])
    assert S.get_equipment("NEW") is None


def test_excel_roundtrip(tmp_path):
    p = tmp_path / "e.xlsx"
    S.import_template().to_excel(p, index=False)
    r = S.parse_equipment_table(pd.read_excel(p))
    assert len(r["valid"]) == 1 and r["valid"][0]["asset_id"] == "PMP-001" and r["valid"][0]["installation_date"] == "2021-03-15"


def test_maintenance_requires_registered_asset_and_valid_values():
    with pytest.raises(S.StorageError, match="not in the equipment registry"):
        S.create_maintenance({"asset_id": "NOPE", "reported_issue": "x"})
    S.add_equipment(EQ)
    for bad, msg in (({"reported_issue": ""}, "Reported issue"), ({"priority": "Urgent"}, "Priority"), ({"type": "Magic"}, "Maintenance type"),
                     ({"status": "Done"}, "Status"), ({"planned_date": "soon"}, "Planned date"), ({"completion_date": "2025-01-01"}, "completion date")):
        with pytest.raises(S.StorageError, match=msg):
            S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "Vibration", **bad})
    with pytest.raises(S.StorageError, match="Related Asset ID"):
        S.create_maintenance({"asset_id": "", "reported_issue": "x"})


def test_full_maintenance_lifecycle_and_history_retained():
    S.add_equipment(EQ)
    rid = S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "High vibration", "priority": "High", "type": "Inspection"})
    assert S.get_maintenance(rid)["status"] == "Pending" and S.mr_label(rid) == "MR-00001"
    S.update_maintenance(rid, {"technician": "A. Hassan"})
    S.update_maintenance(rid, {"status": "In Progress"})
    S.update_maintenance(rid, {"work_performed": "Checked alignment", "notes": "bearing temp ok"})
    with pytest.raises(S.StorageError, match="valid date"):
        S.update_maintenance(rid, {"status": "Completed", "completion_date": "x"})
    assert S.get_maintenance(rid)["status"] == "In Progress"       # failed update changed nothing
    done = S.update_maintenance(rid, {"status": "Completed", "completion_date": "2026-11-02"})
    assert done["status"] == "Completed" and done["completion_date"] == "2026-11-02" and done["technician"] == "A. Hassan"
    with pytest.raises(S.StorageError, match="cannot change"):
        S.update_maintenance(rid, {"status": "Pending"})
    with pytest.raises(S.StorageError, match="retained"):
        S.update_maintenance(rid, {"priority": "Low"})
    S.update_maintenance(rid, {"notes": "post-check ok"})       # notes still amendable
    ev = S.maintenance_events(rid)
    assert list(ev["event"])[:3] == ["Created", "Technician assigned", "Status changed"] and "Completed" in " ".join(ev["detail"])
    h = S.list_maintenance("PMP-001")
    assert len(h) == 1 and h.iloc[0]["status"] == "Completed"


def test_equipment_with_history_cannot_be_deleted_but_empty_can():
    S.add_equipment(EQ)
    S.add_equipment({**EQ, "asset_id": "EMPTY"})
    S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "x"})
    with pytest.raises(S.StorageError, match="history is retained"):
        S.delete_equipment("PMP-001")
    S.delete_equipment("EMPTY")
    assert S.get_equipment("EMPTY") is None and S.get_equipment("PMP-001")


def test_foreign_key_enforced_at_database_level(db):
    S.add_equipment(EQ)
    with S.conn() as c:
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO maintenance_records (asset_id,type,reported_issue,priority,created_date,status,created_at,updated_at) "
                      "VALUES ('GHOST','Other','x','Low','2026-01-01','Pending','t','t')")


def test_persistence_across_new_connections_and_process_restart(db):
    S.add_equipment(EQ)
    rid = S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "x", "priority": "Critical"})
    import subprocess, textwrap
    code = textwrap.dedent(f"""
        import sys, os; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        os.environ['{C.DB_PATH_ENV}'] = {str(db)!r}
        from src import storage as S
        print(len(S.list_equipment()), S.get_maintenance({rid})['priority'])""")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()
    assert out == ["1", "Critical"]


def test_summary_counts_only_stored_records_and_attention_definition():
    z = S.summary()
    assert z["equipment_total"] == 0 and z["maintenance_total"] == 0 and z["critical_open"] == 0
    S.add_equipment(EQ)
    S.add_equipment({**EQ, "asset_id": "B", "status": "Degraded"})
    S.add_equipment({**EQ, "asset_id": "C"})
    r1 = S.create_maintenance({"asset_id": "C", "reported_issue": "x", "priority": "Critical"})
    r2 = S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "y", "priority": "Low"})
    S.update_maintenance(r2, {"status": "In Progress"})
    r3 = S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "z", "priority": "Low", "status": "Completed"})
    z = S.summary()
    assert z["equipment_total"] == 3 and z["equipment_by_status"]["Degraded"] == 1
    assert z["maintenance_by_status"] == {"Pending": 1, "In Progress": 1, "Completed": 1, "Cancelled": 0}
    assert z["critical_open"] == 1 and z["equipment_attention"] == 2        # B (degraded) + C (open critical)


def test_ai_support_levels_are_distinct():
    S.add_equipment(EQ)
    S.add_equipment({**EQ, "asset_id": "ENG-1", "type": "Aircraft Engines"})
    rec_ai = {"label": "NASA", "token": "tok1", "capability": "ai", "source_file": None}
    rec_sens = {"label": "pumps.csv", "token": "tok2", "capability": "sensor_only", "source_file": "pumps.csv"}
    e_p, e_e = S.get_equipment("PMP-001"), S.get_equipment("ENG-1")
    assert S.ai_support_level(e_p, S.list_links("PMP-001"))[0] == 1
    S.link_dataset("PMP-001", rec_sens, "PUMP-A")
    assert S.ai_support_level(e_p, S.list_links("PMP-001"))[0] == 2
    S.link_dataset("PMP-001", rec_ai, "5")          # a pump linked to AI-capable data is still NOT AI-supported
    assert S.ai_support_level(e_p, S.list_links("PMP-001"))[0] == 2
    assert S.ai_support_level(e_e, S.list_links("ENG-1"))[0] == 1
    S.link_dataset("ENG-1", rec_ai, "5")
    assert S.ai_support_level(e_e, S.list_links("ENG-1"))[0] == 3


def test_recommendation_to_record_link_and_duplicate_lookup():
    S.add_equipment(EQ)
    rid = S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "Sustained sensor drift", "priority": "Medium", "source": "Recommendation",
                                "source_ref": "R-SENSOR-SHIFT-E1", "dataset_ref": "tok2", "recommended_action": "Review trends"})
    f = S.find_record_by_source("R-SENSOR-SHIFT-E1", "tok2")
    assert f["id"] == rid and f["source"] == "Recommendation"
    assert S.find_record_by_source("R-SENSOR-SHIFT-E1", "other") is None


def test_analysis_and_report_records_stored_per_asset():
    S.add_equipment(EQ)
    rec = {"label": "pumps.csv", "token": "tok2", "capability": "sensor_only"}
    S.save_analysis("PMP-001", rec, "PUMP-A", {"state": "Drift detected", "priority": "Medium", "n_findings": 1, "last_cycle": 60, "summary": "s"})
    S.save_report("PMP-001", "Asset report", "r.pdf", rec, b"%PDF-1.4 x")
    assert list(S.list_analyses("PMP-001")["state"]) == ["Drift detected"]
    rp = S.list_reports("PMP-001")
    assert S.get_report_content(int(rp.iloc[0]["id"])) == b"%PDF-1.4 x"
    with pytest.raises(S.StorageError):
        S.save_analysis("GHOST", rec, "x", {})


def test_clear_session_does_not_touch_stored_data():
    S.add_equipment(EQ)
    S.create_maintenance({"asset_id": "PMP-001", "reported_issue": "x"})
    ss = {"a": 1}
    DS.clear_session(ss)
    assert len(S.list_equipment()) == 1 and len(S.list_maintenance()) == 1


def test_no_schema_destruction_on_reopen(db):
    S.add_equipment(EQ)
    for _ in range(3):
        with S.conn():
            pass
    assert len(S.list_equipment()) == 1
