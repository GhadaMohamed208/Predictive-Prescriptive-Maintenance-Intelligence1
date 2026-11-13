"""Persistent storage (SQLite) for the equipment registry and maintenance management.

* Database file: $MAINT_DB_PATH or <project>/storage/maintenance.db (created on first use; never dropped or truncated).
* Foreign keys are enforced; equipment that has maintenance history cannot be deleted (history is retained).
* All writes are validated here, not only in the UI. Functions raise ``StorageError`` with a user-readable message.
* Schema changes use CREATE TABLE IF NOT EXISTS + ``PRAGMA user_version`` so existing data is never altered destructively.

NOTE (Streamlit Community Cloud): the container file system is ephemeral across redeploys/reboots. Data survive app restarts
on the same container and all local runs; export CSV regularly or point MAINT_DB_PATH at a persistent volume."""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import config as C

SCHEMA_VERSION = 1


class StorageError(ValueError):
    """A validation / integrity problem with a user-readable message."""


def db_path() -> Path:
    return Path(os.environ.get(C.DB_PATH_ENV) or C.DEFAULT_DB_PATH)


@contextmanager
def conn():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path), timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    try:
        _init(c)
        yield c
        c.commit()
    except sqlite3.IntegrityError as e:
        c.rollback()
        raise StorageError(f"Database integrity check failed: {e}") from e
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _q(items):
    return ",".join("'" + i.replace("'", "''") + "'" for i in items)


def _init(c: sqlite3.Connection) -> None:
    if c.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
        return
    c.executescript(f"""
    CREATE TABLE IF NOT EXISTS equipment (
        asset_id TEXT PRIMARY KEY, name TEXT NOT NULL,
        type TEXT NOT NULL CHECK (type IN ({_q(C.EQUIPMENT_TYPES)})),
        manufacturer TEXT, model TEXT, serial_number TEXT, installation_date TEXT, location TEXT, department TEXT,
        status TEXT NOT NULL CHECK (status IN ({_q(C.EQUIPMENT_STATUSES)})), notes TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS maintenance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT NOT NULL REFERENCES equipment(asset_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        type TEXT NOT NULL CHECK (type IN ({_q(C.MAINT_TYPES)})), reported_issue TEXT NOT NULL, description TEXT,
        recommended_action TEXT, technician TEXT,
        priority TEXT NOT NULL CHECK (priority IN ({_q(C.MAINT_PRIORITIES)})),
        created_date TEXT NOT NULL, planned_date TEXT,
        status TEXT NOT NULL CHECK (status IN ({_q(C.MAINT_STATUSES)})),
        completion_date TEXT, work_performed TEXT, notes TEXT,
        source TEXT NOT NULL DEFAULT 'Manual', source_ref TEXT, dataset_ref TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS ix_mr_asset ON maintenance_records(asset_id);
    CREATE INDEX IF NOT EXISTS ix_mr_status ON maintenance_records(status);
    CREATE TABLE IF NOT EXISTS maintenance_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id INTEGER NOT NULL REFERENCES maintenance_records(id) ON DELETE RESTRICT,
        at TEXT NOT NULL, event TEXT NOT NULL, detail TEXT);
    CREATE TABLE IF NOT EXISTS asset_datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT NOT NULL REFERENCES equipment(asset_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        dataset_label TEXT NOT NULL, source_file TEXT, dataset_token TEXT NOT NULL, capability TEXT NOT NULL,
        dataset_asset TEXT NOT NULL, linked_at TEXT NOT NULL, UNIQUE(asset_id, dataset_token, dataset_asset));
    CREATE TABLE IF NOT EXISTS analysis_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT NOT NULL REFERENCES equipment(asset_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        dataset_label TEXT NOT NULL, dataset_token TEXT NOT NULL, dataset_asset TEXT, analyzed_at TEXT NOT NULL,
        state TEXT, rul REAL, priority TEXT, n_findings INTEGER, last_cycle INTEGER, summary TEXT);
    CREATE TABLE IF NOT EXISTS report_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id TEXT REFERENCES equipment(asset_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        report_type TEXT NOT NULL, filename TEXT NOT NULL, dataset_label TEXT, dataset_token TEXT,
        generated_at TEXT NOT NULL, content BLOB);
    PRAGMA user_version = {SCHEMA_VERSION};
    """)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today() -> str:
    return date.today().isoformat()


def _clean(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def parse_date(v, field: str, required: bool = False) -> str | None:
    s = _clean(v)
    if s is None:
        if required:
            raise StorageError(f"{field} is required.")
        return None
    try:
        return pd.to_datetime(s, errors="raise").date().isoformat()
    except Exception as e:
        raise StorageError(f"{field} '{s}' is not a valid date (use YYYY-MM-DD).") from e


# ============================================================ equipment
EQ_FIELDS = ["asset_id", "name", "type", "manufacturer", "model", "serial_number", "installation_date", "location",
             "department", "status", "notes"]
_ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/ ]{0,63}$")


def normalise_equipment(d: dict) -> dict:
    """Validate and normalise one equipment dict (raises StorageError)."""
    out = {k: _clean(d.get(k)) for k in EQ_FIELDS}
    if not out["asset_id"]:
        raise StorageError("Asset ID is required.")
    if not _ID_OK.match(out["asset_id"]):
        raise StorageError(f"Asset ID '{out['asset_id']}' is invalid (letters, digits, . _ - / and spaces; max 64 characters).")
    if not out["name"]:
        raise StorageError("Equipment name is required.")
    t = next((x for x in C.EQUIPMENT_TYPES if out["type"] and x.lower() == out["type"].lower()), None)
    if t is None:
        raise StorageError(f"Type '{out['type']}' is not allowed. Allowed: {', '.join(C.EQUIPMENT_TYPES)}.")
    out["type"] = t
    s = next((x for x in C.EQUIPMENT_STATUSES if out["status"] and x.lower() == out["status"].lower()), None) if out["status"] else "Operational"
    if s is None:
        raise StorageError(f"Operational status '{out['status']}' is not allowed. Allowed: {', '.join(C.EQUIPMENT_STATUSES)}.")
    out["status"] = s
    out["installation_date"] = parse_date(out["installation_date"], "Installation date")
    return out


def add_equipment(d: dict) -> dict:
    e = normalise_equipment(d)
    with conn() as c:
        if c.execute("SELECT 1 FROM equipment WHERE lower(asset_id)=lower(?)", (e["asset_id"],)).fetchone():
            raise StorageError(f"Asset ID '{e['asset_id']}' already exists. Asset IDs must be unique.")
        e["created_at"] = e["updated_at"] = now()
        c.execute(f"INSERT INTO equipment ({','.join(e)}) VALUES ({','.join('?' * len(e))})", list(e.values()))
    return e


def update_equipment(asset_id: str, d: dict) -> dict:
    d = {**d, "asset_id": asset_id}            # Asset ID is the key and cannot be changed
    e = normalise_equipment(d)
    with conn() as c:
        if not c.execute("SELECT 1 FROM equipment WHERE asset_id=?", (asset_id,)).fetchone():
            raise StorageError(f"Equipment '{asset_id}' does not exist.")
        e["updated_at"] = now()
        sets = [k for k in e if k not in ("asset_id",)]
        c.execute(f"UPDATE equipment SET {','.join(k + '=?' for k in sets)} WHERE asset_id=?", [e[k] for k in sets] + [asset_id])
    return e


def get_equipment(asset_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM equipment WHERE asset_id=?", (asset_id,)).fetchone()
    return dict(r) if r else None


def list_equipment() -> pd.DataFrame:
    with conn() as c:
        rows = c.execute("SELECT * FROM equipment ORDER BY asset_id").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=EQ_FIELDS + ["created_at", "updated_at"])


def delete_equipment(asset_id: str) -> None:
    """Delete only equipment without maintenance history; links/analyses/reports of a deletable asset go with it."""
    with conn() as c:
        if not c.execute("SELECT 1 FROM equipment WHERE asset_id=?", (asset_id,)).fetchone():
            raise StorageError(f"Equipment '{asset_id}' does not exist.")
        n = c.execute("SELECT COUNT(*) FROM maintenance_records WHERE asset_id=?", (asset_id,)).fetchone()[0]
        if n:
            raise StorageError(f"'{asset_id}' has {n} maintenance record(s). Maintenance history is retained, so the equipment cannot be "
                               "deleted. Set its operational status to 'Out of Service' instead.")
        for t in ("asset_datasets", "analysis_records", "report_records"):
            c.execute(f"DELETE FROM {t} WHERE asset_id=?", (asset_id,))
        c.execute("DELETE FROM equipment WHERE asset_id=?", (asset_id,))


_ALIASES = {
    "asset_id": {"assetid", "asset", "id", "equipmentid", "tag", "assetnumber", "assetno"},
    "name": {"name", "equipmentname", "assetname", "description", "equipment"},
    "type": {"type", "equipmenttype", "assettype", "category"},
    "manufacturer": {"manufacturer", "make", "vendor", "brand"},
    "model": {"model", "modelnumber"},
    "serial_number": {"serialnumber", "serial", "serialno", "sn"},
    "installation_date": {"installationdate", "installdate", "installed", "dateinstalled", "commissioningdate"},
    "location": {"location", "facility", "site", "plant"},
    "department": {"department", "productionarea", "area", "dept"},
    "status": {"operationalstatus", "status", "state"},
    "notes": {"notes", "note", "remarks", "comments"},
}


def _norm(c) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def parse_equipment_table(df: pd.DataFrame) -> dict:
    """Validate an imported table row by row. Never writes. Returns
    {'valid': [dict], 'rejected': DataFrame(row, asset_id, reason), 'unmapped': [cols], 'missing_columns': [..]}"""
    colmap: dict[str, str] = {}
    for c in df.columns:
        n = _norm(c)
        for k, al in _ALIASES.items():
            if k not in colmap and n in al:
                colmap[k] = c
                break
    missing = [k for k in ("asset_id", "name", "type") if k not in colmap]
    res = {"valid": [], "rejected": [], "unmapped": [c for c in df.columns if c not in colmap.values()], "missing_columns": missing}
    if missing:
        return {**res, "rejected": pd.DataFrame(columns=["row", "asset_id", "reason"])}
    existing = {a.lower() for a in list_equipment()["asset_id"]}
    seen: dict[str, int] = {}
    for i, r in df.iterrows():
        rown = int(i) + 2                      # +2: header row + 1-based
        raw = {k: r[col] for k, col in colmap.items()}
        aid = _clean(raw.get("asset_id"))
        try:
            e = normalise_equipment(raw)
            if e["asset_id"].lower() in existing:
                raise StorageError(f"Asset ID '{e['asset_id']}' already exists in the registry.")
            if e["asset_id"].lower() in seen:
                raise StorageError(f"Asset ID '{e['asset_id']}' is duplicated in the file (first at row {seen[e['asset_id'].lower()]}).")
            seen[e["asset_id"].lower()] = rown
            res["valid"].append(e)
        except StorageError as ex:
            res["rejected"].append({"row": rown, "asset_id": aid or "", "reason": str(ex)})
    res["rejected"] = pd.DataFrame(res["rejected"], columns=["row", "asset_id", "reason"])
    return res


def import_equipment(valid_rows: list[dict]) -> int:
    """Insert already-validated rows atomically (all or nothing). Returns the number inserted."""
    with conn() as c:
        for e in valid_rows:
            if c.execute("SELECT 1 FROM equipment WHERE lower(asset_id)=lower(?)", (e["asset_id"],)).fetchone():
                raise StorageError(f"Asset ID '{e['asset_id']}' already exists; nothing was imported.")
            e = {**e, "created_at": now(), "updated_at": now()}
            c.execute(f"INSERT INTO equipment ({','.join(e)}) VALUES ({','.join('?' * len(e))})", list(e.values()))
    return len(valid_rows)


def import_template() -> pd.DataFrame:
    return pd.DataFrame([{"Asset ID": "PMP-001", "Equipment Name": "Cooling water pump 1", "Type": "Pumps", "Manufacturer": "ExampleCo",
                          "Model": "CW-200", "Serial Number": "SN12345", "Installation Date": "2021-03-15", "Location": "Plant A",
                          "Department": "Utilities", "Operational Status": "Operational", "Notes": ""}])


def ai_support_level(eq: dict, links: pd.DataFrame) -> tuple[int, str]:
    """(level, text): 1 registered only, 2 sensor data linked, 3 supported by an implemented model."""
    if eq["type"] in C.AI_SUPPORTED_TYPES and len(links) and (links["capability"] == "ai").any():
        return 3, "Registered · sensor data linked · supported by the implemented AI model (C-MAPSS turbofan)"
    if len(links):
        return 2, "Registered · sensor data linked · statistical sensor analysis only (no AI model for this equipment type or data)"
    if eq["type"] in C.AI_SUPPORTED_TYPES:
        return 1, "Registered only · AI-capable type, but no sensor data linked yet"
    return 1, "Registered only · no sensor data linked · no AI model exists for this equipment type"


# ============================================================ maintenance
_TRANSITIONS = {"Pending": {"In Progress", "Completed", "Cancelled"}, "In Progress": {"Pending", "Completed", "Cancelled"},
                "Completed": set(), "Cancelled": {"Pending"}}
OPEN = ("Pending", "In Progress")


def allowed_transitions(status: str) -> list[str]:
    return [status] + sorted(_TRANSITIONS[status])


def mr_label(i: int) -> str:
    return f"MR-{int(i):05d}"


def _check_enum(v, allowed, field):
    if v not in allowed:
        raise StorageError(f"{field} '{v}' is not allowed. Allowed: {', '.join(allowed)}.")
    return v


def _event(c, rid, event, detail=""):
    c.execute("INSERT INTO maintenance_events (record_id, at, event, detail) VALUES (?,?,?,?)", (rid, now(), event, detail))


def create_maintenance(d: dict) -> int:
    asset_id = _clean(d.get("asset_id"))
    if not asset_id:
        raise StorageError("Related Asset ID is required.")
    issue = _clean(d.get("reported_issue"))
    if not issue:
        raise StorageError("Reported issue is required.")
    rec = {"asset_id": asset_id, "type": _check_enum(_clean(d.get("type")) or "Corrective", C.MAINT_TYPES, "Maintenance type"),
           "reported_issue": issue, "description": _clean(d.get("description")), "recommended_action": _clean(d.get("recommended_action")),
           "technician": _clean(d.get("technician")), "priority": _check_enum(_clean(d.get("priority")) or "Medium", C.MAINT_PRIORITIES, "Priority"),
           "created_date": parse_date(d.get("created_date"), "Created date") or today(),
           "planned_date": parse_date(d.get("planned_date"), "Planned date"),
           "status": _check_enum(_clean(d.get("status")) or "Pending", C.MAINT_STATUSES, "Status"),
           "completion_date": parse_date(d.get("completion_date"), "Completion date"), "work_performed": _clean(d.get("work_performed")),
           "notes": _clean(d.get("notes")), "source": _clean(d.get("source")) or "Manual", "source_ref": _clean(d.get("source_ref")),
           "dataset_ref": _clean(d.get("dataset_ref"))}
    if rec["status"] == "Completed" and not rec["completion_date"]:
        rec["completion_date"] = today()
    if rec["status"] != "Completed" and rec["completion_date"]:
        raise StorageError("A completion date can only be set when the status is Completed.")
    with conn() as c:
        if not c.execute("SELECT 1 FROM equipment WHERE asset_id=?", (asset_id,)).fetchone():
            raise StorageError(f"Asset ID '{asset_id}' is not in the equipment registry. Register the equipment first.")
        rec["created_at"] = rec["updated_at"] = now()
        cur = c.execute(f"INSERT INTO maintenance_records ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", list(rec.values()))
        _event(c, cur.lastrowid, "Created", f"Source: {rec['source']}")
        return int(cur.lastrowid)


def get_maintenance(rid: int) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM maintenance_records WHERE id=?", (int(rid),)).fetchone()
    return dict(r) if r else None


def list_maintenance(asset_id: str | None = None) -> pd.DataFrame:
    with conn() as c:
        if asset_id:
            rows = c.execute("SELECT m.*, e.name AS asset_name FROM maintenance_records m JOIN equipment e USING(asset_id) "
                             "WHERE m.asset_id=? ORDER BY m.id DESC", (asset_id,)).fetchall()
        else:
            rows = c.execute("SELECT m.*, e.name AS asset_name FROM maintenance_records m JOIN equipment e USING(asset_id) ORDER BY m.id DESC").fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if len(df):
        df.insert(0, "record", df["id"].map(mr_label))
    return df


def update_maintenance(rid: int, changes: dict) -> dict:
    """Edit fields (technician, notes, work performed, priority, dates, status ...). Completed records stay retained."""
    allowed = {"type", "reported_issue", "description", "recommended_action", "technician", "priority", "planned_date", "status",
               "completion_date", "work_performed", "notes"}
    bad = set(changes) - allowed
    if bad:
        raise StorageError("Fields cannot be changed: " + ", ".join(sorted(bad)))
    with conn() as c:
        cur = c.execute("SELECT * FROM maintenance_records WHERE id=?", (int(rid),)).fetchone()
        if not cur:
            raise StorageError(f"Maintenance record {rid} does not exist.")
        cur = dict(cur)
        new = {k: (_clean(v) if not k.endswith("date") else parse_date(v, k.replace("_", " ").capitalize())) for k, v in changes.items()}
        if "type" in new:
            _check_enum(new["type"], C.MAINT_TYPES, "Maintenance type")
        if "priority" in new:
            _check_enum(new["priority"], C.MAINT_PRIORITIES, "Priority")
        if "reported_issue" in new and not new["reported_issue"]:
            raise StorageError("Reported issue cannot be empty.")
        status = new.get("status", cur["status"])
        _check_enum(status, C.MAINT_STATUSES, "Status")
        if status != cur["status"]:
            if status not in _TRANSITIONS[cur["status"]]:
                raise StorageError(f"Status cannot change from {cur['status']} to {status}.")
            if status == "Completed":
                new["completion_date"] = new.get("completion_date") or today()
            else:
                new["completion_date"] = None
        elif cur["status"] == "Completed":
            # completed records are retained; only notes-type fields may still be edited
            locked = {"type", "reported_issue", "priority", "planned_date", "status"} & set(new)
            if any(new[k] != cur[k] for k in locked):
                raise StorageError("A completed record is retained as-is; only technician, notes and work performed can be amended.")
        if new.get("completion_date") and status != "Completed":
            raise StorageError("A completion date can only be set when the status is Completed.")
        diffs = [k for k in new if new[k] != cur.get(k)]
        if not diffs:
            return cur
        new["updated_at"] = now()
        c.execute(f"UPDATE maintenance_records SET {','.join(k + '=?' for k in new)} WHERE id=?", list(new.values()) + [int(rid)])
        for k in diffs:
            if k == "status":
                _event(c, rid, "Status changed", f"{cur['status']} -> {new['status']}")
            elif k == "technician":
                _event(c, rid, "Technician assigned", str(new[k]))
            elif k in ("notes", "work_performed"):
                _event(c, rid, f"{k.replace('_', ' ').capitalize()} updated", (new[k] or "")[:200])
            elif k != "completion_date":
                _event(c, rid, f"{k.replace('_', ' ').capitalize()} changed", str(new[k]))
    return get_maintenance(rid)


def maintenance_events(rid: int) -> pd.DataFrame:
    with conn() as c:
        rows = c.execute("SELECT at, event, detail FROM maintenance_events WHERE record_id=? ORDER BY id", (int(rid),)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["at", "event", "detail"])


def find_record_by_source(source_ref: str, dataset_ref: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM maintenance_records WHERE source_ref=? AND dataset_ref=? ORDER BY id DESC LIMIT 1", (source_ref, dataset_ref)).fetchone()
    return dict(r) if r else None


# ============================================================ links, analyses, reports
def link_dataset(asset_id: str, record: dict, dataset_asset: str) -> None:
    with conn() as c:
        if not c.execute("SELECT 1 FROM equipment WHERE asset_id=?", (asset_id,)).fetchone():
            raise StorageError(f"Asset ID '{asset_id}' is not in the equipment registry.")
        c.execute("INSERT OR IGNORE INTO asset_datasets (asset_id, dataset_label, source_file, dataset_token, capability, dataset_asset, linked_at) "
                  "VALUES (?,?,?,?,?,?,?)", (asset_id, record["label"], record.get("source_file"), record["token"], record["capability"], str(dataset_asset), now()))


def list_links(asset_id: str) -> pd.DataFrame:
    with conn() as c:
        rows = c.execute("SELECT * FROM asset_datasets WHERE asset_id=? ORDER BY id DESC", (asset_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "asset_id", "dataset_label", "source_file", "dataset_token", "capability", "dataset_asset", "linked_at"])


def find_equipment_for_dataset_asset(token: str, dataset_asset: str) -> str | None:
    with conn() as c:
        r = c.execute("SELECT asset_id FROM asset_datasets WHERE dataset_token=? AND dataset_asset=? LIMIT 1", (token, str(dataset_asset))).fetchone()
        if r:
            return r["asset_id"]
        r = c.execute("SELECT asset_id FROM equipment WHERE lower(asset_id)=lower(?)", (str(dataset_asset),)).fetchone()
    return r["asset_id"] if r else None


def save_analysis(asset_id: str, record: dict, dataset_asset: str, row: dict) -> int:
    with conn() as c:
        if not c.execute("SELECT 1 FROM equipment WHERE asset_id=?", (asset_id,)).fetchone():
            raise StorageError(f"Asset ID '{asset_id}' is not in the equipment registry.")
        cur = c.execute("INSERT INTO analysis_records (asset_id, dataset_label, dataset_token, dataset_asset, analyzed_at, state, rul, priority, n_findings, last_cycle, summary) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (asset_id, record["label"], record["token"], str(dataset_asset), now(), row.get("state"), row.get("rul"),
                                                           row.get("priority"), row.get("n_findings"), row.get("last_cycle"), row.get("summary")))
        return int(cur.lastrowid)


def list_analyses(asset_id: str) -> pd.DataFrame:
    with conn() as c:
        rows = c.execute("SELECT * FROM analysis_records WHERE asset_id=? ORDER BY id DESC", (asset_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "asset_id", "dataset_label", "dataset_token", "dataset_asset", "analyzed_at", "state", "rul", "priority", "n_findings", "last_cycle", "summary"])


def save_report(asset_id: str | None, report_type: str, filename: str, record: dict, content: bytes) -> int:
    with conn() as c:
        if asset_id and not c.execute("SELECT 1 FROM equipment WHERE asset_id=?", (asset_id,)).fetchone():
            raise StorageError(f"Asset ID '{asset_id}' is not in the equipment registry.")
        cur = c.execute("INSERT INTO report_records (asset_id, report_type, filename, dataset_label, dataset_token, generated_at, content) VALUES (?,?,?,?,?,?,?)",
                        (asset_id, report_type, filename, record["label"], record["token"], now(), content))
        return int(cur.lastrowid)


def list_reports(asset_id: str) -> pd.DataFrame:
    with conn() as c:
        rows = c.execute("SELECT id, asset_id, report_type, filename, dataset_label, generated_at FROM report_records WHERE asset_id=? ORDER BY id DESC", (asset_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "asset_id", "report_type", "filename", "dataset_label", "generated_at"])


def get_report_content(rid: int) -> bytes | None:
    with conn() as c:
        r = c.execute("SELECT content FROM report_records WHERE id=?", (int(rid),)).fetchone()
    return bytes(r["content"]) if r and r["content"] is not None else None


# ============================================================ dashboard numbers (stored records only)
def summary() -> dict:
    with conn() as c:
        eq = [dict(r) for r in c.execute("SELECT status, COUNT(*) n FROM equipment GROUP BY status")]
        mr = [dict(r) for r in c.execute("SELECT status, COUNT(*) n FROM maintenance_records GROUP BY status")]
        crit = c.execute("SELECT COUNT(*) FROM maintenance_records WHERE priority='Critical' AND status IN ('Pending','In Progress')").fetchone()[0]
        attn = c.execute("SELECT COUNT(*) FROM equipment e WHERE e.status IN ('Degraded','Under Maintenance','Out of Service') OR EXISTS "
                         "(SELECT 1 FROM maintenance_records m WHERE m.asset_id=e.asset_id AND m.status IN ('Pending','In Progress') "
                         "AND m.priority IN ('Critical','High'))").fetchone()[0]
    ec = {s: 0 for s in C.EQUIPMENT_STATUSES}
    ec.update({r["status"]: r["n"] for r in eq})
    mc = {s: 0 for s in C.MAINT_STATUSES}
    mc.update({r["status"]: r["n"] for r in mr})
    return {"equipment_total": sum(ec.values()), "equipment_by_status": ec, "maintenance_by_status": mc,
            "maintenance_total": sum(mc.values()), "critical_open": int(crit), "equipment_attention": int(attn)}
