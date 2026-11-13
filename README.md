# Industrial Rotating Equipment Maintenance Intelligence Platform 

A Streamlit decision-support application for condition monitoring and maintenance management of rotating equipment.
It combines (1) **dataset-driven analysis** with one globally active dataset, (2) an **AI model for one validated equipment type**
(NASA C-MAPSS FD001 aircraft turbofan engines: remaining-life estimate, anomaly detection, recommendations), (3) **statistical sensor analysis**
for any other valid sensor data, (4) an **equipment registry** and (5) **maintenance management** stored in SQLite, plus PDF/CSV reporting.

> **Scope & honesty.** The built-in dataset is simulated. The AI model supports **aircraft turbofan engines (C-MAPSS-compatible data) only**.
> Pumps, motors, compressors, fans and bearings can be *registered* and analysed with sensor statistics, but **no remaining-life prediction is made for them**.
> Findings are observations or model-based suspicions; only a physical inspection can confirm a fault.

## What changed in v2
* One **central dataset state** (`src/dataset_state.py`): load once, active on every page; no hidden fallback to NASA data; every result, report and cache is bound to a dataset token; changing or clearing the dataset invalidates everything derived from it.
* New navigation: Control Room · Fleet Overview · Equipment Management · Maintenance Management · Sensor & Degradation Analysis · Maintenance Recommendations · Reports & Export · Data Source · Settings. Model-development pages are hidden unless `MAINT_DEV_MODE=1`.
* Plain-language recommendations (detected condition, why, sensor evidence, priority, action, verification, status; finding type = detected anomaly / suspected degradation / data-quality notice; "confirmed fault" is never produced by the system).
* Upload validation for CSV/TSV/TXT/XLSX with specific errors; valid non-NASA data runs as **sensor-only analysis** and says why no remaining-life estimate exists.
* Reports (asset + fleet PDF) stamped with data source, dataset name and input mode; no model-development metrics.
* Priority 2: equipment registry (manual + CSV/Excel import with row-level rejection), maintenance records with full lifecycle and audit trail, recommendation-to-record conversion, dashboard counts from stored records.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+ (developed on 3.13)
pip install -r requirements.txt
pytest -q                          # 76 tests
streamlit run app.py               # http://localhost:8501
```
Optional environment variables: `MAINT_DEV_MODE=1` (show internal model pages), `MAINT_DB_PATH=/path/maintenance.db` (database location; default `storage/maintenance.db`).
Trained artifacts are in `models/` and `artifacts/` (`python -m src.train_models` reproduces them). If they are missing the app shows a setup message.

## Typical workflow
1. Open the app (no data is loaded). 2. **Data Source**: load the built-in NASA demo or upload a file → validation → *Make this the active dataset*.
3. Review **Control Room**, **Fleet Overview**, **Sensor & Degradation Analysis**. 4. **Maintenance Recommendations**: review findings, update action status, optionally *Create maintenance record*.
5. **Equipment Management**: register equipment (manual/import), link it to a dataset asset, save analyses/reports. 6. **Maintenance Management**: assign, progress and complete work; history is retained.
7. **Reports & Export**: asset or fleet PDF, CSV. 8. **Clear session** resets all dataset-dependent state (stored equipment/maintenance records are kept).

## Data and upload rules
Formats: CSV, TSV, whitespace TXT, XLSX (20 MB). Required: asset id column, cycle/time column (number or timestamp), ≥1 numeric measurement column; text asset ids (e.g. `PUMP-01`) and timestamps are supported. Missing values, duplicates, non-numeric values and unreadable files are rejected with specific messages and **never replace the active dataset**.
The AI path needs the 14 C-MAPSS sensors in the supported value range; otherwise the file is analysed as sensor-only.

## Equipment registry and maintenance (SQLite)
Tables: `equipment` (unique Asset ID), `maintenance_records` (FK → equipment, enum-checked), `maintenance_events` (audit trail), `asset_datasets` (equipment ↔ dataset asset links), `analysis_records`, `report_records`. Foreign keys are enforced; equipment with maintenance history cannot be deleted; completed records are retained; imports are validated row by row and inserted atomically.
Three levels are kept distinct: registered → sensor data linked → supported by the implemented AI model (only Aircraft Engines with C-MAPSS-compatible data).
**Streamlit Community Cloud:** the container disk is ephemeral across redeploys/reboots, so the database persists across app restarts only while the container lives. For durable storage set `MAINT_DB_PATH` to a persistent volume or export regularly.

## Deployment (Streamlit Community Cloud)
Push the repository, create the app with `app.py` as entry point, Python 3.11+, `requirements.txt` as dependency file. No secrets are required. To update, push to the tracked branch; the app redeploys. Set `MAINT_DEV_MODE` only on private deployments.

## Method
| Item | Choice |
|---|---|
| Train target | `RUL = final_cycle − cycle`; raw and capped (`RUL_CAP = 125`, configurable in `src/config.py`) kept in separate columns |
| Test target | One official RUL per engine, valid **only at the last observed cycle**; never assigned row-wise |
| Features (70) | per sensor: current value, 10-cycle rolling mean / std / slope, shift vs the engine's own early baseline (first 5 cycles). All **causal** (past data only; unit-tested) |
| Validation | Engine-grouped 5-fold CV (`GroupKFold`) for feature-set choice, tuning and model selection. Official test evaluated **once**, at final cycles |
| Candidates | Dummy (mean), Ridge, Random Forest, HistGradientBoosting, XGBoost — small tuning grids, seed 42 |
| Selection | Lowest CV RMSE; among models within 2% prefer a SHAP-supported one → **XGBoost** |

### Measured results (cycles; produced by `src.train_models`)
| Model | CV MAE | CV RMSE | CV R² | RMSE late-life (RUL≤60) | RMSE random cut-off/engine | Train s |
|---|---|---|---|---|---|---|
| **XGBoost** | 8.71 | **12.58** | 0.91 | 10.53 | 13.71 | 1.3 |
| HistGradientBoosting | 8.72 | 12.68 | 0.91 | 10.65 | 13.79 | 1.5 |
| Random Forest | 9.04 | 12.80 | 0.91 | 11.13 | 14.10 | 12.6 |
| Ridge | 12.40 | 15.63 | 0.86 | 17.92 | 16.98 | 0.02 |
| Dummy (mean) | 36.97 | 41.67 | −0.00 | 59.49 | 42.32 | 0.0 |

**Official test (100 engines, final cycle, XGBoost):** vs. capped truth MAE 9.58, RMSE 13.66, R² 0.884, NASA score 275; vs. raw (uncapped) truth MAE 10.65, RMSE 14.67, R² 0.875, NASA score 301.
The capped comparison puts prediction and truth on the same scale; the raw comparison is shown for transparency (engines with true RUL > 125 are under-predicted by construction).
XGBoost and HistGradientBoosting differ by <1% CV RMSE — not a meaningful gap.

**Feature ablation** (HistGB, same CV): current sensors only 18.64 RMSE → + rolling stats 18.00 → + shift vs engine baseline **12.81**. Adding the raw cycle count lowered CV RMSE further (11.11), but it was deliberately **not used**:
the pipeline then depends only on sensor behaviour (not on cycle bookkeeping that is awkward for uploads and what-if). This is a trade-off, listed under future work.

**Derived states** (Healthy > 60, Warning 30–60, Critical < 30 predicted cycles; configurable) are decision-support rules. They have no independent labels, so **no accuracy is reported** for them.

## Model reference (internal / development mode)
* **SHAP** (`TreeExplainer`): global importance + summary plot on a documented 1,500-row training sample (seed 42); local explanation per engine/cycle, grouped by sensor; text generated from the computed values. SHAP = model attribution, **not causation**; no component diagnosis is made. If an explainer is unavailable the UI says so.
* **Anomaly detector** (separate from RUL): Isolation Forest fitted on early-life (cycles 10–50) 10-cycle-mean shifts from each engine's baseline; flag threshold = 99th percentile of reference scores. Sanity check on training data (no fault labels exist): flagged 9.9% of rows when true RUL > 125, 64.3% for 60–125 and 99.9% for ≤ 60 (Spearman ρ = 0.88 with elapsed life). This shows the score tracks progression — it is not a fault-classification accuracy.
* **Rules registry** (`src/recommendation_engine.py`): each rule has ID, trigger, evidence, priority, evidence-strength category (Strong / Moderate / Informational, defined in the code and UI), source, limitation, verification step and status (Pending / In Progress / Completed / Dismissed — session-only, exportable). Rules are application conventions; none maps a sensor to a component or promises a quantified life extension. Sensor-shift findings read "potential contributing pattern identified; further engineering inspection required".

## Application map
Control Room · Fleet Overview · Equipment Management · Maintenance Management · Sensor & Degradation Analysis · Maintenance Recommendations · Reports & Export · Data Source · Settings.
Development mode adds: Engine Intelligence · Explainable AI · What-if · Model Performance.

## Project layout
```
app.py, pages/*.py (+ pages/dev/)   Streamlit UI (thin; logic in src/)
src/dataset_state.py                central dataset/session management        src/app_state.py   Streamlit glue, caches, sidebar
src/upload_flow.py, upload_mapping.py, data_validation.py, data_loader.py     upload validation and classification
src/analysis_service.py, generic_analysis.py, fleet_analysis.py, engine_analysis.py   analysis of the active dataset
src/recommendation_engine.py, anomaly_detection.py, inference.py, explainability.py, what_if_engine.py   model-side logic
src/storage.py, dashboard_panels.py equipment registry + maintenance (SQLite)
src/report_generator.py, ui_components.py, config.py
models/ artifacts/ data/ docs/ tests/ (test_pipeline.py, test_p1.py, test_p2.py, e2e/)
```

## Testing
`pytest -q` → **76 passed** (32 original pipeline tests, 20 Priority-1 tests, 24 Priority-2 tests). Browser evidence (real Chromium): `tests/e2e/p1_journey.py` (35 checks) and `tests/e2e/p2_workflow.py` (32 checks incl. an application restart); results in `docs/`.

## Troubleshooting
* *"Trained model artifacts were not found"* → `python -m src.train_models`.
* *Upload rejected* → read the message; fix missing/duplicate/non-numeric values or choose the right columns.
* *Database path not writable* → set `MAINT_DB_PATH`.

## Limitations
Simulated single-condition data for the AI model; AI only for turbofan engines; estimates have an upper limit and error; sensors are not mapped to components; no live telemetry; recommendation action statuses are session-only (maintenance records are persistent); Cloud storage is ephemeral; no user accounts or permissions.

## References
A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08 (`data/docs`). NASA PCoE C-MAPSS dataset. S. Lundberg & S.-I. Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.
