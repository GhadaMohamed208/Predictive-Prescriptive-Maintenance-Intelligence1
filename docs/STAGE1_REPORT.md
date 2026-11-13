# Stage 1 – Inspection report: current architecture, issues, plan

Date: 2026-11-05 · Baseline: v1.0.0 (backup kept in `industrial-maintenance-ai-backup-before-dev/`).

## 1. Current architecture (verified from source and a real-browser run)

* One Streamlit process (`app.py`, `st.navigation`, 11 pages in `pages/`), logic in `src/`, trained artifacts in `models/` + `artifacts/`, NASA FD001 files in `data/`. No backend/API/DB; inference is local.
* Data source model: `src/app_state.py::dataset()` returns the built-in test set, the training set or the uploaded frame, chosen by `st.session_state["source"]`.
* Upload: `pages/upload_analyze.py` keeps its own state in `session_state["upload"]` and a one-shot `_pending_source`.
* Results are re-computed per page; some are cached with `st.cache_data` (fleet table, per-engine recommendations); action statuses, scenarios and PDFs live in `session_state`.
* Reports: ReportLab (`src/report_generator.py`), fed by `src/engine_analysis.py`.
* 11 pages, many exposing ML internals (Model Performance, Explainable AI incl. raw SHAP, What-if, Settings JSON, train/test tables, RMSE/MAE, "out-of-range features", "at cap", "official true RUL").

## 2. Identified problems (each mapped to your requirements)

| # | Problem (evidence) | Req. |
|---|---|---|
| A | **Active source is a widget key** (`source`, sidebar selectbox). Streamlit drops widget-key state on page change, `setup_page()` then falls back to `SRC_TEST`. Reproduced in the browser: after *Run analysis → Open in Fleet Overview* the page shows the official NASA data. | P1-1, 9 |
| B | `dataset()` silently returns the NASA test set when `source == Uploaded` but no upload exists (hidden fallback). | P1-1 |
| C | Dependent results are not invalidated on dataset change: `recs_for` cache key uses dataset *name* + engine id (two uploads with the same name collide); `rec_status` keys are `RULE-E<id>` (engine ids collide across datasets); `analysis`, `scenarios`, `pdf_engine`, `pdf_fleet` survive a source change. | P1-1.7 |
| D | **Sensor & Degradation** EDA tab always uses built-in train/test; degradation reference band uses NASA training data regardless of the active dataset. | P1-2 |
| E | **Clear session** removes only 6 keys; upload, source, selected engine, thresholds-independent caches, widget state remain. | P1-3 |
| F | Navigation is ML-development oriented (11 items incl. Model Performance, Explainable AI, What-if, Settings JSON). | P1-4, 5 |
| G | UI shows train/test, CV, RMSE/MAE/R², SHAP plots, model config JSON, OOD jargon. | P1-5 |
| H | Reports: *Input mode* always "Compatible engine trajectory"; "model and evaluation context" shows RMSE; no sensor-evidence table tied to the dataset; no dataset/source stamp that distinguishes built-in vs uploaded; stale PDF can be downloaded after the source changes. | P1-8 |
| I | Upload only accepts the 14-sensor schema; any other schema is a dead end (Mode C = rejected), extra columns dropped silently, RUL file matched by order only. | P1-9 |
| J | Recommendation wording mixes finding types; no explicit Detected anomaly / Suspected fault / Confirmed fault distinction. | P1-7 |

## 3. Implementation plan

**Stage 2 (Priority 1)**
1. New `src/dataset_state.py` – single source of truth (`ActiveDataset` record, `activate_nasa`, `activate_upload`, `clear_session`, `dataset_token`, invalidation of dependent keys). Never falls back; "no dataset" is an explicit state. Sidebar selector becomes a view over that state (widget value copied back each run).
2. `app_state.dataset()` delegates to it; pages call `require_dataset()` which shows a "choose a data source" panel instead of loading anything.
3. Upload workflow: validation messages, capability levels – *AI-supported* (14 sensors, RUL available) vs *sensor-data only* (valid trajectories without model-compatible schema: trends/statistics, no RUL); RUL file matched with a verification step; extra columns reported.
4. New navigation: Control Room · Fleet Overview · Sensor & Degradation Analysis · Maintenance Recommendations · Reports & Export · Data Upload · Settings. Dev pages (Model Performance, Explainable AI, What-if, model internals) moved under `pages/dev/`, visible only with `MAINT_DEV_MODE=1`.
5. Redesigned Control Room / Fleet Overview (supported KPIs only, filters/search, status & priority distribution, "historical/uploaded data" labelling).
6. Maintenance Recommendations page: plain-language findings with explicit finding class (Detected anomaly / Suspected degradation / Confirmed fault = only by human inspection / Recommended action), sensor evidence, verification, status.
7. Reports: real input-mode label, dataset stamp, sensor evidence, no model-development metrics, stale-PDF protection (PDF bound to dataset token).
8. Clear Session wipes every dataset-dependent key.

**Stage 3** – unit tests for state management, upload validation and reports + a real-browser run of the 13-step journey and invalid uploads.

**Stage 4 (Priority 2)** – SQLite (`storage/maintenance.db`, path overridable) with foreign keys: `equipment`, `maintenance_records`, `analysis_records`, `report_records`; repositories, CSV/Excel import with row-level validation, two new pages, dashboard integration, "create maintenance record from recommendation" (manual path works without AI). openpyxl added for Excel.

**Stage 5** – repository tests + browser workflow incl. app restart.

## 4. Conflicts / decisions needing a note
* Streamlit Community Cloud's disk is ephemeral across redeploys/reboots; SQLite persists across *app restarts on the same container* and local runs. The DB path is configurable and the README documents backup/export; a hosted DB is the safe next step if long-term retention on Cloud is required.
* Upload is not in the suggested navigation but is required to meet the upload requirements; it is kept as "Data Upload". Settings is kept in slim form for alert thresholds and Clear Session.
