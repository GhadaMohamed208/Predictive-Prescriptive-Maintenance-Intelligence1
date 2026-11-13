"""Central, documented configuration. Nothing here is a NASA standard unless stated."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
ASSETS_DIR = ROOT / "assets"

PRODUCT_NAME = "Industrial Rotating Equipment Maintenance Intelligence Platform"
PRODUCT_SHORT = "Maintenance Intelligence Platform"
APP_VERSION = "2.0.0"
RANDOM_SEED = 42

# ---- Dataset schema (NASA C-MAPSS: 26 whitespace separated columns, no header) ----
ID_COL = "unit"
CYCLE_COL = "cycle"
SETTING_COLS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLUMNS = [ID_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS  # 26
N_RAW_COLUMNS = len(RAW_COLUMNS)

# Sensors that are (near) constant in FD001 train data are excluded from modelling.
# This list is *re-derived from the data at training time*; this is only the fallback.
# Rule: <= 2 distinct values in the training data (constant, or Sensor 6 which only takes 2 values).
FD001_CONSTANT_SENSORS = [1, 5, 6, 10, 16, 18, 19]
MODEL_SENSOR_IDS = [i for i in range(1, 22) if i not in FD001_CONSTANT_SENSORS]
MODEL_SENSORS = [f"s{i}" for i in MODEL_SENSOR_IDS]


def sensor_label(col: str) -> str:
    """'s11' -> 'Sensor 11'. Sensors are deliberately NOT given physical names."""
    if col.startswith("s") and col[1:].isdigit():
        return f"Sensor {int(col[1:])}"
    if col.startswith("op") and col[2:].isdigit():
        return f"Op. setting {int(col[2:])}"
    return col


# ---- RUL target ----
RUL_CAP = 125  # configurable modelling convention, not a NASA definition

# ---- Derived health state (decision-support rule, NOT an official label) ----
# Healthy: RUL > warning ; Warning: critical <= RUL <= warning ; Critical: RUL < critical
DEFAULT_THRESHOLDS = {"warning": 60.0, "critical": 30.0}
STATES = ["Healthy", "Warning", "Critical"]
STATE_COLORS = {"Healthy": "#34d399", "Warning": "#f59e0b", "Critical": "#ef4444"}
STATE_ICONS = {"Healthy": "✔", "Warning": "▲", "Critical": "✖"}

# ---- Feature engineering defaults ----
ROLL_WINDOW = 10
BASELINE_CYCLES = 5

# ---- Anomaly detection ----
ANOMALY_REF_CYCLES = (10, 50)  # early-life reference window (cycles, inclusive)
ANOMALY_THRESHOLD_QUANTILE = 0.99

# ---- Equipment registry (Priority 2) ----
EQUIPMENT_TYPES = ["Electric Motors", "Pumps", "Compressors", "Fans", "Bearings", "Aircraft Engines", "Other Rotating Equipment"]
# Only these registered types have an implemented, validated AI model (NASA C-MAPSS turbofan). Registration alone never implies AI support.
AI_SUPPORTED_TYPES = ["Aircraft Engines"]
EQUIPMENT_STATUSES = ["Operational", "Standby", "Degraded", "Under Maintenance", "Out of Service"]
MAINT_TYPES = ["Corrective", "Preventive", "Predictive", "Inspection", "Other"]
MAINT_PRIORITIES = ["Low", "Medium", "High", "Critical"]
MAINT_STATUSES = ["Pending", "In Progress", "Completed", "Cancelled"]
DB_PATH_ENV = "MAINT_DB_PATH"
DEFAULT_DB_PATH = ROOT / "storage" / "maintenance.db"

# ---- Upload ----
MAX_UPLOAD_MB = 20
MAX_UPLOAD_ROWS = 500_000
MIN_HISTORY_CYCLES = ROLL_WINDOW

# ---- Artifact paths ----
MODEL_PATH = MODELS_DIR / "rul_model.joblib"
ANOMALY_PATH = MODELS_DIR / "anomaly_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
ABLATION_PATH = ARTIFACTS_DIR / "feature_ablation.csv"
OOF_PATH = ARTIFACTS_DIR / "oof_predictions.csv"
TEST_PRED_PATH = ARTIFACTS_DIR / "test_predictions.csv"
SHAP_GLOBAL_PATH = ARTIFACTS_DIR / "shap_global.npz"
VALIDATION_LOG_PATH = ARTIFACTS_DIR / "validation_log.json"
REFERENCE_STATS_PATH = ARTIFACTS_DIR / "reference_stats.json"

DISCLAIMER = (
    "This application is a decision-support tool. It supports, and does not replace, the judgement of an authorised "
    "maintenance engineer or any certified industrial safety system. Estimates are statistical model outputs and sensor "
    "findings are observations; neither proves a physical fault. The built-in dataset is simulated NASA C-MAPSS data."
)
