from __future__ import annotations

import os
from datetime import date
from pathlib import Path

APP_TITLE = "Bad Debt Early-Warning API"

# Project root: Model API/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_REGISTRY = {
    "stacked": {
        "model_path": str(BASE_DIR / "artifacts/stacked_recall_driven_model.joblib"),
        "schema_path": str(BASE_DIR / "artifacts/feature_cols_stacked.json"),
        "label": "Model Stacked (LightGBM + LR)",
        "training_flow": "Test Stacked SMOTE, Auto Search Parameter & Updated",
        "label_strategy": "y_bad_debt_ever (3 kondisi)",
        "credit_memo_policy": "netting PREVIOUS_CUSTOMER_TRX_ID sampai snapshot_date",
    },
    "lgbm_hyper_smote": {
        "model_path": str(
            BASE_DIR / "artifacts/bad_debt_snapshot_lgbm_hyper_smote_16_features.joblib"
        ),
        "schema_path": str(
            BASE_DIR / "artifacts/feature_cols_snapshot_16_features.json"
        ),
        "label": "LightGBM",
        "training_flow": "Test_New_CM_SMOTE",
        "label_strategy": "y_bad_debt_ever (3 kondisi)",
        "credit_memo_policy": "netting PREVIOUS_CUSTOMER_TRX_ID sampai snapshot_date",
    },
}

DEFAULT_MODEL_KEY = "stacked"
DEFAULT_SNAPSHOT_DATE = os.getenv("BAD_DEBT_SNAPSHOT_DATE", date.today().isoformat())

THRESHOLD_LOW = float(os.getenv("THRESHOLD_LOW", "0.3"))
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.6"))
if THRESHOLD_LOW >= THRESHOLD_HIGH:
    raise ValueError(
        "Invalid thresholds: THRESHOLD_LOW must be less than THRESHOLD_HIGH."
    )

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
API_KEY = os.getenv("API_KEY")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
if _cors_origins_env.strip():
    ALLOW_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    ALLOW_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# ── Scheduler & Compute ──────────────────────────────────────────────
COMPUTE_SCHEDULE_HOUR = int(os.getenv("COMPUTE_SCHEDULE_HOUR", "6"))
COMPUTE_AUTO_ENABLED = os.getenv("COMPUTE_AUTO_ENABLED", "true").lower() == "true"
COMPUTE_DEFAULT_TIME_RANGE = os.getenv("COMPUTE_DEFAULT_TIME_RANGE", "3m")
COMPUTE_KEEP_DAYS = int(os.getenv("COMPUTE_KEEP_DAYS", "30"))
COMPUTE_MAX_RUNNING_MINUTES = int(os.getenv("COMPUTE_MAX_RUNNING_MINUTES", "180"))
COMPUTE_AUTO_RECOVER_STALE = (
    os.getenv("COMPUTE_AUTO_RECOVER_STALE", "true").lower() == "true"
)
COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL = (
    os.getenv("COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL", "true").lower() == "true"
)
COMPUTE_PUBLISH_TARGET_TABLE = os.getenv(
    "COMPUTE_PUBLISH_TARGET_TABLE", "hasil_baddebt"
)
COMPUTE_PUBLISH_REPLACE_PARTITION = (
    os.getenv("COMPUTE_PUBLISH_REPLACE_PARTITION", "false").lower() == "true"
)

# ── Pagination ────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
