"""
Central configuration for Model A - Maintenance Duration Prediction.

All column-name decisions made during Phase 1 dataset profiling live here,
so every module (training, preprocessing, API) references the SAME lists.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "historical_training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgboost_maintenance_duration_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
PLOTS_DIR = MODEL_DIR / "plots"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COLUMN = "actual_duration_hours"

# ---------------------------------------------------------------------------
# Feature groups (decided during Phase 1 profiling + user confirmation)
# ---------------------------------------------------------------------------
# Numeric features available BEFORE the maintenance task is executed
NUMERIC_FEATURES = [
    "asset_age_years",
    "condition_score",          # structurally missing for Signal/Traction - handled in preprocessing
    "severity",
    "criticality_score",
    "urgency_score",
    "safety_risk_score",
    "overdue_days",
    "estimated_duration_hours",  # planner's pre-task estimate - NOT the outcome, legitimate feature
]

# Categorical features available BEFORE the maintenance task is executed
CATEGORICAL_FEATURES = [
    "department",
    "asset_type",                # structurally missing for Track - handled in preprocessing
    "maintenance_type",          # structurally missing for Signal - handled in preprocessing
    "traffic_density",
    "risk_level",
    "section_id",                # included per user decision (regional/track effects)
]

# Engineered features added by src/feature_engineering.py. Every one of
# these is computable from raw inputs BEFORE the maintenance task happens -
# see feature_engineering.py for the justification table.
ENGINEERED_NUMERIC_FEATURES = [
    "severity_x_criticality",
    "condition_x_severity",
]
ENGINEERED_CATEGORICAL_FEATURES = [
    "asset_age_bucket",
    "is_overdue",
]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
FEATURE_COLUMNS = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Columns EXCLUDED and why (documented per Development Rule #28)
# ---------------------------------------------------------------------------
EXCLUDED_COLUMNS = {
    "task_id": "Unique identifier, carries no predictive signal.",
    "date": "Not fed to the model directly (used only to build the "
            "time-based train/test split); a raw date string is not a "
            "usable numeric/categorical feature.",
    "block_duration_hours": "LEAKAGE - outcome of the traffic block actually "
        "taken during execution; only known AFTER the maintenance work "
        "happens. Near-zero correlation (0.07) with target confirms it is "
        "not a planning-time signal.",
    "block_extension_minutes": "LEAKAGE - only exists once a block has "
        "overrun; a post-execution outcome, not known beforehand.",
    "actual_delay_minutes": "LEAKAGE - consequence of the maintenance "
        "activity on train traffic, only measurable after the fact.",
    "trains_affected": "LEAKAGE - operational impact measured during/after "
        "block execution, not known at planning/prediction time.",
    "combined_block": "LEAKAGE / irrelevant - describes how the block was "
        "actually coordinated during execution, determined after "
        "scheduling, not before.",
    "successful_block": "LEAKAGE - block execution outcome; -1 for 83% of "
        "rows (not applicable), 0/1 only known after the block occurred.",
    "number_of_departments": "LEAKAGE - reflects how many departments "
        "ended up sharing the executed block; a scheduling/execution "
        "outcome, not a pre-task characteristic.",
    "train_count_in_window": "Excluded per user decision - scheduling-"
        "context feature that keeps Model A independent of block-timing "
        "decisions, which belong to Model B.",
    "high_priority_train_count": "Excluded per user decision - same "
        "reasoning as train_count_in_window.",
}

TARGET_LEAKAGE_NOTE = (
    "actual_duration_hours is the TARGET (y) and must never appear in the "
    "feature matrix X."
)
