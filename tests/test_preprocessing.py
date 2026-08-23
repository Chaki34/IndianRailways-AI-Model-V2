import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src import config
from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor, run_data_quality_checks


def _sample_df(n=20):
    return pd.DataFrame({
        "department": ["Track"] * n,
        "asset_type": [None] * n,
        "maintenance_type": ["Rail Repair"] * n,
        "traffic_density": ["Low"] * n,
        "risk_level": ["Low"] * n,
        "section_id": ["SEC0001"] * n,
        "asset_age_years": [10.0] * n,
        "condition_score": [70.0] * n,
        "severity": [5] * n,
        "criticality_score": [5.0] * n,
        "urgency_score": [5] * n,
        "safety_risk_score": [5] * n,
        "overdue_days": [0] * n,
        "estimated_duration_hours": [2.0] * n,
        "actual_duration_hours": [2.1] * n,
    })


def test_preprocessor_handles_missing_categoricals():
    df = add_engineered_features(_sample_df())
    preprocessor = build_preprocessor()
    X = df[config.FEATURE_COLUMNS]
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(df)
    assert not pd.isnull(transformed).any()


def test_preprocessor_handles_unknown_category_at_inference():
    df = add_engineered_features(_sample_df())
    preprocessor = build_preprocessor()
    X = df[config.FEATURE_COLUMNS]
    preprocessor.fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["section_id"] = "SEC_NEVER_SEEN"
    # Should not raise - handle_unknown="ignore"
    transformed = preprocessor.transform(new_row)
    assert transformed.shape[0] == 1


def test_data_quality_checks_detect_negative_duration():
    df = _sample_df()
    df.loc[0, "actual_duration_hours"] = -1.0
    report = run_data_quality_checks(df)
    assert report["invalid_values"]["negative_or_zero_duration"] == 1


def test_data_quality_checks_no_false_positives_on_clean_data():
    df = _sample_df()
    report = run_data_quality_checks(df)
    assert report["invalid_values"]["negative_or_zero_duration"] == 0
    assert report["duplicate_rows"] == df.duplicated().sum()
