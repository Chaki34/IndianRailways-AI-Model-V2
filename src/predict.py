"""
Local prediction test script.

    python -m src.predict

Loads the saved pipeline and metadata, runs a sample prediction, and
prints it in the human-readable format required by the spec.
"""

import json

import joblib
import pandas as pd

from src import config


def load_model_and_metadata():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.train` first.")
    pipeline = joblib.load(config.MODEL_PATH)
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata


def _engineer_single(record: dict) -> pd.DataFrame:
    """Apply the exact same feature engineering used in training to a
    single raw record (dict of the fields a user would actually supply)."""
    from src.feature_engineering import add_engineered_features
    df = pd.DataFrame([record])
    return add_engineered_features(df)


def predict_duration(pipeline, metadata, record: dict) -> dict:
    """
    record must contain the RAW input fields (pre-feature-engineering):
    department, asset_type, maintenance_type, traffic_density, risk_level,
    section_id, asset_age_years, condition_score, severity,
    criticality_score, urgency_score, safety_risk_score, overdue_days,
    estimated_duration_hours
    """
    engineered = _engineer_single(record)
    X = engineered[config.FEATURE_COLUMNS]
    point_pred = float(pipeline.predict(X)[0])

    lower_off, upper_off = metadata["residual_range_10_90_pct"]
    lower = max(0.0, point_pred + lower_off)
    upper = point_pred + upper_off

    hours = int(point_pred)
    minutes = round((point_pred - hours) * 60)

    return {
        "predicted_duration_hours": round(point_pred, 2),
        "predicted_duration_minutes": round(point_pred * 60),
        "human_readable": f"{point_pred:.2f} hours (\u2248 {hours} hours {minutes} minutes)",
        "estimated_range": {
            "lower_hours": round(lower, 2),
            "upper_hours": round(upper, 2),
        },
    }


SAMPLE_RECORD = {
    "department": "Track",
    "asset_type": "Track Circuit",
    "maintenance_type": "Rail Repair",
    "traffic_density": "High",
    "risk_level": "High",
    "section_id": "SEC0007",
    "asset_age_years": 20,
    "condition_score": 45,
    "severity": 7,
    "criticality_score": 8.2,
    "urgency_score": 10,
    "safety_risk_score": 7,
    "overdue_days": 10,
    "estimated_duration_hours": 3.0,
}


if __name__ == "__main__":
    pipeline, metadata = load_model_and_metadata()
    result = predict_duration(pipeline, metadata, SAMPLE_RECORD)

    print("Input:")
    print(json.dumps(SAMPLE_RECORD, indent=2))
    print("\nEstimated Maintenance Duration:")
    print(f"  {result['predicted_duration_hours']} hours")
    print(f"  {result['human_readable']}")
    print(f"\nEstimated Range: {result['estimated_range']['lower_hours']} - "
          f"{result['estimated_range']['upper_hours']} hours")
    print(f"\n(Model trained with: {metadata['model_backend']})")
