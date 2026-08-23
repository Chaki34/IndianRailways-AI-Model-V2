import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config
from src.predict import SAMPLE_RECORD, load_model_and_metadata, predict_duration

MODEL_EXISTS = config.MODEL_PATH.exists()


@pytest.mark.skipif(not MODEL_EXISTS, reason="Run `python -m src.train` first")
class TestPrediction:
    @classmethod
    def setup_class(cls):
        cls.pipeline, cls.metadata = load_model_and_metadata()

    def test_valid_prediction_returns_positive_duration(self):
        result = predict_duration(self.pipeline, self.metadata, SAMPLE_RECORD)
        assert result["predicted_duration_hours"] > 0

    def test_prediction_range_is_ordered(self):
        result = predict_duration(self.pipeline, self.metadata, SAMPLE_RECORD)
        rng = result["estimated_range"]
        assert rng["lower_hours"] <= result["predicted_duration_hours"] <= rng["upper_hours"]

    def test_unknown_category_does_not_crash(self):
        record = dict(SAMPLE_RECORD)
        record["section_id"] = "SEC_NEVER_SEEN_BEFORE"
        record["asset_type"] = "Totally New Asset Type"
        result = predict_duration(self.pipeline, self.metadata, record)
        assert result["predicted_duration_hours"] > 0

    def test_extreme_values_do_not_crash(self):
        record = dict(SAMPLE_RECORD)
        record["asset_age_years"] = 100
        record["overdue_days"] = 5000
        record["severity"] = 10
        result = predict_duration(self.pipeline, self.metadata, record)
        assert result["predicted_duration_hours"] > 0
