import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import config

MODEL_EXISTS = config.MODEL_PATH.exists()

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


VALID_PAYLOAD = {
    "department": "Track",
    "asset_type": "Track Circuit",
    "maintenance_type": "Rail Repair",
    "traffic_density": "Medium",
    "risk_level": "High",
    "section_id": "SEC0007",
    "asset_age_years": 12,
    "condition_score": 45,
    "severity": 7,
    "criticality_score": 8.2,
    "urgency_score": 6,
    "safety_risk_score": 7,
    "overdue_days": 3,
    "estimated_duration_hours": 3.0,
}


@pytest.mark.skipif(not (FASTAPI_AVAILABLE and MODEL_EXISTS),
                     reason="Requires fastapi installed and a trained model")
class TestAPI:
    @classmethod
    def setup_class(cls):
        from api.app import app
        cls.client = TestClient(app)

    def test_valid_request_returns_200(self):
        res = self.client.post("/predict-duration", json=VALID_PAYLOAD)
        assert res.status_code == 200
        body = res.json()
        assert body["predicted_duration_hours"] > 0
        assert "lower_hours" in body["estimated_range"]

    def test_missing_field_returns_422(self):
        payload = dict(VALID_PAYLOAD)
        del payload["severity"]
        res = self.client.post("/predict-duration", json=payload)
        assert res.status_code == 422

    def test_invalid_category_returns_422(self):
        payload = dict(VALID_PAYLOAD)
        payload["department"] = "NotARealDepartment"
        res = self.client.post("/predict-duration", json=payload)
        assert res.status_code == 422

    def test_out_of_range_score_returns_422(self):
        payload = dict(VALID_PAYLOAD)
        payload["severity"] = 99
        res = self.client.post("/predict-duration", json=payload)
        assert res.status_code == 422

    def test_health_endpoint(self):
        res = self.client.get("/health")
        assert res.status_code == 200
