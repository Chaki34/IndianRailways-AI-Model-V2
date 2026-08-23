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


VALID_REQUEST = {
    "date": "2026-09-01",
    "tasks": [
        {"task_id": "T1", "department": "Track", "section_id": "SEC0007",
         "risk_level": "High", "predicted_duration_hours": 3.0},
        {"task_id": "T2", "department": "Traction", "section_id": "SEC0007",
         "risk_level": "Medium", "predicted_duration_hours": 2.0},
        {"task_id": "T3", "department": "Signal", "section_id": "SEC0007",
         "risk_level": "Low", "predicted_duration_hours": 1.5},
    ],
}

RAW_FIELD_REQUEST = {
    "date": "2026-09-01",
    "tasks": [
        {
            "task_id": "T1", "department": "Track", "section_id": "SEC0007",
            "risk_level": "High", "severity": 7, "criticality_score": 8.2,
            "urgency_score": 6, "safety_risk_score": 7, "overdue_days": 3,
            "asset_type": "Track Circuit", "maintenance_type": "Rail Repair",
            "traffic_density": "Medium", "asset_age_years": 12,
            "condition_score": 45, "estimated_duration_hours": 3.0,
        }
    ],
}


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="Requires fastapi installed")
class TestModelBAPI:
    @classmethod
    def setup_class(cls):
        from api.app import app
        cls.client = TestClient(app)

    def test_valid_request_returns_one_block(self):
        res = self.client.post("/schedule-maintenance", json=VALID_REQUEST)
        assert res.status_code == 200
        body = res.json()
        assert len(body["blocks"]) == 1
        assert body["blocks"][0]["duration_hours"] == 3.0
        assert set(body["blocks"][0]["departments"]) == {"Track", "Traction", "Signal"}

    def test_critical_risk_forces_two_blocks(self):
        payload = {
            "date": "2026-09-01",
            "tasks": [
                {"task_id": "T1", "department": "Track", "section_id": "SEC0007",
                 "risk_level": "Critical", "predicted_duration_hours": 3.0},
                {"task_id": "T2", "department": "Traction", "section_id": "SEC0007",
                 "risk_level": "Medium", "predicted_duration_hours": 2.0},
                {"task_id": "T3", "department": "Signal", "section_id": "SEC0007",
                 "risk_level": "Low", "predicted_duration_hours": 1.5},
            ],
        }
        res = self.client.post("/schedule-maintenance", json=payload)
        assert res.status_code == 200
        assert len(res.json()["blocks"]) == 2

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Requires a trained Model A model")
    def test_raw_field_convenience_path_calls_model_a_internally(self):
        res = self.client.post("/schedule-maintenance", json=RAW_FIELD_REQUEST)
        assert res.status_code == 200
        body = res.json()
        assert len(body["blocks"]) == 1
        assert body["blocks"][0]["tasks"][0]["predicted_duration_hours"] > 0

    def test_empty_tasks_list_returns_422(self):
        payload = {"date": "2026-09-01", "tasks": []}
        res = self.client.post("/schedule-maintenance", json=payload)
        assert res.status_code == 422

    def test_missing_duration_and_missing_raw_fields_returns_422(self):
        payload = {
            "date": "2026-09-01",
            "tasks": [{"task_id": "T1", "department": "Track", "section_id": "SEC0007"}],
        }
        res = self.client.post("/schedule-maintenance", json=payload)
        assert res.status_code == 422

    def test_invalid_department_returns_422(self):
        payload = {
            "date": "2026-09-01",
            "tasks": [{"task_id": "T1", "department": "NotADepartment",
                       "section_id": "SEC0007", "predicted_duration_hours": 2.0}],
        }
        res = self.client.post("/schedule-maintenance", json=payload)
        assert res.status_code == 422

    def test_task_exceeding_max_duration_reported_as_conflict(self):
        payload = {
            "date": "2026-09-01",
            "tasks": [{"task_id": "T1", "department": "Track", "section_id": "SEC0001",
                       "predicted_duration_hours": 8.0}],
        }
        res = self.client.post("/schedule-maintenance", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert len(body["blocks"]) == 0
        assert len(body["conflicts"]) == 1

    def test_custom_max_block_duration_override(self):
        payload = dict(VALID_REQUEST)
        payload["max_block_duration_hours"] = 1.0  # forces everything to exceed cap
        res = self.client.post("/schedule-maintenance", json=payload)
        assert res.status_code == 200
        assert len(res.json()["blocks"]) == 0
        assert len(res.json()["conflicts"]) == 3

    def test_predict_duration_endpoint_still_works_unchanged(self):
        """Regression check - Model B's addition must not break Model A."""
        payload = {
            "department": "Track", "asset_type": "Track Circuit",
            "maintenance_type": "Rail Repair", "traffic_density": "Medium",
            "risk_level": "High", "section_id": "SEC0007", "asset_age_years": 12,
            "condition_score": 45, "severity": 7, "criticality_score": 8.2,
            "urgency_score": 6, "safety_risk_score": 7, "overdue_days": 3,
            "estimated_duration_hours": 3.0,
        }
        res = self.client.post("/predict-duration", json=payload)
        if MODEL_EXISTS:
            assert res.status_code == 200
            assert res.json()["predicted_duration_hours"] > 0
        else:
            assert res.status_code == 500
