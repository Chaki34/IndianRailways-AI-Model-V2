"""
FastAPI service for Model A - Maintenance Duration Prediction.

    uvicorn api.app:app --reload

POST /predict-duration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.predict import load_model_and_metadata, predict_duration
from api.schedule import router as model_b_router

app = FastAPI(
    title="Railway Maintenance Duration Prediction API",
    description="Model A (duration prediction) + Model B (maintenance "
                 "block scheduling/coordination) of the Indian Railways "
                 "Smart Maintenance Scheduling prototype.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model B - maintenance block scheduling. Kept in its own router/module
# (api/schedule.py) so Model A's routes below are never touched.
app.include_router(model_b_router)

_pipeline = None
_metadata = None


@app.on_event("startup")
def _load_model():
    global _pipeline, _metadata
    try:
        _pipeline, _metadata = load_model_and_metadata()
    except FileNotFoundError as e:
        # Service can still start; /predict-duration will 500 with a clear
        # message until the model is trained.
        print(f"[startup] {e}")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class MaintenanceRequest(BaseModel):
    department: Literal["Track", "Traction", "Signal"]
    asset_type: str = Field(..., description="e.g. 'Track Circuit', 'Transformer'. "
                                               "Use 'Not Recorded' if unknown.")
    maintenance_type: str = Field(..., description="e.g. 'Rail Repair', 'Inspection'. "
                                                     "Use 'Not Recorded' if unknown.")
    traffic_density: Literal["Low", "Medium", "High"]
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    section_id: str = Field(..., examples=["SEC0007"])
    asset_age_years: float = Field(..., ge=0, le=100)
    condition_score: float = Field(..., ge=0, le=100,
        description="0-100 condition rating. Not recorded for all "
                     "departments in the historical data - pass a best "
                     "estimate if unavailable.")
    severity: int = Field(..., ge=1, le=10)
    criticality_score: float = Field(..., ge=1, le=10)
    urgency_score: int = Field(..., ge=1, le=10)
    safety_risk_score: int = Field(..., ge=1, le=10)
    overdue_days: int = Field(..., ge=0)
    estimated_duration_hours: float = Field(..., gt=0,
        description="Planner's pre-task duration estimate, in hours.")

    class Config:
        json_schema_extra = {
            "example": {
                "department": "Track",
                "asset_type": "Track",
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
        }


class EstimatedRange(BaseModel):
    lower_hours: float
    upper_hours: float


class MaintenanceResponse(BaseModel):
    predicted_duration_hours: float
    predicted_duration_minutes: int
    estimated_range: EstimatedRange


@app.get("/")
def root():
    return {"service": "railway-maintenance-duration-prediction", "status": "ok"}


@app.get("/health")
def health():
    return {"model_loaded": _pipeline is not None}


@app.post("/predict-duration", response_model=MaintenanceResponse)
def predict_duration_endpoint(request: MaintenanceRequest):
    if _pipeline is None or _metadata is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Run `python -m src.train` to "
                   "train and save a model, then restart the API.",
        )
    try:
        result = predict_duration(_pipeline, _metadata, request.model_dump())
    except Exception:
        # Never leak stack traces to the client.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        )
    return MaintenanceResponse(
        predicted_duration_hours=result["predicted_duration_hours"],
        predicted_duration_minutes=result["predicted_duration_minutes"],
        estimated_range=EstimatedRange(**result["estimated_range"]),
    )
