"""
Model B API - maintenance block scheduling / coordination.

    POST /schedule-maintenance

Kept in its own router/module and included into api/app.py with a single
line, so Model A's existing /predict-duration route is never touched.
"""

from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.model_b.config import SchedulerConfig
from src.model_b.constraints import Task as SchedulerTask
from src.model_b.scheduler import schedule_tasks

router = APIRouter()


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class TaskInput(BaseModel):
    task_id: str
    department: Literal["Track", "Traction", "Signal"]
    section_id: str
    risk_level: Literal["Low", "Medium", "High", "Critical"] = "Low"
    severity: Optional[int] = Field(None, ge=1, le=10)
    criticality_score: Optional[float] = Field(None, ge=1, le=10)
    urgency_score: Optional[int] = Field(None, ge=1, le=10)
    safety_risk_score: Optional[int] = Field(None, ge=1, le=10)
    overdue_days: Optional[int] = Field(None, ge=0)

    # Path A (preferred, strict separation): supply the Model A output directly.
    predicted_duration_hours: Optional[float] = Field(None, gt=0)

    # Path B (convenience, approved as opt-in): supply raw Model A input
    # fields instead, and the API calls Model A's prediction internally.
    asset_type: Optional[str] = None
    maintenance_type: Optional[str] = None
    traffic_density: Optional[Literal["Low", "Medium", "High"]] = None
    asset_age_years: Optional[float] = Field(None, ge=0, le=100)
    condition_score: Optional[float] = Field(None, ge=0, le=100)
    estimated_duration_hours: Optional[float] = Field(None, gt=0)

    @model_validator(mode="after")
    def _require_duration_or_raw_fields(self):
        if self.predicted_duration_hours is not None:
            return self
        raw_required = ["asset_type", "maintenance_type", "traffic_density",
                         "asset_age_years", "condition_score",
                         "estimated_duration_hours", "severity",
                         "criticality_score", "urgency_score",
                         "safety_risk_score", "overdue_days"]
        missing = [f for f in raw_required if getattr(self, f) is None]
        if missing:
            raise ValueError(
                "Either 'predicted_duration_hours' must be supplied, or all "
                f"of the raw Model A fields must be present. Missing: {missing}"
            )
        return self


class ScheduleRequest(BaseModel):
    date: str = Field(..., examples=["2026-09-01"])
    tasks: List[TaskInput] = Field(..., min_length=1)
    max_block_duration_hours: Optional[float] = Field(None, gt=0)
    window_length_hours: Optional[float] = Field(None, gt=0)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class ScheduledTaskOut(BaseModel):
    task_id: str
    department: str
    predicted_duration_hours: float
    mode: str


class BlockOut(BaseModel):
    block_id: str
    section_id: str
    start_time: str
    end_time: str
    duration_hours: float
    departments: List[str]
    tasks: List[ScheduledTaskOut]
    reason: str


class ConflictOut(BaseModel):
    task_ids: List[str]
    reason: str


class ScheduleResponse(BaseModel):
    blocks: List[BlockOut]
    conflicts: List[ConflictOut]
    backend: str
    optimization_summary: dict


def _format_hour(h: float) -> str:
    total_minutes = round(h * 60)
    hh, mm = divmod(total_minutes, 60)
    hh = hh % 24
    return f"{hh:02d}:{mm:02d}"


def _resolve_duration(task: TaskInput) -> float:
    """Path A: use predicted_duration_hours as-is. Path B: call Model A's
    prediction pipeline internally using the raw fields supplied."""
    if task.predicted_duration_hours is not None:
        return task.predicted_duration_hours

    # Path B - convenience: call Model A internally. Imported lazily so
    # Model B tests don't require a trained Model A model to be present.
    from src.predict import load_model_and_metadata, predict_duration
    try:
        pipeline, metadata = load_model_and_metadata()
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Task {task.task_id} did not supply "
                   f"predicted_duration_hours and Model A's trained model "
                   f"is not available to compute it. Run "
                   f"`python -m src.train` first, or supply "
                   f"predicted_duration_hours directly.",
        )
    record = {
        "department": task.department,
        "asset_type": task.asset_type,
        "maintenance_type": task.maintenance_type,
        "traffic_density": task.traffic_density,
        "risk_level": task.risk_level,
        "section_id": task.section_id,
        "asset_age_years": task.asset_age_years,
        "condition_score": task.condition_score,
        "severity": task.severity,
        "criticality_score": task.criticality_score,
        "urgency_score": task.urgency_score,
        "safety_risk_score": task.safety_risk_score,
        "overdue_days": task.overdue_days,
        "estimated_duration_hours": task.estimated_duration_hours,
    }
    result = predict_duration(pipeline, metadata, record)
    return result["predicted_duration_hours"]


@router.post("/schedule-maintenance", response_model=ScheduleResponse)
def schedule_maintenance(request: ScheduleRequest):
    config = SchedulerConfig()
    if request.max_block_duration_hours is not None:
        config.max_block_duration_hours = request.max_block_duration_hours
    if request.window_length_hours is not None:
        config.window_length_hours = request.window_length_hours

    try:
        scheduler_tasks = []
        for t in request.tasks:
            duration = _resolve_duration(t)
            scheduler_tasks.append(SchedulerTask(
                task_id=t.task_id,
                department=t.department,
                section_id=t.section_id,
                predicted_duration_hours=duration,
                risk_level=t.risk_level,
                severity=t.severity,
                urgency_score=t.urgency_score,
                safety_risk_score=t.safety_risk_score,
                overdue_days=t.overdue_days,
            ))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500,
                             detail="Failed to resolve task durations due to an internal error.")

    result = schedule_tasks(scheduler_tasks, config)

    blocks_out = [
        BlockOut(
            block_id=b.block_id,
            section_id=b.section_id,
            start_time=_format_hour(b.start_hour),
            end_time=_format_hour(b.end_hour),
            duration_hours=b.duration_hours,
            departments=b.departments,
            tasks=[ScheduledTaskOut(
                task_id=st.task_id, department=st.department,
                predicted_duration_hours=st.predicted_duration_hours,
                mode=st.mode,
            ) for st in b.tasks],
            reason=b.reason,
        ) for b in result.blocks
    ]
    conflicts_out = [ConflictOut(task_ids=c.task_ids, reason=c.reason)
                      for c in result.conflicts]

    return ScheduleResponse(
        blocks=blocks_out,
        conflicts=conflicts_out,
        backend=result.backend,
        optimization_summary=result.summary,
    )
