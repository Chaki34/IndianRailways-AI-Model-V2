"""
Task data model and pairwise compatibility checks for Model B.

Pure Python, no FastAPI/Pydantic dependency here - keeps the scheduling
core independently testable and reusable outside the API.
"""

from dataclasses import dataclass
from typing import Optional

from src.model_b.config import SchedulerConfig


@dataclass
class Task:
    task_id: str
    department: str
    section_id: str
    predicted_duration_hours: float
    risk_level: str = "Low"
    severity: Optional[int] = None
    urgency_score: Optional[int] = None
    safety_risk_score: Optional[int] = None
    overdue_days: Optional[int] = None


def is_isolating(task: Task, config: SchedulerConfig) -> bool:
    """True if this task must always get its own block."""
    return task.risk_level in config.isolating_risk_levels


def are_compatible(task_a: Task, task_b: Task, config: SchedulerConfig) -> tuple[bool, str]:
    """
    Returns (compatible, reason). Two tasks can share a block only if:
      1. Neither is safety-isolating (e.g. Critical risk_level).
      2. They are either from different, compatible departments, OR from
         the same department AND same-department parallelism is allowed.
    """
    if is_isolating(task_a, config):
        return False, f"{task_a.task_id} has isolating risk_level " \
                       f"'{task_a.risk_level}' and cannot be combined with any other task."
    if is_isolating(task_b, config):
        return False, f"{task_b.task_id} has isolating risk_level " \
                       f"'{task_b.risk_level}' and cannot be combined with any other task."

    if task_a.department == task_b.department:
        if config.allow_same_department_parallel:
            return True, f"Same department ({task_a.department}) parallelism is allowed by config."
        return False, f"{task_a.task_id} and {task_b.task_id} are both " \
                       f"'{task_a.department}' - same-department tasks are " \
                       f"assumed to need the same crew/possession and are " \
                       f"sequenced, not parallelized."

    pair = frozenset({task_a.department, task_b.department})
    if pair in config.compatible_department_pairs:
        return True, f"{task_a.department} and {task_b.department} are " \
                      f"configured as a compatible department pair."
    return False, f"{task_a.department} and {task_b.department} are not " \
                   f"configured as a compatible department pair."


def task_fits_duration_cap(task: Task, config: SchedulerConfig) -> bool:
    return task.predicted_duration_hours <= config.max_block_duration_hours
