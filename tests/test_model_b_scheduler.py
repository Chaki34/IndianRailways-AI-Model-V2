import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.model_b.config import SchedulerConfig
from src.model_b.constraints import Task, are_compatible
from src.model_b.scheduler import schedule_tasks


def test_all_compatible_departments_form_one_block():
    """Matches the spec's worked example: Track=3h, Traction=2h, Signal=1.5h,
    all compatible -> one block, duration = max, not sum."""
    tasks = [
        Task("T1", "Track", "SEC0007", 3.0, risk_level="High"),
        Task("T2", "Traction", "SEC0007", 2.0, risk_level="Medium"),
        Task("T3", "Signal", "SEC0007", 1.5, risk_level="Low"),
    ]
    result = schedule_tasks(tasks)
    assert len(result.blocks) == 1
    assert result.blocks[0].duration_hours == 3.0
    assert set(result.blocks[0].departments) == {"Track", "Traction", "Signal"}
    assert result.summary["tasks_scheduled"] == 3
    assert result.summary["tasks_unscheduled"] == 0


def test_critical_risk_task_is_always_isolated():
    tasks = [
        Task("T1", "Track", "SEC0007", 3.0, risk_level="Critical"),
        Task("T2", "Traction", "SEC0007", 2.0, risk_level="Medium"),
        Task("T3", "Signal", "SEC0007", 1.5, risk_level="Low"),
    ]
    result = schedule_tasks(tasks)
    assert len(result.blocks) == 2
    track_block = next(b for b in result.blocks if "Track" in b.departments)
    assert track_block.departments == ["Track"]
    assert len(track_block.tasks) == 1
    other_block = next(b for b in result.blocks if "Track" not in b.departments)
    assert set(other_block.departments) == {"Traction", "Signal"}
    assert other_block.duration_hours == 2.0


def test_same_department_same_section_is_sequenced_not_parallel():
    tasks = [
        Task("T1", "Track", "SEC0002", 2.0),
        Task("T2", "Track", "SEC0002", 3.0),
    ]
    result = schedule_tasks(tasks)
    assert len(result.blocks) == 2
    for b in result.blocks:
        assert len(b.tasks) == 1
    # sequenced: second block starts where the first ends
    starts = sorted(b.start_hour for b in result.blocks)
    ends = sorted(b.end_hour for b in result.blocks)
    assert starts[1] == ends[0]


def test_task_exceeding_max_duration_is_reported_not_dropped():
    tasks = [Task("T1", "Track", "SEC0001", 8.0)]
    result = schedule_tasks(tasks)
    assert len(result.blocks) == 0
    assert len(result.conflicts) == 1
    assert "T1" in result.conflicts[0].task_ids
    assert result.summary["tasks_unscheduled"] == 1


def test_window_overflow_is_reported_not_dropped():
    """Two 5h Critical (isolated) tasks in one section exceed the default
    6h window when sequenced -> second is flagged as a conflict, not lost."""
    tasks = [
        Task("A", "Track", "SEC0003", 5.0, risk_level="Critical"),
        Task("B", "Traction", "SEC0003", 5.0, risk_level="Critical"),
    ]
    result = schedule_tasks(tasks)
    assert len(result.blocks) == 1
    assert len(result.conflicts) == 1
    assert result.summary["tasks_scheduled"] == 1
    assert result.summary["tasks_unscheduled"] == 1


def test_empty_task_list_returns_empty_schedule():
    result = schedule_tasks([])
    assert result.blocks == []
    assert result.summary["total_blocks"] == 0


def test_multiple_sections_are_scheduled_independently():
    tasks = [
        Task("T1", "Track", "SEC0001", 2.0),
        Task("T2", "Signal", "SEC0002", 1.0),
    ]
    result = schedule_tasks(tasks)
    assert len(result.blocks) == 2
    sections = {b.section_id for b in result.blocks}
    assert sections == {"SEC0001", "SEC0002"}


def test_custom_config_can_disable_department_pair():
    config = SchedulerConfig()
    config.compatible_department_pairs = set()  # nothing compatible
    tasks = [
        Task("T1", "Track", "SEC0007", 2.0),
        Task("T2", "Signal", "SEC0007", 1.0),
    ]
    result = schedule_tasks(tasks, config)
    assert len(result.blocks) == 2  # forced apart


def test_are_compatible_reports_reason_string():
    config = SchedulerConfig()
    t1 = Task("T1", "Track", "SEC0007", 2.0, risk_level="Low")
    t2 = Task("T2", "Signal", "SEC0007", 1.0, risk_level="Low")
    compatible, reason = are_compatible(t1, t2, config)
    assert compatible is True
    assert "compatible department pair" in reason
