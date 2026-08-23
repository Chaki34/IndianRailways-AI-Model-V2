"""
Model B - maintenance block scheduling / coordination engine.

Consumes Task objects (which carry predicted_duration_hours produced by
Model A) and produces a set of recommended maintenance blocks.

Algorithm (see README "Model B - scheduling algorithm" for the worked
example):
  1. Group tasks by section_id (tasks in different sections never share
     a block in this prototype).
  2. Within each section group, tasks whose predicted_duration_hours
     exceeds the configured max block duration are flagged unschedulable
     immediately (reported in `conflicts`, not silently dropped).
  3. Build pairwise compatibility using src.model_b.constraints.
  4. Partition tasks into the minimum number of mutually-compatible
     groups ("blocks") - this is graph coloring with side constraints:
       - Preferred: Google OR-Tools CP-SAT (exact, minimizes block count
         first, then total duration).
       - Fallback: pure-Python first-fit-decreasing heuristic, used
         automatically if `ortools` is not installed. Produces the same
         style of result and is what actually runs in this sandbox
         (no internet access to install ortools here) - flagged in the
         result's `backend` field, same pattern as Model A's XGBoost
         fallback.
  5. Sequence the resulting blocks within the section's available
     window (see config.window_length_hours); blocks that don't fit are
     reported as unscheduled rather than silently dropped.
"""

from dataclasses import dataclass, field
from typing import List

from src.model_b.config import SchedulerConfig
from src.model_b.constraints import are_compatible, task_fits_duration_cap
from src.model_b.constraints import Task

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


@dataclass
class ScheduledTask:
    task_id: str
    department: str
    predicted_duration_hours: float
    mode: str  # "parallel" or "isolated"


@dataclass
class Block:
    block_id: str
    section_id: str
    start_hour: float
    end_hour: float
    duration_hours: float
    departments: List[str]
    tasks: List[ScheduledTask]
    reason: str


@dataclass
class Conflict:
    task_ids: List[str]
    reason: str


@dataclass
class ScheduleResult:
    blocks: List[Block] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    backend: str = "greedy"
    summary: dict = field(default_factory=dict)


def _format_hour(h: float) -> str:
    total_minutes = round(h * 60)
    hh, mm = divmod(total_minutes, 60)
    hh = hh % 24
    return f"{hh:02d}:{mm:02d}"


def _build_conflict_pairs(tasks: List[Task], config: SchedulerConfig):
    """Returns set of (i, j) index pairs (i < j) that CANNOT share a block,
    plus a dict of reasons for reporting."""
    conflict_pairs = set()
    reasons = {}
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            compatible, reason = are_compatible(tasks[i], tasks[j], config)
            if not compatible:
                conflict_pairs.add((i, j))
                reasons[(i, j)] = reason
    return conflict_pairs, reasons


def _greedy_partition(tasks: List[Task], conflict_pairs, config: SchedulerConfig):
    """First-fit-decreasing: sort tasks longest-first, place each task in
    the first existing group it doesn't conflict with; else start a new
    group. This tends to minimize both the number of groups and the
    resulting max-duration-per-group total."""
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i].predicted_duration_hours)
    groups: List[List[int]] = []

    for i in order:
        placed = False
        for group in groups:
            conflicts_with_group = any(
                (min(i, j), max(i, j)) in conflict_pairs for j in group
            )
            if conflicts_with_group:
                continue
            # respect max block duration cap
            prospective_duration = max(
                [tasks[k].predicted_duration_hours for k in group] +
                [tasks[i].predicted_duration_hours]
            )
            if prospective_duration > config.max_block_duration_hours:
                continue
            group.append(i)
            placed = True
            break
        if not placed:
            groups.append([i])
    return groups


def _ortools_partition(tasks: List[Task], conflict_pairs, config: SchedulerConfig):
    """Exact CP-SAT formulation: minimize number of blocks first, then
    (in a second solve) minimize total block duration given that block
    count. NOTE: not executable in this sandbox (no ortools installed /
    no internet to install it) - written for correctness on a machine
    where `pip install ortools` succeeds; the greedy fallback above is
    what has actually been tested end-to-end here."""
    n = len(tasks)
    max_blocks = n  # worst case: every task isolated

    def build_model():
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f"x_{i}_{k}") for k in range(max_blocks)] for i in range(n)]
        y = [model.NewBoolVar(f"y_{k}") for k in range(max_blocks)]

        for i in range(n):
            model.Add(sum(x[i][k] for k in range(max_blocks)) == 1)
        for k in range(max_blocks):
            for i in range(n):
                model.Add(x[i][k] <= y[k])
        for (i, j) in conflict_pairs:
            for k in range(max_blocks):
                model.Add(x[i][k] + x[j][k] <= 1)
        return model, x, y

    # Pass 1: minimize number of active blocks
    model, x, y = build_model()
    model.Minimize(sum(y))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT could not find a feasible schedule.")
    best_block_count = sum(int(solver.Value(yk)) for yk in y)

    # Pass 2: fix block count, minimize total duration
    model2, x2, y2 = build_model()
    model2.Add(sum(y2) == best_block_count)
    dur_vars = []
    for k in range(max_blocks):
        dk = model2.NewIntVar(0, 1000, f"dur_{k}")
        for i in range(n):
            # duration in tenths of an hour to keep integer domain
            model2.Add(dk >= int(tasks[i].predicted_duration_hours * 10) * x2[i][k])
        dur_vars.append(dk)
    model2.Minimize(sum(dur_vars))
    solver2 = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds = 5.0
    status2 = solver2.Solve(model2)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT could not find a feasible schedule (pass 2).")

    groups = []
    for k in range(max_blocks):
        members = [i for i in range(n) if solver2.Value(x2[i][k]) == 1]
        if members:
            groups.append(members)
    return groups


def schedule_section(tasks: List[Task], section_id: str, config: SchedulerConfig,
                      block_id_prefix: str = "B") -> ScheduleResult:
    """Schedule all tasks belonging to a single section_id."""
    result = ScheduleResult()

    # Step 2: filter out tasks that individually exceed the duration cap
    schedulable = []
    for t in tasks:
        if not task_fits_duration_cap(t, config):
            result.conflicts.append(Conflict(
                task_ids=[t.task_id],
                reason=f"predicted_duration_hours ({t.predicted_duration_hours}h) "
                       f"exceeds max_block_duration_hours "
                       f"({config.max_block_duration_hours}h) - cannot be scheduled "
                       f"as a single block.",
            ))
        else:
            schedulable.append(t)

    if not schedulable:
        result.summary = {"section_id": section_id, "total_blocks": 0,
                           "total_block_duration_hours": 0.0,
                           "tasks_scheduled": 0, "tasks_unscheduled": len(tasks)}
        return result

    conflict_pairs, reasons = _build_conflict_pairs(schedulable, config)

    backend_used = "greedy"
    groups = None
    if ORTOOLS_AVAILABLE:
        try:
            groups = _ortools_partition(schedulable, conflict_pairs, config)
            backend_used = "ortools"
        except Exception:
            groups = None
    if groups is None:
        groups = _greedy_partition(schedulable, conflict_pairs, config)
        backend_used = "greedy"
    result.backend = backend_used

    # Step 5: sequence groups (blocks) within the section's window
    cursor = config.window_start_hour
    total_duration = 0.0
    scheduled_count = 0
    for idx, group in enumerate(groups):
        group_tasks = [schedulable[i] for i in group]
        block_duration = max(t.predicted_duration_hours for t in group_tasks)

        if (cursor - config.window_start_hour) + block_duration > config.window_length_hours:
            result.conflicts.append(Conflict(
                task_ids=[t.task_id for t in group_tasks],
                reason=f"Block would run from {_format_hour(cursor)} for "
                       f"{block_duration}h, exceeding the available "
                       f"{config.window_length_hours}h window for section "
                       f"{section_id}. Needs a separate maintenance night/window.",
            ))
            continue

        start = cursor
        end = cursor + block_duration
        cursor = end
        total_duration += block_duration
        scheduled_count += len(group_tasks)

        departments = sorted({t.department for t in group_tasks})
        if len(group_tasks) == 1:
            reason = f"{group_tasks[0].task_id} scheduled in its own block " \
                     f"({group_tasks[0].department}); duration " \
                     f"{block_duration}h."
            mode = "isolated"
        else:
            pair_reasons = []
            for a in range(len(group_tasks)):
                for b in range(a + 1, len(group_tasks)):
                    _, r = are_compatible(group_tasks[a], group_tasks[b], config)
                    pair_reasons.append(r)
            reason = (
                f"{', '.join(departments)} are mutually compatible and run "
                f"in parallel. Block duration is set by the longest task "
                f"({block_duration}h), not the sum. " + " ".join(pair_reasons)
            )
            mode = "parallel"

        result.blocks.append(Block(
            block_id=f"{block_id_prefix}{idx + 1}",
            section_id=section_id,
            start_hour=start,
            end_hour=end,
            duration_hours=round(block_duration, 2),
            departments=departments,
            tasks=[ScheduledTask(
                task_id=t.task_id, department=t.department,
                predicted_duration_hours=t.predicted_duration_hours,
                mode=mode,
            ) for t in group_tasks],
            reason=reason,
        ))

    result.summary = {
        "section_id": section_id,
        "total_blocks": len(result.blocks),
        "total_block_duration_hours": round(total_duration, 2),
        "tasks_scheduled": scheduled_count,
        "tasks_unscheduled": len(tasks) - scheduled_count,
    }
    return result


def schedule_tasks(tasks: List[Task], config: SchedulerConfig = None) -> ScheduleResult:
    """Entry point: groups tasks by section_id, schedules each section
    independently, and merges the results."""
    config = config or SchedulerConfig()
    if not tasks:
        return ScheduleResult(summary={"total_blocks": 0, "total_block_duration_hours": 0.0,
                                        "tasks_scheduled": 0, "tasks_unscheduled": 0})

    sections = {}
    for t in tasks:
        sections.setdefault(t.section_id, []).append(t)

    merged = ScheduleResult()
    backends_used = set()
    for section_id, section_tasks in sections.items():
        section_result = schedule_section(
            section_tasks, section_id, config,
            block_id_prefix=f"{section_id}-B")
        merged.blocks.extend(section_result.blocks)
        merged.conflicts.extend(section_result.conflicts)
        backends_used.add(section_result.backend)

    merged.backend = "mixed" if len(backends_used) > 1 else (backends_used.pop() if backends_used else "greedy")
    merged.summary = {
        "total_blocks": len(merged.blocks),
        "total_block_duration_hours": round(sum(b.duration_hours for b in merged.blocks), 2),
        "tasks_scheduled": sum(len(b.tasks) for b in merged.blocks),
        "tasks_unscheduled": len(tasks) - sum(len(b.tasks) for b in merged.blocks),
        "sections_processed": len(sections),
    }
    return merged
