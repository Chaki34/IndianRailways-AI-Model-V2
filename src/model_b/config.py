"""
Model B configuration.

Everything in this file is a PROTOTYPE ASSUMPTION, not learned from data.
The historical dataset has no time-of-day fields, no pairwise task-linking
key, and no explicit department-compatibility or resource data (see
README "Model B - data gaps" section for the full inspection findings).
These defaults are deliberately conservative and are meant to be replaced
with real operating rules by domain experts before production use.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Department compatibility (ASSUMPTION - not present in the dataset)
# ---------------------------------------------------------------------------
# Cross-department pairs assumed safe to run in parallel in a shared block.
# Real Indian Railways possession rules should replace this matrix.
COMPATIBLE_DEPARTMENT_PAIRS = {
    frozenset({"Track", "Signal"}),
    frozenset({"Track", "Traction"}),
    frozenset({"Traction", "Signal"}),
}

# Two tasks from the SAME department in the same section are assumed to
# need the same crew/possession and therefore CANNOT run in parallel
# (they are sequenced into separate blocks instead). Set True to allow it.
ALLOW_SAME_DEPARTMENT_PARALLEL = False

# ---------------------------------------------------------------------------
# Safety override (ASSUMPTION)
# ---------------------------------------------------------------------------
# A task at any of these risk levels is never combined with another task -
# it always gets its own isolated block, regardless of department
# compatibility. Mirrors the standard "single critical possession at a
# time" safety principle.
ISOLATING_RISK_LEVELS = {"Critical"}

# ---------------------------------------------------------------------------
# Block window & duration limits (ASSUMPTION - no time-of-day data exists)
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_START_HOUR = 0.0     # 00:00 - placeholder "nightly block"
DEFAULT_WINDOW_LENGTH_HOURS = 6.0   # total available time per section/date
DEFAULT_MAX_BLOCK_DURATION_HOURS = 6.0  # historical combined-block max was 7.9h;
                                          # 6h is a conservative cap

# ---------------------------------------------------------------------------
# Optimization backend
# ---------------------------------------------------------------------------
# "ortools" (preferred, constraint-programming) or "greedy" (pure-Python
# first-fit-decreasing heuristic, used automatically if ortools is not
# installed in the current environment).
PREFERRED_BACKEND = "ortools"


@dataclass
class SchedulerConfig:
    """Bundles the above defaults so they can be overridden per-request
    (e.g. a different max block duration for a specific section) without
    editing this file."""
    compatible_department_pairs: set = field(
        default_factory=lambda: set(COMPATIBLE_DEPARTMENT_PAIRS))
    allow_same_department_parallel: bool = ALLOW_SAME_DEPARTMENT_PARALLEL
    isolating_risk_levels: set = field(
        default_factory=lambda: set(ISOLATING_RISK_LEVELS))
    window_start_hour: float = DEFAULT_WINDOW_START_HOUR
    window_length_hours: float = DEFAULT_WINDOW_LENGTH_HOURS
    max_block_duration_hours: float = DEFAULT_MAX_BLOCK_DURATION_HOURS
