"""
Feature engineering.

Every feature created here is available BEFORE the maintenance task is
carried out - none of them use any post-execution/outcome column.

| Feature                  | Reason                                         | Available before maintenance? |
|---------------------------|------------------------------------------------|--------------------------------|
| severity_x_criticality    | Captures compounding risk: a high-severity     | Yes - both inputs are          |
|                            | fault on a high-criticality asset likely takes | pre-task assessments.          |
|                            | disproportionately longer than either alone.   |                                 |
| condition_x_severity      | A severe fault on an already poor-condition    | Yes - condition_score is       |
|                            | asset is expected to take longer to fix than   | recorded at inspection time,   |
|                            | the same severity on a healthy asset.          | before repair work starts.     |
| asset_age_bucket           | Buckets a noisy continuous age into stable     | Yes - asset age is known       |
|                            | groups (New/Mid/Old/Very Old) that tree models | from asset records.            |
|                            | can split on cleanly.                          |                                 |
| is_overdue                | Whether the task is already overdue - overdue  | Yes - overdue_days is a        |
|                            | backlog items often require more extensive     | scheduling-time field.         |
|                            | remedial work than on-schedule ones.           |                                 |

Rejected features (explicitly NOT created): anything derived from
block_duration_hours, actual_delay_minutes, trains_affected,
combined_block, successful_block, number_of_departments,
train_count_in_window, high_priority_train_count - all of these are only
known after the maintenance activity/traffic block has occurred (see
src/config.py EXCLUDED_COLUMNS), so no engineered feature is built from
them.
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Interaction terms
    out["severity_x_criticality"] = out["severity"] * out["criticality_score"]
    out["condition_x_severity"] = out["condition_score"] * out["severity"]

    # Asset age bucket
    out["asset_age_bucket"] = pd.cut(
        out["asset_age_years"],
        bins=[-0.01, 10, 20, 30, 100],
        labels=["New (0-10y)", "Mid (10-20y)", "Old (20-30y)", "Very Old (30y+)"],
    ).astype(str)

    # Overdue flag
    out["is_overdue"] = (out["overdue_days"] > 0).map({True: "Overdue", False: "On Schedule"})

    return out
