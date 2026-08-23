# Railway Maintenance Duration Prediction — Model A

Indian Railways Smart Maintenance Scheduling prototype — **Model A only**
(AI-based maintenance duration prediction). Model B (block/schedule
optimization) is a later phase and is **not** implemented here.

```
Maintenance Requests
        │
        ▼
┌───────────────────┐
│ Model A (this repo)│
│ XGBoost Regression │
│ Duration Prediction│
└─────────┬──────────┘
          ▼
  Predicted Durations
          ▼
┌───────────────────┐
│ Model B (future)   │
│ Scheduling &        │
│ Coordination        │
└─────────┬──────────┘
          ▼
     Optimal Block
```

## 1. Objective

Given the characteristics of a maintenance task **before it is carried
out** (department, asset type, severity, criticality, asset condition,
age, etc.), predict how many hours it will take to complete —
`Predicted Duration = 3.17 hours`.

## 2. Dataset

`data/historical_training_dataset.csv` — 30,000 historical maintenance
records across three departments (Track, Traction, Signal), 26 raw
columns. Full profiling output is produced by:

```bash
python -m src.data_loader
```

### Structural missingness (important, not random)
Different departments record different fields:
- `asset_type` — recorded for Signal & Traction only (missing for 100% of Track rows)
- `maintenance_type` — recorded for Track & Traction only (missing for 100% of Signal rows)
- `condition_score` — recorded for Track only (missing for 100% of Signal/Traction rows)

These are handled by the preprocessing pipeline's imputers
(`"Not Recorded"` for categoricals, median for numerics) — **no rows are
dropped**.

## 3. ML problem

Supervised learning → Regression → **XGBoost Regressor**
(`random_state=42`), compared against a `DummyRegressor` baseline and a
`RandomForestRegressor`.

> **Sandbox note:** the development environment used to build this
> prototype had no internet access, so `xgboost`/`shap` could not be
> pip-installed there. `src/train.py` detects this and falls back to
> `sklearn.HistGradientBoostingRegressor` (a comparable gradient-boosted
> tree model) purely so the pipeline could be validated end-to-end with
> real numbers. **On a normal machine, `pip install -r requirements.txt`
> makes real XGBoost + SHAP load automatically** — no code changes
> needed. Check `models/model_metadata.json` → `model_backend` to see
> which one actually trained your saved model.

## 4. Target variable & leakage

**Target (y): `actual_duration_hours`** — the real, completed duration of
a historical maintenance task.

Columns removed as leakage (only knowable **after** the task/traffic
block executed) or excluded per project-owner decision — full reasoning
in `src/config.py::EXCLUDED_COLUMNS`:

| Column | Reason |
|---|---|
| `block_duration_hours` | Outcome of the traffic block actually taken; corr. 0.07 with target |
| `block_extension_minutes` | Only exists after a block overrun |
| `actual_delay_minutes` | Consequence of the activity on traffic, post-hoc |
| `trains_affected` | Operational impact measured during/after execution |
| `combined_block` | Describes how the block was actually coordinated, post-hoc |
| `successful_block` | Block execution outcome (-1/0/1) |
| `number_of_departments` | Reflects who ended up sharing the executed block |
| `train_count_in_window`, `high_priority_train_count` | Excluded by project decision — keeps Model A independent of Model B's block-timing concerns |
| `task_id` | Identifier only |
| `date` | Used only to build the time-based split, not as a raw model input |

`estimated_duration_hours` (the **planner's pre-task estimate**) is kept
as a feature — it is available before work starts and is not the
outcome itself, so it is not leakage. See §9 for a caveat on its
dominant importance.

## 5. Data preprocessing

`src/preprocessing.py` builds one `sklearn.compose.ColumnTransformer`,
persisted inside the saved pipeline and reused identically for training,
testing, and API inference (no separate manual encoding path):

- Numeric features → median imputation
- Categorical features → constant `"Not Recorded"` imputation → one-hot encoding (`handle_unknown="ignore"`, so unseen categories at inference don't crash)

### Data quality (from `run_data_quality_checks`, printed by `train.py`)
- Duplicate rows: **0**
- Negative/zero durations: **0**
- Negative asset ages: **0**
- Condition scores outside 0–100: **0**
- Target distribution: min 0.3h, max 12.3h, mean 2.57h, median 2.3h, std 1.15h, IQR [1.8, 3.1]
- Tukey outliers: 1,050 records (3.5%) above the upper bound — **not removed**. Railway emergency/major repairs can legitimately run long, and the observed max (12.3h) is operationally plausible; no records were data-entry errors.

## 6. Feature engineering

All features below are computable **before** the maintenance task
happens — see the justification table in `src/feature_engineering.py`.

| Feature | Reason | Available before maintenance? |
|---|---|---|
| `severity_x_criticality` | Compounding risk: severe fault × critical asset | Yes |
| `condition_x_severity` | Severe fault on poor-condition asset takes longer | Yes |
| `asset_age_bucket` | Stable buckets (New/Mid/Old/Very Old) for tree splits | Yes |
| `is_overdue` | Overdue backlog items often need more remedial work | Yes |

Nothing is engineered from any excluded/leakage column.

## 7. Train/test strategy

**Time-based split is primary** (oldest 80% train / newest 20% test) —
the dataset spans 2022‑01‑01 to 2026‑06‑28, and this is a deployment
system that will predict durations for *future* tasks, so testing on the
chronologically newest slice simulates real deployment more realistically
than a random split. A random 80/20 split (`random_state=42`) is also
run for comparison (see `train.py` output) — results are similar,
confirming no major temporal drift.

## 8. Evaluation results (this run — real numbers, see `models/model_metadata.json`)

Trained on 24,000 records, tested on 6,000 held-out (time-based, never
used in tuning).

| Model | MAE (hours) | RMSE (hours) | R² |
|---|---|---|---|
| Baseline (Dummy — predicts median) | 0.86 | 1.20 | -0.08 |
| Random Forest | 0.51 | 0.78 | 0.54 |
| **XGBoost-backend, tuned (final)** | **0.4754** | **0.746** | **0.5837** |

```
MAE  = 0.4754 hours
0.4754 × 60 = 28.5 minutes
Average prediction error ≈ 28.5 minutes
```

**Cross-validation (5-fold, training data only):** mean MAE 0.4725h, std
0.0104h — stable, no sign of overfitting.

**Hyperparameter tuning:** `RandomizedSearchCV` (`scoring="neg_mean_absolute_error"`)
over `n_estimators/max_iter`, `max_depth`, `learning_rate` (+ `subsample`,
`colsample_bytree`, `min_child_weight` when real XGBoost is used). Best
params for this run: `learning_rate≈0.022, max_depth=3, max_iter=148`.
The test set was never touched during tuning.

## 9. Prediction confidence

`predict.py` adds an empirical range: the 10th/90th percentile of
**training-set residuals** (`[-0.66h, +0.66h]` in this run) around each
point prediction. This is a residual-based estimate, not a formal
statistical confidence interval — labeled as such everywhere it's shown.

## 10. Explainability

`src/explain.py` uses SHAP `TreeExplainer` when available, else
scikit-learn `permutation_importance` on the real fitted pipeline (never
fabricated). **Finding worth flagging to your team:** in this run,
`estimated_duration_hours` (the planner's own pre-task estimate)
dominates feature importance (~0.62 of total), with every other feature
contributing far less. This is legitimate (not leakage — it's known
before work starts), but it means the model is currently mostly
*refining the planner's own estimate* rather than learning
independently from asset/severity/condition signals. Worth deciding with
your team whether that's the intended product behavior, or whether a
second model variant without this feature is worth training for
comparison.

## 11. Visualizations

Saved to `models/plots/` after training: `actual_vs_predicted.png`,
`residuals.png`, `target_distribution.png`, `feature_importance.png`,
`error_distribution.png`.

## 12. Project structure

```
railway-maintenance-ai/
├── data/historical_training_dataset.csv
├── models/
│   ├── xgboost_maintenance_duration_model.pkl
│   ├── model_metadata.json
│   └── plots/
├── src/
│   ├── config.py            # feature lists, excluded columns, paths
│   ├── data_loader.py        # Phase 1 profiling
│   ├── preprocessing.py      # data quality checks + ColumnTransformer
│   ├── feature_engineering.py
│   ├── train.py              # Phases 2-8 entry point
│   ├── evaluate.py           # metrics + plots
│   ├── explain.py            # SHAP / permutation importance
│   └── predict.py            # local prediction + CLI test
├── api/app.py                 # FastAPI POST /predict-duration
├── frontend/                  # index.html, styles.css, app.js
├── tests/                     # preprocessing, prediction, API tests
├── requirements.txt
└── README.md
```

## 13. Install & run

```bash
pip install -r requirements.txt

# Train (place historical_training_dataset.csv in data/ first)
python -m src.train

# Local single-prediction test
python -m src.predict

# Start the API
uvicorn api.app:app --reload

# Run tests
pytest tests/ -v

# Frontend: open frontend/index.html in a browser (API must be running on :8000)
```

## 14. Example API request/response

```bash
curl -X POST http://127.0.0.1:8000/predict-duration \
  -H "Content-Type: application/json" \
  -d '{
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
    "estimated_duration_hours": 3.0
  }'
```

```json
{
  "predicted_duration_hours": 3.18,
  "predicted_duration_minutes": 191,
  "estimated_range": { "lower_hours": 2.52, "upper_hours": 3.84 }
}
```

`422` is returned for missing fields, invalid categories, or out-of-range
values (Pydantic-validated). `500` is returned for internal errors, with
no stack trace exposed to the client.

## 15. Reproducibility

- `random_state = 42` everywhere applicable (splits, RF, XGBoost, search CV)
- Python 3.12.3, scikit-learn 1.8.0 (exact versions this run used are in `model_metadata.json`)
- Same preprocessing pipeline object used for train/test/API — no drift possible

## 16. Limitations

- MAE ≈ 28.5 minutes / R² ≈ 0.58 — useful as a decision-support estimate, not a guaranteed exact duration.
- Model is currently heavily anchored to the planner's own `estimated_duration_hours` input (§10) — worth revisiting.
- `section_id` (36 categories) is included per project-owner decision; watch for overfitting to specific sections as more data accumulates.
- Sandbox-trained artifacts in this environment used the HistGradientBoostingRegressor fallback, not real XGBoost/SHAP — re-run `python -m src.train` with `requirements.txt` installed to get the production XGBoost model before deploying.
- `train_count_in_window` / `high_priority_train_count` were excluded to keep Model A decoupled from block-timing — if useful signal is later confirmed safe (e.g. from confirmed pre-planned timetables only), they could be reconsidered.

## 17. Future integration with Model B

This model's `/predict-duration` output (`predicted_duration_hours`) is
designed to be the direct input to Model B's block/schedule optimization
— Model A does not perform any scheduling, multi-department
coordination, or resource allocation itself.

---

# Model B — Maintenance Scheduling & Coordination

Model B answers: *"Which tasks can be coordinated into a common
maintenance block, when should they run, and what's the best feasible
schedule?"* It consumes `predicted_duration_hours` from Model A and is
kept **fully separate** — no shared code path with Model A's training or
prediction internals, only a data contract.

```
Model A (predicted_duration_hours)
        │
        ▼
Model B: constraint validation → OR-Tools CP-SAT / greedy fallback
        │
        ▼
Recommended maintenance blocks
```

## Data gaps found during inspection (Phase A)

The historical dataset has **no time-of-day field** (only `date`), **no
pairwise task-linking key** (so `combined_block`/`number_of_departments`
tell us aggregate stats but never *which* tasks were historically
combined), **no department-compatibility matrix**, and **no resource
(crew/equipment) data**. Everything below that fills these gaps is a
clearly-labeled prototype assumption in `src/model_b/config.py`, not
something learned from data.

| Assumption | Default | Source |
|---|---|---|
| Candidate grouping key | same `section_id` + same `date` | Matches how tasks actually co-occur (up to 6 tasks/section/date in the raw data) |
| Department compatibility | Track↔Signal, Track↔Traction, Traction↔Signal all ✅; same department ❌ (sequenced) | Standard co-possession practice; not present in the data |
| Safety override | `risk_level == "Critical"` → always isolated | Single-critical-possession principle |
| Default block window | 00:00, 6h length | No real window data exists |
| Max block duration | 6h | Historical combined-block max observed was 7.9h |

## Files

New: `src/model_b/config.py`, `constraints.py`, `scheduler.py`,
`api/schedule.py`, `tests/test_model_b_scheduler.py`,
`tests/test_model_b_api.py`. Modified: `api/app.py` (one import + one
`include_router` line — `/predict-duration` untouched),
`requirements.txt` (added `ortools`).

## Algorithm

Graph-coloring with side constraints, per `section_id` group:
1. Tasks whose `predicted_duration_hours` exceeds the max block duration
   are reported in `conflicts` immediately (never silently dropped).
2. Pairwise compatibility computed via `constraints.py`.
3. Partition into the minimum number of mutually-compatible groups:
   - **Preferred:** OR-Tools CP-SAT (exact; minimizes block count, then
     total duration). Written and included in `scheduler.py`, but this
     sandbox has no internet access to `pip install ortools`, so it has
     **not** been executed here — verify it on a machine with the
     package installed.
   - **Fallback (what actually ran and was tested in this sandbox):** a
     pure-Python first-fit-decreasing heuristic — sort tasks longest
     first, place each into the first compatible existing block, else
     start a new one. `scheduler.py` tries `ortools` first and falls
     back automatically; the response's `backend` field always reports
     which one ran.
4. Blocks are sequenced within the section's window; a block that
   wouldn't fit is reported in `conflicts`, not dropped.

### Worked example (from the spec)
Track=3h (High risk), Traction=2h (Medium), Signal=1.5h (Low), same
section — all cross-department pairs compatible → **1 block, 3 hours**
(the longest task, not the sum). Verified by
`test_all_compatible_departments_form_one_block`.

If Track were `Critical` risk: Track is force-isolated → **2 blocks**
(Track alone for 3h, then Traction+Signal for 2h, sequenced
`00:00–03:00` / `03:00–05:00`). Verified by
`test_critical_risk_task_is_always_isolated`.

## API

```
POST /schedule-maintenance
```
Request — either supply `predicted_duration_hours` directly per task
(preferred, strict separation from Model A), or supply the raw Model A
input fields and Model B will call Model A's prediction internally
(convenience path, opt-in per task):

```json
{
  "date": "2026-09-01",
  "tasks": [
    {"task_id": "T1", "department": "Track", "section_id": "SEC0007",
     "risk_level": "High", "predicted_duration_hours": 3.0},
    {"task_id": "T2", "department": "Traction", "section_id": "SEC0007",
     "risk_level": "Medium", "predicted_duration_hours": 2.0},
    {"task_id": "T3", "department": "Signal", "section_id": "SEC0007",
     "risk_level": "Low", "predicted_duration_hours": 1.5}
  ]
}
```

Response:

```json
{
  "blocks": [
    {
      "block_id": "SEC0007-B1", "section_id": "SEC0007",
      "start_time": "00:00", "end_time": "03:00", "duration_hours": 3.0,
      "departments": ["Signal", "Track", "Traction"],
      "tasks": [
        {"task_id": "T1", "department": "Track", "predicted_duration_hours": 3.0, "mode": "parallel"},
        {"task_id": "T2", "department": "Traction", "predicted_duration_hours": 2.0, "mode": "parallel"},
        {"task_id": "T3", "department": "Signal", "predicted_duration_hours": 1.5, "mode": "parallel"}
      ],
      "reason": "Signal, Track, Traction are mutually compatible and run in parallel. Block duration is set by the longest task (3.0h), not the sum. ..."
    }
  ],
  "conflicts": [],
  "backend": "greedy",
  "optimization_summary": {
    "total_blocks": 1, "total_block_duration_hours": 3.0,
    "tasks_scheduled": 3, "tasks_unscheduled": 0, "sections_processed": 1
  }
}
```

`422` for missing/invalid fields (e.g. an unknown department, an empty
`tasks` list, or a task with neither `predicted_duration_hours` nor a
complete set of raw Model A fields). `500` only for internal errors (no
stack trace exposed). `max_block_duration_hours` / `window_length_hours`
can be overridden per-request.

## Tests

`tests/test_model_b_scheduler.py` — 9 tests against the pure-Python
scheduler core, **run and passing in this sandbox** (no dependencies
beyond the standard library): all-compatible → 1 block, Critical-risk
isolation, same-department sequencing, over-max-duration task reported
not dropped, window overflow reported not dropped, empty input,
multi-section independence, custom config override, compatibility
reason strings.

`tests/test_model_b_api.py` — 9 tests against the FastAPI layer via
`TestClient`, including a regression check that `/predict-duration`
still works unchanged. Requires `fastapi` (not installed in this
sandbox — same limitation noted for Model A's `test_api.py`); written
and reviewed for correctness, run with `pytest tests/ -v` on a machine
with `requirements.txt` installed.

## Limitations

- Department-compatibility matrix and block windows are **prototype
  assumptions**, not learned from data or confirmed with domain
  experts — replace before production use.
- No resource (crew/equipment) constraints enforced yet — hook point
  exists in `SchedulerConfig` for a future addition.
- OR-Tools CP-SAT path is written but unexecuted in this sandbox;
  validate it on a machine with `ortools` installed before relying on
  it for larger task pools (the greedy fallback is fine for the small
  per-section pools — typically ≤6 tasks — seen in the historical data,
  but CP-SAT's exactness matters more as pool size grows).

