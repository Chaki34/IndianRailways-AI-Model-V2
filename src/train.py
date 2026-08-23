"""
Model A - Maintenance Duration Prediction - training entry point.

    python -m src.train

Runs Phases 2-8 of the project sequence: cleaning, baseline, primary
model, evaluation, tuning, explainability, and model saving.
"""

import json
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config, evaluate, explain
from src.data_loader import load_raw_data, profile_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import (build_preprocessor, clean_dataset,
                                print_data_quality_report,
                                run_data_quality_checks)

# ---------------------------------------------------------------------------
# Model backend selection - XGBoost is the specified primary algorithm.
# Falls back to HistGradientBoostingRegressor ONLY if xgboost is not
# installed in the current environment (this sandbox has no internet
# access to install it). On a normal machine with `pip install -r
# requirements.txt`, XGBoost is used automatically.
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    BACKEND = "xgboost.XGBRegressor"

    def make_primary_model(**params):
        return XGBRegressor(random_state=config.RANDOM_STATE,
                             objective="reg:squarederror", **params)

    PARAM_DISTRIBUTIONS = {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
    }
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn.HistGradientBoostingRegressor (XGBOOST FALLBACK - " \
              "xgboost not installed in this environment)"
    print("\n[WARNING] xgboost is not installed in this environment. "
          "Falling back to sklearn's HistGradientBoostingRegressor so the "
          "pipeline can still be validated end-to-end with REAL metrics. "
          "Run `pip install -r requirements.txt` on a machine with "
          "internet access, then re-run this script, to train the actual "
          "XGBoost model specified in the project requirements.\n")

    def make_primary_model(**params):
        return HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, **params)

    PARAM_DISTRIBUTIONS = {
        "model__max_iter": randint(100, 400),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.02, 0.28),
    }


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Older records train, newest records test - simulates real deployment."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]
    print(f"\nTime-based split: train dates {train_df['date'].min()} -> "
          f"{train_df['date'].max()}, test dates {test_df['date'].min()} -> "
          f"{test_df['date'].max()}")
    return train_df, test_df


def build_pipeline(model) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def main():
    print("=" * 70)
    print("MODEL A - MAINTENANCE DURATION PREDICTION - TRAINING")
    print("=" * 70)
    print(f"Model backend in use: {BACKEND}")

    # --- Phase 1: profiling (brief) ---------------------------------------
    raw_df = load_raw_data()
    print(f"\nDataset loaded: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")

    # --- Phase 2: data quality + cleaning ----------------------------------
    dq_report = run_data_quality_checks(raw_df)
    print_data_quality_report(dq_report)
    df = clean_dataset(raw_df)
    df = add_engineered_features(df)

    # --- Train/test split ---------------------------------------------------
    # Time-based split (primary, realistic-deployment metric)
    train_df, test_df = time_based_split(df, test_size=0.2)
    X_train, y_train = train_df[config.FEATURE_COLUMNS], train_df[config.TARGET_COLUMN]
    X_test, y_test = test_df[config.FEATURE_COLUMNS], test_df[config.TARGET_COLUMN]
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Testing samples : {len(X_test):,}")

    # Random split (secondary, for comparison only)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        df[config.FEATURE_COLUMNS], df[config.TARGET_COLUMN],
        test_size=0.2, random_state=config.RANDOM_STATE)

    # --- Phase 3: baseline ---------------------------------------------------
    results = {}

    dummy_pipeline = build_pipeline(DummyRegressor(strategy="median"))
    dummy_pipeline.fit(X_train, y_train)
    dummy_pred = dummy_pipeline.predict(X_test)
    results["Baseline (Dummy)"] = evaluate.compute_metrics(y_test, dummy_pred)

    rf_pipeline = build_pipeline(RandomForestRegressor(
        n_estimators=150, random_state=config.RANDOM_STATE, n_jobs=1))
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    results["Random Forest"] = evaluate.compute_metrics(y_test, rf_pred)

    # --- Phase 4: primary model (default config) -----------------------------
    primary_pipeline = build_pipeline(make_primary_model())
    primary_pipeline.fit(X_train, y_train)
    primary_pred = primary_pipeline.predict(X_test)
    results["XGBoost (default)"] = evaluate.compute_metrics(y_test, primary_pred)

    print("\n" + "=" * 70)
    print("PHASE 3-4: BASELINE COMPARISON (time-based test set)")
    print("=" * 70)
    evaluate.print_comparison_table(results)

    # --- Phase 6: hyperparameter tuning --------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6: HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 70)
    search_pipeline = build_pipeline(make_primary_model())
    search = RandomizedSearchCV(
        search_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=15,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV MAE (hours): {-search.best_score_:.4f}")

    # --- Phase 13: cross-validation on tuned model ---------------------------
    cv_scores = cross_val_score(
        best_pipeline, X_train, y_train, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=1)
    cv_mae = -cv_scores
    print(f"\nCross-validation MAE (5-fold, training data only):")
    print(f"  Mean: {cv_mae.mean():.4f} hours   Std: {cv_mae.std():.4f} hours")

    # --- Phase 5: final evaluation on untouched test set ----------------------
    final_pred = best_pipeline.predict(X_test)
    final_metrics = evaluate.compute_metrics(y_test, final_pred)
    results["XGBoost (tuned) - FINAL"] = final_metrics

    print("\n" + "=" * 70)
    print("PHASE 5: FINAL MODEL EVALUATION (held-out test set, never used in tuning)")
    print("=" * 70)
    evaluate.print_comparison_table(results)
    evaluate.print_metrics("XGBoost (tuned) - FINAL", final_metrics)

    # Random-split comparison (secondary/informational)
    random_split_pipeline = build_pipeline(make_primary_model(**{
        k.replace("model__", ""): v for k, v in search.best_params_.items()}))
    random_split_pipeline.fit(Xr_train, yr_train)
    random_pred = random_split_pipeline.predict(Xr_test)
    random_metrics = evaluate.compute_metrics(yr_test, random_pred)
    print("\n--- For comparison: same tuned config, RANDOM 80/20 split ---")
    evaluate.print_metrics("XGBoost (tuned) - random split", random_metrics)

    # --- Phase 7: explainability -----------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 7: MODEL EXPLAINABILITY")
    print("=" * 70)
    importance_result = explain.compute_global_importance(best_pipeline, X_test, y_test)
    explain.print_top_factors(importance_result)

    # --- Phase 16: visualizations -----------------------------------------------
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate.plot_actual_vs_predicted(
        y_test.values, final_pred, config.PLOTS_DIR / "actual_vs_predicted.png")
    evaluate.plot_residuals(
        y_test.values, final_pred, config.PLOTS_DIR / "residuals.png")
    evaluate.plot_target_distribution(
        df[config.TARGET_COLUMN], config.PLOTS_DIR / "target_distribution.png")
    evaluate.plot_feature_importance(
        importance_result["feature_names"], importance_result["importances"],
        config.PLOTS_DIR / "feature_importance.png")
    evaluate.plot_error_distribution(
        y_test.values, final_pred, config.PLOTS_DIR / "error_distribution.png")
    print(f"\nPlots saved to {config.PLOTS_DIR}")

    # --- Phase 14: uncertainty estimate (quantile-based, from CV residuals) ---
    train_pred = best_pipeline.predict(X_train)
    train_residuals = (y_train.values - train_pred)
    resid_lower, resid_upper = np.percentile(train_residuals, [10, 90])
    print(f"\nPrediction range methodology: 10th/90th percentile of training "
          f"residuals ([{resid_lower:.2f}h, {resid_upper:.2f}h]) added to "
          f"each point prediction. This is an empirical residual-based "
          f"range, not a formal statistical confidence interval.")

    # --- Phase 8: save model + metadata ------------------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_pipeline, config.MODEL_PATH)

    metadata = {
        "model_name": "railway_maintenance_duration_predictor",
        "model_backend": BACKEND,
        "training_date": datetime.now().isoformat(),
        "dataset_path": str(config.DATA_PATH),
        "dataset_row_count": int(len(df)),
        "feature_names": config.FEATURE_COLUMNS,
        "target_name": config.TARGET_COLUMN,
        "training_row_count": int(len(X_train)),
        "test_row_count": int(len(X_test)),
        "split_strategy": "time_based (oldest 80% train / newest 20% test)",
        "best_hyperparameters": search.best_params_,
        "MAE_hours": final_metrics["MAE_hours"],
        "MAE_minutes": final_metrics["MAE_minutes"],
        "RMSE_hours": final_metrics["RMSE_hours"],
        "R2": final_metrics["R2"],
        "cv_mae_mean_hours": round(float(cv_mae.mean()), 4),
        "cv_mae_std_hours": round(float(cv_mae.std()), 4),
        "residual_range_10_90_pct": [round(float(resid_lower), 3), round(float(resid_upper), 3)],
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "excluded_columns": config.EXCLUDED_COLUMNS,
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Metadata saved to: {config.METADATA_PATH}")
    print("Model saved successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
