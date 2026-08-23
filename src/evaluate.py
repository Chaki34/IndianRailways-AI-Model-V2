"""
Regression evaluation metrics and diagnostic plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE_hours": round(float(mae), 4),
        "MAE_minutes": round(float(mae) * 60, 1),
        "RMSE_hours": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name} Performance")
    print(f"  MAE  : {metrics['MAE_hours']} hours  (~{metrics['MAE_minutes']} minutes)")
    print(f"  RMSE : {metrics['RMSE_hours']} hours")
    print(f"  R2   : {metrics['R2']}")


def print_comparison_table(results: dict) -> None:
    print("\n" + "-" * 60)
    print(f"{'Model':<22}{'MAE (h)':<12}{'RMSE (h)':<12}{'R2':<10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<22}{m['MAE_hours']:<12}{m['RMSE_hours']:<12}{m['R2']:<10}")
    print("-" * 60)


def plot_actual_vs_predicted(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="#2563eb")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Maintenance Duration (hours)")
    ax.set_ylabel("Predicted Maintenance Duration (hours)")
    ax.set_title("Actual vs Predicted Maintenance Duration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residuals(y_true, y_pred, path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=12, color="#059669")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Duration (hours)")
    ax.set_ylabel("Residual = Actual - Predicted (hours)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_target_distribution(y, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y, bins=40, color="#7c3aed", edgecolor="white")
    ax.set_xlabel("Actual Maintenance Duration (hours)")
    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of Historical Maintenance Duration")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_feature_importance(feature_names, importances, path, top_n=15):
    order = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order], np.array(importances)[order], color="#f97316")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_distribution(y_true, y_pred, path):
    errors = (y_true - y_pred) * 60  # minutes
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=40, color="#dc2626", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Prediction Error (minutes) = Actual - Predicted")
    ax.set_ylabel("Number of Records")
    ax.set_title("Prediction Error Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
