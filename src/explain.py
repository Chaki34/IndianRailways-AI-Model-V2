"""
Model explainability.

Uses SHAP when it is installed (the standard, preferred approach per spec).
If SHAP is not available in the current environment, falls back to
scikit-learn's permutation_importance, which is also computed from the
actual fitted model on real held-out data - nothing here is fabricated.
"""

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance

from src import config


def get_feature_names(preprocessor) -> list:
    """Extract output feature names from the fitted ColumnTransformer."""
    num_names = config.ALL_NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(config.ALL_CATEGORICAL_FEATURES))
    return list(num_names) + cat_names


def compute_global_importance(pipeline, X_test, y_test) -> dict:
    """
    Returns {method, feature_names, importances} using the best available
    technique. `pipeline` is the full fitted sklearn Pipeline
    (preprocessor + model).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    if SHAP_AVAILABLE and hasattr(model, "predict"):
        try:
            X_test_transformed = preprocessor.transform(X_test)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_transformed)
            importances = np.abs(shap_values).mean(axis=0)
            return {
                "method": "SHAP (TreeExplainer, mean |SHAP value|)",
                "feature_names": feature_names,
                "importances": importances,
                "shap_values": shap_values,
                "X_test_transformed": X_test_transformed,
            }
        except Exception as e:
            print(f"  SHAP explainability failed ({e}); falling back to "
                  f"permutation importance.")

    # Fallback: permutation importance on the real fitted pipeline/test data
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=8,
        random_state=config.RANDOM_STATE, scoring="neg_mean_absolute_error",
    )
    return {
        "method": "Permutation importance (fallback - SHAP not installed "
                  "in this environment)",
        "feature_names": X_test.columns.tolist(),
        "importances": result.importances_mean,
    }


def print_top_factors(importance_result: dict, top_n: int = 10) -> None:
    names = importance_result["feature_names"]
    importances = importance_result["importances"]
    order = np.argsort(importances)[::-1][:top_n]
    print(f"\nExplainability method: {importance_result['method']}")
    print(f"Top {top_n} contributing factors:")
    for i in order:
        print(f"  {names[i]:<35} {importances[i]:.4f}")
