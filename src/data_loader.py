"""
Data loading and dataset profiling.

Run standalone for the Phase 1 profiling report:
    python -m src.data_loader
"""

import pandas as pd

from src import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the historical training dataset from disk."""
    path = path or config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place "
            f"'historical_training_dataset.csv' inside the data/ folder."
        )
    df = pd.read_csv(path)
    return df


def profile_dataset(df: pd.DataFrame) -> None:
    """Print a full profiling report. Never assumes schema - reads it live."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("=" * 70)
    print("DATASET PROFILING REPORT")
    print("=" * 70)

    print(f"\nRows: {df.shape[0]:,}   Columns: {df.shape[1]}")

    print("\n--- Column dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    print(report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False))

    print("\n--- Duplicate records ---")
    dup_rows = df.duplicated().sum()
    dup_ids = df["task_id"].duplicated().sum() if "task_id" in df.columns else "n/a"
    print(f"Full-row duplicates: {dup_rows} ({dup_rows / len(df) * 100:.2f}%)")
    print(f"task_id duplicates: {dup_ids}")

    print("\n--- Numeric ranges ---")
    print(df.describe().T)

    print("\n--- Categorical unique values ---")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col} -> {df[col].nunique()} unique")
        print(df[col].value_counts(dropna=False).head(10))

    if config.TARGET_COLUMN in df.columns:
        print(f"\n--- Target column: {config.TARGET_COLUMN} ---")
        print(df[config.TARGET_COLUMN].describe())
        n_leq0 = (df[config.TARGET_COLUMN] <= 0).sum()
        print(f"Records with target <= 0: {n_leq0}")

    print("\n--- Sample rows ---")
    print(df.head(5).to_string())
    print("=" * 70)


if __name__ == "__main__":
    data = load_raw_data()
    profile_dataset(data)
