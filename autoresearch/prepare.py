import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "merged_features.csv"
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"

TARGETS = ("Depression_label", "PTSD_label")
EXPECTED_SPLITS = ("train", "dev", "test")
REQUIRED_COLUMNS = ("Participant", "split", *TARGETS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate merged_features.csv and write a compact data summary.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to merged feature CSV.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for data summary artifact.",
    )
    return parser.parse_args()


def summarize_target(df: pd.DataFrame, target: str):
    grouped = (
        df.groupby("split")[target]
        .agg(count="size", positives="sum")
        .reindex(EXPECTED_SPLITS)
        .fillna(0)
    )
    grouped["positive_rate"] = grouped["positives"] / grouped["count"].replace(0, pd.NA)
    return grouped.round(6).reset_index().to_dict(orient="records")


def main():
    args = parse_args()
    csv_path = args.csv_path.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    present_splits = set(df["split"].dropna().astype(str))
    missing_splits = [split for split in EXPECTED_SPLITS if split not in present_splits]
    if missing_splits:
        raise ValueError(f"Missing required splits: {missing_splits}")

    feature_columns = [
        col for col in df.columns if col not in {"Participant", "split", *TARGETS}
    ]
    categorical_columns = (
        df[feature_columns]
        .select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )
    numeric_columns = [col for col in feature_columns if col not in categorical_columns]

    summary = {
        "csv_path": str(csv_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "feature_columns": int(len(feature_columns)),
        "numeric_feature_columns": int(len(numeric_columns)),
        "categorical_feature_columns": int(len(categorical_columns)),
        "split_counts": df["split"].value_counts().reindex(EXPECTED_SPLITS).fillna(0).astype(int).to_dict(),
        "targets": {
            target: summarize_target(df, target)
            for target in TARGETS
        },
    }

    summary_path = artifacts_dir / "data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")
    print(f"Feature columns: {summary['feature_columns']}")
    print(f"Numeric feature columns: {summary['numeric_feature_columns']}")
    print(f"Categorical feature columns: {summary['categorical_feature_columns']}")
    print("Split counts:")
    for split, count in summary["split_counts"].items():
        print(f"  {split}: {count}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
