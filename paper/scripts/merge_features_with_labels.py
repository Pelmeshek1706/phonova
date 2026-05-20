#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import dataframe_sha256, normalize_id, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge extracted summary features with labels, severity scores, metadata, and optional split assignments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-csv", type=Path, required=True, help="Feature table from extract_paper_features.py.")
    parser.add_argument("--labels-csv", type=Path, required=True, help="Labels/metadata table.")
    parser.add_argument("--output", type=Path, required=True, help="Merged output CSV.")
    parser.add_argument("--id-column", default="Participant", help="Join column present in both tables.")
    parser.add_argument("--split-csv", type=Path, default=None, help="Optional CSV containing split assignments.")
    parser.add_argument("--split-column", default="split", help="Split column to merge from --split-csv.")
    parser.add_argument("--how", choices=["inner", "left"], default="inner", help="Merge mode.")
    return parser


def _normalize_join_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Missing join column '{column}'")
    out = df.copy()
    out[column] = out[column].map(normalize_id)
    return out


def main() -> int:
    args = build_parser().parse_args()
    features = _normalize_join_column(pd.read_csv(args.features_csv), args.id_column)
    labels = _normalize_join_column(pd.read_csv(args.labels_csv), args.id_column)

    label_cols = [column for column in labels.columns if column != args.id_column]
    merged = features.merge(labels[[args.id_column] + label_cols], on=args.id_column, how=args.how, validate="one_to_one")

    if args.split_csv is not None:
        splits = _normalize_join_column(pd.read_csv(args.split_csv), args.id_column)
        if args.split_column not in splits.columns:
            raise KeyError(f"Missing split column '{args.split_column}' in {args.split_csv}")
        if args.split_column in merged.columns:
            merged = merged.drop(columns=[args.split_column])
        merged = merged.merge(
            splits[[args.id_column, args.split_column]],
            on=args.id_column,
            how="left",
            validate="one_to_one",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    write_json(
        args.output.with_suffix(args.output.suffix + ".meta.json"),
        {
            "features_csv": str(args.features_csv),
            "labels_csv": str(args.labels_csv),
            "split_csv": str(args.split_csv) if args.split_csv else None,
            "output": str(args.output),
            "id_column": args.id_column,
            "how": args.how,
            "n_feature_rows": int(len(features)),
            "n_label_rows": int(len(labels)),
            "n_output_rows": int(len(merged)),
            "output_sha256": dataframe_sha256(merged),
        },
    )
    print(f"Saved merged table with {len(merged)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
