#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import dataframe_sha256, get_feature_set, load_config, metadata_columns, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the fixed paper feature set from a full Phonova summary table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV with Phonova summary columns.")
    parser.add_argument("--output", type=Path, required=True, help="CSV containing metadata plus the selected feature set.")
    parser.add_argument("--config", type=Path, default=None, help="Paper configuration JSON.")
    parser.add_argument("--feature-set", default=None, help="Feature set name from the config.")
    parser.add_argument("--allow-missing", action="store_true", help="Fill missing feature columns with NaN.")
    parser.add_argument("--metadata-column", action="append", default=[], help="Extra metadata column to preserve.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    features = get_feature_set(config, args.feature_set)
    df = pd.read_csv(args.input)

    missing = [feature for feature in features if feature not in df.columns]
    if missing and not args.allow_missing:
        raise KeyError(
            f"Input is missing {len(missing)} paper features. "
            "Re-run with --allow-missing only for smoke tests or partial feature tables. "
            f"First missing features: {missing[:10]}"
        )
    for feature in missing:
        df[feature] = np.nan

    meta_cols = metadata_columns(config, df)
    for column in args.metadata_column:
        if column in df.columns and column not in meta_cols:
            meta_cols.append(column)

    out = df[meta_cols + features].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    write_json(
        args.output.with_suffix(args.output.suffix + ".meta.json"),
        {
            "input": str(args.input),
            "output": str(args.output),
            "feature_set": args.feature_set or config.get("default_feature_set"),
            "n_rows": int(len(out)),
            "n_features": int(len(features)),
            "metadata_columns": meta_cols,
            "missing_features": missing,
            "output_sha256": dataframe_sha256(out),
        },
    )
    print(f"Saved {len(out)} rows and {len(features)} features to {args.output}")
    if missing:
        print(f"Filled {len(missing)} missing features with NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
