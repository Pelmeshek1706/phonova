#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from _common import get_feature_set, load_config, metadata_columns, normalize_id, write_json


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify paper scripts against frozen paper artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paper-input-en", type=Path, required=True, help="Frozen EN paper input feature CSV.")
    parser.add_argument("--paper-input-uk", type=Path, required=True, help="Frozen UK paper input feature CSV.")
    parser.add_argument(
        "--reference-selected-models",
        type=Path,
        required=True,
        help="Frozen section3_en_selected_models.csv artifact.",
    )
    parser.add_argument(
        "--reference-test-predictions",
        type=Path,
        required=True,
        help="Frozen section3_en_test_predictions.csv artifact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for verifier outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Paper configuration JSON.")
    parser.add_argument("--feature-set", default=None, help="Feature set to verify.")
    parser.add_argument("--model", default="paper_b0_rho05_vaderfull", help="Configured model name to train.")
    parser.add_argument("--family", default="B0_rho05_vaderfull", help="Reference paper family name.")
    parser.add_argument("--metric-atol", type=float, default=1e-12, help="Allowed absolute metric drift.")
    parser.add_argument("--prediction-atol", type=float, default=1e-8, help="Allowed probability drift.")
    return parser


def _run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _compare_selected(
    source_path: Path,
    selected_path: Path,
    config: dict[str, Any],
    features: list[str],
) -> dict[str, Any]:
    source = _load_required(source_path)
    selected = _load_required(selected_path)
    columns = metadata_columns(config, source) + features
    expected = source[columns].copy()
    assert_frame_equal(selected, expected, check_dtype=False, check_exact=False, rtol=0.0, atol=0.0)
    return {
        "rows": int(len(selected)),
        "columns": int(len(selected.columns)),
        "metadata_columns": int(len(columns) - len(features)),
        "feature_columns": int(len(features)),
        "cell_differences": 0,
        "columns_exact": list(selected.columns) == columns,
    }


def _reference_rows(path: Path, family: str) -> pd.DataFrame:
    reference = _load_required(path)
    rows = reference.loc[reference["family"].astype(str) == family].copy()
    if rows.empty:
        raise ValueError(f"No rows found for family={family!r} in {path}")
    return rows


def _compare_feature_contract(reference_rows: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    by_target: dict[str, Any] = {}
    for _, row in reference_rows.iterrows():
        payload = json.loads(row["config_json"])
        reference_features = list(payload["feature_columns"])
        target = str(row["target"])
        by_target[target] = {
            "reference_count": len(reference_features),
            "config_count": len(features),
            "set_equal": set(reference_features) == set(features),
            "order_equal": reference_features == features,
        }
        if reference_features != features:
            missing = [feature for feature in reference_features if feature not in features]
            extra = [feature for feature in features if feature not in reference_features]
            by_target[target]["missing_from_config"] = missing
            by_target[target]["extra_in_config"] = extra
            raise AssertionError(f"Feature contract mismatch for {target}: missing={missing}, extra={extra}")
    return by_target


def _compare_metrics(summary_path: Path, reference_rows: pd.DataFrame, atol: float) -> dict[str, Any]:
    summary = _load_required(summary_path)
    metric_pairs = {
        "n_features": "n_features",
        "threshold": "threshold",
        "test_pr_auc": "official_test_pr",
        "test_f1_macro": "official_test_f1",
        "test_f1_binary": "official_test_f1_binary",
        "test_accuracy": "official_test_accuracy",
        "test_roc_auc": "official_test_roc",
    }
    by_target: dict[str, Any] = {}
    max_abs_diff = 0.0
    for _, ref in reference_rows.iterrows():
        target = str(ref["target"])
        actual_rows = summary.loc[summary["target"].astype(str) == target]
        if len(actual_rows) != 1:
            raise AssertionError(f"Expected one summary row for {target}, found {len(actual_rows)}")
        actual = actual_rows.iloc[0]
        target_report: dict[str, Any] = {}
        for actual_col, ref_col in metric_pairs.items():
            actual_value = float(actual[actual_col])
            ref_value = float(ref[ref_col])
            diff = abs(actual_value - ref_value)
            max_abs_diff = max(max_abs_diff, diff)
            target_report[actual_col] = {
                "actual": actual_value,
                "reference": ref_value,
                "abs_diff": diff,
            }
            if diff > atol:
                raise AssertionError(
                    f"Metric drift for {target} {actual_col}: actual={actual_value}, "
                    f"reference={ref_value}, abs_diff={diff}, atol={atol}"
                )
        by_target[target] = target_report
    return {"max_abs_diff": max_abs_diff, "by_target": by_target}


def _compare_predictions(prediction_dir: Path, reference_path: Path, family: str, model_name: str, atol: float) -> dict[str, Any]:
    reference = _load_required(reference_path)
    reference = reference.loc[reference["family"].astype(str) == family].copy()
    if reference.empty:
        raise ValueError(f"No prediction rows found for family={family!r} in {reference_path}")

    by_target: dict[str, Any] = {}
    for target, ref_rows in reference.groupby("target", sort=True):
        pred_path = prediction_dir / f"predictions_{model_name}_{target}.csv"
        actual = _load_required(pred_path)

        left = actual.copy()
        right = ref_rows.copy()
        left["Participant_norm"] = left["Participant"].map(normalize_id)
        right["Participant_norm"] = right["Participant"].map(normalize_id)
        merged = left.merge(right, on=["Participant_norm", "target"], how="outer", indicator=True)
        unmatched = merged.loc[merged["_merge"] != "both"]
        if not unmatched.empty:
            raise AssertionError(f"Prediction participant mismatch for {target}: {len(unmatched)} unmatched rows")

        merged = merged.sort_values("Participant_norm").reset_index(drop=True)
        proba_diff = np.abs(merged["proba"].astype(float) - merged["proba_en_test"].astype(float))
        y_true_diff = int((merged["y_true"].astype(int) != merged["y_true_en_test"].astype(int)).sum())
        y_pred_diff = int((merged["y_pred"].astype(int) != merged["y_pred_en_test"].astype(int)).sum())
        max_proba_abs_diff = float(proba_diff.max()) if len(proba_diff) else 0.0
        if y_true_diff or y_pred_diff or max_proba_abs_diff > atol:
            raise AssertionError(
                f"Prediction drift for {target}: y_true_diff={y_true_diff}, y_pred_diff={y_pred_diff}, "
                f"max_proba_abs_diff={max_proba_abs_diff}, atol={atol}"
            )
        by_target[str(target)] = {
            "rows": int(len(merged)),
            "y_true_differences": y_true_diff,
            "y_pred_differences": y_pred_diff,
            "max_proba_abs_diff": max_proba_abs_diff,
        }
    return by_target


def _write_markdown(report_path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Paper Parity Verification",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Selected Features",
    ]
    for language, payload in report["selected_features"].items():
        lines.append(
            f"- {language}: {payload['rows']} rows, {payload['feature_columns']} features, "
            f"{payload['metadata_columns']} metadata columns, cell differences = {payload['cell_differences']}"
        )
    lines.extend(["", "## Model Metrics"])
    for target, metrics in report["model_metrics"]["by_target"].items():
        lines.append(f"- {target}:")
        for name, values in metrics.items():
            lines.append(
                f"  - {name}: actual {values['actual']:.16g}, reference {values['reference']:.16g}, "
                f"abs diff {values['abs_diff']:.3g}"
            )
    lines.extend(["", "## Predictions"])
    for target, payload in report["predictions"].items():
        lines.append(
            f"- {target}: {payload['rows']} rows, y_true diffs = {payload['y_true_differences']}, "
            f"y_pred diffs = {payload['y_pred_differences']}, "
            f"max probability abs diff = {payload['max_proba_abs_diff']:.3g}"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    features = get_feature_set(config, args.feature_set)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_dir = str(SCRIPT_DIR.parents[1] / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    selected_en = args.output_dir / "selected_en_paper_82.csv"
    selected_uk = args.output_dir / "selected_uk_paper_82.csv"
    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "select_paper_features.py"),
            "--input",
            str(args.paper_input_en),
            "--output",
            str(selected_en),
            "--config",
            str(args.config) if args.config else str(SCRIPT_DIR.parents[1] / "paper" / "paper_reproducibility_config.json"),
            "--feature-set",
            args.feature_set or config.get("default_feature_set", "paper_82"),
        ],
        env,
    )
    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "select_paper_features.py"),
            "--input",
            str(args.paper_input_uk),
            "--output",
            str(selected_uk),
            "--config",
            str(args.config) if args.config else str(SCRIPT_DIR.parents[1] / "paper" / "paper_reproducibility_config.json"),
            "--feature-set",
            args.feature_set or config.get("default_feature_set", "paper_82"),
        ],
        env,
    )

    model_dir = args.output_dir / "model_results"
    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "train_evaluate_models.py"),
            "--input",
            str(selected_en),
            "--output-dir",
            str(model_dir),
            "--config",
            str(args.config) if args.config else str(SCRIPT_DIR.parents[1] / "paper" / "paper_reproducibility_config.json"),
            "--feature-set",
            args.feature_set or config.get("default_feature_set", "paper_82"),
            "--models",
            args.model,
            "--bootstrap",
            "0",
        ],
        env,
    )

    reference_rows = _reference_rows(args.reference_selected_models, args.family)
    report = {
        "status": "passed",
        "feature_set": args.feature_set or config.get("default_feature_set", "paper_82"),
        "model": args.model,
        "family": args.family,
        "selected_features": {
            "EN": _compare_selected(args.paper_input_en, selected_en, config, features),
            "UK": _compare_selected(args.paper_input_uk, selected_uk, config, features),
        },
        "feature_contract": _compare_feature_contract(reference_rows, features),
        "model_metrics": _compare_metrics(model_dir / "summary_results.csv", reference_rows, args.metric_atol),
        "predictions": _compare_predictions(
            model_dir,
            args.reference_test_predictions,
            args.family,
            args.model,
            args.prediction_atol,
        ),
    }

    json_path = args.output_dir / "parity_report.json"
    markdown_path = args.output_dir / "parity_report.md"
    write_json(json_path, report)
    _write_markdown(markdown_path, report)
    print(f"Parity verification passed. Wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
