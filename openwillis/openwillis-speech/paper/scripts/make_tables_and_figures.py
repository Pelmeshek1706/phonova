#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import feature_group, get_feature_set, load_config, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper/supplement tables and lightweight figures from model outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory produced by train_evaluate_models.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for tables and figures.")
    parser.add_argument("--config", type=Path, default=None, help="Paper configuration JSON.")
    parser.add_argument("--feature-set", default=None, help="Feature set name from config.")
    parser.add_argument("--top-n", type=int, default=20, help="Top coefficient rows to export per model/target.")
    return parser


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    rows = []
    columns = [str(column) for column in df.columns]
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = [str(row[column]) for column in df.columns]
        rows.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _maybe_write_metric_figure(summary: pd.DataFrame, output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    metric_cols = ["test_f1_macro", "test_pr_auc", "test_roc_auc"]
    rows = []
    for _, row in summary.iterrows():
        for metric in metric_cols:
            rows.append(
                {
                    "label": f"{row['target']} / {row['model_name']}",
                    "metric": metric.replace("test_", ""),
                    "value": row[metric],
                }
            )
    plot_df = pd.DataFrame(rows)
    labels = plot_df["label"].drop_duplicates().tolist()
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    for offset, metric in enumerate(metric_cols):
        values = [
            plot_df[(plot_df["label"] == label) & (plot_df["metric"] == metric.replace("test_", ""))]["value"].iloc[0]
            for label in labels
        ]
        positions = [idx + (offset - 1) * width for idx in x]
        ax.bar(positions, values, width=width, label=metric.replace("test_", ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "model_metric_summary.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    features = get_feature_set(config, args.feature_set)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.results_dir / "summary_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    table_cols = [
        "target",
        "model_name",
        "n_features",
        "threshold",
        "oof_pr_auc",
        "oof_f1_macro",
        "test_pr_auc",
        "test_f1_macro",
        "test_f1_binary",
        "test_accuracy",
        "test_roc_auc",
    ]
    paper_metrics = summary[[column for column in table_cols if column in summary.columns]].copy()
    paper_metrics.to_csv(args.output_dir / "paper_model_metrics.csv", index=False)
    _write_markdown_table(paper_metrics.round(4), args.output_dir / "paper_model_metrics.md")

    coef_rows = []
    for coef_path in sorted(args.results_dir.glob("coefficients_*.csv")):
        coef = pd.read_csv(coef_path).head(args.top_n)
        stem = coef_path.stem.replace("coefficients_", "")
        coef.insert(0, "model_target", stem)
        coef_rows.append(coef)
    if coef_rows:
        coef_out = pd.concat(coef_rows, ignore_index=True)
        coef_out.to_csv(args.output_dir / "top_coefficients.csv", index=False)

    feature_groups = pd.DataFrame({"feature": features})
    feature_groups["feature_group"] = feature_groups["feature"].map(feature_group)
    feature_groups.to_csv(args.output_dir / "paper_feature_groups.csv", index=False)
    feature_groups.groupby("feature_group").size().reset_index(name="n_features").to_csv(
        args.output_dir / "paper_feature_group_counts.csv",
        index=False,
    )

    figure_path = _maybe_write_metric_figure(summary, args.output_dir)
    write_json(
        args.output_dir / "tables_figures_manifest.json",
        {
            "results_dir": str(args.results_dir),
            "output_dir": str(args.output_dir),
            "metrics_csv": str(args.output_dir / "paper_model_metrics.csv"),
            "metrics_md": str(args.output_dir / "paper_model_metrics.md"),
            "top_coefficients_csv": str(args.output_dir / "top_coefficients.csv") if coef_rows else None,
            "feature_groups_csv": str(args.output_dir / "paper_feature_groups.csv"),
            "metric_figure_png": figure_path,
        },
    )
    print(f"Saved paper tables and figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
