#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny end-to-end paper pipeline smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "paper_smoke")
    parser.add_argument("--keep-output", action="store_true", help="Do not delete a previous smoke output directory.")
    return parser


def run_step(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists() and not args.keep_output:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transcripts = REPO_ROOT / "paper" / "examples" / "transcripts"
    labels = REPO_ROOT / "paper" / "examples" / "labels.csv"
    features_dir = args.output_dir / "features"
    merged_csv = args.output_dir / "features_with_labels.csv"
    selected_csv = args.output_dir / "paper_82_model_input.csv"
    model_dir = args.output_dir / "model_results"
    tables_dir = args.output_dir / "tables_figures"

    py = sys.executable
    run_step(
        [
            py,
            "paper/scripts/extract_paper_features.py",
            "--input",
            str(transcripts),
            "--output-dir",
            str(features_dir),
            "--language",
            "en",
            "--speaker-label",
            "participant",
            "--option",
            "simple",
            "--feature-groups",
            "structure,pause,repetition",
        ]
    )
    run_step(
        [
            py,
            "paper/scripts/merge_features_with_labels.py",
            "--features-csv",
            str(features_dir / "summary_features.csv"),
            "--labels-csv",
            str(labels),
            "--output",
            str(merged_csv),
        ]
    )
    run_step(
        [
            py,
            "paper/scripts/select_paper_features.py",
            "--input",
            str(merged_csv),
            "--output",
            str(selected_csv),
            "--allow-missing",
        ]
    )
    run_step(
        [
            py,
            "paper/scripts/train_evaluate_models.py",
            "--input",
            str(selected_csv),
            "--output-dir",
            str(model_dir),
            "--models",
            "normalised_basic",
            "--cv-splits",
            "2",
            "--bootstrap",
            "0",
            "--allow-missing",
        ]
    )
    run_step(
        [
            py,
            "paper/scripts/make_tables_and_figures.py",
            "--results-dir",
            str(model_dir),
            "--output-dir",
            str(tables_dir),
        ]
    )
    print(f"Smoke test completed: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
