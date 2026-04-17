#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/poluidol/Documents/work/airest/airest_perp_research/data_analysis")
OUT_ROOT = ROOT / "tmp" / "model_search_3_4_2026_expanded_stable05_full_vader_model_family_sweep"
CONSISTENCY_PATH = (
    ROOT
    / "tmp"
    / "model_search_3_4_2026_expanded_ua_sentiment_hist_raw_vs_translated"
    / "consistency_en_ua_all_plus_sentiment_hist.csv"
)
SCRIPT_VERSION = "3_4_2026_expanded_stable05_full_vader_model_family_sweep_v1"
FEATURE_GROUP_NAME = "stable_rho_gt_05__full_vader_hist"


sys.path.insert(0, str(ROOT / "tmp"))
import run_3_4_2026_expanded_all_minus_noisy_vader_model_family_sweep as family  # noqa: E402


def stage_dir(name: str) -> Path:
    path = OUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_fixed_feature_columns() -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    consistency_df = pd.read_csv(CONSISTENCY_PATH)
    feat_col = "feature" if "feature" in consistency_df.columns else consistency_df.columns[0]
    rho_col = "spearman_rho" if "spearman_rho" in consistency_df.columns else "rho"
    label_columns = {"Depression_severity", "PTSD_severity"}

    model_df = consistency_df.loc[~consistency_df[feat_col].isin(label_columns), [feat_col, rho_col]].copy()
    rho_map = dict(zip(model_df[feat_col], model_df[rho_col]))
    stable = {feat for feat, rho in rho_map.items() if pd.notna(rho) and float(rho) > 0.5}
    vader_hist = set(family.signal.SENTIMENT_COLUMNS)
    fixed = sorted(stable | vader_hist)

    feature_rows = []
    for feat in fixed:
        in_stable = feat in stable
        in_vader = feat in vader_hist
        if in_stable and in_vader:
            source = "stable_rho_gt_05+vader_hist"
        elif in_stable:
            source = "stable_rho_gt_05"
        else:
            source = "vader_hist_forced_in"
        feature_rows.append(
            {
                "feature": feat,
                "source": source,
                "spearman_rho": rho_map.get(feat),
                "in_stable_rho_gt_05": bool(in_stable),
                "in_full_vader_hist": bool(in_vader),
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    meta = {
        "stable_rho_gt_05_count_before_union": len(stable),
        "full_vader_hist_count": len(vader_hist),
        "stable_vader_overlap_count": len(stable & vader_hist),
        "final_feature_count": len(fixed),
    }
    return fixed, feature_df, meta


def make_config_json(target: str, feature_columns: list[str], spec: family.SearchSpec) -> dict[str, Any]:
    payload = family.make_config_json(target, feature_columns, spec)
    payload["script_version"] = SCRIPT_VERSION
    payload["feature_group"] = FEATURE_GROUP_NAME
    return payload


def evaluate_one_target(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    n_splits: int,
    n_repeats: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, y_train, X_test, y_test = family.base.train_test_split_for_target(df, target, feature_columns)
    search_path = stage_dir("search") / f"search_{target}.csv"
    existing = pd.DataFrame()
    if search_path.exists() and not force:
        existing = pd.read_csv(search_path)
    done_keys = set(existing["config_key"].astype(str)) if not existing.empty else set()
    search_specs = family.build_search_specs(target, y_train)
    rows: list[dict[str, Any]] = []

    for idx, spec in enumerate(search_specs, start=1):
        payload = make_config_json(target, feature_columns, spec)
        config_key = family.make_config_key(payload)
        if config_key in done_keys:
            continue
        summary = family.repeated_cv_summary(payload, X_train, y_train, n_splits=n_splits, n_repeats=n_repeats)
        row = {
            "target": target,
            "feature_group": FEATURE_GROUP_NAME,
            "family": spec.family,
            "model": spec.name,
            "preproc": payload["preproc"]["name"],
            "n_features": len(feature_columns),
            "mean_oof_pr": family.safe_float(summary["mean_oof_pr"]),
            "std_oof_pr": family.safe_float(summary["std_oof_pr"]),
            "mean_oof_f1": family.safe_float(summary["mean_oof_f1"]),
            "std_oof_f1": family.safe_float(summary["std_oof_f1"]),
            "mean_oof_f1_binary": family.safe_float(summary["mean_oof_f1_binary"]),
            "mean_oof_roc": family.safe_float(summary["mean_oof_roc"]),
            "threshold": family.safe_float(summary["threshold_final"]),
            "threshold_rule": summary["threshold_rule"],
            "official_test_pr": np.nan,
            "official_test_f1": np.nan,
            "official_test_f1_binary": np.nan,
            "official_test_accuracy": np.nan,
            "official_test_roc": np.nan,
            "config_key": config_key,
            "config_json": json.dumps(payload, sort_keys=True),
        }
        rows.append(row)
        print(
            f"[family_sweep] {target} {idx}/{len(search_specs)} {spec.name} "
            f"mean_combo={0.5 * (row['mean_oof_pr'] + row['mean_oof_f1']):.4f} "
            f"mean_pr={row['mean_oof_pr']:.4f} mean_f1={row['mean_oof_f1']:.4f}",
            flush=True,
        )
        if len(rows) >= 8:
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            existing = family.sort_cv_df(existing).drop_duplicates(subset=["config_key"], keep="first")
            existing.to_csv(search_path, index=False)
            rows = []

    if rows:
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        existing = family.sort_cv_df(existing).drop_duplicates(subset=["config_key"], keep="first")
        existing.to_csv(search_path, index=False)

    search_df = family.sort_cv_df(existing)
    shortlist_df = family.shortlist_rows(
        search_df.loc[search_df["target"].eq(target)].copy(),
        top_k=20,
        top_per_family=3,
        top_by_metric=2,
    )

    official_rows = []
    for _, row in shortlist_df.iterrows():
        payload = json.loads(row["config_json"])
        metrics, p_test = family.fit_and_eval_official(payload, X_train, y_train, X_test, y_test, float(row["threshold"]))
        out_row = row.to_dict()
        out_row["official_test_pr"] = family.safe_float(metrics["pr_auc"])
        out_row["official_test_f1"] = family.safe_float(metrics["f1_macro"])
        out_row["official_test_f1_binary"] = family.safe_float(metrics["f1_binary"])
        out_row["official_test_accuracy"] = family.safe_float(metrics["accuracy"])
        out_row["official_test_roc"] = family.safe_float(metrics["roc_auc"])
        pred_path = stage_dir("predictions") / f"pred_{target}_{row['config_key']}.csv"
        pd.DataFrame(
            {
                "Participant": df.loc[df["split"].astype(str).str.lower().str.strip().eq("test"), "Participant"].values,
                "y_true": y_test.values,
                "proba": p_test,
                "y_pred": (p_test >= float(row["threshold"])).astype(int),
            }
        ).to_csv(pred_path, index=False)
        out_row["prediction_csv"] = str(pred_path)
        official_rows.append(out_row)

    official_df = family.sort_test_df(pd.DataFrame(official_rows))
    official_df.to_csv(stage_dir("official") / f"official_test_shortlist_{target}.csv", index=False)
    return search_df, official_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-family sweep on stable_rho_gt_05 + full vader_hist.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    fixed_features, feature_df, meta = build_fixed_feature_columns()
    feature_df.to_csv(OUT_ROOT / "fixed_feature_set.csv", index=False)
    (OUT_ROOT / "run_info.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "fixed_feature_group": FEATURE_GROUP_NAME,
                "consistency_path": str(CONSISTENCY_PATH),
                "augmented_en_path": str(family.AUGMENTED_EN_PATH),
                "fixed_feature_meta": meta,
                "splits": args.splits,
                "repeats": args.repeats,
                "xgboost_available": family.HAS_XGB,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    df = pd.read_csv(family.AUGMENTED_EN_PATH)
    missing = [col for col in fixed_features if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing fixed feature columns in augmented EN data: {missing[:10]}")

    all_search = []
    all_official = []
    for target in family.TARGETS:
        search_df, official_df = evaluate_one_target(
            df=df,
            target=target,
            feature_columns=fixed_features,
            n_splits=args.splits,
            n_repeats=args.repeats,
            force=args.force,
        )
        all_search.append(search_df.loc[search_df["target"].eq(target)].copy())
        all_official.append(official_df)

    search_all = family.sort_cv_df(pd.concat(all_search, ignore_index=True))
    search_all.to_csv(OUT_ROOT / "search_all.csv", index=False)

    official_all = family.sort_test_df(pd.concat(all_official, ignore_index=True))
    official_all.to_csv(OUT_ROOT / "official_test_shortlist.csv", index=False)

    best_family_rows = []
    for _, subset in official_all.groupby(["target", "family"], sort=False):
        best_family_rows.append(family.sort_test_df(subset).iloc[0].to_dict())
    best_by_family = family.sort_test_df(pd.DataFrame(best_family_rows))
    best_by_family.to_csv(OUT_ROOT / "best_by_family_target.csv", index=False)

    final_df = family.choose_final_rows(official_all)
    final_df.to_csv(OUT_ROOT / "best_by_target.csv", index=False)

    print(
        final_df[
            [
                "target",
                "feature_group",
                "family",
                "model",
                "n_features",
                "official_test_pr",
                "official_test_f1",
                "official_test_f1_binary",
                "official_test_accuracy",
                "official_test_roc",
                "selection_status",
            ]
        ].to_csv(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
