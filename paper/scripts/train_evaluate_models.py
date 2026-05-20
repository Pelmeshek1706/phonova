#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from _common import dataframe_sha256, get_feature_set, load_config, set_global_seed, write_json


class RobustNumericPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        clip_low: float = 0.005,
        clip_high: float = 0.995,
        add_missing_indicators: bool = True,
        log_perplexity: bool = True,
    ):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.add_missing_indicators = add_missing_indicators
        self.log_perplexity = log_perplexity

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.columns_ = list(frame.columns)
        self.perplexity_cols_ = [column for column in self.columns_ if "semantic_perplexity" in column]
        self.quantiles_ = {}
        self.medians_ = {}
        self.missing_cols_ = []

        transformed = self._apply_log(frame)
        for column in self.columns_:
            values = pd.to_numeric(transformed[column], errors="coerce")
            lo = values.quantile(self.clip_low)
            hi = values.quantile(self.clip_high)
            self.quantiles_[column] = (
                float(lo) if np.isfinite(lo) else np.nan,
                float(hi) if np.isfinite(hi) else np.nan,
            )
            median = values.median(skipna=True)
            self.medians_[column] = float(median) if np.isfinite(median) else 0.0
            if self.add_missing_indicators and values.isna().any():
                self.missing_cols_.append(column)
        return self

    def _apply_log(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if self.log_perplexity:
            for column in self.perplexity_cols_:
                values = pd.to_numeric(out[column], errors="coerce").clip(lower=0)
                out[column] = np.log1p(values)
        return out

    def transform(self, X):
        frame = pd.DataFrame(X).copy().reindex(columns=self.columns_)
        frame = self._apply_log(frame)
        arrays = []
        names = []

        for column in self.columns_:
            values = pd.to_numeric(frame[column], errors="coerce")
            lo, hi = self.quantiles_[column]
            if np.isfinite(lo):
                values = values.clip(lower=lo)
            if np.isfinite(hi):
                values = values.clip(upper=hi)

            if self.add_missing_indicators and column in self.missing_cols_:
                arrays.append(values.isna().astype(float).to_numpy().reshape(-1, 1))
                names.append(f"{column}__isna")

            arrays.append(values.fillna(self.medians_[column]).to_numpy().reshape(-1, 1))
            names.append(column)

        self.feature_names_out_ = names
        return np.concatenate(arrays, axis=1).astype(np.float32)

    def get_feature_names_out(self):
        return np.asarray(getattr(self, "feature_names_out_", self.columns_), dtype=object)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate deterministic paper models from a selected paper feature table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV with metadata, labels, split, and paper features.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics, predictions, and coefficients.")
    parser.add_argument("--config", type=Path, default=None, help="Paper configuration JSON.")
    parser.add_argument("--feature-set", default=None, help="Feature set name from config.")
    parser.add_argument("--targets", default=None, help="Comma-separated target columns. Defaults to config targets.")
    parser.add_argument("--models", default=None, help="Comma-separated model names. Defaults to all config models.")
    parser.add_argument("--cv-splits", type=int, default=None, help="Override cross-validation split count.")
    parser.add_argument("--bootstrap", type=int, default=None, help="Override bootstrap repetitions. Use 0 to disable.")
    parser.add_argument("--allow-missing", action="store_true", help="Fill missing feature columns with NaN.")
    return parser


def _split_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _coerce_class_weight(value: Any) -> Any:
    if isinstance(value, dict):
        return {int(key): float(weight) for key, weight in value.items()}
    return value


def _model_params(model_cfg: dict[str, Any], target: str) -> dict[str, Any]:
    if model_cfg["type"] == "logistic_regression":
        params = dict(model_cfg["params"])
    elif model_cfg["type"] == "target_logistic_regression":
        params = dict(model_cfg["target_params"][target])
    else:
        raise ValueError(f"Unsupported model type: {model_cfg['type']}")
    if "class_weight" in params:
        params["class_weight"] = _coerce_class_weight(params["class_weight"])
    return params


def build_pipeline(config: dict[str, Any], model_cfg: dict[str, Any], target: str, seed: int) -> Pipeline:
    training = config["training"]
    params = _model_params(model_cfg, target)
    params["random_state"] = seed

    steps = [
        (
            "prep",
            RobustNumericPreprocessor(
                clip_low=float(training.get("clip_low", 0.005)),
                clip_high=float(training.get("clip_high", 0.995)),
                add_missing_indicators=bool(training.get("add_missing_indicators", True)),
                log_perplexity=bool(training.get("log_perplexity", True)),
            ),
        )
    ]
    scaler = str(training.get("scaler", "standard")).lower()
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "maxabs":
        steps.append(("scaler", MaxAbsScaler()))
    elif scaler != "none":
        raise ValueError(f"Unsupported scaler: {scaler}")

    steps.append(("model", LogisticRegression(**params)))
    return Pipeline(steps)


def fixed_threshold(model_cfg: dict[str, Any], target: str) -> float | None:
    thresholds = model_cfg.get("fixed_thresholds", {})
    if target not in thresholds:
        return None
    return float(thresholds[target])


def tuned_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float, float]:
    thresholds = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, 101)))
    best = (0.5, -1.0, -1.0)
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        f1_macro = f1_score(y_true, pred, average="macro", zero_division=0)
        f1_binary = f1_score(y_true, pred, average="binary", zero_division=0)
        if (f1_macro > best[1]) or (np.isclose(f1_macro, best[1]) and f1_binary > best[2]):
            best = (float(threshold), float(f1_macro), float(f1_binary))
    return best


def metrics_at_threshold(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    out = {
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, pred, average="binary", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "threshold": float(threshold),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def oof_proba(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int, seed: int) -> np.ndarray:
    min_class_count = int(y.value_counts().min())
    n_splits = min(int(cv_splits), min_class_count)
    if n_splits < 2:
        raise ValueError("Need at least 2 samples in each class for cross-validation.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        model = clone(pipe)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        out[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return out


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    prep = pipe.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    coefs = pipe.named_steps["model"].coef_.ravel()
    out = pd.DataFrame({"feature": names, "coef": coefs, "abs_coef": np.abs(coefs)})
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def bootstrap_ci(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if n_boot <= 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), size=len(y_arr))
        if len(np.unique(y_arr[idx])) < 2 and metric in {"pr_auc", "roc_auc"}:
            continue
        values.append(metrics_at_threshold(pd.Series(y_arr[idx]), proba[idx], threshold)[metric])
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(np.mean(arr)))


def run_one(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split = df["split"].astype(str).str.lower().str.strip()
    train_splits = set(config["training"].get("train_splits", ["train", "dev"]))
    test_splits = set(config["training"].get("test_splits", ["test"]))
    train_mask = split.isin(train_splits)
    test_mask = split.isin(test_splits)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Input must contain train/dev and test rows in the split column.")

    seed = int(config.get("seed", 42))
    cv_splits = int(args.cv_splits or config.get("n_splits", 5))
    X_train = df.loc[train_mask, features].reset_index(drop=True)
    y_train = df.loc[train_mask, target].astype(int).reset_index(drop=True)
    X_test = df.loc[test_mask, features].reset_index(drop=True)
    y_test = df.loc[test_mask, target].astype(int).reset_index(drop=True)

    pipe = build_pipeline(config, model_cfg, target, seed)
    oof = oof_proba(pipe, X_train, y_train, cv_splits=cv_splits, seed=seed)
    tuned_thr, tuned_oof_f1_macro, tuned_oof_f1_binary = tuned_threshold(y_train, oof)
    threshold = fixed_threshold(model_cfg, target)
    if threshold is None:
        threshold = tuned_thr
        oof_f1_macro = tuned_oof_f1_macro
        oof_f1_binary = tuned_oof_f1_binary
    else:
        oof_metrics = metrics_at_threshold(y_train, oof, threshold)
        oof_f1_macro = oof_metrics["f1_macro"]
        oof_f1_binary = oof_metrics["f1_binary"]
    oof_pr_auc = float(average_precision_score(y_train, oof))

    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_proba, threshold)

    prefix = f"{model_name}_{target}"
    pred_path = args.output_dir / f"predictions_{prefix}.csv"
    coef_path = args.output_dir / f"coefficients_{prefix}.csv"
    pd.DataFrame(
        {
            "Participant": df.loc[test_mask, "Participant"].astype(str).to_numpy(),
            "target": target,
            "model_name": model_name,
            "y_true": y_test.to_numpy(),
            "proba": test_proba,
            "y_pred": (test_proba >= threshold).astype(int),
        }
    ).to_csv(pred_path, index=False)
    coefficient_table(pipe).to_csv(coef_path, index=False)

    n_boot = int(args.bootstrap if args.bootstrap is not None else config.get("bootstrap", {}).get("n_boot", 0))
    row: dict[str, Any] = {
        "target": target,
        "model_name": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(features)),
        "threshold": float(threshold),
        "tuned_threshold": float(tuned_thr),
        "oof_pr_auc": oof_pr_auc,
        "oof_f1_macro": float(oof_f1_macro),
        "oof_f1_binary": float(oof_f1_binary),
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_binary": test_metrics["f1_binary"],
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "predictions_csv": str(pred_path),
        "coefficients_csv": str(coef_path),
    }
    for metric in ("f1_macro", "f1_binary", "accuracy", "pr_auc", "roc_auc"):
        lo, hi, mean = bootstrap_ci(y_test, test_proba, threshold, metric, n_boot, seed + len(row))
        row[f"test_{metric}_ci_low"] = lo
        row[f"test_{metric}_ci_high"] = hi
        row[f"test_{metric}_boot_mean"] = mean
    return row


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_global_seed(int(config.get("seed", 42)))
    features = get_feature_set(config, args.feature_set)
    df = pd.read_csv(args.input)

    missing = [feature for feature in features if feature not in df.columns]
    if missing and not args.allow_missing:
        raise KeyError(f"Input is missing {len(missing)} features: {missing[:10]}")
    for feature in missing:
        df[feature] = np.nan

    if "split" not in df.columns:
        raise KeyError("Input must contain a split column with train/dev/test assignments.")
    if "Participant" not in df.columns:
        raise KeyError("Input must contain a Participant column.")

    targets = _split_list(args.targets) or list(config.get("targets", []))
    model_names = _split_list(args.models) or list(config["training"]["models"].keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in targets:
        if target not in df.columns:
            raise KeyError(f"Missing target column '{target}'")
        for model_name in model_names:
            model_cfg = config["training"]["models"][model_name]
            rows.append(run_one(df, features, target, model_name, model_cfg, config, args))

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "summary_results.csv"
    summary.to_csv(summary_path, index=False)
    write_json(
        args.output_dir / "summary_results.meta.json",
        {
            "input": str(args.input),
            "feature_set": args.feature_set or config.get("default_feature_set"),
            "n_rows": int(len(df)),
            "n_features": int(len(features)),
            "targets": targets,
            "models": model_names,
            "seed": int(config.get("seed", 42)),
            "cv_splits": int(args.cv_splits or config.get("n_splits", 5)),
            "bootstrap": int(args.bootstrap if args.bootstrap is not None else config.get("bootstrap", {}).get("n_boot", 0)),
            "summary_csv": str(summary_path),
            "summary_sha256": dataframe_sha256(summary),
        },
    )
    print(f"Saved model summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
