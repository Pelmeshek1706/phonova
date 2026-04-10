import argparse
import json
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


FIXED_EXCLUDE_COLS: List[str] = [
    "Participant",
    "split",
    "Depression_label",
    "PTSD_label",
    "gender",
]

FIXED_REDUNDANT_DROP_COLS: List[str] = [
    "mean_pre_turn_pause",
    "mean_turn_length_minutes",
    "mean_turn_length_words",
    "turn_to_turn_tangeniality_mean",
    "turn_to_previous_speaker_turn_similarity_mean",
    "first_order_sentence_tangeniality_mean",
    "second_order_sentence_tangeniality_mean",
    "semantic_perplexity_mean",
    "semantic_perplexity_5_mean",
    "semantic_perplexity_11_mean",
    "semantic_perplexity_15_mean",
    "semantic_perplexity_11_var",
    "semantic_perplexity_15_var",
]

ALLOWED_MODELS = ["logreg", "sgd", "svm", "tree", "rf", "xgb"]


def parse_csv_arg(value: str, cast=float) -> List[Any]:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(target_col: str, mode: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short = target_col.replace("_label", "").lower()
    return f"{stamp}_{short}_{mode}"


def bootstrap_ci_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    n_boot: int = 2000,
    seed: int = 1706,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return float("nan"), float("nan"), float("nan")

    scores: List[float] = []
    for _ in range(n_boot):
        samp_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        samp_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([samp_pos, samp_neg])
        yt, yp, pr = y_true[idx], y_pred[idx], y_proba[idx]

        if metric == "f1_macro":
            score = f1_score(yt, yp, average="macro")
        elif metric == "auc":
            if len(np.unique(yt)) < 2:
                continue
            score = roc_auc_score(yt, pr)
        elif metric == "pr_auc":
            if len(np.unique(yt)) < 2:
                continue
            score = average_precision_score(yt, pr)
        else:
            raise ValueError("metric must be one of: f1_macro, auc, pr_auc")
        scores.append(float(score))

    if not scores:
        return float("nan"), float("nan"), float("nan")

    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def get_fixed_drop_cols(extra_drop: Optional[Sequence[str]] = None) -> List[str]:
    cols = list(FIXED_EXCLUDE_COLS) + list(FIXED_REDUNDANT_DROP_COLS)
    if extra_drop:
        cols.extend(list(extra_drop))
    # stable unique order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def prepare_xy(
    df: pd.DataFrame,
    *,
    target_col: str,
    split_col: str,
    train_values: Sequence[str],
    eval_value: str,
    extra_drop: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if split_col not in df.columns:
        raise KeyError(f"Missing split column: {split_col}")

    drop_cols = get_fixed_drop_cols(extra_drop)
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    split_lower = df[split_col].astype(str).str.lower().str.strip()
    train_values_set = {str(v).lower().strip() for v in train_values}
    is_train = split_lower.isin(train_values_set)
    is_eval = split_lower.eq(str(eval_value).lower().strip())

    if is_train.sum() == 0:
        raise ValueError(f"No rows found for training splits: {sorted(train_values_set)}")
    if is_eval.sum() == 0:
        raise ValueError(f"No rows found for eval split: {eval_value}")

    y_train = df.loc[is_train, target_col].astype(int).to_numpy()
    y_eval = df.loc[is_eval, target_col].astype(int).to_numpy()

    X_train = df.loc[is_train].drop(columns=existing_drop_cols, errors="ignore").copy()
    X_eval = df.loc[is_eval].drop(columns=existing_drop_cols, errors="ignore").copy()

    if X_train.shape[1] == 0:
        raise ValueError("No features left after applying fixed feature policy.")

    # keep only common columns in original order
    common_cols = [c for c in X_train.columns if c in X_eval.columns]
    X_train = X_train[common_cols].copy()
    X_eval = X_eval[common_cols].copy()

    removed_existing = [c for c in drop_cols if c in df.columns]
    missing_from_dataset = [c for c in drop_cols if c not in df.columns]
    return X_train, X_eval, y_train, y_eval, removed_existing, missing_from_dataset


def build_preprocessor(X: pd.DataFrame, *, imputer_strategy: str, scale_numeric: bool) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if X[c].dtype.name in ["object", "category", "bool"]]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")), ("enc", ohe)]
    num_steps = [("imp", SimpleImputer(strategy=imputer_strategy))]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline(cat_steps), categorical_cols),
            ("num", Pipeline(num_steps), numeric_cols),
        ],
        remainder="drop",
    )


def safe_cv_splits(y: np.ndarray, desired: int) -> int:
    counts = Counter(y.tolist())
    min_class = min(counts.values()) if counts else 0
    return max(2, min(desired, min_class)) if min_class >= 2 else 0


def predict_proba_or_score(pipe: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        try:
            from scipy.special import expit
            return expit(pipe.decision_function(X))
        except Exception:
            return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
        "f1_lo": float("nan"),
        "f1_hi": float("nan"),
        "auc_lo": float("nan"),
        "auc_hi": float("nan"),
        "pr_auc_lo": float("nan"),
        "pr_auc_hi": float("nan"),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        out["brier"] = float(brier_score_loss(y_true, y_proba))
        _, f1_lo, f1_hi = bootstrap_ci_binary(y_true, y_pred, y_proba, "f1_macro", n_boot=n_boot, seed=seed)
        _, auc_lo, auc_hi = bootstrap_ci_binary(y_true, y_pred, y_proba, "auc", n_boot=n_boot, seed=seed)
        _, pr_lo, pr_hi = bootstrap_ci_binary(y_true, y_pred, y_proba, "pr_auc", n_boot=n_boot, seed=seed)
        out.update({
            "f1_lo": float(f1_lo), "f1_hi": float(f1_hi),
            "auc_lo": float(auc_lo), "auc_hi": float(auc_hi),
            "pr_auc_lo": float(pr_lo), "pr_auc_hi": float(pr_hi),
        })
    return out


def cv_pr_auc(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int) -> Tuple[float, float]:
    k = safe_cv_splits(y, n_splits)
    if k < 2:
        return float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1706)
    scores = cross_val_score(clone(pipe), X, y, cv=cv, scoring="average_precision", n_jobs=1)
    return float(np.mean(scores)), float(np.std(scores))


def build_model_specs(args: argparse.Namespace, y_train: np.ndarray) -> List[Tuple[str, Any, Dict[str, List[Any]]]]:
    model_names = args.models or ALLOWED_MODELS
    model_names = [m for m in model_names if m in ALLOWED_MODELS]
    specs: List[Tuple[str, Any, Dict[str, List[Any]]]] = []

    class_weight = None if args.class_weight == "none" else args.class_weight

    if "logreg" in model_names:
        specs.append((
            "logreg",
            LogisticRegression(
                solver="saga",
                max_iter=args.max_iter,
                class_weight=class_weight,
                random_state=args.random_state,
                n_jobs=1,
            ),
            {"C": parse_csv_arg(args.logreg_c_grid, float)} if args.tune_linear_models else {},
        ))

    if "sgd" in model_names:
        specs.append((
            "sgd",
            SGDClassifier(
                loss="log_loss",
                penalty=args.sgd_penalty,
                max_iter=args.max_iter,
                tol=1e-3,
                class_weight=class_weight,
                random_state=args.random_state,
            ),
            {"alpha": parse_csv_arg(args.sgd_alpha_grid, float)} if args.tune_linear_models else {},
        ))

    if "svm" in model_names:
        specs.append((
            "svm",
            SVC(
                probability=True,
                class_weight=class_weight,
                random_state=args.random_state,
            ),
            {
                "C": parse_csv_arg(args.svm_c_grid, float),
                "gamma": [x.strip() for x in args.svm_gamma_grid.split(",") if x.strip()],
                "kernel": [x.strip() for x in args.svm_kernel_grid.split(",") if x.strip()],
            } if args.tune_tree_models else {},
        ))

    if "tree" in model_names:
        specs.append((
            "tree",
            DecisionTreeClassifier(random_state=args.random_state, class_weight=class_weight),
            {
                "max_depth": [None if x.strip().lower() == "none" else int(x.strip()) for x in args.tree_max_depth_grid.split(",") if x.strip()],
                "min_samples_leaf": parse_csv_arg(args.tree_min_samples_leaf_grid, int),
                "min_samples_split": parse_csv_arg(args.tree_min_samples_split_grid, int),
            } if args.tune_tree_models else {},
        ))

    if "rf" in model_names:
        specs.append((
            "rf",
            RandomForestClassifier(random_state=args.random_state, class_weight=class_weight, n_jobs=1),
            {
                "n_estimators": parse_csv_arg(args.rf_n_estimators_grid, int),
                "max_depth": [None if x.strip().lower() == "none" else int(x.strip()) for x in args.rf_max_depth_grid.split(",") if x.strip()],
                "min_samples_leaf": parse_csv_arg(args.rf_min_samples_leaf_grid, int),
                "max_features": [float(x.strip()) if x.strip() not in {"sqrt", "log2"} else x.strip() for x in args.rf_max_features_grid.split(",") if x.strip()],
            } if args.tune_tree_models else {},
        ))

    if "xgb" in model_names and XGB_AVAILABLE:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1)) if args.xgb_use_scale_pos_weight else 1.0
        specs.append((
            "xgb",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=args.random_state,
                n_jobs=1,
                scale_pos_weight=scale_pos_weight,
            ),
            {
                "n_estimators": parse_csv_arg(args.xgb_n_estimators_grid, int),
                "max_depth": parse_csv_arg(args.xgb_max_depth_grid, int),
                "learning_rate": parse_csv_arg(args.xgb_learning_rate_grid, float),
                "subsample": parse_csv_arg(args.xgb_subsample_grid, float),
                "colsample_bytree": parse_csv_arg(args.xgb_colsample_bytree_grid, float),
            } if args.tune_tree_models else {},
        ))

    return specs


def fit_with_optional_grid(
    *,
    estimator: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
    grid_scoring: str,
    cv_splits: int,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, Any], float, bool]:
    pipe = Pipeline([("pre", preprocessor), ("clf", clone(estimator))])
    if not param_grid:
        pipe.fit(X_train, y_train)
        return pipe, {}, float("nan"), False

    k = safe_cv_splits(y_train, cv_splits)
    if k < 2:
        pipe.fit(X_train, y_train)
        return pipe, {}, float("nan"), False

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid={f"clf__{k}": v for k, v in param_grid.items()},
        scoring=grid_scoring,
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )
    gs.fit(X_train, y_train)
    best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    return gs.best_estimator_, best_params, float(gs.best_score_), True


def append_results_tsv(results_tsv: str, res_df: pd.DataFrame) -> None:
    path = Path(results_tsv)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    res_df.to_csv(path, sep="\t", mode="a", header=header, index=False)


def print_feature_policy(X_train: pd.DataFrame, removed_existing: List[str], missing_from_dataset: List[str], expected_feature_count: Optional[int]) -> None:
    print("\n[FEATURE POLICY]")
    print("Using all available features except the fixed excluded and redundant columns.")
    print(f"Removed existing columns ({len(removed_existing)}): {removed_existing}")
    if missing_from_dataset:
        print(f"Columns from fixed drop list not present in dataset ({len(missing_from_dataset)}): {missing_from_dataset}")
    print(f"Remaining feature count: {X_train.shape[1]}")
    if expected_feature_count is not None and X_train.shape[1] != expected_feature_count:
        print(f"[WARN] Expected feature count = {expected_feature_count}, actual = {X_train.shape[1]}")


def run_training(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.dataset_path)

    if args.mode == "experiment":
        train_values = ("train",)
        eval_value = "dev"
    elif args.mode == "final":
        train_values = ("train", "dev")
        eval_value = "test"
    else:
        train_values = tuple(args.train_values)
        eval_value = args.eval_value

    extra_drop = [x.strip() for x in args.extra_drop_cols.split(",") if x.strip()] if args.extra_drop_cols else []
    X_train, X_eval, y_train, y_eval, removed_existing, missing_from_dataset = prepare_xy(
        df,
        target_col=args.target_col,
        split_col=args.split_col,
        train_values=train_values,
        eval_value=eval_value,
        extra_drop=extra_drop,
    )

    print_feature_policy(X_train, removed_existing, missing_from_dataset, args.expected_feature_count)
    print(f"[INFO] dataset_path = {args.dataset_path}")
    print(f"[INFO] target_col = {args.target_col}")
    print(f"[INFO] mode = {args.mode}")
    print(f"[INFO] train_values = {train_values} | eval_value = {eval_value}")
    print(f"[INFO] X_train shape = {X_train.shape} | X_eval shape = {X_eval.shape}")
    print(f"[INFO] y_train counts = {np.bincount(y_train)} | y_eval counts = {np.bincount(y_eval)}")

    pre = build_preprocessor(X_train, imputer_strategy=args.imputer_strategy, scale_numeric=args.scale_numeric)
    model_specs = build_model_specs(args, y_train)

    run_id = make_run_id(args.target_col, args.mode)
    timestamp_utc = utc_now_str()
    all_rows: List[Dict[str, Any]] = []

    for model_name, estimator, grid in model_specs:
        pipe, best_params, best_cv, tuned = fit_with_optional_grid(
            estimator=estimator,
            preprocessor=pre,
            X_train=X_train,
            y_train=y_train,
            param_grid=grid,
            grid_scoring=args.grid_scoring,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
        )
        cv_mean, cv_std = cv_pr_auc(pipe, X_train, y_train, args.cv_splits)
        y_pred = pipe.predict(X_eval)
        y_proba = predict_proba_or_score(pipe, X_eval)
        metrics = compute_metrics(y_eval, y_pred, y_proba, n_boot=args.n_boot, seed=args.random_state)

        row: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": timestamp_utc,
            "experiment_tag": args.experiment_tag,
            "notes": args.notes,
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "target": args.target_col,
            "mode": args.mode,
            "train_values": ",".join(train_values),
            "eval_split": eval_value,
            "feature_policy": "all_available_minus_fixed_excluded_minus_fixed_redundant",
            "removed_columns": json.dumps(removed_existing, ensure_ascii=False),
            "feature_count": int(X_train.shape[1]),
            "expected_feature_count": args.expected_feature_count,
            "imputer_strategy": args.imputer_strategy,
            "scale_numeric": bool(args.scale_numeric),
            "class_weight": args.class_weight,
            "models_cli": ",".join(args.models or ALLOWED_MODELS),
            "cv_splits": args.cv_splits,
            "grid_scoring": args.grid_scoring,
            "model": model_name,
            "variant": "full_tuned" if tuned else "full",
            "tuned": bool(tuned),
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "cv_best_score": best_cv,
            "cv_pr_auc_mean": cv_mean,
            "cv_pr_auc_std": cv_std,
            "n_train": int(len(y_train)),
            "n_eval": int(len(y_eval)),
        }
        row.update(metrics)
        all_rows.append(row)

        print(f"\n=== {model_name} [{'full_tuned' if tuned else 'full'}] ===")
        if tuned:
            print(f"Best CV score ({args.grid_scoring}): {best_cv:.4f} | params: {best_params}")
        print(f"Train CV PR AUC: {cv_mean:.4f} +/- {cv_std:.4f}")
        print(classification_report(y_eval, y_pred, digits=3))
        print("Confusion matrix:")
        print(confusion_matrix(y_eval, y_pred))

    res_df = pd.DataFrame(all_rows).sort_values(["pr_auc", "f1_macro"], ascending=[False, False]).reset_index(drop=True)

    print("\n=== Summary metrics ===")
    print(
        res_df[[
            "target", "mode", "model", "variant", "feature_count", "cv_pr_auc_mean",
            "pr_auc", "f1_macro", "roc_auc", "balanced_acc", "best_params"
        ]].round(4).to_string(index=False)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_name or f"{run_id}_{Path(args.dataset_path).stem}_{args.target_col}_{eval_value}"
    out_csv = save_dir / f"{stem}.csv"
    out_json = save_dir / f"{stem}_best.json"
    res_df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(res_df.iloc[0].to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved run table to: {out_csv}")
    print(f"[INFO] Saved best row to: {out_json}")

    append_results_tsv(args.results_tsv, res_df)
    print(f"[INFO] Appended results to: {args.results_tsv}")

    if args.plot_after_run:
        out_plot = save_dir / f"metrics_progression_{args.target_col}_{eval_value}.svg"
        plot_results(
            results_tsv=args.results_tsv,
            out_path=str(out_plot),
            target=args.target_col,
            eval_split=eval_value,
        )
        print(f"[INFO] Saved plot to: {out_plot}")

    return res_df


def plot_results(
    *,
    results_tsv: str,
    out_path: str,
    target: Optional[str] = None,
    eval_split: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    df = pd.read_csv(results_tsv, sep="\t")
    if target is not None:
        df = df[df["target"] == target].copy()
    if eval_split is not None:
        df = df[df["eval_split"] == eval_split].copy()
    if mode is not None:
        df = df[df["mode"] == mode].copy()

    if df.empty:
        raise ValueError("No rows left after applying plot filters.")

    df = df.reset_index(drop=True)
    df["plot_idx"] = np.arange(1, len(df) + 1)
    best_idx = int(df["pr_auc"].astype(float).idxmax())

    plt.figure(figsize=(10, 6))
    plt.plot(df["plot_idx"], df["pr_auc"], marker="o", label="PR AUC")
    plt.plot(df["plot_idx"], df["f1_macro"], marker="o", label="F1 macro")
    plt.scatter(df.loc[best_idx, "plot_idx"], df.loc[best_idx, "pr_auc"], s=80, label="Best PR AUC")
    plt.xlabel("Logged row order")
    plt.ylabel("Metric value")
    title_parts = ["Metrics progression"]
    if target:
        title_parts.append(target)
    if eval_split:
        title_parts.append(eval_split)
    plt.title(" | ".join(title_parts))
    plt.legend()
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format=out.suffix.lstrip(".") or "svg")
    plt.close()



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-dataset legacy trainer with fixed feature policy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Run training on one merged dataset.")
    p_train.add_argument("--dataset-path", required=True, help="Path to the single merged_labels.csv dataset.")
    p_train.add_argument("--target-col", required=True, choices=["Depression_label", "PTSD_label"])
    p_train.add_argument("--mode", default="experiment", choices=["experiment", "final", "custom"])
    p_train.add_argument("--train-values", nargs="*", default=None, help="Used only with mode=custom")
    p_train.add_argument("--eval-value", default=None, help="Used only with mode=custom")
    p_train.add_argument("--split-col", default="split")
    p_train.add_argument("--extra-drop-cols", default="", help="Optional extra columns to drop, comma-separated.")
    p_train.add_argument("--expected-feature-count", type=int, default=83)

    p_train.add_argument("--models", nargs="*", default=None, choices=ALLOWED_MODELS)
    p_train.add_argument("--imputer-strategy", default="median", choices=["median", "mean", "most_frequent"])
    p_train.add_argument("--scale-numeric", dest="scale_numeric", action="store_true")
    p_train.add_argument("--no-scale-numeric", dest="scale_numeric", action="store_false")
    p_train.set_defaults(scale_numeric=True)
    p_train.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    p_train.add_argument("--cv-splits", type=int, default=5)
    p_train.add_argument("--grid-scoring", default="average_precision")
    p_train.add_argument("--random-state", type=int, default=1706)
    p_train.add_argument("--n-boot", type=int, default=2000)
    p_train.add_argument("--max-iter", type=int, default=5000)

    p_train.add_argument("--tune-linear-models", dest="tune_linear_models", action="store_true")
    p_train.add_argument("--no-tune-linear-models", dest="tune_linear_models", action="store_false")
    p_train.set_defaults(tune_linear_models=True)
    p_train.add_argument("--tune-tree-models", dest="tune_tree_models", action="store_true")
    p_train.add_argument("--no-tune-tree-models", dest="tune_tree_models", action="store_false")
    p_train.set_defaults(tune_tree_models=True)

    p_train.add_argument("--logreg-c-grid", default="0.1,1,10")
    p_train.add_argument("--sgd-alpha-grid", default="0.00001,0.0001,0.001")
    p_train.add_argument("--sgd-penalty", default="l2", choices=["l2", "l1", "elasticnet"])
    p_train.add_argument("--svm-c-grid", default="0.1,1,10")
    p_train.add_argument("--svm-gamma-grid", default="scale,auto")
    p_train.add_argument("--svm-kernel-grid", default="rbf,linear")
    p_train.add_argument("--tree-max-depth-grid", default="3,5,10,None")
    p_train.add_argument("--tree-min-samples-leaf-grid", default="1,2,4,8")
    p_train.add_argument("--tree-min-samples-split-grid", default="2,5,10")
    p_train.add_argument("--rf-n-estimators-grid", default="200,500")
    p_train.add_argument("--rf-max-depth-grid", default="5,10,None")
    p_train.add_argument("--rf-min-samples-leaf-grid", default="1,2,4")
    p_train.add_argument("--rf-max-features-grid", default="sqrt,log2,0.5")
    p_train.add_argument("--xgb-use-scale-pos-weight", dest="xgb_use_scale_pos_weight", action="store_true")
    p_train.add_argument("--no-xgb-use-scale-pos-weight", dest="xgb_use_scale_pos_weight", action="store_false")
    p_train.set_defaults(xgb_use_scale_pos_weight=True)
    p_train.add_argument("--xgb-n-estimators-grid", default="200,500")
    p_train.add_argument("--xgb-max-depth-grid", default="3,6,10")
    p_train.add_argument("--xgb-learning-rate-grid", default="0.03,0.1")
    p_train.add_argument("--xgb-subsample-grid", default="0.8,1.0")
    p_train.add_argument("--xgb-colsample-bytree-grid", default="0.8,1.0")

    p_train.add_argument("--experiment-tag", default="")
    p_train.add_argument("--notes", default="")
    p_train.add_argument("--results-tsv", default="artifacts/results.tsv")
    p_train.add_argument("--save-dir", default="artifacts")
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--plot-after-run", action="store_true")

    p_plot = subparsers.add_parser("plot-results", help="Plot metric history from results.tsv")
    p_plot.add_argument("--results-tsv", required=True)
    p_plot.add_argument("--out-path", required=True)
    p_plot.add_argument("--target", default=None)
    p_plot.add_argument("--eval-split", default=None)
    p_plot.add_argument("--mode", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if args.mode == "custom":
            if not args.train_values or args.eval_value is None:
                raise ValueError("For mode=custom you must provide --train-values and --eval-value")
        run_training(args)
        return

    if args.command == "plot-results":
        plot_results(
            results_tsv=args.results_tsv,
            out_path=args.out_path,
            target=args.target,
            eval_split=args.eval_split,
            mode=args.mode,
        )
        print(f"[INFO] Saved plot to: {args.out_path}")
        return


if __name__ == "__main__":
    main()
