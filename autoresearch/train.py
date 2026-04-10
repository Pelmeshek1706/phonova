import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "merged_features.csv"
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"
DEFAULT_RESULTS_PATH = BASE_DIR / "results.json"
DEFAULT_RESULTS_TSV_PATH = BASE_DIR / "results.tsv"
DEFAULT_PLOT_PATH = DEFAULT_ARTIFACTS_DIR / "metrics_progression_cleaned_fullfeature.svg"

RANDOM_STATE = 42
TARGETS = ("Depression_label", "PTSD_label")
EXPECTED_SPLITS = ("train", "dev", "test")
NON_FEATURE_COLUMNS = ("Participant", "split", *TARGETS, "gender")
REDUNDANT_COLUMNS = (
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
)
FULL_FEATURE_SUBSET_NAME = "cleaned_full_minus_gender"
BLEND_PREPROCESS_LABEL = "blend_of_cleaned_full_feature_pipelines"
RESULTS_TSV_HEADER = (
    "run_id\ttarget_label\tdev_pr_auc\tdev_f1\tmodel\tfeature_subset\tpreprocess\tthreshold\t"
    "cleaned_full_feature_set\tgender_excluded\tremoved_columns\tstatus\tdescription"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    scaler_name: str
    model_factory: Callable[[], object]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate cleaned full-feature tabular classifiers for AiRest voice features.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to merged feature CSV.",
    )
    parser.add_argument(
        "--target",
        choices=("all", *TARGETS),
        default="all",
        help="Evaluate one target or both targets.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also evaluate the hidden test split. Do not use during model search.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for leaderboard, predictions, and model artifacts.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to summary JSON output.",
    )
    parser.add_argument(
        "--results-tsv-path",
        type=Path,
        default=DEFAULT_RESULTS_TSV_PATH,
        help="Append-only experiment log.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path to the metrics progression SVG built from results.tsv.",
    )
    parser.add_argument(
        "--skip-importance",
        action="store_true",
        help="Skip native importance export for faster iteration.",
    )
    return parser.parse_args()


def ensure_valid_dataframe(df: pd.DataFrame):
    missing_cols = [col for col in ("Participant", "split", *TARGETS) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    split_values = set(df["split"].dropna().astype(str))
    missing_splits = [split for split in EXPECTED_SPLITS if split not in split_values]
    if missing_splits:
        raise ValueError(f"Missing required splits: {missing_splits}")


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ensure_valid_dataframe(df)
    return df


def get_removed_redundant_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REDUNDANT_COLUMNS if column in df.columns]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    removed_columns = set(get_removed_redundant_columns(df))
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and column not in removed_columns
    ]


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, get_feature_columns(df)].copy()


def numeric_scaler_factory(scale_name: str, n_train_samples: int):
    n_quantiles = min(100, n_train_samples)
    mapping = {
        "none": None,
        "standard": StandardScaler,
        "robust": RobustScaler,
        "power": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
        "quantile_normal": lambda: QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            random_state=RANDOM_STATE,
        ),
        "quantile_uniform": lambda: QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=n_quantiles,
            random_state=RANDOM_STATE,
        ),
        "minmax": MinMaxScaler,
    }
    factory = mapping.get(scale_name)
    if factory is None:
        return None
    return factory()


def build_preprocessor(scale_name: str, n_train_samples: int) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    scaler = numeric_scaler_factory(scale_name, n_train_samples)
    if scaler is not None:
        steps.append(("scaler", scaler))
    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), slice(0, None))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_factories():
    return {
        "raw_hgb3": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            l2_regularization=1.0,
            max_depth=3,
            max_iter=600,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
        "raw_hgb3b": lambda: HistGradientBoostingClassifier(
            learning_rate=0.04,
            l2_regularization=0.7,
            max_depth=3,
            max_iter=900,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
        "raw_hgbdeep": lambda: HistGradientBoostingClassifier(
            learning_rate=0.03,
            l2_regularization=2.0,
            max_leaf_nodes=31,
            max_iter=900,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
        "raw_hgbmf70": lambda: HistGradientBoostingClassifier(
            learning_rate=0.03,
            l2_regularization=1.0,
            max_depth=3,
            max_features=0.7,
            max_iter=900,
            min_samples_leaf=6,
            random_state=RANDOM_STATE,
        ),
        "hgb_bal_d2": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            l2_regularization=0.5,
            max_depth=2,
            max_iter=800,
            min_samples_leaf=5,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
        "hgb_bal_d3": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            l2_regularization=1.0,
            max_depth=3,
            max_iter=800,
            min_samples_leaf=10,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
        "rf_bal": lambda: RandomForestClassifier(
            class_weight="balanced_subsample",
            max_features=0.5,
            min_samples_leaf=3,
            n_estimators=1000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "et_bal": lambda: ExtraTreesClassifier(
            class_weight="balanced",
            max_features=0.5,
            min_samples_leaf=1,
            n_estimators=1000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "logreg_l2_tight": lambda: LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=12000,
            random_state=RANDOM_STATE,
        ),
        "logreg_l2": lambda: LogisticRegression(
            C=0.4,
            class_weight="balanced",
            max_iter=12000,
            random_state=RANDOM_STATE,
        ),
        "logreg_en_sparse": lambda: LogisticRegression(
            C=0.4,
            class_weight="balanced",
            l1_ratio=0.7,
            max_iter=12000,
            penalty="elasticnet",
            random_state=RANDOM_STATE,
            solver="saga",
        ),
        "logreg_en_dense": lambda: LogisticRegression(
            C=0.8,
            class_weight="balanced",
            l1_ratio=0.3,
            max_iter=12000,
            penalty="elasticnet",
            random_state=RANDOM_STATE,
            solver="saga",
        ),
        "lsvc_cal": lambda: CalibratedClassifierCV(
            estimator=LinearSVC(
                C=0.5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv=3,
            method="sigmoid",
        ),
        "bnb": lambda: BernoulliNB(alpha=0.5),
        "gnb": lambda: GaussianNB(var_smoothing=1e-8),
    }


def get_candidate_specs(target: str) -> list[CandidateSpec]:
    factories = build_model_factories()
    base = [
        CandidateSpec("raw_hgb3", "hgb", "none", factories["raw_hgb3"]),
        CandidateSpec("raw_hgbdeep", "hgb", "none", factories["raw_hgbdeep"]),
        CandidateSpec("raw_hgbmf70", "hgb", "none", factories["raw_hgbmf70"]),
        CandidateSpec("hgb_bal_d2", "hgb", "none", factories["hgb_bal_d2"]),
        CandidateSpec("et_bal", "tree", "none", factories["et_bal"]),
        CandidateSpec("rf_bal", "tree", "none", factories["rf_bal"]),
        CandidateSpec("logreg_l2_standard", "linear", "standard", factories["logreg_l2"]),
        CandidateSpec("logreg_l2_robust", "linear", "robust", factories["logreg_l2_tight"]),
        CandidateSpec("logreg_en_power", "linear", "power", factories["logreg_en_sparse"]),
        CandidateSpec("logreg_en_quantile", "linear", "quantile_normal", factories["logreg_en_dense"]),
        CandidateSpec("raw_bnb_standard", "nb", "standard", factories["bnb"]),
        CandidateSpec("raw_bnb_quantile", "nb", "quantile_uniform", factories["bnb"]),
        CandidateSpec("raw_bnb_minmax", "nb", "minmax", factories["bnb"]),
        CandidateSpec("gnb_power", "nb", "power", factories["gnb"]),
    ]

    if target == "Depression_label":
        base.extend(
            [
                CandidateSpec("hgb_bal_d3", "hgb", "none", factories["hgb_bal_d3"]),
                CandidateSpec("logreg_en_robust", "linear", "robust", factories["logreg_en_sparse"]),
                CandidateSpec("lsvc_cal_standard", "svm", "standard", factories["lsvc_cal"]),
            ]
        )
    else:
        base.extend(
            [
                CandidateSpec("raw_hgb3b", "hgb", "none", factories["raw_hgb3b"]),
                CandidateSpec("hgb_bal_d3", "hgb", "none", factories["hgb_bal_d3"]),
                CandidateSpec("logreg_l2_quantile", "linear", "quantile_normal", factories["logreg_l2"]),
                CandidateSpec("lsvc_cal_standard", "svm", "standard", factories["lsvc_cal"]),
            ]
        )
    return base


def build_pipeline(spec: CandidateSpec, n_train_samples: int):
    return Pipeline(
        steps=[
            ("pre", build_preprocessor(spec.scaler_name, n_train_samples)),
            ("model", spec.model_factory()),
        ]
    )


def safe_average_precision(y_true, proba):
    return float(average_precision_score(y_true, proba))


def safe_roc_auc(y_true, proba):
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return None


def tune_threshold(y_true, proba):
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 197),
                np.asarray(proba, dtype=float),
            ]
        )
    )
    best_threshold = 0.5
    best_f1 = -1.0
    best_balanced_acc = -1.0

    for threshold in candidates:
        pred = (proba >= threshold).astype(int)
        balanced_acc = balanced_accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1 + 1e-12:
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_balanced_acc = float(balanced_acc)
            continue
        is_tie = abs(f1 - best_f1) <= 1e-12
        better_tie_break = balanced_acc > best_balanced_acc + 1e-12
        closer_to_half = abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        if is_tie and (better_tie_break or (abs(balanced_acc - best_balanced_acc) <= 1e-12 and closer_to_half)):
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_balanced_acc = float(balanced_acc)
    return best_threshold


def confusion_counts(y_true, pred):
    tp = int(((y_true == 1) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_predictions(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    pr_auc = safe_average_precision(y_true, proba)
    metrics = {
        "pr_auc": pr_auc,
        "average_precision": pr_auc,
        "roc_auc": safe_roc_auc(y_true, proba),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
        "threshold": float(threshold),
    }
    metrics.update(confusion_counts(np.asarray(y_true), pred))
    return metrics


def save_predictions(path: Path, participants, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    pd.DataFrame(
        {
            "Participant": participants,
            "y_true": np.asarray(y_true, dtype=int),
            "y_proba": proba,
            "y_pred": pred,
        }
    ).to_csv(path, index=False)


def native_importance_df(pipe: Pipeline):
    model = pipe.named_steps["model"]
    feature_names = pipe.named_steps["pre"].get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.asarray(model.feature_importances_, dtype=float),
            }
        ).sort_values("importance", ascending=False)

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        return pd.DataFrame(
            {
                "feature": feature_names,
                "coef": coef,
                "abs_coef": np.abs(coef),
            }
        ).sort_values("abs_coef", ascending=False)

    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).ravel()
        return pd.DataFrame(
            {
                "feature": feature_names,
                "coef": coef,
                "abs_coef": np.abs(coef),
            }
        ).sort_values("abs_coef", ascending=False)

    return None


def preprocess_label(scale_name: str):
    mapping = {
        "none": "median_impute",
        "standard": "median_impute+standard",
        "robust": "median_impute+robust",
        "power": "median_impute+power",
        "quantile_normal": "median_impute+quantile_normal",
        "quantile_uniform": "median_impute+quantile_uniform",
        "minmax": "median_impute+minmax",
    }
    return mapping.get(scale_name, f"median_impute+{scale_name}")


def fit_and_score_candidate(spec: CandidateSpec, X_train: pd.DataFrame, y_train, X_dev: pd.DataFrame, y_dev):
    pipe = build_pipeline(spec, n_train_samples=len(X_train))
    pipe.fit(X_train, y_train)
    dev_proba = pipe.predict_proba(X_dev)[:, 1]
    threshold = tune_threshold(y_dev, dev_proba)
    dev_metrics = evaluate_predictions(y_dev, dev_proba, threshold)
    return pipe, dev_proba, dev_metrics


def choose_shortlist(single_rows: list[dict]) -> list[dict]:
    shortlist = []
    for family in ("hgb", "tree", "linear", "nb", "svm"):
        family_rows = [row for row in single_rows if row["family"] == family]
        if family_rows:
            shortlist.append(family_rows[0])
    for row in single_rows:
        if row not in shortlist:
            shortlist.append(row)
        if len(shortlist) >= 6:
            break
    deduped = []
    seen_names = set()
    for row in shortlist:
        if row["model"] in seen_names:
            continue
        seen_names.add(row["model"])
        deduped.append(row)
    return deduped[:6]


def build_blend_name(model_names, weights):
    weight_suffix = "_".join(f"{int(round(weight * 100)):02d}" for weight in weights)
    return f"blend_{'_'.join(model_names)}_{weight_suffix}"


def blend_predictions(prediction_map, model_names, weights):
    return sum(weight * prediction_map[name] for name, weight in zip(model_names, weights))


def iter_blend_weights(num_terms: int, step: float = 0.1):
    values = np.round(np.arange(step, 1.0, step), 10)
    if num_terms == 2:
        for w1 in values:
            w2 = 1.0 - w1
            if w2 >= step - 1e-12:
                yield (float(w1), float(w2))
        return
    if num_terms == 3:
        for w1 in values:
            for w2 in values:
                w3 = 1.0 - w1 - w2
                if w3 >= step - 1e-12:
                    yield (float(w1), float(w2), float(w3))
        return
    raise ValueError(f"Unsupported blend size: {num_terms}")


def iter_local_blend_weights(base_weights, step: float = 0.025, radius: float = 0.1):
    base = np.asarray(base_weights, dtype=float)
    values = np.round(np.arange(-radius, radius + step / 2, step), 10)

    if len(base) == 2:
        for delta in values:
            weights = np.asarray([base[0] + delta, base[1] - delta], dtype=float)
            if np.all(weights >= step - 1e-12):
                yield tuple(float(weight) for weight in weights)
        return

    if len(base) == 3:
        for delta_1 in values:
            for delta_2 in values:
                delta_3 = -(delta_1 + delta_2)
                if abs(delta_3) > radius + 1e-12:
                    continue
                weights = np.asarray(
                    [base[0] + delta_1, base[1] + delta_2, base[2] + delta_3],
                    dtype=float,
                )
                if np.all(weights >= step - 1e-12):
                    yield tuple(float(weight) for weight in weights)
        return


def build_saved_model(spec, fitted_pipelines):
    if spec["kind"] == "single":
        return fitted_pipelines[spec["base_models"][0]]
    return {
        "type": "blend",
        "weights": dict(zip(spec["base_models"], spec["weights"])),
        "models": {name: fitted_pipelines[name] for name in spec["base_models"]},
    }


def describe_target_search(target: str, selected_model: str):
    baseline_note = (
        "baseline_rebuild=run_01 family mix approximated with raw_hgb/raw_bnb style models "
        "under cleaned_full_minus_gender constraints"
    )
    if target == "Depression_label":
        return (
            f"{baseline_note}; cleaned full feature set used; gender excluded; duplicate columns removed; "
            f"target-specific cleaned full-feature model and blend search; selected={selected_model}"
        )
    return (
        f"{baseline_note}; cleaned full feature set used; gender excluded; duplicate columns removed; "
        f"PTSD search emphasizes raw_hgb/raw_bnb/logreg families on cleaned full features; selected={selected_model}"
    )


def run_single_target(
    df: pd.DataFrame,
    target: str,
    include_test: bool,
    artifacts_dir: Path,
    skip_importance: bool,
):
    feature_columns = get_feature_columns(df)
    removed_columns = get_removed_redundant_columns(df)
    X = df.loc[:, feature_columns].copy()
    y = df[target].astype(int)

    split_mask = {split: df["split"].eq(split) for split in EXPECTED_SPLITS}
    X_train = X.loc[split_mask["train"]].reset_index(drop=True)
    y_train = y.loc[split_mask["train"]].reset_index(drop=True)
    X_dev = X.loc[split_mask["dev"]].reset_index(drop=True)
    y_dev = y.loc[split_mask["dev"]].reset_index(drop=True)
    X_test = X.loc[split_mask["test"]].reset_index(drop=True)
    y_test = y.loc[split_mask["test"]].reset_index(drop=True)
    participant_dev = df.loc[split_mask["dev"], "Participant"].reset_index(drop=True)
    participant_test = df.loc[split_mask["test"], "Participant"].reset_index(drop=True)

    candidate_specs = get_candidate_specs(target)
    leaderboard_rows = []
    fitted_pipelines = {}
    dev_predictions = {}
    test_predictions = {}
    candidate_meta = {}

    for spec in candidate_specs:
        pipe, dev_proba, dev_metrics = fit_and_score_candidate(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            X_dev=X_dev,
            y_dev=y_dev,
        )
        fitted_pipelines[spec.name] = pipe
        dev_predictions[spec.name] = dev_proba
        if include_test:
            test_predictions[spec.name] = pipe.predict_proba(X_test)[:, 1]

        candidate_meta[spec.name] = {
            "kind": "single",
            "family": spec.family,
            "scaler_name": spec.scaler_name,
            "feature_subset": FULL_FEATURE_SUBSET_NAME,
            "preprocess": preprocess_label(spec.scaler_name),
        }
        leaderboard_rows.append(
            {
                "target": target,
                "model": spec.name,
                "kind": "single",
                "family": spec.family,
                "dev_pr_auc": dev_metrics["pr_auc"],
                "dev_f1": dev_metrics["f1"],
                "threshold": dev_metrics["threshold"],
                "preprocess": preprocess_label(spec.scaler_name),
                "feature_subset": FULL_FEATURE_SUBSET_NAME,
            }
        )

    single_df = pd.DataFrame(leaderboard_rows).sort_values(
        by=["dev_pr_auc", "dev_f1", "model"],
        ascending=[False, False, True],
        na_position="last",
    )
    shortlist = choose_shortlist(single_df.to_dict(orient="records"))

    blend_rows = []
    blend_predictions_cache = {}
    blend_test_predictions_cache = {}
    blend_spec_lookup = {}

    blend_step = 0.05 if target == "PTSD_label" else 0.1
    for num_terms in (2, 3):
        if len(shortlist) < num_terms:
            continue
        for combo in combinations([row["model"] for row in shortlist], num_terms):
            for weights in iter_blend_weights(num_terms, step=blend_step):
                blend_name = build_blend_name(combo, weights)
                dev_proba = blend_predictions(dev_predictions, combo, weights)
                threshold = tune_threshold(y_dev, dev_proba)
                dev_metrics = evaluate_predictions(y_dev, dev_proba, threshold)
                blend_rows.append(
                    {
                        "target": target,
                        "model": blend_name,
                        "kind": "blend",
                        "family": "blend",
                        "dev_pr_auc": dev_metrics["pr_auc"],
                        "dev_f1": dev_metrics["f1"],
                        "threshold": dev_metrics["threshold"],
                        "preprocess": BLEND_PREPROCESS_LABEL,
                        "feature_subset": FULL_FEATURE_SUBSET_NAME,
                    }
                )
                blend_predictions_cache[blend_name] = dev_proba
                blend_spec_lookup[blend_name] = {
                    "name": blend_name,
                    "kind": "blend",
                    "base_models": list(combo),
                    "weights": list(weights),
                    "feature_subset": FULL_FEATURE_SUBSET_NAME,
                    "preprocess": BLEND_PREPROCESS_LABEL,
                }
                if include_test:
                    blend_test_predictions_cache[blend_name] = blend_predictions(test_predictions, combo, weights)

    if target == "PTSD_label" and blend_rows:
        ranked_blends = sorted(
            blend_rows,
            key=lambda row: (row["dev_pr_auc"], row["dev_f1"]),
            reverse=True,
        )
        refined_combos = set()
        for row in ranked_blends:
            base_spec = blend_spec_lookup[row["model"]]
            combo = tuple(base_spec["base_models"])
            if combo in refined_combos:
                continue
            refined_combos.add(combo)
            for weights in iter_local_blend_weights(base_spec["weights"], step=0.025, radius=0.1):
                if list(weights) == base_spec["weights"]:
                    continue
                blend_name = build_blend_name(combo, weights)
                dev_proba = blend_predictions(dev_predictions, combo, weights)
                threshold = tune_threshold(y_dev, dev_proba)
                dev_metrics = evaluate_predictions(y_dev, dev_proba, threshold)
                blend_rows.append(
                    {
                        "target": target,
                        "model": blend_name,
                        "kind": "blend",
                        "family": "blend",
                        "dev_pr_auc": dev_metrics["pr_auc"],
                        "dev_f1": dev_metrics["f1"],
                        "threshold": dev_metrics["threshold"],
                        "preprocess": BLEND_PREPROCESS_LABEL,
                        "feature_subset": FULL_FEATURE_SUBSET_NAME,
                    }
                )
                blend_predictions_cache[blend_name] = dev_proba
                blend_spec_lookup[blend_name] = {
                    "name": blend_name,
                    "kind": "blend",
                    "base_models": list(combo),
                    "weights": list(weights),
                    "feature_subset": FULL_FEATURE_SUBSET_NAME,
                    "preprocess": BLEND_PREPROCESS_LABEL,
                }
                if include_test:
                    blend_test_predictions_cache[blend_name] = blend_predictions(test_predictions, combo, weights)
            if len(refined_combos) >= 3:
                break

    leaderboard_df = pd.DataFrame([*leaderboard_rows, *blend_rows]).sort_values(
        by=["dev_pr_auc", "dev_f1", "model"],
        ascending=[False, False, True],
        na_position="last",
    )
    leaderboard_path = artifacts_dir / f"leaderboard_{target}.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    best_row = leaderboard_df.iloc[0].to_dict()
    best_model_name = best_row["model"]
    best_is_blend = best_row["kind"] == "blend"

    if best_is_blend:
        best_spec = blend_spec_lookup[best_model_name]
        best_dev_proba = blend_predictions_cache[best_model_name]
        best_saved_model = build_saved_model(best_spec, fitted_pipelines)
        best_base_name = best_spec["base_models"][int(np.argmax(best_spec["weights"]))]
    else:
        best_spec = {
            "name": best_model_name,
            "kind": "single",
            "base_models": [best_model_name],
            "weights": [1.0],
            "feature_subset": FULL_FEATURE_SUBSET_NAME,
            "preprocess": candidate_meta[best_model_name]["preprocess"],
        }
        best_dev_proba = dev_predictions[best_model_name]
        best_saved_model = fitted_pipelines[best_model_name]
        best_base_name = best_model_name

    best_threshold = tune_threshold(y_dev, best_dev_proba)
    best_dev_metrics = evaluate_predictions(y_dev, best_dev_proba, best_threshold)
    best_test_metrics = None
    if include_test:
        if best_is_blend:
            best_test_proba = blend_test_predictions_cache[best_model_name]
        else:
            best_test_proba = test_predictions[best_model_name]
        best_test_metrics = evaluate_predictions(y_test, best_test_proba, best_threshold)

    dev_pred_path = artifacts_dir / f"predictions_{target}_dev.csv"
    save_predictions(dev_pred_path, participant_dev, y_dev, best_dev_proba, best_threshold)

    test_pred_path = None
    if include_test:
        if best_is_blend:
            best_test_proba = blend_test_predictions_cache[best_model_name]
        else:
            best_test_proba = test_predictions[best_model_name]
        test_pred_path = artifacts_dir / f"predictions_{target}_test.csv"
        save_predictions(test_pred_path, participant_test, y_test, best_test_proba, best_threshold)

    model_path = artifacts_dir / f"best_model_{target}.joblib"
    joblib.dump(best_saved_model, model_path)

    native_importance_path = None
    if not skip_importance:
        native_df = native_importance_df(fitted_pipelines[best_base_name])
        if native_df is not None:
            native_importance_path = artifacts_dir / f"native_importance_{target}.csv"
            native_df.to_csv(native_importance_path, index=False)

    result = {
        "selected_model": best_model_name,
        "feature_subset": FULL_FEATURE_SUBSET_NAME,
        "preprocess": best_spec["preprocess"],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "removed_redundant_columns": removed_columns,
        "gender_excluded": True,
        "shortlist_models": [row["model"] for row in shortlist],
        "dev": best_dev_metrics,
        "artifacts": {
            "leaderboard": str(leaderboard_path),
            "dev_predictions": str(dev_pred_path),
            "model": str(model_path),
        },
    }
    if native_importance_path is not None:
        result["artifacts"]["native_importance"] = str(native_importance_path)
    if include_test:
        result["test"] = best_test_metrics
        result["artifacts"]["test_predictions"] = str(test_pred_path)
    result["best_model_family"] = candidate_meta.get(best_base_name, {}).get("family", "blend")
    return result


def build_summary(target_results, include_test):
    dev_pr_aucs = [target_results[target]["dev"]["pr_auc"] for target in target_results]
    dev_aucs = [
        target_results[target]["dev"]["roc_auc"]
        for target in target_results
        if target_results[target]["dev"]["roc_auc"] is not None
    ]
    summary = {
        "primary_metric": "mean_dev_pr_auc",
        "primary_score": float(np.mean(dev_pr_aucs)),
        "mean_dev_pr_auc": float(np.mean(dev_pr_aucs)),
        "min_dev_pr_auc": float(np.min(dev_pr_aucs)),
        "mean_dev_roc_auc": float(np.mean(dev_aucs)) if dev_aucs else None,
        "targets_evaluated": list(target_results.keys()),
        "test_included": include_test,
    }
    if include_test:
        test_pr_aucs = [target_results[target]["test"]["pr_auc"] for target in target_results]
        summary["mean_test_pr_auc"] = float(np.mean(test_pr_aucs))
    return summary


def print_summary(payload):
    summary = payload["summary"]
    print(f"Primary metric: {summary['primary_metric']} = {summary['primary_score']:.6f}")
    print(f"Secondary metric: min_dev_pr_auc = {summary['min_dev_pr_auc']:.6f}")
    for target, target_result in payload["targets"].items():
        dev = target_result["dev"]
        line = (
            f"{target}: model={target_result['selected_model']} "
            f"dev_pr_auc={dev['pr_auc']:.6f} "
            f"dev_f1={dev['f1']:.6f} "
            f"dev_auc={dev['roc_auc']:.6f} "
            f"threshold={dev['threshold']:.4f}"
        )
        if payload["summary"]["test_included"]:
            test = target_result["test"]
            line += f" test_pr_auc={test['pr_auc']:.6f} test_auc={test['roc_auc']:.6f}"
        print(line)
        print(
            f"  feature_space={target_result['feature_subset']} "
            f"feature_count={target_result['feature_count']} "
            f"removed_columns={','.join(target_result['removed_redundant_columns'])}"
        )


def ensure_results_tsv_header(results_tsv_path: Path):
    if not results_tsv_path.exists():
        results_tsv_path.write_text(RESULTS_TSV_HEADER + "\n", encoding="utf-8")


def append_results_tsv(results_tsv_path: Path, run_id: str, target: str, target_result: dict, status: str, description: str):
    ensure_results_tsv_header(results_tsv_path)
    row = [
        run_id,
        target,
        f"{target_result['dev']['pr_auc']:.6f}",
        f"{target_result['dev']['f1']:.6f}",
        target_result["selected_model"],
        FULL_FEATURE_SUBSET_NAME,
        target_result["preprocess"],
        f"{target_result['dev']['threshold']:.6f}",
        "yes",
        "yes",
        "|".join(target_result["removed_redundant_columns"]),
        status,
        description,
    ]
    with open(results_tsv_path, "a", encoding="utf-8") as handle:
        handle.write("\t".join(row) + "\n")


def parse_results_records(results_tsv_path: Path) -> list[dict]:
    if not results_tsv_path.exists():
        return []
    records = []
    with open(results_tsv_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if parts[0] in {"commit", "run_id"}:
                continue
            if len(parts) < 10:
                continue
            if len(parts) >= 2 and parts[1] not in TARGETS:
                continue
            try:
                record = {
                    "run_id": parts[0],
                    "target_label": parts[1],
                    "dev_pr_auc": float(parts[2]),
                    "dev_f1": float(parts[3]),
                    "model": parts[4],
                }
            except ValueError:
                continue
            if len(parts) >= 13:
                record.update(
                    {
                        "feature_subset": parts[5],
                        "preprocess": parts[6],
                        "threshold": parts[7],
                        "cleaned_full_feature_set": parts[8],
                        "gender_excluded": parts[9],
                        "removed_columns": parts[10],
                        "status": parts[11],
                        "description": "\t".join(parts[12:]),
                    }
                )
            else:
                record.update(
                    {
                        "feature_subset": parts[5],
                        "preprocess": parts[6],
                        "threshold": parts[7],
                        "cleaned_full_feature_set": "",
                        "gender_excluded": "",
                        "removed_columns": "",
                        "status": parts[8],
                        "description": "\t".join(parts[9:]),
                    }
                )
            records.append(record)
    return records


def generate_metrics_plot(results_tsv_path: Path, plot_path: Path):
    records = parse_results_records(results_tsv_path)
    if not records:
        return None

    df = pd.DataFrame(records)
    if df.empty:
        return None

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    height = 820
    margin_left = 70
    margin_right = 24
    margin_top = 52
    margin_bottom = 48
    panel_gap = 52
    panel_height = (height - margin_top - margin_bottom - panel_gap) / 2
    plot_width = width - margin_left - margin_right
    metric_specs = [("dev_pr_auc", "Dev PR AUC"), ("dev_f1", "Dev F1")]
    colors = {
        "Depression_label": "#1f77b4",
        "PTSD_label": "#d62728",
    }
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="28" text-anchor="middle" font-size="18" font-family="Arial">AiRest Dev Metrics Progression From results.tsv</text>',
    ]

    for panel_index, (metric_name, title) in enumerate(metric_specs):
        panel_top = margin_top + panel_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        svg_lines.append(
            f'<text x="{margin_left}" y="{panel_top - 14}" font-size="14" font-family="Arial">{escape(title)}</text>'
        )
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{panel_bottom}" x2="{margin_left + plot_width}" y2="{panel_bottom}" stroke="#444" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<line x1="{margin_left}" y1="{panel_top}" x2="{margin_left}" y2="{panel_bottom}" stroke="#444" stroke-width="1"/>'
        )

        axis_min = 0.0
        axis_max = 1.0
        for tick_value in np.linspace(axis_min, axis_max, 6):
            y = panel_bottom - (tick_value - axis_min) / (axis_max - axis_min) * panel_height
            svg_lines.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>'
            )
            svg_lines.append(
                f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{tick_value:.1f}</text>'
            )

        max_run_count = 1
        target_frames = {}
        for target in TARGETS:
            target_df = df[df["target_label"] == target].reset_index(drop=True).copy()
            if target_df.empty:
                continue
            target_df["run_number"] = np.arange(1, len(target_df) + 1)
            target_frames[target] = target_df
            max_run_count = max(max_run_count, int(target_df["run_number"].max()))

        for run_number in range(1, max_run_count + 1):
            if max_run_count == 1:
                x = margin_left + plot_width / 2
            else:
                x = margin_left + (run_number - 1) / (max_run_count - 1) * plot_width
            svg_lines.append(
                f'<line x1="{x:.1f}" y1="{panel_top}" x2="{x:.1f}" y2="{panel_bottom}" stroke="#f2f2f2" stroke-width="1"/>'
            )
            svg_lines.append(
                f'<text x="{x:.1f}" y="{panel_bottom + 18}" text-anchor="middle" font-size="10" font-family="Arial">{run_number}</text>'
            )

        legend_x = margin_left + plot_width - 180
        legend_y = panel_top + 16
        for legend_index, target in enumerate(TARGETS):
            if target not in target_frames:
                continue
            legend_item_y = legend_y + legend_index * 16
            svg_lines.append(
                f'<line x1="{legend_x}" y1="{legend_item_y}" x2="{legend_x + 18}" y2="{legend_item_y}" stroke="{colors[target]}" stroke-width="3"/>'
            )
            svg_lines.append(
                f'<text x="{legend_x + 24}" y="{legend_item_y + 4}" font-size="11" font-family="Arial">{escape(target)}</text>'
            )

        for target, target_df in target_frames.items():
            points = []
            for _, row in target_df.iterrows():
                run_number = int(row["run_number"])
                value = float(row[metric_name])
                if max_run_count == 1:
                    x = margin_left + plot_width / 2
                else:
                    x = margin_left + (run_number - 1) / (max_run_count - 1) * plot_width
                y = panel_bottom - (value - axis_min) / (axis_max - axis_min) * panel_height
                points.append((x, y, value))
            point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            svg_lines.append(
                f'<polyline fill="none" stroke="{colors[target]}" stroke-width="2.2" points="{point_string}"/>'
            )
            for x, y, _ in points:
                svg_lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colors[target]}"/>'
                )
            best_row = target_df.loc[target_df[metric_name].idxmax()]
            best_run = int(best_row["run_number"])
            best_value = float(best_row[metric_name])
            if max_run_count == 1:
                best_x = margin_left + plot_width / 2
            else:
                best_x = margin_left + (best_run - 1) / (max_run_count - 1) * plot_width
            best_y = panel_bottom - (best_value - axis_min) / (axis_max - axis_min) * panel_height
            svg_lines.append(
                f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="5.2" fill="white" stroke="{colors[target]}" stroke-width="2"/>'
            )
            svg_lines.append(
                f'<text x="{best_x + 8:.1f}" y="{best_y - 8:.1f}" font-size="10" font-family="Arial" fill="{colors[target]}">best {best_value:.3f}</text>'
            )

        svg_lines.append(
            f'<text x="{margin_left + plot_width / 2:.1f}" y="{panel_bottom + 34}" text-anchor="middle" font-size="11" font-family="Arial">Run Number</text>'
        )

    svg_lines.append("</svg>")
    plot_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")
    return plot_path


def main():
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    results_path = args.results_path.resolve()
    results_tsv_path = args.results_tsv_path.resolve()
    plot_path = args.plot_path.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.csv_path.resolve())
    targets = TARGETS if args.target == "all" else (args.target,)

    target_results = {}
    for target in targets:
        target_results[target] = run_single_target(
            df=df,
            target=target,
            include_test=args.include_test,
            artifacts_dir=artifacts_dir,
            skip_importance=args.skip_importance,
        )

    payload = {
        "csv_path": str(args.csv_path.resolve()),
        "summary": build_summary(target_results, include_test=args.include_test),
        "feature_space": {
            "name": FULL_FEATURE_SUBSET_NAME,
            "gender_excluded": True,
            "removed_redundant_columns": get_removed_redundant_columns(df),
            "non_feature_columns": list(NON_FEATURE_COLUMNS),
        },
        "targets": target_results,
    }

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print_summary(payload)
    print(f"Saved summary JSON to {results_path}")

    run_id = datetime.now().astimezone().isoformat(timespec="seconds")
    for target in targets:
        description = describe_target_search(target, target_results[target]["selected_model"])
        append_results_tsv(
            results_tsv_path=results_tsv_path,
            run_id=run_id,
            target=target,
            target_result=target_results[target],
            status="keep",
            description=description,
        )

    generated_plot = generate_metrics_plot(results_tsv_path, plot_path)
    if generated_plot is not None:
        print(f"Saved metrics plot to {generated_plot}")


if __name__ == "__main__":
    main()
