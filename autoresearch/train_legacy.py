import os
import json
import argparse
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False


# =========================
# 1) Legacy feature blocks
# =========================
B0_SUMMARY: List[str] = []

B0_TURNS: List[str] = [
    "words_per_min",
    "speech_percentage",
    "mean_pause_length",
    "pause_variability",
    "word_repeat_percentage",
    "phrase_repeat_percentage",
    "turn_length_words",
]

L_SUMMARY: List[str] = [
    "word_coherence_mean",
    "word_coherence_10_mean",
    "semantic_perplexity_mean",
    "semantic_perplexity_5_mean",
    "first_order_sentence_tangeniality_mean",
]

L_TURNS: List[str] = [
    "syllables_per_min",
    "sentiment_neg",
    "sentiment_pos",
    "sentiment_overall",
    "mattr_10",
    "mattr_50",
    "first_person_percentage",
    "first_person_sentiment_negative",
]

SPINE_COLS: List[str] = [
    "Participant",
    "gender",
    "age",
    "split",
    "Depression_label",
    "PTSD_label",
    "PTSD_severity",
    "Depression_severity",
]


def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _union(a: List[str], b: List[str]) -> List[str]:
    return _unique_preserve_order(list(a) + list(b))


def merge_turns_and_summary_featureblock(
    detailed_labels_path: str,
    data_dir: str,
    *,
    participant_col: str = "Participant",
    turns_cols: Optional[List[str]] = None,
    summary_cols: Optional[List[str]] = None,
    turns_prefix: str = "turns_",
    summary_prefix: str = "summary_sc_",
    turns_suffix: str = ".csv",
    summary_suffix: str = ".csv",
    turns_agg: str | Callable[[pd.Series], Any] = "mean",
    resolve_name_collisions: bool = True,
    turns_collision_suffix: str = "_turns",
    summary_collision_suffix: str = "_summary",
    turns_speaker_col: Optional[str] = None,
    participant_speaker_value: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    if os.path.isfile(data_dir):
        data_dir = os.path.dirname(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    df_labels = pd.read_csv(detailed_labels_path)
    df_labels[participant_col] = pd.to_numeric(df_labels[participant_col], errors="coerce").astype("Int64")
    pids = df_labels[participant_col].dropna().unique().astype(int)

    turns_cols = _unique_preserve_order(turns_cols or [])
    summary_cols = _unique_preserve_order(summary_cols or [])

    rows: List[Dict[str, Any]] = []
    for pid in pids:
        rec: Dict[str, Any] = {participant_col: pid}

        t_path = os.path.join(data_dir, f"{turns_prefix}{pid}{turns_suffix}")
        if os.path.exists(t_path) and turns_cols:
            t_header = pd.read_csv(t_path, nrows=0).columns.tolist()
            t_use = [c for c in turns_cols if c in t_header]

            if turns_speaker_col and turns_speaker_col in t_header and turns_speaker_col not in t_use:
                t_use_with_speaker = [turns_speaker_col] + t_use
            else:
                t_use_with_speaker = t_use

            if t_use:
                t_df = pd.read_csv(t_path, usecols=t_use_with_speaker)
                if turns_speaker_col and participant_speaker_value is not None and turns_speaker_col in t_df.columns:
                    t_df = t_df[t_df[turns_speaker_col] == participant_speaker_value].copy()

                t_vals: Dict[str, Any] = {}
                if callable(turns_agg):
                    for c in t_use:
                        t_vals[c] = turns_agg(t_df[c])
                elif turns_agg == "mean":
                    for c in t_use:
                        col = pd.to_numeric(t_df[c], errors="coerce")
                        t_vals[c] = float(col.mean()) if len(col) else np.nan
                elif turns_agg == "first":
                    for c in t_use:
                        t_vals[c] = t_df[c].iloc[0] if len(t_df) else np.nan
                else:
                    raise ValueError(f"Unsupported turns_agg: {turns_agg!r}")

                if resolve_name_collisions:
                    renamed = {}
                    for k, v in t_vals.items():
                        out_k = k if k not in rec else f"{k}{turns_collision_suffix}"
                        renamed[out_k] = v
                    t_vals = renamed

                rec.update(t_vals)
            elif verbose:
                print(f"[pid={pid}] turns file exists but none of requested cols found.")

        s_path = os.path.join(data_dir, f"{summary_prefix}{pid}{summary_suffix}")
        if os.path.exists(s_path) and summary_cols:
            s_header = pd.read_csv(s_path, nrows=0).columns.tolist()
            s_use = [c for c in summary_cols if c in s_header]
            if s_use:
                s_df = pd.read_csv(s_path, usecols=s_use)
                s_vals = {c: (s_df[c].iloc[0] if len(s_df) else np.nan) for c in s_use}

                if resolve_name_collisions:
                    renamed = {}
                    for k, v in s_vals.items():
                        out_k = k if k not in rec else f"{k}{summary_collision_suffix}"
                        renamed[out_k] = v
                    s_vals = renamed

                rec.update(s_vals)
            elif verbose:
                print(f"[pid={pid}] summary file exists but none of requested cols found.")

        rows.append(rec)

    df_feat = pd.DataFrame(rows)
    return df_labels.merge(df_feat, on=participant_col, how="left")


def build_legacy_datasets(
    detailed_labels_path: str,
    data_dir: str,
    out_dir: str,
    *,
    language: str = "eng",
    iteration: str = "3",
    participant_col: str = "Participant",
) -> Dict[str, str]:
    out_dir = str(Path(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    outputs: Dict[str, str] = {}

    df_l = merge_turns_and_summary_featureblock(
        detailed_labels_path=detailed_labels_path,
        data_dir=data_dir,
        participant_col=participant_col,
        turns_cols=L_TURNS,
        summary_cols=L_SUMMARY,
        turns_agg="mean",
        resolve_name_collisions=True,
    )
    l_cols = [c for c in df_l.columns if c in SPINE_COLS] + \
             [c for c in df_l.columns if (c in L_TURNS or c.endswith("_turns"))] + \
             [c for c in df_l.columns if (c in L_SUMMARY or c.endswith("_summary"))]
    df_l_out = df_l[_unique_preserve_order(l_cols)].copy()
    l_path = os.path.join(out_dir, f"dataset_L_gemma_{language}_small_test{iteration}.csv")
    df_l_out.to_csv(l_path, index=False)
    outputs["L"] = l_path

    df_b0 = merge_turns_and_summary_featureblock(
        detailed_labels_path=detailed_labels_path,
        data_dir=data_dir,
        participant_col=participant_col,
        turns_cols=B0_TURNS,
        summary_cols=B0_SUMMARY,
        turns_agg="mean",
        resolve_name_collisions=True,
    )
    b0_cols = [c for c in df_b0.columns if c in SPINE_COLS] + \
              [c for c in df_b0.columns if (c in B0_TURNS or c.endswith("_turns"))] + \
              [c for c in df_b0.columns if (c in B0_SUMMARY or c.endswith("_summary"))]
    df_b0_out = df_b0[_unique_preserve_order(b0_cols)].copy()
    b0_path = os.path.join(out_dir, f"dataset_B0_gemma_{language}_small_test{iteration}.csv")
    df_b0_out.to_csv(b0_path, index=False)
    outputs["B0"] = b0_path

    df_b0l = merge_turns_and_summary_featureblock(
        detailed_labels_path=detailed_labels_path,
        data_dir=data_dir,
        participant_col=participant_col,
        turns_cols=_union(B0_TURNS, L_TURNS),
        summary_cols=_union(B0_SUMMARY, L_SUMMARY),
        turns_agg="mean",
        resolve_name_collisions=True,
    )
    b0l_turns = _union(B0_TURNS, L_TURNS)
    b0l_summary = _union(B0_SUMMARY, L_SUMMARY)
    b0l_cols = [c for c in df_b0l.columns if c in SPINE_COLS] + \
               [c for c in df_b0l.columns if (c in b0l_turns or c.endswith("_turns"))] + \
               [c for c in df_b0l.columns if (c in b0l_summary or c.endswith("_summary"))]
    df_b0l_out = df_b0l[_unique_preserve_order(b0l_cols)].copy()
    b0l_path = os.path.join(out_dir, f"dataset_B0_plus_L_gemma_{language}_full{iteration}.csv")
    df_b0l_out.to_csv(b0l_path, index=False)
    outputs["B0+L"] = b0l_path

    return outputs


def bootstrap_ci_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    n_boot: int = 2000,
    seed: int = 1706,
    stratified: bool = True,
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
        if stratified:
            samp_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            samp_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            idx = np.concatenate([samp_pos, samp_neg])
        else:
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        yt, yp, pr = y_true[idx], y_pred[idx], y_proba[idx]
        if metric == "f1_macro":
            s = f1_score(yt, yp, average="macro")
        elif metric == "auc":
            if len(np.unique(yt)) < 2:
                continue
            s = roc_auc_score(yt, pr)
        elif metric in ("pr_auc", "average_precision"):
            if len(np.unique(yt)) < 2:
                continue
            s = average_precision_score(yt, pr)
        else:
            raise ValueError("metric must be 'f1_macro', 'auc', or 'pr_auc'")
        scores.append(float(s))

    if not scores:
        return float("nan"), float("nan"), float("nan")

    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def build_standard_drop_cols(
    df: pd.DataFrame,
    *,
    target_col: str,
    split_col: str = "split",
    id_col: str = "Participant",
    drop_demographics: bool = True,
    extra_drop: Optional[Sequence[str]] = None,
) -> List[str]:
    drop = set()
    for c in [id_col, split_col, target_col]:
        if c in df.columns:
            drop.add(c)
    for c in df.columns:
        cl = c.lower()
        if cl.endswith("_label") and c != target_col:
            drop.add(c)
    for c in df.columns:
        if "severity" in c.lower():
            drop.add(c)
    if drop_demographics:
        for c in ["age", "sex", "gender"]:
            if c in df.columns:
                drop.add(c)
    if extra_drop:
        for c in extra_drop:
            if c in df.columns:
                drop.add(c)
    return sorted(drop)


def assert_no_leakage_cols_left(X: pd.DataFrame) -> None:
    bad = []
    for c in X.columns:
        cl = c.lower()
        if cl.endswith("_label") or "severity" in cl or cl == "split":
            bad.append(c)
    if bad:
        raise ValueError(
            "Leakage columns detected in features X (you must drop them): "
            + ", ".join(sorted(set(bad)))
        )


def run_models_pipeline_more_models_tuned_prauc(
    dataset_path: str,
    *,
    target_col: str = "Depression_label",
    split_col: str = "split",
    id_col: str = "Participant",
    train_values: Sequence[str] = ("train", "dev"),
    eval_value: str = "test",
    drop_cols: Optional[Sequence[str]] = None,
    drop_demographics: bool = True,
    random_state: int = 1706,
    n_boot: int = 2000,
    compute_importance: bool = True,
    perm_scoring: str = "average_precision",
    perm_n_repeats: int = 10,
    plot_top_k: int = 15,
    train_on_importance: bool = True,
    importance_top_k: int = 5,
    importance_corr_thr: float = 0.90,
    importance_n_repeats: int = 20,
    importance_cv_splits: int = 5,
    tune_tree_models: bool = True,
    cv_splits: int = 5,
    grid_scoring: str = "average_precision",
    grid_n_jobs: int = 1,
    grid_verbose: int = 0,
    report_train_cv_pr_auc: bool = True,
    save_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    results_tsv: Optional[str] = None,
    run_meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    if split_col not in df.columns:
        raise KeyError(f"Missing '{split_col}' column in dataset: {dataset_path}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target_col='{target_col}' in dataset: {dataset_path}")

    std_drop = build_standard_drop_cols(
        df,
        target_col=target_col,
        split_col=split_col,
        id_col=id_col,
        drop_demographics=drop_demographics,
        extra_drop=drop_cols,
    )

    split_lower = df[split_col].astype(str).str.lower().str.strip()
    train_values_set = {str(v).lower() for v in train_values}
    is_train = split_lower.isin(train_values_set)
    is_eval = split_lower.eq(str(eval_value).lower())

    if is_train.sum() == 0 or is_eval.sum() == 0:
        raise ValueError(
            f"Train/Eval rows not found. train in {sorted(train_values_set)} count={is_train.sum()}, "
            f"eval='{eval_value}' count={is_eval.sum()}."
        )

    y_train = df.loc[is_train, target_col].astype(int).to_numpy()
    y_eval = df.loc[is_eval, target_col].astype(int).to_numpy()

    uniq = np.unique(y_train)
    if len(uniq) != 2:
        raise ValueError(f"Target '{target_col}' is not binary on training split(s). Unique values: {uniq}")

    X_train = df.loc[is_train].drop(columns=std_drop, errors="ignore")
    X_eval = df.loc[is_eval].drop(columns=std_drop, errors="ignore")

    if X_train.shape[1] == 0:
        raise ValueError("No features left after dropping standardized leakage/meta columns.")

    assert_no_leakage_cols_left(X_train)

    def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
        categorical_cols = [c for c in X.columns if X[c].dtype.name in ["object", "category", "bool"]]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", ohe),
        ])
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, categorical_cols),
                ("num", num_pipe, numeric_cols),
            ],
            remainder="drop",
        )
        return pre, categorical_cols, numeric_cols

    def _predict_proba_or_score(pipe: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            return pipe.predict_proba(X)[:, 1]
        except Exception:
            try:
                from scipy.special import expit
                return expit(pipe.decision_function(X))
            except Exception:
                return None

    def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, Any]:
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

            _, f1_lo, f1_hi = bootstrap_ci_binary(y_true, y_pred, y_proba, metric="f1_macro", n_boot=n_boot, seed=random_state)
            _, auc_lo, auc_hi = bootstrap_ci_binary(y_true, y_pred, y_proba, metric="auc", n_boot=n_boot, seed=random_state)
            _, pr_lo, pr_hi = bootstrap_ci_binary(y_true, y_pred, y_proba, metric="pr_auc", n_boot=n_boot, seed=random_state)
            out["f1_lo"], out["f1_hi"] = float(f1_lo), float(f1_hi)
            out["auc_lo"], out["auc_hi"] = float(auc_lo), float(auc_hi)
            out["pr_auc_lo"], out["pr_auc_hi"] = float(pr_lo), float(pr_hi)
        return out

    def _safe_cv_splits(y: np.ndarray, desired: int) -> int:
        counts = Counter(y.tolist())
        min_class = min(counts.values()) if counts else 0
        return max(2, min(desired, min_class)) if min_class >= 2 else 0

    def _cv_pr_auc(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, desired_splits: int) -> Tuple[float, float]:
        k = _safe_cv_splits(y, desired_splits)
        if k < 2:
            return float("nan"), float("nan")
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        scores = cross_val_score(clone(pipe), X, y, cv=cv, scoring="average_precision", n_jobs=1)
        return float(np.mean(scores)), float(np.std(scores))

    def _warn_if_degenerate_proba(y_proba: Optional[np.ndarray], title: str) -> None:
        if y_proba is None:
            return
        uniqp = np.unique(np.round(y_proba, 6))
        if len(uniqp) <= 5:
            print(f"[WARN] {title}: very few unique predicted probabilities ({len(uniqp)}).")

    def _perm_importance_raw(pipe: Pipeline, X_curr: pd.DataFrame, y_curr: np.ndarray, title: str) -> None:
        try:
            perm = permutation_importance(pipe, X_curr, y_curr, n_repeats=perm_n_repeats, random_state=random_state, scoring=perm_scoring)
            perm_df = pd.DataFrame({
                "feature": list(X_curr.columns),
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
            print(f"\n[{title}] Top-{min(10, len(perm_df))} permutation importances ({perm_scoring}, raw):")
            print(perm_df.head(10).to_string(index=False))
            top_n = min(plot_top_k, len(perm_df))
            if top_n > 0:
                plt.figure(figsize=(9, max(4, 0.35 * top_n)))
                plt.barh(perm_df.loc[: top_n - 1, "feature"][::-1], perm_df.loc[: top_n - 1, "importance_mean"][::-1])
                plt.title(f"Permutation importance (top-{top_n}) — {title} [raw]")
                plt.xlabel(f"Mean importance on EVAL ({perm_scoring})")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"[{title}] permutation importance failed: {e}")

    def _select_cols_train_only_cv(base_pipe: Pipeline, X_tr: pd.DataFrame, y_tr: np.ndarray) -> List[str]:
        k = _safe_cv_splits(y_tr, importance_cv_splits)
        if k < 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
            (idx_sub, idx_val), = sss.split(X_tr, y_tr)
            X_sub, y_sub = X_tr.iloc[idx_sub], y_tr[idx_sub]
            X_val, y_val = X_tr.iloc[idx_val], y_tr[idx_val]
            pipe = clone(base_pipe)
            pipe.fit(X_sub, y_sub)
            perm = permutation_importance(pipe, X_val, y_val, n_repeats=importance_n_repeats, random_state=random_state, scoring=perm_scoring)
            imp = pd.Series(perm.importances_mean, index=X_val.columns)
        else:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            imp_accum = pd.Series(0.0, index=X_tr.columns)
            folds = 0
            for tr_idx, val_idx in cv.split(X_tr, y_tr):
                X_sub, y_sub = X_tr.iloc[tr_idx], y_tr[tr_idx]
                X_val, y_val = X_tr.iloc[val_idx], y_tr[val_idx]
                pipe = clone(base_pipe)
                pipe.fit(X_sub, y_sub)
                perm = permutation_importance(pipe, X_val, y_val, n_repeats=importance_n_repeats, random_state=random_state, scoring=perm_scoring)
                imp_accum = imp_accum.add(pd.Series(perm.importances_mean, index=X_val.columns), fill_value=0.0)
                folds += 1
            imp = imp_accum / max(folds, 1)

        imp = imp.sort_values(ascending=False)
        cand = imp.head(max(1, importance_top_k)).index.tolist()
        X_cand = X_tr[cand]
        num_cols = X_cand.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            corr = X_cand[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = set()
            for c in upper.columns:
                for r in upper.index[upper[c] > importance_corr_thr].tolist():
                    drop = c if imp.get(c, -np.inf) < imp.get(r, -np.inf) else r
                    to_drop.add(drop)
            cand = [c for c in cand if c not in to_drop]
        if not cand:
            cand = imp.head(5).index.tolist()
        return cand

    def _grid_search_tree_if_enabled(*, estimator, preprocessor: ColumnTransformer, X_tr: pd.DataFrame, y_tr: np.ndarray, param_grid: Dict[str, List[Any]]):
        pipe = Pipeline([("pre", preprocessor), ("clf", clone(estimator))])
        if (not tune_tree_models) or (not param_grid):
            pipe.fit(X_tr, y_tr)
            return pipe, {}, float("nan"), False
        cvk = _safe_cv_splits(y_tr, cv_splits)
        if cvk < 2:
            pipe.fit(X_tr, y_tr)
            return pipe, {}, float("nan"), False
        cv = StratifiedKFold(n_splits=cvk, shuffle=True, random_state=random_state)
        grid = {f"clf__{k}": v for k, v in param_grid.items()}
        gs = GridSearchCV(estimator=pipe, param_grid=grid, scoring=grid_scoring, cv=cv, n_jobs=grid_n_jobs, refit=True, verbose=grid_verbose, error_score=np.nan)
        gs.fit(X_tr, y_tr)
        return gs.best_estimator_, gs.best_params_, float(gs.best_score_), True

    models: List[Tuple[str, Any, bool, Dict[str, List[Any]]]] = []
    models.append(("LogisticRegression", LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced", random_state=random_state, n_jobs=1), False, {}))
    models.append(("SVM", SVC(probability=True, class_weight="balanced", random_state=random_state), True, {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf", "linear"]}))
    models.append(("SGDClassifier(log_loss)", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=5000, tol=1e-3, class_weight="balanced", random_state=random_state), False, {}))
    models.append(("DecisionTree", DecisionTreeClassifier(random_state=random_state, class_weight="balanced"), True, {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4, 8]}))
    models.append(("RandomForest", RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=1), True, {"n_estimators": [200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 0.5], "bootstrap": [True]}))
    if xgb_available:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg / max(pos, 1)) if pos > 0 else 1.0
        base_xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=random_state, n_jobs=1, scale_pos_weight=spw)
        models.append(("XGBoost", base_xgb, True, {"n_estimators": [200, 500], "max_depth": [3, 6, 10], "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "min_child_weight": [1, 5], "reg_lambda": [1.0, 5.0]}))

    print(f"\n[INFO] dataset = {dataset_path}")
    print(f"[INFO] target_col = {target_col}")
    print(f"[INFO] evaluation split = {eval_value}")
    print(f"[INFO] standardized drop_cols ({len(std_drop)}): {std_drop}")
    print(f"[INFO] X_train shape: {X_train.shape} | X_eval shape: {X_eval.shape}")
    print(f"[INFO] class counts train: {np.bincount(y_train)} | eval: {np.bincount(y_eval)}")

    results: List[Dict[str, Any]] = []
    for model_name, estimator, is_tree, grid in models:
        pre, _, _ = _build_preprocessor(X_train)
        if is_tree:
            pipe_full, best_params, best_cv, tuned = _grid_search_tree_if_enabled(estimator=estimator, preprocessor=pre, X_tr=X_train, y_tr=y_train, param_grid=grid)
        else:
            pipe_full = Pipeline([("pre", pre), ("clf", clone(estimator))])
            pipe_full.fit(X_train, y_train)
            best_params, best_cv, tuned = {}, float("nan"), False
        cv_pr_auc_mean, cv_pr_auc_std = (float("nan"), float("nan"))
        if report_train_cv_pr_auc:
            cv_pr_auc_mean, cv_pr_auc_std = _cv_pr_auc(pipe_full, X_train, y_train, desired_splits=cv_splits)
        y_pred = pipe_full.predict(X_eval)
        y_proba = _predict_proba_or_score(pipe_full, X_eval)
        _warn_if_degenerate_proba(y_proba, f"{model_name}[full]")
        row_full = {"dataset": dataset_path, "target": target_col, "eval_split": eval_value, "model": model_name, "variant": "full_tuned" if tuned else "full", "tuned": bool(tuned), "cv_best_score": best_cv, "cv_pr_auc_mean": cv_pr_auc_mean, "cv_pr_auc_std": cv_pr_auc_std, "best_params": str(best_params) if best_params else "", "n_train": int(len(y_train)), "n_eval": int(len(y_eval)), "n_features_raw": int(X_train.shape[1]), "n_features_selected": np.nan, "selected_cols": "", **_metrics(y_eval, y_pred, y_proba)}
        print(f"\n=== {model_name} [{row_full['variant']}] ({dataset_path}) ===")
        if tuned:
            print("Best CV score:", round(best_cv, 4), "Best params:", best_params)
        if report_train_cv_pr_auc:
            print("Train CV PR AUC:", row_full["cv_pr_auc_mean"], "+/-", row_full["cv_pr_auc_std"])
        print("\nClassification report:\n", classification_report(y_eval, y_pred, digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_eval, y_pred))
        if compute_importance:
            _perm_importance_raw(pipe_full, X_eval, y_eval, f"{model_name} [{row_full['variant']}]")
        results.append(row_full)

        if train_on_importance:
            pre_sel, _, _ = _build_preprocessor(X_train)
            base_pipe = Pipeline([("pre", pre_sel), ("clf", clone(estimator))])
            selected_cols = _select_cols_train_only_cv(base_pipe, X_train, y_train)
            X_train_sel = X_train[selected_cols].copy()
            X_eval_sel = X_eval[selected_cols].copy()
            pre2, _, _ = _build_preprocessor(X_train_sel)
            if is_tree:
                pipe_sel, best_params2, best_cv2, tuned2 = _grid_search_tree_if_enabled(estimator=estimator, preprocessor=pre2, X_tr=X_train_sel, y_tr=y_train, param_grid=grid)
            else:
                pipe_sel = Pipeline([("pre", pre2), ("clf", clone(estimator))])
                pipe_sel.fit(X_train_sel, y_train)
                best_params2, best_cv2, tuned2 = {}, float("nan"), False
            cv_pr_auc_mean2, cv_pr_auc_std2 = (float("nan"), float("nan"))
            if report_train_cv_pr_auc:
                cv_pr_auc_mean2, cv_pr_auc_std2 = _cv_pr_auc(pipe_sel, X_train_sel, y_train, desired_splits=cv_splits)
            y_pred2 = pipe_sel.predict(X_eval_sel)
            y_proba2 = _predict_proba_or_score(pipe_sel, X_eval_sel)
            _warn_if_degenerate_proba(y_proba2, f"{model_name}[selected]")
            row_sel = {"dataset": dataset_path, "target": target_col, "eval_split": eval_value, "model": model_name, "variant": "selected_tuned" if tuned2 else f"selected(top={importance_top_k},corr<{importance_corr_thr})", "tuned": bool(tuned2), "cv_best_score": best_cv2, "cv_pr_auc_mean": cv_pr_auc_mean2, "cv_pr_auc_std": cv_pr_auc_std2, "best_params": str(best_params2) if best_params2 else "", "n_train": int(len(y_train)), "n_eval": int(len(y_eval)), "n_features_raw": int(X_train.shape[1]), "n_features_selected": int(len(selected_cols)), "selected_cols": ",".join(selected_cols), **_metrics(y_eval, y_pred2, y_proba2)}
            print(f"\n=== {model_name} [{row_sel['variant']}] ({dataset_path}) ===")
            print("Selected cols:", selected_cols)
            if tuned2:
                print("Best CV score:", round(best_cv2, 4), "Best params:", best_params2)
            if report_train_cv_pr_auc:
                print("Train CV PR AUC:", row_sel["cv_pr_auc_mean"], "+/-", row_sel["cv_pr_auc_std"])
            print("\nClassification report:\n", classification_report(y_eval, y_pred2, digits=3))
            print("Confusion matrix:\n", confusion_matrix(y_eval, y_pred2))
            if compute_importance:
                _perm_importance_raw(pipe_sel, X_eval_sel, y_eval, f"{model_name} [{row_sel['variant']}]")
            results.append(row_sel)

    res_df = pd.DataFrame(results)[[
        "dataset", "target", "eval_split", "model", "variant", "tuned", "cv_best_score", "cv_pr_auc_mean", "cv_pr_auc_std", "best_params",
        "n_train", "n_eval", "n_features_raw", "n_features_selected",
        "accuracy", "f1_macro", "f1_lo", "f1_hi",
        "balanced_acc", "roc_auc", "auc_lo", "auc_hi",
        "pr_auc", "pr_auc_lo", "pr_auc_hi",
        "brier", "selected_cols",
    ]]
    print(f"\n=== Summary metrics on EVAL ({eval_value}) ===")
    print(res_df.round(4).to_string(index=False))

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        stem = run_name or f"legacy_{Path(dataset_path).stem}_{target_col}_{eval_value}"
        out_csv = save_path / f"{stem}.csv"
        res_df.to_csv(out_csv, index=False)
        best_row = res_df.sort_values(["pr_auc", "f1_macro"], ascending=[False, False]).iloc[0].to_dict()
        out_json = save_path / f"{stem}_best.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(best_row, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved results to: {out_csv}")
        print(f"[INFO] Saved best-run summary to: {out_json}")

    if results_tsv is not None:
        meta = dict(run_meta or {})
        meta.setdefault('timestamp_utc', datetime.utcnow().isoformat(timespec='seconds'))
        meta.setdefault('run_name', run_name or '')
        meta.setdefault('dataset_path', dataset_path)
        meta.setdefault('target_cli', target_col)
        meta.setdefault('eval_split_cli', eval_value)
        meta.setdefault('train_values_cli', list(train_values))
        log_path = append_results_tsv(results_tsv, meta, res_df)
        print(f"[INFO] Appended experiment rows to: {log_path}")

    return res_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy dataset builder + model training baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-datasets", help="Build legacy B0 / L / B0+L datasets.")
    p_build.add_argument("--detailed-labels-path", required=True)
    p_build.add_argument("--data-dir", required=True)
    p_build.add_argument("--out-dir", required=True)
    p_build.add_argument("--language", default="eng")
    p_build.add_argument("--iteration", default="3")

    p_train = subparsers.add_parser("train", help="Run legacy model pipeline.")
    p_train.add_argument("--dataset-path", required=True)
    p_train.add_argument("--target-col", required=True, choices=["Depression_label", "PTSD_label"])
    p_train.add_argument("--mode", default="experiment", choices=["experiment", "final", "custom"])
    p_train.add_argument("--train-values", nargs="*", default=None)
    p_train.add_argument("--eval-value", default=None)
    p_train.add_argument("--save-dir", default="artifacts")
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--results-tsv", default="artifacts/results.tsv")
    p_train.add_argument("--experiment-tag", default="")
    p_train.add_argument("--notes", default="")
    p_train.add_argument("--plot-after-run", action="store_true")
    p_train.add_argument("--plot-path", default=None)
    p_train.add_argument("--importance-top-k", type=int, default=5)
    p_train.add_argument("--importance-corr-thr", type=float, default=0.90)
    p_train.add_argument("--no-importance-selection", action="store_true")
    p_train.add_argument("--no-importance", action="store_true")
    p_train.add_argument("--no-tree-tuning", action="store_true")
    p_train.add_argument("--keep-demographics", action="store_true")

    p_plot = subparsers.add_parser("plot-results", help="Plot metric history from results.tsv.")
    p_plot.add_argument("--results-tsv", required=True)
    p_plot.add_argument("--out-path", required=True)
    p_plot.add_argument("--target", default=None)
    p_plot.add_argument("--dataset", default=None)
    p_plot.add_argument("--eval-split", default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "build-datasets":
        outputs = build_legacy_datasets(
            detailed_labels_path=args.detailed_labels_path,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            language=args.language,
            iteration=args.iteration,
        )
        print("Built datasets:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")
        return

    if args.command == "plot-results":
        out = plot_results_history(
            results_tsv=args.results_tsv,
            out_path=args.out_path,
            target=args.target,
            dataset=args.dataset,
            eval_split=args.eval_split,
        )
        print(f"Saved plot to: {out}")
        return

    if args.command == "train":
        if args.mode == "experiment":
            train_values = ("train",)
            eval_value = "dev"
        elif args.mode == "final":
            train_values = ("train", "dev")
            eval_value = "test"
        else:
            if not args.train_values or args.eval_value is None:
                raise ValueError("For mode=custom you must provide --train-values and --eval-value.")
            train_values = tuple(args.train_values)
            eval_value = args.eval_value

        run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}__{args.target_col}__{args.mode}"
        run_meta = {
            'run_id': run_id,
            'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds'),
            'experiment_tag': args.experiment_tag,
            'notes': args.notes,
            'mode': args.mode,
            'dataset_path': args.dataset_path,
            'target_cli': args.target_col,
            'train_values_cli': list(train_values),
            'eval_value_cli': eval_value,
            'importance_top_k_cli': args.importance_top_k,
            'importance_corr_thr_cli': args.importance_corr_thr,
            'importance_selection_cli': not args.no_importance_selection,
            'compute_importance_cli': not args.no_importance,
            'tree_tuning_cli': not args.no_tree_tuning,
            'drop_demographics_cli': not args.keep_demographics,
        }

        run_models_pipeline_more_models_tuned_prauc(
            dataset_path=args.dataset_path,
            target_col=args.target_col,
            train_values=train_values,
            eval_value=eval_value,
            compute_importance=not args.no_importance,
            train_on_importance=not args.no_importance_selection,
            importance_top_k=args.importance_top_k,
            importance_corr_thr=args.importance_corr_thr,
            tune_tree_models=not args.no_tree_tuning,
            drop_demographics=not args.keep_demographics,
            save_dir=args.save_dir,
            run_name=args.run_name or run_id,
            results_tsv=args.results_tsv,
            run_meta=run_meta,
        )

        if args.plot_after_run:
            default_plot = Path(args.save_dir) / f"metrics_progression_{args.target_col}_{eval_value}.svg"
            out = plot_results_history(
                results_tsv=args.results_tsv,
                out_path=args.plot_path or str(default_plot),
                target=args.target_col,
                eval_split=eval_value,
            )
            print(f"Saved plot to: {out}")
        return


if __name__ == "__main__":
    main()
