#!/usr/bin/env python3
"""Track A sentiment model evaluation.

This script converts the sentiment cells from `notebooks/airest_article.ipynb`
into a reusable command. It evaluates the notebook backends on a labeled
sentiment dataset and writes aggregate accuracy/F1 outputs. Transformer models
are optional and are loaded only when requested.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_TO_INT = {"negative": -1.0, "neutral": 0.0, "positive": 1.0, "mixed": 0.0}
INT_LABELS = [-1.0, 0.0, 1.0]
TARGET_NAMES = ["negative (-1)", "neutral (0)", "positive (+1)"]


def require_module(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing optional dependency '{module_name}'. Install {install_hint}.") from exc


def normalize_label(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in LABEL_TO_INT:
            return LABEL_TO_INT[key]
    try:
        return float(value)
    except Exception:
        return None


def load_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        if args.input.suffix.lower() == ".jsonl":
            return pd.read_json(args.input, lines=True)
        return pd.read_csv(args.input)
    datasets = require_module("datasets", "paper/track-a-requirements.txt")
    dataset = datasets.load_dataset(args.dataset_name, split=args.dataset_split)
    return dataset.to_pandas()


class TwitterXlmRobertaSentiment:
    def __init__(self, device: int):
        transformers = require_module("transformers", "paper/track-a-requirements.txt")
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.pipe = transformers.pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=device,
        )

    def polarity_scores(self, text: str) -> dict[str, float]:
        result = self.pipe(text)
        scores = {item["label"]: item["score"] for item in result[0]}
        return {
            "neg": scores.get("negative", 0.0),
            "neu": scores.get("neutral", 0.0),
            "pos": scores.get("positive", 0.0),
        }

    def major_label(self, text: str) -> tuple[str, float]:
        scores = self.polarity_scores(text)
        best = max(("neg", "neu", "pos"), key=scores.get)
        label = {"neg": "negative", "neu": "neutral", "pos": "positive"}[best]
        return label, LABEL_TO_INT[label]


class UkrRobertaCosmusSentiment:
    map_labels = {
        "LABEL_0": "mixed",
        "LABEL_1": "negative",
        "LABEL_2": "neutral",
        "LABEL_3": "positive",
    }

    def __init__(self, device: int):
        transformers = require_module("transformers", "paper/track-a-requirements.txt")
        hub = require_module("huggingface_hub", "paper/track-a-requirements.txt")
        safetensors = require_module("safetensors.torch", "paper/track-a-requirements.txt")
        repo_id = "YShynkarov/ukr-roberta-cosmus-sentiment"
        weights_path = hub.hf_hub_download(repo_id=repo_id, filename="ukrroberta_cosmus_sentiment.safetensors")
        config = transformers.RobertaConfig.from_pretrained("youscan/ukr-roberta-base", num_labels=4)
        tokenizer = transformers.RobertaTokenizer.from_pretrained("youscan/ukr-roberta-base")
        model = transformers.RobertaForSequenceClassification(config)
        model.load_state_dict(safetensors.load_file(weights_path))
        model.eval()
        self.pipe = transformers.pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            truncation=True,
        )

    def polarity_scores(self, text: str) -> dict[str, float]:
        result = self.pipe(text)
        return {self.map_labels[item["label"]]: item["score"] for item in result[0]}

    def major_label(self, text: str) -> tuple[str, float]:
        scores = self.polarity_scores(text)
        best = max(scores, key=scores.get)
        return best, LABEL_TO_INT[best]


class GenericHfPipelineSentiment:
    def __init__(self, model_name: str, device: int):
        transformers = require_module("transformers", "paper/track-a-requirements.txt")
        self.pipe = transformers.pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=device,
        )

    def major_label(self, text: str) -> tuple[str, float]:
        result = self.pipe(text)
        items = result[0] if result and isinstance(result[0], list) else result
        scores: dict[str, float] = {}
        for item in items:
            label = str(item["label"]).lower()
            if "neg" in label or label in {"label_0", "0"}:
                scores["negative"] = item["score"]
            elif "neu" in label or label in {"label_1", "1"}:
                scores["neutral"] = item["score"]
            elif "pos" in label or label in {"label_2", "2"}:
                scores["positive"] = item["score"]
        if not scores:
            label = str(items[0]["label"]).lower()
            scores[label] = float(items[0]["score"])
        best = max(scores, key=scores.get)
        canonical = best if best in LABEL_TO_INT else "neutral"
        return canonical, LABEL_TO_INT[canonical]


class VaderTranslatedSentiment:
    def __init__(self):
        vader = require_module("vaderSentiment.vaderSentiment", "paper/track-a-requirements.txt")
        self.analyzer = vader.SentimentIntensityAnalyzer()

    def major_label(self, text: str) -> tuple[str, float]:
        scores = self.analyzer.polarity_scores(text)
        top = max(("neg", "neu", "pos"), key=lambda key: scores[key])
        label = {"neg": "negative", "neu": "neutral", "pos": "positive"}[top]
        return label, LABEL_TO_INT[label]


def build_analyzer(args: argparse.Namespace):
    if args.backend == "twitter-xlm-roberta":
        return TwitterXlmRobertaSentiment(args.device)
    if args.backend == "ukr-roberta-cosmus":
        return UkrRobertaCosmusSentiment(args.device)
    if args.backend == "vader-translated":
        return VaderTranslatedSentiment()
    if args.backend == "hf-pipeline":
        if not args.hf_model:
            raise SystemExit("--hf-model is required for --backend hf-pipeline")
        return GenericHfPipelineSentiment(args.hf_model, args.device)
    raise SystemExit(f"Unsupported analyzer backend: {args.backend}")


def maybe_progress(values, label: str):
    try:
        from tqdm import tqdm
        return tqdm(values, desc=label)
    except ImportError:
        return values


def evaluate_predictions(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = require_module("sklearn.metrics", "paper/requirements.txt")
    y_true = df["gold_label_numeric"].astype(float)
    y_pred = df["pred_label_numeric"].astype(float)
    metric_row = {
        "backend": args.backend,
        "model_label": args.model_label or args.backend,
        "n": int(len(df)),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "macro_f1": float(metrics.f1_score(y_true, y_pred, labels=INT_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true, y_pred, labels=INT_LABELS, average="weighted", zero_division=0)),
    }
    report = metrics.classification_report(
        y_true,
        y_pred,
        labels=INT_LABELS,
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )
    report["confusion_matrix"] = metrics.confusion_matrix(y_true, y_pred, labels=INT_LABELS).tolist()
    return pd.DataFrame([metric_row]), report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="CSV or JSONL sentiment dataset. If omitted, load HF dataset.")
    parser.add_argument("--dataset-name", default="YShynkarov/COSMUS")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--backend",
        choices=["ukr-roberta-cosmus", "twitter-xlm-roberta", "vader-translated", "hf-pipeline", "precomputed"],
        required=True,
    )
    parser.add_argument("--hf-model", default=None, help="Generic Hugging Face model for --backend hf-pipeline.")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--text-column", default="document_content")
    parser.add_argument("--translated-text-column", default="translated_text")
    parser.add_argument("--label-column", default="annotator_sentiment")
    parser.add_argument("--prediction-column", default=None, help="Numeric prediction column for --backend precomputed.")
    parser.add_argument("--exclude-mixed", action="store_true", default=True)
    parser.add_argument("--include-mixed", dest="exclude_mixed", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true", help="Validate input columns without loading a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_input(args)
    if args.limit:
        df = df.head(args.limit).copy()
    if args.label_column not in df.columns:
        raise SystemExit(f"Input is missing label column: {args.label_column}")
    text_column = args.translated_text_column if args.backend == "vader-translated" else args.text_column
    if args.backend != "precomputed" and text_column not in df.columns:
        raise SystemExit(f"Input is missing text column for {args.backend}: {text_column}")
    if args.backend == "precomputed" and not args.prediction_column:
        raise SystemExit("--prediction-column is required for --backend precomputed")
    if args.backend == "precomputed" and args.prediction_column not in df.columns:
        raise SystemExit(f"Input is missing prediction column: {args.prediction_column}")

    out = df.copy()
    out["gold_label_numeric"] = out[args.label_column].map(normalize_label)
    out = out.dropna(subset=["gold_label_numeric"]).copy()
    if args.exclude_mixed and args.label_column in out.columns:
        out = out[out[args.label_column].astype(str).str.lower() != "mixed"].copy()
    if args.dry_run:
        (args.output_dir / "sentiment_validation_meta.json").write_text(
            json.dumps({"rows_after_label_filter": int(len(out)), "backend": args.backend}, indent=2),
            encoding="utf-8",
        )
        return

    if args.backend == "precomputed":
        out["pred_label_numeric"] = out[args.prediction_column].map(normalize_label)
        out["pred_label_text"] = out["pred_label_numeric"].map({-1.0: "negative", 0.0: "neutral", 1.0: "positive"})
    else:
        analyzer = build_analyzer(args)
        labels, values = [], []
        for text in maybe_progress(out[text_column].fillna("").astype(str).tolist(), args.backend):
            label, value = analyzer.major_label(text)
            labels.append(label)
            values.append(value)
        out["pred_label_text"] = labels
        out["pred_label_numeric"] = values

    out = out.dropna(subset=["gold_label_numeric", "pred_label_numeric"]).copy()
    metrics_df, report = evaluate_predictions(out, args)
    out.to_csv(args.output_dir / "sentiment_predictions.csv", index=False)
    metrics_df.to_csv(args.output_dir / "sentiment_metrics.csv", index=False)
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
