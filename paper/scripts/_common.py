from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "paper" / "paper_reproducibility_config.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or DEFAULT_CONFIG_PATH)


def get_feature_set(config: dict[str, Any], feature_set: str | None = None) -> list[str]:
    selected = feature_set or config.get("default_feature_set", "paper_82")
    feature_sets = config.get("feature_sets", {})
    if selected not in feature_sets:
        available = ", ".join(sorted(feature_sets))
        raise KeyError(f"Unknown feature set '{selected}'. Available: {available}")
    features = list(feature_sets[selected])
    duplicates = sorted({feature for feature in features if features.count(feature) > 1})
    if duplicates:
        raise ValueError(f"Feature set '{selected}' contains duplicate features: {duplicates}")
    return features


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def dataframe_sha256(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def metadata_columns(config: dict[str, Any], df: pd.DataFrame) -> list[str]:
    return existing_columns(df, config.get("metadata_columns", []))


def feature_group(feature: str) -> str:
    if feature.startswith("vader_hist_") or feature.startswith("vader_") or feature.startswith("sentiment_"):
        return "sentiment"
    if feature.startswith("first_person"):
        return "first_person"
    if feature.startswith("mattr_") or feature in {"prop_verb_past", "prop_function_words"}:
        return "lexical_style"
    if feature.startswith("word_coherence"):
        return "coherence"
    if "tangeniality" in feature or feature.startswith("turn_to_previous_speaker_turn_similarity"):
        return "coherence"
    if feature.startswith("semantic_perplexity"):
        return "semantic_perplexity"
    if feature in {
        "speech_length_minutes",
        "speech_length_words",
        "words_per_min",
        "syllables_per_min",
        "mean_pre_word_pause",
        "mean_pause_variability",
        "speech_percentage",
        "num_turns",
        "num_one_word_turns",
        "num_phrases",
        "num_one_word_phrases",
        "mean_turn_length_minutes",
        "mean_turn_length_words",
        "mean_phrases_per_turn",
        "mean_pre_turn_pause",
        "speaker_percentage",
        "num_interrupts",
        "participant_utterance_count",
    }:
        return "structure_timing"
    return "other"


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
