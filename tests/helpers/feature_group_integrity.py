import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENWILLIS_SRC = REPO_ROOT / "openwillis" / "openwillis-speech" / "src"
TEST_DATA_ROOT = REPO_ROOT / "tests" / "data"
LANGUAGE_ALIASES = {
    "en": "eng",
    "eng": "eng",
    "ua": "ukr",
    "uk": "ukr",
    "ukr": "ukr",
}
CANONICAL_PIPELINE_LANGUAGES = {
    "eng": "en",
    "ukr": "uk",
}
TEST_FIXTURE_ROOTS = {
    "eng": TEST_DATA_ROOT / "role_labeled_whisper_like_stub_batch_eng_26-03-2026",
    "ukr": TEST_DATA_ROOT / "role_labeled_whisper_like_stub_batch_ukr_26-03-2026",
}
PARTICIPANT_LABEL = "participant"
MATTR_COLUMNS = ["mattr_5", "mattr_10", "mattr_25", "mattr_50", "mattr_100"]
WORD_COHERENCE_COLUMNS = [
    ("word_coherence", "word_coherence_mean", "word_coherence_var"),
    ("word_coherence_5", "word_coherence_5_mean", "word_coherence_5_var"),
    ("word_coherence_10", "word_coherence_10_mean", "word_coherence_10_var"),
]
TURN_OPTIONAL_COLUMNS = [
    ("turn_to_turn_tangeniality", "turn_to_turn_tangeniality_mean", "turn_to_turn_tangeniality_var"),
    ("semantic_perplexity", "semantic_perplexity_mean", "semantic_perplexity_var"),
    ("semantic_perplexity_5", "semantic_perplexity_5_mean", "semantic_perplexity_5_var"),
    ("semantic_perplexity_11", "semantic_perplexity_11_mean", "semantic_perplexity_11_var"),
    ("semantic_perplexity_15", "semantic_perplexity_15_mean", "semantic_perplexity_15_var"),
    (
        "first_order_sentence_tangeniality",
        "first_order_sentence_tangeniality_mean",
        "first_order_sentence_tangeniality_var",
    ),
    (
        "second_order_sentence_tangeniality",
        "second_order_sentence_tangeniality_mean",
        "second_order_sentence_tangeniality_var",
    ),
]
CRITICAL_SENTIMENT_SUMMARY_COLUMNS = [
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_overall",
    "sentiment_vader_pos",
    "sentiment_vader_neg",
    "sentiment_vader_neu",
    "sentiment_vader_overall",
    "mattr_5",
    "mattr_10",
    "mattr_25",
    "mattr_50",
    "mattr_100",
    "first_person_percentage",
    "first_person_sentiment_positive",
    "first_person_sentiment_negative",
    "first_person_sentiment_overall",
    "first_person_sentiment_positive_vader",
    "first_person_sentiment_negative_vader",
    "first_person_sentiment_overall_vader",
    "prop_verb_past",
    "prop_function_words",
]
CRITICAL_WORD_COHERENCE_SUMMARY_COLUMNS = [
    "word_coherence_mean",
    "word_coherence_var",
    "word_coherence_5_mean",
    "word_coherence_5_var",
    "word_coherence_10_mean",
    "word_coherence_10_var",
    *[f"word_coherence_variability_{k}_mean" for k in range(2, 11)],
    *[f"word_coherence_variability_{k}_var" for k in range(2, 11)],
]


def _normalize_language(language):
    normalized = str(language).strip().lower()
    if normalized not in LANGUAGE_ALIASES:
        raise KeyError(f"Unsupported feature-group language: {language}")
    return LANGUAGE_ALIASES[normalized]


def _fixture_path(language):
    canonical_language = _normalize_language(language)
    tracked_path = TEST_FIXTURE_ROOTS[canonical_language] / "300.json"
    if not tracked_path.exists():
        raise FileNotFoundError(f"Missing feature-group fixture for {canonical_language}: {tracked_path}")
    return tracked_path


def _drop_stubbed_runtime_modules():
    for module_name in ("nltk", "spacy"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if getattr(module, "__file__", None) is None:
            sys.modules.pop(module_name, None)


def _force_cpu_runtime():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import torch
    except Exception:
        return

    try:
        torch.cuda.is_available = lambda: False
    except Exception:
        pass

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None:
        try:
            mps_backend.is_available = lambda: False
        except Exception:
            pass


def _load_speech_characteristics():
    _drop_stubbed_runtime_modules()
    _force_cpu_runtime()
    if str(OPENWILLIS_SRC) not in sys.path:
        sys.path.insert(0, str(OPENWILLIS_SRC))
    importlib.invalidate_caches()
    module = importlib.import_module("openwillis.speech.speech_attribute")
    return module.speech_characteristics


@lru_cache(maxsize=1)
def _load_speech_attribute_module():
    _drop_stubbed_runtime_modules()
    _force_cpu_runtime()
    if str(OPENWILLIS_SRC) not in sys.path:
        sys.path.insert(0, str(OPENWILLIS_SRC))
    importlib.invalidate_caches()
    return importlib.import_module("openwillis.speech.speech_attribute")


@lru_cache(maxsize=None)
def _load_payload_cached(language):
    return json.loads(_fixture_path(language).read_text())


def load_feature_group_payload(language):
    return json.loads(json.dumps(_load_payload_cached(language)))


@lru_cache(maxsize=None)
def _run_feature_group_case_cached(language, feature_groups, option):
    canonical_language = _normalize_language(language)
    source_payload = json.loads(json.dumps(_load_payload_cached(canonical_language)))
    payload = json.loads(json.dumps(source_payload))
    speech_characteristics = _load_speech_characteristics()
    words, turns, summary = speech_characteristics(
        payload,
        language=CANONICAL_PIPELINE_LANGUAGES[canonical_language],
        speaker_label=PARTICIPANT_LABEL,
        option=option,
        feature_groups=list(feature_groups),
        whisper_turn_mode="speaker",
    )
    return source_payload, words, turns, summary


def load_feature_group_case(language, feature_groups, option):
    canonical_language = _normalize_language(language)
    payload, words, turns, summary = _run_feature_group_case_cached(canonical_language, tuple(feature_groups), option)
    return {
        "language": canonical_language,
        "json": json.loads(json.dumps(payload)),
        "words": words.copy(deep=True),
        "turns": turns.copy(deep=True),
        "summary": summary.copy(deep=True),
    }


def _summary_row(summary_df):
    assert len(summary_df) == 1
    return summary_df.iloc[0]


def _numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def _numeric_value(value):
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _assert_summary_notna(summary, columns):
    missing = [column for column in columns if pd.isna(summary[column])]
    assert not missing, f"Unexpected NaN summary columns: {missing}"


def _valid_segment_words(segment):
    return [word for word in segment.get("words", []) if "start" in word and "end" in word]


def _merge_speaker_turns(json_payload):
    merged_turns = []
    current = None

    for segment in json_payload.get("segments", []):
        words = _valid_segment_words(segment)
        if not words:
            continue

        speaker = segment.get("speaker")
        segment_start = float(words[0]["start"])
        segment_end = float(words[-1]["end"])

        if current is not None and current["speaker"] == speaker:
            current["words"].extend(words)
            current["end"] = segment_end
        else:
            if current is not None:
                merged_turns.append(current)
            current = {
                "speaker": speaker,
                "start": segment_start,
                "end": segment_end,
                "words": list(words),
            }

    if current is not None:
        merged_turns.append(current)

    return merged_turns


def _participant_stats(json_payload):
    merged_turns = [
        turn for turn in _merge_speaker_turns(json_payload) if turn["speaker"] == PARTICIPANT_LABEL
    ]
    participant_segments = [
        segment
        for segment in json_payload.get("segments", [])
        if segment.get("speaker") == PARTICIPANT_LABEL and "start" in segment and "end" in segment
    ]
    file_length = max((float(segment.get("end", 0.0) or 0.0) for segment in json_payload.get("segments", [])), default=0.0)
    segment_speech_minutes = sum(
        float(segment.get("end", 0.0) or 0.0) - float(segment.get("start", 0.0) or 0.0)
        for segment in participant_segments
    ) / 60.0

    return {
        "participant_word_count": sum(len(turn["words"]) for turn in merged_turns),
        "participant_turn_count": len(merged_turns),
        "participant_turn_minutes": sum((turn["end"] - turn["start"]) for turn in merged_turns) / 60.0,
        "participant_one_word_turn_count": sum(1 for turn in merged_turns if len(turn["words"]) == 1),
        "participant_segment_minutes": segment_speech_minutes,
        "file_length_minutes": file_length / 60.0,
    }


@lru_cache(maxsize=None)
def _pipeline_structural_stats(language):
    canonical_language = _normalize_language(language)
    speech_attribute = _load_speech_attribute_module()
    payload = json.loads(json.dumps(_load_payload_cached(canonical_language)))
    measures = speech_attribute.get_config(str(speech_attribute.__file__), "text.json")
    filtered_words, utterances = speech_attribute.filter_whisper(
        payload,
        measures,
        whisper_turn_mode="speaker",
    )
    participant_utterances = utterances[utterances[measures["speaker_label"]] == PARTICIPANT_LABEL].copy()
    one_word_turns = 0
    if not participant_utterances.empty:
        one_word_turns = int(
            participant_utterances[measures["words_ids"]].apply(lambda items: len(items) == 1).sum()
        )

    return {
        "participant_word_count": len([item for item in filtered_words if item.get("speaker") == PARTICIPANT_LABEL]),
        "participant_turn_count": int(len(participant_utterances)),
        "participant_one_word_turn_count": one_word_turns,
    }


def _assert_optional_summary_relationship(summary, df, raw_col, mean_col, var_col):
    series = _numeric_series(df, raw_col).dropna()
    if len(series) == 0:
        assert pd.isna(summary[mean_col])
        assert pd.isna(summary[var_col])
        return

    assert _numeric_value(summary[mean_col]) == pytest.approx(float(series.mean()))
    if len(series) == 1:
        assert pd.isna(summary[var_col]) or _numeric_value(summary[var_col]) == pytest.approx(0.0)
    else:
        assert _numeric_value(summary[var_col]) == pytest.approx(float(series.var()))


def assert_structural_group(language):
    case = load_feature_group_case(language, feature_groups=("pause", "repetition"), option="simple")
    summary = _summary_row(case["summary"])
    turns = case["turns"]
    words = case["words"]
    stats = _participant_stats(case["json"])
    pipeline_stats = _pipeline_structural_stats(language)

    assert len(words) == pipeline_stats["participant_word_count"]
    assert len(turns) == pipeline_stats["participant_turn_count"]
    assert int(_numeric_value(summary["speech_length_words"])) == pipeline_stats["participant_word_count"]
    assert int(_numeric_value(summary["num_turns"])) == pipeline_stats["participant_turn_count"]
    assert int(_numeric_value(summary["num_one_word_turns"])) == pipeline_stats["participant_one_word_turn_count"]
    assert int(_numeric_value(summary["num_interrupts"])) == int(turns["interrupt_flag"].fillna(False).astype(bool).sum())

    assert _numeric_value(summary["file_length"]) == pytest.approx(stats["file_length_minutes"])
    assert _numeric_value(summary["speech_length_minutes"]) == pytest.approx(stats["participant_turn_minutes"])
    assert _numeric_value(summary["speaker_percentage"]) == pytest.approx(
        100.0 * stats["participant_segment_minutes"] / stats["file_length_minutes"]
    )
    assert _numeric_value(summary["speech_percentage"]) == pytest.approx(
        100.0 * stats["participant_turn_minutes"] / stats["file_length_minutes"]
    )
    assert _numeric_value(summary["speech_length_words"]) == pytest.approx(float(_numeric_series(turns, "turn_length_words").sum()))
    assert _numeric_value(summary["speech_length_minutes"]) == pytest.approx(
        float(_numeric_series(turns, "turn_length_minutes").sum())
    )
    assert _numeric_value(summary["mean_turn_length_words"]) == pytest.approx(
        float(_numeric_series(turns, "turn_length_words").mean())
    )
    assert _numeric_value(summary["mean_turn_length_minutes"]) == pytest.approx(
        float(_numeric_series(turns, "turn_length_minutes").mean())
    )
    assert _numeric_value(summary["mean_pre_turn_pause"]) == pytest.approx(
        float(_numeric_series(turns, "pre_turn_pause").dropna().mean())
    )
    assert _numeric_value(summary["mean_pre_word_pause"]) == pytest.approx(
        float(_numeric_series(words, "pre_word_pause").dropna().mean())
    )
    assert _numeric_value(summary["mean_pause_variability"]) == pytest.approx(
        float(_numeric_series(words, "pre_word_pause").dropna().var())
    )
    assert _numeric_value(summary["words_per_min"]) == pytest.approx(
        _numeric_value(summary["speech_length_words"]) / _numeric_value(summary["speech_length_minutes"])
    )
    assert _numeric_value(summary["word_repeat_percentage"]) == pytest.approx(
        float(_numeric_series(turns, "word_repeat_percentage").mean())
    )
    assert _numeric_value(summary["phrase_repeat_percentage"]) == pytest.approx(
        float(_numeric_series(turns, "phrase_repeat_percentage").mean())
    )

    assert int(words["pre_word_pause"].isna().sum()) == int(_numeric_value(summary["num_turns"]))
    assert int(turns["pre_turn_pause"].isna().sum()) == 1
    assert int(turns["mean_pause_length"].isna().sum()) == int(_numeric_value(summary["num_one_word_turns"]))
    assert int(turns["pause_variability"].isna().sum()) == int(_numeric_value(summary["num_one_word_turns"]))


def assert_sentiment_and_first_person_group(language):
    case = load_feature_group_case(language, feature_groups=("sentiment", "first_person"), option="simple")
    summary = _summary_row(case["summary"])
    turns = case["turns"]
    words = case["words"]
    stats = _participant_stats(case["json"])

    pipeline_stats = _pipeline_structural_stats(language)

    assert len(words) == pipeline_stats["participant_word_count"]
    assert len(turns) == pipeline_stats["participant_turn_count"]
    assert words["part_of_speech"].notna().all()
    _assert_summary_notna(summary, CRITICAL_SENTIMENT_SUMMARY_COLUMNS)

    for column in ("sentiment_pos", "sentiment_neg", "sentiment_neu"):
        series = _numeric_series(turns, column)
        assert series.notna().all()
        assert ((series >= 0.0) & (series <= 1.0)).all()

    for column in ("sentiment_vader_pos", "sentiment_vader_neg", "sentiment_vader_neu"):
        series = _numeric_series(turns, column)
        assert series.notna().all()
        assert ((series >= 0.0) & (series <= 1.0)).all()

    turn_sentiment_total = (
        _numeric_series(turns, "sentiment_pos")
        + _numeric_series(turns, "sentiment_neg")
        + _numeric_series(turns, "sentiment_neu")
    )
    vader_turn_total = (
        _numeric_series(turns, "sentiment_vader_pos")
        + _numeric_series(turns, "sentiment_vader_neg")
        + _numeric_series(turns, "sentiment_vader_neu")
    )

    assert float((turn_sentiment_total - 1.0).abs().max()) < 1e-6
    assert float((vader_turn_total - 1.0).abs().max()) < 2e-3
    assert (
        _numeric_value(summary["sentiment_pos"])
        + _numeric_value(summary["sentiment_neg"])
        + _numeric_value(summary["sentiment_neu"])
    ) == pytest.approx(1.0)
    assert _numeric_value(summary["sentiment_overall"]) == pytest.approx(
        _numeric_value(summary["sentiment_pos"]) - _numeric_value(summary["sentiment_neg"])
    )
    assert (
        _numeric_value(summary["sentiment_vader_pos"])
        + _numeric_value(summary["sentiment_vader_neg"])
        + _numeric_value(summary["sentiment_vader_neu"])
    ) == pytest.approx(1.0, abs=2e-3)

    for column in (
        "first_person_percentage",
        "first_person_sentiment_positive",
        "first_person_sentiment_negative",
        "first_person_sentiment_positive_vader",
        "first_person_sentiment_negative_vader",
    ):
        series = _numeric_series(turns, column)
        assert (series.dropna() >= 0.0).all()

    assert 0.0 <= _numeric_value(summary["first_person_percentage"]) <= 100.0
    prop_verb_past = _numeric_series(pd.DataFrame([summary]), "prop_verb_past").iloc[0]
    prop_function_words = _numeric_series(pd.DataFrame([summary]), "prop_function_words").iloc[0]
    assert pd.isna(prop_verb_past) or 0.0 <= float(prop_verb_past) <= 1.0
    assert pd.isna(prop_function_words) or 0.0 <= float(prop_function_words) <= 1.0


def assert_lexical_diversity_group(language):
    case = load_feature_group_case(language, feature_groups=("sentiment", "first_person"), option="simple")
    summary = _summary_row(case["summary"])
    turns = case["turns"]

    summary_values = [_numeric_series(pd.DataFrame([summary]), column).iloc[0] for column in MATTR_COLUMNS]
    assert all(pd.notna(value) for value in summary_values)
    assert all(0.0 <= value <= 1.0 for value in summary_values)
    assert summary_values == sorted(summary_values, reverse=True)

    for column in MATTR_COLUMNS:
        series = _numeric_series(turns, column)
        assert series.notna().any()
        assert ((series.dropna() >= 0.0) & (series.dropna() <= 1.0)).all()


def assert_coherence_and_perplexity_group(language):
    case = load_feature_group_case(language, feature_groups=("coherence",), option="coherence")
    summary = _summary_row(case["summary"])
    turns = case["turns"]
    words = case["words"]
    stats = _participant_stats(case["json"])

    pipeline_stats = _pipeline_structural_stats(language)

    assert len(words) == pipeline_stats["participant_word_count"]
    assert _numeric_value(summary["file_length"]) == pytest.approx(stats["file_length_minutes"])
    assert _numeric_value(summary["speaker_percentage"]) == pytest.approx(
        100.0 * stats["participant_segment_minutes"] / stats["file_length_minutes"]
    )
    _assert_summary_notna(summary, CRITICAL_WORD_COHERENCE_SUMMARY_COLUMNS)

    for raw_col, mean_col, var_col in WORD_COHERENCE_COLUMNS:
        _assert_optional_summary_relationship(summary, words, raw_col, mean_col, var_col)

    for k in range(2, 11):
        raw_col = f"word_coherence_variability_{k}"
        _assert_optional_summary_relationship(summary, words, raw_col, f"{raw_col}_mean", f"{raw_col}_var")

    for raw_col, mean_col, var_col in TURN_OPTIONAL_COLUMNS:
        _assert_optional_summary_relationship(summary, turns, raw_col, mean_col, var_col)

    assert words["word_coherence"].notna().sum() > 0
    assert words["word_coherence_5"].notna().sum() > 0
    assert words["word_coherence_10"].notna().sum() > 0
    assert int(words["word_coherence_10"].isna().sum()) >= int(words["word_coherence_5"].isna().sum())


def assert_cross_level_relationships(language):
    pause_case = load_feature_group_case(language, feature_groups=("pause", "repetition"), option="simple")
    sentiment_case = load_feature_group_case(language, feature_groups=("sentiment", "first_person"), option="simple")
    coherence_case = load_feature_group_case(language, feature_groups=("coherence",), option="coherence")

    pause_summary = _summary_row(pause_case["summary"])
    sentiment_summary = _summary_row(sentiment_case["summary"])
    coherence_summary = _summary_row(coherence_case["summary"])
    stats = _participant_stats(pause_case["json"])

    pipeline_stats = _pipeline_structural_stats(language)

    assert len(pause_case["words"]) == pipeline_stats["participant_word_count"]
    assert len(sentiment_case["words"]) == pipeline_stats["participant_word_count"]
    assert len(coherence_case["words"]) == pipeline_stats["participant_word_count"]
    assert len(pause_case["turns"]) == pipeline_stats["participant_turn_count"]
    assert len(sentiment_case["turns"]) == pipeline_stats["participant_turn_count"]
    assert int(_numeric_value(pause_summary["num_turns"])) == pipeline_stats["participant_turn_count"]

    assert _numeric_value(pause_summary["file_length"]) == pytest.approx(_numeric_value(sentiment_summary["file_length"]))
    assert _numeric_value(pause_summary["file_length"]) == pytest.approx(_numeric_value(coherence_summary["file_length"]))
    assert _numeric_value(pause_summary["speaker_percentage"]) == pytest.approx(
        _numeric_value(sentiment_summary["speaker_percentage"])
    )
    assert _numeric_value(pause_summary["speaker_percentage"]) == pytest.approx(
        _numeric_value(coherence_summary["speaker_percentage"])
    )

    turn_word_rate = _numeric_series(pause_case["turns"], "turn_length_words") / _numeric_series(
        pause_case["turns"], "turn_length_minutes"
    )
    assert float((_numeric_series(pause_case["turns"], "words_per_min") - turn_word_rate).abs().max()) < 1e-6
    assert int(pause_case["words"]["pre_word_pause"].isna().sum()) == int(_numeric_value(pause_summary["num_turns"]))
    assert int(_numeric_value(pause_summary["num_interrupts"])) == int(
        pause_case["turns"]["interrupt_flag"].fillna(False).astype(bool).sum()
    )
