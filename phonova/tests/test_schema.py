from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phonova.speech.util import characteristics_util as cutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURES_PATH = PROJECT_ROOT / "phonova" / "speech" / "config" / "text.json"


def load_measures() -> dict:
    return json.loads(MEASURES_PATH.read_text(encoding="utf-8"))


def test_create_empty_dataframes_declares_expected_columns() -> None:
    measures = load_measures()
    _, turn_df, summ_df = cutil.create_empty_dataframes(measures)

    assert measures["phrase_count"] in turn_df.columns
    assert measures["first_person_sentiment_positive_vader"] in turn_df.columns
    assert measures["first_person_sentiment_negative_vader"] in turn_df.columns
    assert measures["first_person_sentiment_overall_vader"] in turn_df.columns

    for idx in range(1, 24):
        assert measures[f"vader_hist_bin_{idx:02d}"] in summ_df.columns

    for key in (
        "vader_hist_entropy",
        "vader_neg_tail_mass",
        "vader_pos_tail_mass",
        "vader_neutral_mass",
        "participant_utterance_count",
        "num_phrases",
        "num_one_word_phrases",
        "phrase_count_mean",
    ):
        assert measures[key] in summ_df.columns


def test_add_phrase_count_features_uses_valid_turns_only() -> None:
    measures = load_measures()
    word_df, turn_df, summ_df = cutil.create_empty_dataframes(measures)
    utterances = pd.DataFrame(
        [
            {
                measures["words_texts"]: ["I", "am", "fine"],
                measures["phrases_texts"]: ["I", "am fine"],
            },
            {
                measures["words_texts"]: ["yes"],
                measures["phrases_texts"]: ["yes"],
            },
            {
                measures["words_texts"]: ["this", "is", "better"],
                measures["phrases_texts"]: ["this is better"],
            },
        ]
    )

    _, turn_df, summ_df = cutil.add_phrase_count_features(
        [word_df, turn_df, summ_df],
        utterances,
        min_turn_length=2,
        measures=measures,
    )

    np.testing.assert_allclose(turn_df[measures["phrase_count"]].to_numpy(dtype=float), [2.0, 1.0])
    assert summ_df.loc[0, measures["num_phrases"]] == 3
    assert summ_df.loc[0, measures["num_one_word_phrases"]] == 1
    assert summ_df.loc[0, measures["phrase_count_mean"]] == 1.5


def test_extract_segment_texts_prefers_language_specific_fields() -> None:
    payload = {
        "segments": [
            {
                "id": 2,
                "start": 2.0,
                "role": "participant",
                "source_text_en": "english second",
                "text": "ukrainian second",
            },
            {
                "id": 1,
                "start": 1.0,
                "speaker": "participant",
                "source_text_en": "english first",
                "text": "ukrainian first",
            },
            {
                "id": 3,
                "start": 3.0,
                "speaker": "interviewer",
                "source_text_en": "question",
                "text": "question uk",
            },
        ]
    }

    assert cutil.extract_segment_texts_for_speaker(payload, "participant", source="whisper", language="en") == [
        "english first",
        "english second",
    ]
    assert cutil.extract_segment_texts_for_speaker(payload, "participant", source="whisper", language="uk") == [
        "ukrainian first",
        "ukrainian second",
    ]
