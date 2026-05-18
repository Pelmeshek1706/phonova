import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_local_coherence():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "openwillis"
        / "speech"
        / "util"
        / "speech"
        / "coherence.py"
    )

    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    module_name = "local_openwillis_coherence_test"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeEncoder:
    def __init__(self, embeddings):
        self.embeddings = {
            key: np.asarray(value, dtype=np.float32) for key, value in embeddings.items()
        }

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        return np.asarray([self.embeddings[text] for text in texts], dtype=np.float32)


def build_measures():
    return {
        "utterance_text": "utterance_text",
        "words_texts": "words_texts",
        "phrases_texts": "phrases_texts",
        "speaker_label": "speaker_label",
        "sentence_tangeniality1": "first_order_sentence_tangeniality",
        "sentence_tangeniality2": "second_order_sentence_tangeniality",
        "perplexity": "semantic_perplexity",
        "perplexity_5": "semantic_perplexity_5",
        "perplexity_11": "semantic_perplexity_11",
        "perplexity_15": "semantic_perplexity_15",
        "turn_to_turn_tangeniality": "turn_to_turn_tangeniality",
        "turn_to_previous_speaker_turn_similarity": "turn_to_previous_speaker_turn_similarity",
    }


def test_turn_to_previous_speaker_turn_similarity_uses_full_dialogue_context(monkeypatch):
    coherence = load_local_coherence()
    monkeypatch.setattr(
        coherence,
        "calculate_phrase_tangeniality",
        lambda *args, **kwargs: (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    )

    measures = build_measures()
    participant_turns = pd.DataFrame(
        [
            {"utterance_text": "a1", "words_texts": ["a", "1"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "a2", "words_texts": ["a", "2"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "a3", "words_texts": ["a", "3"], "phrases_texts": [], "speaker_label": "participant"},
        ]
    )
    dialogue_turns = pd.DataFrame(
        [
            {"utterance_text": "q1", "words_texts": ["q", "1"], "phrases_texts": [], "speaker_label": "interviewer"},
            {"utterance_text": "a1", "words_texts": ["a", "1"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "q2", "words_texts": ["q", "2"], "phrases_texts": [], "speaker_label": "interviewer"},
            {"utterance_text": "a2", "words_texts": ["a", "2"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "q3", "words_texts": ["q", "3"], "phrases_texts": [], "speaker_label": "interviewer"},
            {"utterance_text": "a3", "words_texts": ["a", "3"], "phrases_texts": [], "speaker_label": "participant"},
        ]
    )
    encoder = FakeEncoder(
        {
            "q1": [1.0, 0.0],
            "a1": [1.0, 0.0],
            "q2": [0.0, 1.0],
            "a2": [0.0, 1.0],
            "q3": [-1.0, 0.0],
            "a3": [1.0, 0.0],
        }
    )
    turn_df = pd.DataFrame(index=range(len(participant_turns)))

    result = coherence.calculate_turn_coherence(
        participant_turns,
        turn_df,
        min_coherence_turn_length=2,
        speaker_label="participant",
        sentence_encoder=encoder,
        lm_model=None,
        tokenizer=None,
        measures=measures,
        dialogue_utterances_filtered=dialogue_turns,
    )

    np.testing.assert_allclose(
        result[measures["turn_to_turn_tangeniality"]].to_numpy(dtype=float),
        np.array([np.nan, 0.0, 0.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result[measures["turn_to_previous_speaker_turn_similarity"]].to_numpy(dtype=float),
        np.array([1.0, 1.0, -1.0]),
        equal_nan=True,
    )


def test_turn_to_previous_speaker_turn_similarity_uses_its_own_short_turn_rule(monkeypatch):
    coherence = load_local_coherence()
    monkeypatch.setattr(
        coherence,
        "calculate_phrase_tangeniality",
        lambda *args, **kwargs: (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    )

    measures = build_measures()
    participant_turns = pd.DataFrame(
        [
            {"utterance_text": "a1", "words_texts": ["a", "1"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "a2", "words_texts": ["a", "2"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "a3_short", "words_texts": ["a"], "phrases_texts": [], "speaker_label": "participant"},
        ]
    )
    dialogue_turns = pd.DataFrame(
        [
            {"utterance_text": "q1", "words_texts": ["q", "1"], "phrases_texts": [], "speaker_label": "interviewer"},
            {"utterance_text": "a1", "words_texts": ["a", "1"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "q2_short", "words_texts": ["q"], "phrases_texts": [], "speaker_label": "interviewer"},
            {"utterance_text": "a2", "words_texts": ["a", "2"], "phrases_texts": [], "speaker_label": "participant"},
            {"utterance_text": "q3", "words_texts": ["q", "3"], "phrases_texts": [], "speaker_label": "interviewer"},
            {"utterance_text": "a3_short", "words_texts": ["a"], "phrases_texts": [], "speaker_label": "participant"},
        ]
    )
    encoder = FakeEncoder(
        {
            "q1": [1.0, 0.0],
            "a1": [1.0, 0.0],
            "q2_short": [0.0, 1.0],
            "a2": [0.0, 1.0],
            "q3": [1.0, 0.0],
            "a3_short": [1.0, 0.0],
        }
    )
    turn_df = pd.DataFrame(index=range(len(participant_turns)))

    result = coherence.calculate_turn_coherence(
        participant_turns,
        turn_df,
        min_coherence_turn_length=2,
        speaker_label="participant",
        sentence_encoder=encoder,
        lm_model=None,
        tokenizer=None,
        measures=measures,
        dialogue_utterances_filtered=dialogue_turns,
    )

    np.testing.assert_allclose(
        result[measures["turn_to_turn_tangeniality"]].to_numpy(dtype=float),
        np.array([np.nan, 0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result[measures["turn_to_previous_speaker_turn_similarity"]].to_numpy(dtype=float),
        np.array([1.0, 1.0, 1.0]),
        equal_nan=True,
    )
