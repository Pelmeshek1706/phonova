from __future__ import annotations

import pandas as pd
import pytest

import phonova.analyzer as analyzer_module
from phonova import SpeechAnalyzer


class DummyBackend:
    backend_name = "gemma"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.sentence_encoder = None

    def supports_word_coherence(self) -> bool:
        return False

    def supports_phrase_coherence(self) -> bool:
        return False


def build_whisper_payload() -> dict:
    return {
        "segments": [
            {
                "speaker": "participant",
                "text": "hello there",
                "words": [
                    {"start": 0.0, "end": 0.1, "word": "hello"},
                    {"start": 0.1, "end": 0.2, "word": "there"},
                ],
            }
        ]
    }


def test_speech_analyzer_runs_simple_feature_flow_without_phonova_dependency(monkeypatch) -> None:
    monkeypatch.setattr(SpeechAnalyzer, "_prepare_language_resources", lambda self: None)
    monkeypatch.setattr(
        analyzer_module,
        "build_coherence_backend",
        lambda settings, measures: DummyBackend(settings),
    )

    analyzer = SpeechAnalyzer(language="ua", coherence_backend="gemma")
    words, turns, summary = analyzer.analyze_transcript(
        json_conf=build_whisper_payload(),
        option="simple",
        feature_groups=["structure"],
        speaker_label="participant",
        min_turn_length=1,
        whisper_turn_mode="speaker",
    )

    assert isinstance(words, pd.DataFrame)
    assert isinstance(turns, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)
    assert not turns.empty
    assert not summary.empty


def test_speech_analyzer_raises_when_coherence_backend_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(SpeechAnalyzer, "_prepare_language_resources", lambda self: None)
    monkeypatch.setattr(
        analyzer_module,
        "build_coherence_backend",
        lambda settings, measures: DummyBackend(settings),
    )

    analyzer = SpeechAnalyzer(language="ua", coherence_backend="gemma")
    with pytest.raises(RuntimeError, match="would otherwise return NaN coherence metrics"):
        analyzer.analyze_transcript(
            json_conf=build_whisper_payload(),
            option="coherence",
            feature_groups=["coherence"],
            speaker_label="participant",
            min_turn_length=1,
            whisper_turn_mode="speaker",
        )
