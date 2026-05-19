from __future__ import annotations

from pathlib import Path

from airest.speech import speech_attribute
from phonova.transcripts import TranscriptPreprocessor


def build_measures() -> dict:
    return speech_attribute.get_config(str(Path(speech_attribute.__file__).resolve()), "text.json")


def build_whisper_payload() -> dict:
    return {
        "segments": [
            {
                "id": 1,
                "speaker": "participant",
                "text": "hello there",
                "words": [
                    {"start": 0.0, "end": 0.2, "word": "hello"},
                    {"start": 0.2, "end": 0.4, "word": "there"},
                ],
                "phrases": [{"word_start": 0, "word_end": 1, "text": "hello there"}],
            },
            {
                "id": 2,
                "speaker": "participant",
                "text": "general kenobi",
                "words": [
                    {"start": 0.5, "end": 0.7, "word": "general"},
                    {"start": 0.7, "end": 0.9, "word": "kenobi"},
                ],
            },
            {
                "id": 3,
                "speaker": "interviewer",
                "text": "nice to meet you",
                "words": [
                    {"start": 1.0, "end": 1.2, "word": "nice"},
                    {"start": 1.2, "end": 1.3, "word": "to"},
                    {"start": 1.3, "end": 1.4, "word": "meet"},
                    {"start": 1.4, "end": 1.6, "word": "you"},
                ],
            },
        ]
    }


def test_transcript_preprocessor_merges_consecutive_speaker_segments() -> None:
    preprocessor = TranscriptPreprocessor(build_measures())
    prepared = preprocessor.prepare(build_whisper_payload(), whisper_turn_mode="speaker")

    assert prepared.source == "whisper"
    assert prepared.time_columns == ["start", "end"]
    assert len(prepared.filtered_json) == 8
    assert len(prepared.utterances) == 2
    assert prepared.utterances.iloc[0][preprocessor.measures["speaker_label"]] == "participant"
    assert prepared.utterances.iloc[0][preprocessor.measures["utterance_text"]] == "hello there general kenobi"


def test_transcript_preprocessor_keeps_segment_turns_when_requested() -> None:
    preprocessor = TranscriptPreprocessor(build_measures())
    prepared = preprocessor.prepare(build_whisper_payload(), whisper_turn_mode="segment")

    assert prepared.source == "whisper"
    assert len(prepared.utterances) == 3
