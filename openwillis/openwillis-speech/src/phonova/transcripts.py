"""Transcript preprocessing utilities for the refinement API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from openwillis.speech import speech_attribute as legacy_speech
from openwillis.speech.util import characteristics_util as legacy_cutil

from .config import normalize_turn_mode


@dataclass(slots=True)
class PreparedTranscript:
    """Normalized transcript data ready for downstream feature extraction."""

    source: str
    raw_json: Any
    filtered_json: list[dict[str, Any]]
    utterances: pd.DataFrame
    time_columns: list[str]


class TranscriptPreprocessor:
    """Prepare incoming transcript JSON into the legacy dataframe schema."""

    def __init__(self, measures: dict) -> None:
        self.measures = measures

    def prepare(self, json_conf: Any, whisper_turn_mode: str = "auto") -> PreparedTranscript:
        """Detect transcript origin and return normalized word- and turn-level structures."""
        if legacy_speech.is_whisper_transcribe(json_conf):
            filtered_json, utterances = self._filter_whisper(
                json_conf,
                whisper_turn_mode=whisper_turn_mode,
            )
            source = "whisper"
        elif legacy_speech.is_amazon_transcribe(json_conf):
            filtered_json, utterances = legacy_speech.filter_transcribe(json_conf, self.measures)
            source = "aws"
        else:
            filtered_json, utterances = legacy_speech.filter_vosk(json_conf, self.measures)
            source = "vosk"

        return PreparedTranscript(
            source=source,
            raw_json=json_conf,
            filtered_json=filtered_json,
            utterances=utterances,
            time_columns=legacy_speech.get_time_columns(source),
        )

    def _filter_whisper(self, json_conf: dict[str, Any], whisper_turn_mode: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        """Normalize Whisper-like transcripts with configurable turn aggregation."""
        item_data = json_conf["segments"]
        item_data = legacy_cutil.create_index_column(item_data, self.measures)
        utterances = self._create_turns_whisper(item_data, whisper_turn_mode=whisper_turn_mode)
        filter_json = legacy_cutil.filter_json_transcribe(item_data, self.measures)
        return filter_json, utterances

    def _create_turns_whisper(self, item_data: list[dict[str, Any]], whisper_turn_mode: str) -> pd.DataFrame:
        """Build turn rows either per segment or by merging consecutive speaker spans."""
        turn_mode = normalize_turn_mode(whisper_turn_mode)
        has_speaker = any(("speaker" in segment and segment["speaker"] is not None) for segment in item_data)

        if turn_mode == "segment" or not has_speaker:
            data = [self._segment_to_turn(item) for item in item_data]
            data = [row for row in data if row is not None]
            return pd.DataFrame(data)

        data = []
        current_speaker = None
        aggregated_text = ""
        aggregated_ids: list[int] = []
        word_ids: list[int] = []
        word_texts: list[str] = []
        phrase_ids: list[tuple[int, int]] = []
        phrase_texts: list[str] = []

        for item in item_data:
            item_turn = self._segment_to_turn(item)
            if item_turn is None:
                continue

            speaker = item_turn[self.measures["speaker_label"]]
            if speaker == current_speaker:
                if item_turn[self.measures["utterance_text"]]:
                    aggregated_text = (
                        f"{aggregated_text} {item_turn[self.measures['utterance_text']]}".strip()
                        if aggregated_text
                        else item_turn[self.measures["utterance_text"]]
                    )
                utterance_ids = item_turn[self.measures["utterance_ids"]]
                aggregated_ids.extend([utterance_ids[0], utterance_ids[1]])
                word_ids.extend(item_turn[self.measures["words_ids"]])
                word_texts.extend(item_turn[self.measures["words_texts"]])
                phrase_ids.extend(item_turn[self.measures["phrases_ids"]])
                phrase_texts.extend(item_turn[self.measures["phrases_texts"]])
                continue

            if aggregated_ids:
                data.append(
                    self._build_aggregated_turn(
                        aggregated_ids=aggregated_ids,
                        aggregated_text=aggregated_text,
                        word_ids=word_ids,
                        word_texts=word_texts,
                        phrase_ids=phrase_ids,
                        phrase_texts=phrase_texts,
                        speaker=current_speaker,
                    )
                )

            current_speaker = speaker
            aggregated_text = item_turn[self.measures["utterance_text"]]
            utterance_ids = item_turn[self.measures["utterance_ids"]]
            aggregated_ids = [utterance_ids[0], utterance_ids[1]]
            word_ids = item_turn[self.measures["words_ids"]].copy()
            word_texts = item_turn[self.measures["words_texts"]].copy()
            phrase_ids = item_turn[self.measures["phrases_ids"]].copy()
            phrase_texts = item_turn[self.measures["phrases_texts"]].copy()

        if aggregated_ids:
            data.append(
                self._build_aggregated_turn(
                    aggregated_ids=aggregated_ids,
                    aggregated_text=aggregated_text,
                    word_ids=word_ids,
                    word_texts=word_texts,
                    phrase_ids=phrase_ids,
                    phrase_texts=phrase_texts,
                    speaker=current_speaker,
                )
            )

        return pd.DataFrame(data)

    def _segment_to_turn(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a single Whisper segment to one turn row."""
        words = [word for word in item.get("words", []) if "start" in word]
        idxs = [word[self.measures["old_index"]] for word in words]
        if not idxs:
            return None

        text = (item.get("text") or "").strip()
        words_texts = [word.get("word", "") for word in words]
        phrase_ids, phrase_texts = self._extract_phrase_payload(item, idxs, words_texts, text)
        return {
            self.measures["utterance_ids"]: (idxs[0], idxs[-1]),
            self.measures["utterance_text"]: text,
            self.measures["phrases_ids"]: phrase_ids,
            self.measures["phrases_texts"]: phrase_texts,
            self.measures["words_ids"]: idxs,
            self.measures["words_texts"]: words_texts,
            self.measures["speaker_label"]: item.get("speaker"),
        }

    def _build_aggregated_turn(
        self,
        aggregated_ids: list[int],
        aggregated_text: str,
        word_ids: list[int],
        word_texts: list[str],
        phrase_ids: list[tuple[int, int]],
        phrase_texts: list[str],
        speaker: str | None,
    ) -> dict[str, Any]:
        """Build one merged turn from consecutive same-speaker segments."""
        normalized_ids = sorted(set(aggregated_ids))
        return {
            self.measures["utterance_ids"]: (normalized_ids[0], normalized_ids[-1]),
            self.measures["utterance_text"]: aggregated_text.strip(),
            self.measures["phrases_ids"]: phrase_ids,
            self.measures["phrases_texts"]: phrase_texts,
            self.measures["words_ids"]: word_ids,
            self.measures["words_texts"]: word_texts,
            self.measures["speaker_label"]: speaker,
        }

    def _extract_phrase_payload(
        self,
        item: dict[str, Any],
        idxs: list[int],
        words_texts: list[str],
        text: str,
    ) -> tuple[list[tuple[int, int]], list[str]]:
        """Extract phrase spans from a Whisper segment or fall back to one full-span phrase."""
        fallback_ids, fallback_texts = self._default_phrase_payload(idxs, text, words_texts)
        raw_phrases = item.get("phrases")
        if not isinstance(raw_phrases, list) or not raw_phrases:
            return fallback_ids, fallback_texts

        parsed_ranges: list[tuple[int, int]] = []
        parsed_texts: list[str] = []
        for phrase in raw_phrases:
            if not isinstance(phrase, dict):
                continue

            try:
                word_start = int(phrase.get("word_start"))
                word_end = int(phrase.get("word_end"))
            except (TypeError, ValueError):
                continue

            if word_start < 0 or word_end < word_start or word_end >= len(idxs):
                continue

            phrase_text = (phrase.get("text") or "").strip()
            if not phrase_text:
                phrase_text = " ".join(words_texts[word_start:word_end + 1]).strip()
            parsed_ranges.append((word_start, word_end))
            parsed_texts.append(phrase_text)

        if not parsed_ranges:
            return fallback_ids, fallback_texts

        ordering = sorted(range(len(parsed_ranges)), key=lambda pos: (parsed_ranges[pos][0], parsed_ranges[pos][1]))
        parsed_ranges = [parsed_ranges[pos] for pos in ordering]
        parsed_texts = [parsed_texts[pos] for pos in ordering]

        expected_start = 0
        for word_start, word_end in parsed_ranges:
            if word_start != expected_start:
                return fallback_ids, fallback_texts
            expected_start = word_end + 1

        if expected_start != len(idxs):
            return fallback_ids, fallback_texts

        phrase_ids = [(idxs[word_start], idxs[word_end]) for word_start, word_end in parsed_ranges]
        return phrase_ids, parsed_texts

    @staticmethod
    def _default_phrase_payload(
        idxs: list[int],
        text: str,
        words_texts: list[str],
    ) -> tuple[list[tuple[int, int]], list[str]]:
        """Build a single fallback phrase that spans the full segment."""
        if not idxs:
            return [], []

        fallback_text = (text or "").strip()
        if not fallback_text:
            fallback_text = " ".join(words_texts).strip()
        return [(idxs[0], idxs[-1])], [fallback_text]
