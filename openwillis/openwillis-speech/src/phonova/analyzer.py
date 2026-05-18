"""Class-based speech analyzer that preserves legacy output semantics."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np

from openwillis.speech import speech_attribute as legacy_speech
from openwillis.speech.util import characteristics_util as legacy_cutil
from openwillis.speech.util.speech.coherence import (
    WORD_STREAM_CHUNK_SIZE,
    PREVIOUS_SPEAKER_SIMILARITY_MIN_TURN_LENGTH,
    _new_coherence_lists,
    _extend_coherence_lists,
    _normalize_embeddings,
    _cosine_for_offset,
    _previous_speaker_turn_similarity,
    _phrase_tangeniality_from_embeddings,
    _release_accelerator_cache,
    _word_coherence_from_embeddings,
    append_nan_values,
    calculate_slope,
    get_word_coherence_summary,
)
from openwillis.speech.util.speech.lexical import get_pos_tag, get_repetitions, get_sentiment
from openwillis.speech.util.speech.pause import get_pause_feature

from .backends import BaseCoherenceBackend, build_coherence_backend
from .config import SpeechAnalyzerSettings
from .transcripts import PreparedTranscript, TranscriptPreprocessor

logger = logging.getLogger(__name__)


class CoherenceAnalyzer:
    """Apply coherence metrics using one preloaded backend instance."""

    def __init__(self, backend: BaseCoherenceBackend, measures: dict) -> None:
        self.backend = backend
        self.measures = measures

    def analyze(
        self,
        df_list: list,
        utterances_speaker,
        utterances_speaker_filtered,
        utterances_dialogue_filtered,
        min_coherence_turn_length: int,
        speaker_label: str | None,
    ) -> list:
        """Run word- and phrase-level coherence analysis using cached resources."""
        df_list = self._apply_word_coherence(
            df_list,
            utterances_speaker,
            min_coherence_turn_length=min_coherence_turn_length,
        )
        df_list = self._apply_phrase_coherence(
            df_list,
            utterances_speaker_filtered,
            utterances_dialogue_filtered,
            min_coherence_turn_length=min_coherence_turn_length,
            speaker_label=speaker_label,
        )
        return df_list

    def _apply_word_coherence(self, df_list: list, utterances_speaker, min_coherence_turn_length: int) -> list:
        """Mirror legacy word-level coherence while reusing one backend instance."""
        if not self.backend.supports_word_coherence():
            logger.info(
                "Coherence backend %s does not have token-level resources for language %s; skipping word coherence.",
                self.backend.backend_name,
                self.backend.settings.language,
            )
            return df_list

        word_df, turn_df, summ_df = df_list
        overall_lists = _new_coherence_lists()
        chunk_rows = []

        def process_chunk(rows_chunk: list) -> None:
            """Compute coherence metrics for a buffered chunk of utterance rows."""
            if not rows_chunk:
                return

            chunk_lists = _new_coherence_lists()
            eligible_words: list[list[str]] = []
            for row in rows_chunk:
                words = row[self.measures["words_texts"]]
                if len(words) >= min_coherence_turn_length:
                    eligible_words.append(words)

            row_embeddings: list[np.ndarray | None] = [None] * len(rows_chunk)
            if eligible_words:
                flat_words = [word for words in eligible_words for word in words]
                try:
                    flat_embeddings = self.backend.embed_words(flat_words)
                    offset = 0
                    for row_idx, row in enumerate(rows_chunk):
                        words = row[self.measures["words_texts"]]
                        if len(words) < min_coherence_turn_length:
                            continue
                        count = len(words)
                        row_embeddings[row_idx] = flat_embeddings[offset:offset + count]
                        offset += count
                except Exception as exc:
                    logger.info("Error in batch word embedding analysis: %s", exc)

            for row_idx, row in enumerate(rows_chunk):
                words = row[self.measures["words_texts"]]
                try:
                    if len(words) < min_coherence_turn_length:
                        append_nan_values(chunk_lists, len(words))
                        continue

                    embeddings = row_embeddings[row_idx]
                    if embeddings is None:
                        embeddings = self.backend.embed_words(words)

                    coherence, coherence_5, coherence_10, variability = _word_coherence_from_embeddings(embeddings)
                    chunk_lists["word_coherence"] += coherence
                    chunk_lists["word_coherence_5"] += coherence_5
                    chunk_lists["word_coherence_10"] += coherence_10
                    for k in range(2, 11):
                        chunk_lists["variability"][k] += variability[k]
                except Exception as exc:
                    logger.info("Error in word coherence analysis for row: %s", exc)
                    append_nan_values(chunk_lists, len(words))

            _extend_coherence_lists(overall_lists, chunk_lists)
            _release_accelerator_cache()

        for _, row in utterances_speaker.iterrows():
            chunk_rows.append(row)
            if len(chunk_rows) >= WORD_STREAM_CHUNK_SIZE:
                process_chunk(chunk_rows)
                chunk_rows = []

        process_chunk(chunk_rows)

        word_df[self.measures["word_coherence"]] = overall_lists["word_coherence"]
        word_df[self.measures["word_coherence_5"]] = overall_lists["word_coherence_5"]
        word_df[self.measures["word_coherence_10"]] = overall_lists["word_coherence_10"]
        for k in range(2, 11):
            word_df[self.measures[f"word_coherence_variability_{k}"]] = overall_lists["variability"][k]

        summ_df = get_word_coherence_summary(word_df, summ_df, self.measures)
        return [word_df, turn_df, summ_df]

    def _apply_phrase_coherence(
        self,
        df_list: list,
        utterances_speaker_filtered,
        utterances_dialogue_filtered,
        min_coherence_turn_length: int,
        speaker_label: str | None,
    ) -> list:
        """Mirror legacy phrase-level coherence while avoiding global backend mutation."""
        if not self.backend.supports_phrase_coherence():
            logger.info(
                "Coherence backend %s has no phrase-level resources for language %s; skipping phrase coherence.",
                self.backend.backend_name,
                self.backend.settings.language,
            )
            return df_list

        word_df, turn_df, summ_df = df_list
        if len(turn_df) == 0:
            return df_list

        turn_df = self._calculate_turn_coherence(
            utterances_speaker_filtered,
            turn_df,
            min_coherence_turn_length=min_coherence_turn_length,
            speaker_label=speaker_label,
            dialogue_utterances_filtered=utterances_dialogue_filtered,
        )

        for measure in [
            "sentence_tangeniality1",
            "sentence_tangeniality2",
            "perplexity",
            "perplexity_5",
            "perplexity_11",
            "perplexity_15",
            "turn_to_turn_tangeniality",
            "turn_to_previous_speaker_turn_similarity",
        ]:
            if turn_df[self.measures[measure]].isnull().all():
                continue
            summ_df[self.measures[f"{measure}_mean"]] = turn_df[self.measures[measure]].mean(skipna=True)
            summ_df[self.measures[f"{measure}_var"]] = turn_df[self.measures[measure]].var(skipna=True)

        if not turn_df[self.measures["turn_to_turn_tangeniality"]].isnull().all():
            summ_df[self.measures["turn_to_turn_tangeniality_slope"]] = calculate_slope(
                turn_df[self.measures["turn_to_turn_tangeniality"]]
            )
        if not turn_df[self.measures["turn_to_previous_speaker_turn_similarity"]].isnull().all():
            summ_df[self.measures["turn_to_previous_speaker_turn_similarity_slope"]] = calculate_slope(
                turn_df[self.measures["turn_to_previous_speaker_turn_similarity"]]
            )

        return [word_df, turn_df, summ_df]

    def _calculate_turn_coherence(
        self,
        utterances_filtered,
        turn_df,
        min_coherence_turn_length: int,
        speaker_label: str | None,
        dialogue_utterances_filtered,
    ):
        """Reproduce legacy turn-level coherence semantics with instance-scoped resources."""
        utterances_texts = utterances_filtered[self.measures["utterance_text"]].values.tolist()
        adjacent_turn_similarity = None
        phrase_embeddings_by_row = {}
        previous_speaker_turn_similarity_list = _previous_speaker_turn_similarity(
            dialogue_utterances_filtered,
            speaker_label,
            PREVIOUS_SPEAKER_SIMILARITY_MIN_TURN_LENGTH,
            self.backend.sentence_encoder,
            self.measures,
        )

        if self.backend.sentence_encoder is not None:
            utterances_embeddings = self.backend.encode_phrases(utterances_texts)
            normalized_turns = _normalize_embeddings(utterances_embeddings)
            adjacent_turn_similarity = np.full((len(utterances_filtered),), np.nan, dtype=np.float32)
            if normalized_turns.shape[0] > 1:
                adjacent_turn_similarity[1:] = _cosine_for_offset(normalized_turns, 1)

            flat_phrases: list[str] = []
            phrase_row_indices: list[object] = []
            phrase_row_counts: list[int] = []
            for row_idx, row in utterances_filtered.iterrows():
                if len(row[self.measures["words_texts"]]) < min_coherence_turn_length:
                    continue
                phrases = row[self.measures["phrases_texts"]]
                if len(phrases) == 0:
                    continue
                phrase_row_indices.append(row_idx)
                phrase_row_counts.append(len(phrases))
                flat_phrases.extend(phrases)

            if flat_phrases:
                all_phrase_embeddings = self.backend.encode_phrases(flat_phrases)
                offset = 0
                for row_idx, count in zip(phrase_row_indices, phrase_row_counts):
                    phrase_embeddings_by_row[row_idx] = all_phrase_embeddings[offset:offset + count]
                    offset += count

        sentence_tangeniality1_list = []
        sentence_tangeniality2_list = []
        perplexity_list = []
        perplexity_5_list = []
        perplexity_11_list = []
        perplexity_15_list = []
        turn_to_turn_tangeniality_list = []
        if len(previous_speaker_turn_similarity_list) != len(utterances_filtered):
            previous_speaker_turn_similarity_list = [np.nan] * len(utterances_filtered)

        for i, row in utterances_filtered.iterrows():
            current_speaker = row[self.measures["speaker_label"]]
            if speaker_label is not None and current_speaker != speaker_label:
                continue
            if len(row[self.measures["words_texts"]]) < min_coherence_turn_length:
                sentence_tangeniality1_list.append(np.nan)
                sentence_tangeniality2_list.append(np.nan)
                perplexity_list.append(np.nan)
                perplexity_5_list.append(np.nan)
                perplexity_11_list.append(np.nan)
                perplexity_15_list.append(np.nan)
                turn_to_turn_tangeniality_list.append(np.nan)
                continue

            phrases_texts = row[self.measures["phrases_texts"]]
            utterance_text = row[self.measures["utterance_text"]]
            sentence_tangeniality1, sentence_tangeniality2 = self._calculate_phrase_similarity(
                phrases_texts,
                phrase_embeddings=phrase_embeddings_by_row.get(i),
            )
            perplexity, perplexity_5, perplexity_11, perplexity_15 = self.backend.calculate_perplexity(utterance_text)

            sentence_tangeniality1_list.append(sentence_tangeniality1)
            sentence_tangeniality2_list.append(sentence_tangeniality2)
            perplexity_list.append(perplexity)
            perplexity_5_list.append(perplexity_5)
            perplexity_11_list.append(perplexity_11)
            perplexity_15_list.append(perplexity_15)

            if (
                i == 0
                or len(utterances_filtered.iloc[i - 1][self.measures["words_texts"]]) < min_coherence_turn_length
                or adjacent_turn_similarity is None
            ):
                turn_to_turn_tangeniality_list.append(np.nan)
            else:
                turn_to_turn_tangeniality_list.append(float(adjacent_turn_similarity[i]))

        turn_df[self.measures["sentence_tangeniality1"]] = sentence_tangeniality1_list
        turn_df[self.measures["sentence_tangeniality2"]] = sentence_tangeniality2_list
        turn_df[self.measures["perplexity"]] = perplexity_list
        turn_df[self.measures["perplexity_5"]] = perplexity_5_list
        turn_df[self.measures["perplexity_11"]] = perplexity_11_list
        turn_df[self.measures["perplexity_15"]] = perplexity_15_list
        turn_df[self.measures["turn_to_turn_tangeniality"]] = turn_to_turn_tangeniality_list
        turn_df[self.measures["turn_to_previous_speaker_turn_similarity"]] = previous_speaker_turn_similarity_list
        return turn_df

    def _calculate_phrase_similarity(
        self,
        phrases_texts: list[str],
        phrase_embeddings=None,
    ) -> tuple[float, float]:
        """Compute first- and second-order tangentiality for one utterance."""
        if self.backend.sentence_encoder is None or len(phrases_texts) == 0:
            return np.nan, np.nan

        embeddings = phrase_embeddings
        if embeddings is None:
            embeddings = self.backend.encode_phrases(phrases_texts)
        return _phrase_tangeniality_from_embeddings(embeddings)


class SpeechAnalyzer:
    """Class-based speech analyzer with one-time language and backend initialization."""

    def __init__(
        self,
        language: str,
        coherence_backend: str = "gemma",
        device_hint: str | None = None,
    ) -> None:
        self.settings = SpeechAnalyzerSettings(
            language=language,
            coherence_backend=coherence_backend,
            device_hint=device_hint,
        )
        self.measures = legacy_speech.get_config(os.path.abspath(legacy_speech.__file__), "text.json")
        self._prepare_language_resources()
        self.preprocessor = TranscriptPreprocessor(self.measures)
        self.backend = build_coherence_backend(self.settings, self.measures)
        self.coherence_analyzer = CoherenceAnalyzer(self.backend, self.measures)

    def analyze_transcript(
        self,
        json_conf,
        speaker_label: str | None = None,
        min_turn_length: int = 1,
        min_coherence_turn_length: int = 5,
        option: str = "coherence",
        feature_groups: Iterable[str] | str | None = None,
        whisper_turn_mode: str = "auto",
    ) -> list:
        """Analyze one transcript while reusing the instance language and backend configuration."""
        df_list = list(legacy_cutil.create_empty_dataframes(self.measures))

        try:
            if option not in {"simple", "coherence"}:
                raise ValueError("Invalid option. Please use 'simple' or 'coherence'")

            if bool(json_conf):
                prepared = self.preprocessor.prepare(json_conf, whisper_turn_mode=whisper_turn_mode)
                legacy_speech.common_summary_feature(df_list[2], json_conf, prepared.source, speaker_label)

                if len(prepared.filtered_json) > 0 and len(prepared.utterances) > 0:
                    df_list = self._process_language_features(
                        df_list,
                        prepared=prepared,
                        speaker_label=speaker_label,
                        min_turn_length=min_turn_length,
                        min_coherence_turn_length=min_coherence_turn_length,
                        option=option,
                        feature_groups=feature_groups,
                    )
        except Exception as exc:
            logger.info("Error in SpeechAnalyzer.analyze_transcript: %s", exc)

        return self._finalize_output(df_list)

    def analyze(self, *args, **kwargs) -> list:
        """Alias for callers who prefer a shorter method name."""
        return self.analyze_transcript(*args, **kwargs)

    def _prepare_language_resources(self) -> None:
        """Load NLP resources once for the configured analyzer language."""
        if self.settings.language in self.measures["english_langs"]:
            legacy_cutil.download_nltk_resources()
        if self.settings.language in {"ua", "uk"}:
            legacy_cutil.download_ua_resources()

    def _process_language_features(
        self,
        df_list: list,
        prepared: PreparedTranscript,
        speaker_label: str | None,
        min_turn_length: int,
        min_coherence_turn_length: int,
        option: str,
        feature_groups: Iterable[str] | str | None,
    ) -> list:
        """Reuse legacy feature extractors around the new coherence orchestrator."""
        groups = self._normalize_feature_groups(feature_groups)
        want_structure = "structure" in groups
        want_pause = "pause" in groups
        want_repetition = "repetition" in groups
        want_coherence = "coherence" in groups and option == "coherence"
        want_sentiment = "sentiment" in groups or "first_person" in groups
        want_first_person = "first_person" in groups
        speaker_filter_label = speaker_label
        coherence_speaker_label = speaker_label

        utterances_speaker, json_conf_speaker = legacy_cutil.filter_speaker(
            prepared.utterances,
            prepared.filtered_json,
            speaker_filter_label,
            self.measures,
        )
        text_list, turn_indices = legacy_cutil.create_text_list(
            utterances_speaker,
            speaker_label,
            min_turn_length,
            self.measures,
        )
        utterances_dialogue_filtered, utterances_speaker_filtered = legacy_cutil.filter_length(
            prepared.utterances,
            utterances_speaker,
            speaker_filter_label,
            min_turn_length,
            self.measures,
        )

        if want_structure:
            df_list = legacy_cutil.add_phrase_count_features(
                df_list,
                utterances_speaker,
                min_turn_length,
                self.measures,
            )
        if want_pause:
            df_list = get_pause_feature(
                json_conf_speaker,
                df_list,
                text_list,
                turn_indices,
                self.measures,
                prepared.time_columns,
                self.settings.language,
            )
        if want_repetition:
            df_list = get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, self.measures)
        if want_coherence:
            df_list = self.coherence_analyzer.analyze(
                df_list,
                utterances_speaker=utterances_speaker,
                utterances_speaker_filtered=utterances_speaker_filtered,
                utterances_dialogue_filtered=utterances_dialogue_filtered,
                min_coherence_turn_length=min_coherence_turn_length,
                speaker_label=coherence_speaker_label,
            )
        if self.settings.language in self.measures["english_langs"] or self.settings.language in {"ua", "uk"}:
            if want_sentiment:
                vader_distribution_texts = legacy_cutil.extract_segment_texts_for_speaker(
                    prepared.raw_json,
                    speaker_filter_label,
                    prepared.source,
                    self.settings.language,
                )
                df_list = get_sentiment(
                    df_list,
                    text_list,
                    self.measures,
                    lang=self.settings.language,
                    vader_distribution_texts=vader_distribution_texts,
                )
            if want_first_person:
                df_list = get_pos_tag(df_list, text_list, self.measures, lang=self.settings.language)

        return df_list

    def _normalize_feature_groups(self, feature_groups: Iterable[str] | str | None) -> set[str]:
        """Normalize the optional feature group selector to the legacy set-based format."""
        if feature_groups is None:
            return {"structure", "pause", "repetition", "coherence", "sentiment", "first_person"}
        if isinstance(feature_groups, str):
            return {feature_groups.strip().lower()}
        return {str(group).strip().lower() for group in feature_groups if group}

    @staticmethod
    def _finalize_output(df_list: list) -> list:
        """Preserve the legacy empty-dataframe contract used by downstream notebooks."""
        for df in df_list:
            if df.empty:
                df.loc[0] = np.nan
            else:
                df.loc[0] = df.loc[0]
        return df_list
