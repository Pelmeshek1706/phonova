"""Backend abstractions for coherence analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from phonova.speech.util.speech import coherence as phonova_coherence

from .config import CoherenceBackendName, SpeechAnalyzerSettings


@dataclass(slots=True)
class CoherenceResources:
    """Loaded model objects required by a coherence backend."""

    sentence_encoder: Optional[Any] = None
    tokenizer: Optional[Any] = None
    word_model: Optional[Any] = None
    language_model: Optional[Any] = None
    model_max_length: Optional[int] = None


class BaseCoherenceBackend(ABC):
    """Shared interface for backend-specific coherence resources."""

    backend_name: CoherenceBackendName

    def __init__(self, settings: SpeechAnalyzerSettings, measures: dict) -> None:
        self.settings = settings
        self.measures = measures
        self.resources = self._load_resources()

    @abstractmethod
    def _load_resources(self) -> CoherenceResources:
        """Load and cache all resources required by the backend."""

    @abstractmethod
    def _embed_words(self, words: list[str]) -> np.ndarray:
        """Create token-level embeddings for a list of words."""

    @abstractmethod
    def calculate_perplexity(self, text: str) -> Tuple[float, float, float, float]:
        """Compute backend-specific perplexity metrics for a turn."""

    @property
    def sentence_encoder(self) -> Optional[Any]:
        """Return the sentence-level encoder used for phrase similarity."""
        return self.resources.sentence_encoder

    @property
    def tokenizer(self) -> Optional[Any]:
        """Return the backend tokenizer used for perplexity calculations."""
        return self.resources.tokenizer

    @property
    def word_model(self) -> Optional[Any]:
        """Return the token-embedding model when the backend needs one."""
        return self.resources.word_model

    @property
    def language_model(self) -> Optional[Any]:
        """Return the language model used for perplexity calculations."""
        return self.resources.language_model

    @property
    def model_max_length(self) -> Optional[int]:
        """Return the effective max token length for the language model."""
        return self.resources.model_max_length

    def supports_word_coherence(self) -> bool:
        """Report whether the backend can produce token-level coherence."""
        return (
            self.sentence_encoder is not None
            if self.backend_name == "gemma"
            else self.tokenizer is not None and self.word_model is not None
        )

    def supports_phrase_coherence(self) -> bool:
        """Report whether the backend can produce phrase-level coherence."""
        has_similarity = self.sentence_encoder is not None
        has_perplexity = self.tokenizer is not None and self.language_model is not None
        return has_similarity or has_perplexity

    def embed_words(self, words: list[str]) -> np.ndarray:
        """Embed words with the backend-specific implementation."""
        if not words:
            return np.zeros((0, 0), dtype=np.float32)
        return self._embed_words(words)

    def encode_phrases(self, phrases: list[str]) -> np.ndarray:
        """Encode phrase texts through the shared sentence encoder pipeline."""
        if self.sentence_encoder is None or not phrases:
            return np.zeros((0, 0), dtype=np.float32)
        return phonova_coherence._encode_in_chunks(
            self.sentence_encoder,
            phrases,
            phonova_coherence.EMBEDDING_BATCH_SIZE,
        )


class GemmaCoherenceBackend(BaseCoherenceBackend):
    """Gemma-backed coherence engine with sentence embeddings and causal LM perplexity."""

    backend_name: CoherenceBackendName = "gemma"

    def _load_resources(self) -> CoherenceResources:
        bundle = phonova_coherence.get_model_bundle(
            self.settings.language,
            device_hint=self.settings.device_hint,
        )
        return CoherenceResources(
            sentence_encoder=bundle.get("sentence_encoder"),
            tokenizer=bundle.get("tokenizer"),
            language_model=bundle.get("lm_model"),
            model_max_length=bundle.get("model_max_length"),
        )

    def _embed_words(self, words: list[str]) -> np.ndarray:
        return phonova_coherence.get_word_embeddings(words, self.sentence_encoder)

    def calculate_perplexity(self, text: str) -> Tuple[float, float, float, float]:
        return phonova_coherence.calculate_perplexity(
            text,
            self.language_model,
            self.tokenizer,
            model_max_length=self.model_max_length,
        )


class BertCoherenceBackend(BaseCoherenceBackend):
    """BERT-backed coherence engine with multilingual token embeddings and pseudo-perplexity."""

    backend_name: CoherenceBackendName = "bert"

    def _load_resources(self) -> CoherenceResources:
        bundle = phonova_coherence.get_bert_bundle(
            self.settings.language,
            self.measures,
            device_hint=self.settings.device_hint,
        )
        return CoherenceResources(
            sentence_encoder=bundle.get("sentence_encoder"),
            tokenizer=bundle.get("tokenizer"),
            word_model=bundle.get("word_model"),
            language_model=bundle.get("mlm_model"),
        )

    def _embed_words(self, words: list[str]) -> np.ndarray:
        return phonova_coherence.get_word_embeddings_bert(words, self.tokenizer, self.word_model)

    def calculate_perplexity(self, text: str) -> Tuple[float, float, float, float]:
        return phonova_coherence.calculate_perplexity_bert(
            text,
            self.language_model,
            self.tokenizer,
        )


def build_coherence_backend(settings: SpeechAnalyzerSettings, measures: dict) -> BaseCoherenceBackend:
    """Construct the backend implementation configured for a speech analyzer."""
    if settings.coherence_backend == "bert":
        return BertCoherenceBackend(settings=settings, measures=measures)
    return GemmaCoherenceBackend(settings=settings, measures=measures)
