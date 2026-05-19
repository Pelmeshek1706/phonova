"""Configuration objects for the class-based speech refinement API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CoherenceBackendName = Literal["gemma", "bert"]
WhisperTurnMode = Literal["auto", "speaker", "segment"]


def normalize_language(language: str) -> str:
    """Normalize a language hint to the legacy two-letter format."""
    if not language:
        return "na"
    return language[:2].lower()


def normalize_backend(backend: str) -> str:
    """Normalize and validate the coherence backend name."""
    normalized = (backend or "gemma").strip().lower()
    if normalized not in {"gemma", "bert"}:
        raise ValueError("Invalid coherence backend. Please use 'gemma' or 'bert'.")
    return normalized


def normalize_turn_mode(turn_mode: str) -> str:
    """Normalize and validate the Whisper turn-construction mode."""
    normalized = (turn_mode or "auto").strip().lower()
    if normalized not in {"auto", "speaker", "segment"}:
        raise ValueError("Invalid whisper_turn_mode. Please use 'auto', 'speaker', or 'segment'.")
    return normalized


@dataclass(frozen=True, slots=True)
class SpeechAnalyzerSettings:
    """One-time initialization settings for the class-based speech analyzer."""

    language: str
    coherence_backend: CoherenceBackendName = "gemma"
    device_hint: Optional[str] = None

    def __post_init__(self) -> None:
        """Store normalized values so downstream services can rely on them."""
        object.__setattr__(self, "language", normalize_language(self.language))
        object.__setattr__(self, "coherence_backend", normalize_backend(self.coherence_backend))
