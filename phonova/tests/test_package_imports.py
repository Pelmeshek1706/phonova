from __future__ import annotations

from phonova.speech import speech_characteristics
from phonova import SpeechAnalyzer, SpeechAnalyzerSettings


def test_public_imports_are_exposed() -> None:
    assert SpeechAnalyzer is not None
    assert SpeechAnalyzerSettings is not None
    assert callable(speech_characteristics)
