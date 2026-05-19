"""Repo-root shim exposing the local phonova package consistently."""

from __future__ import annotations

import importlib
import sys

_analyzer = importlib.import_module(".phonova.analyzer", __name__)
_config = importlib.import_module(".phonova.config", __name__)
_backends = importlib.import_module(".phonova.backends", __name__)
_transcripts = importlib.import_module(".phonova.transcripts", __name__)
_speech = importlib.import_module(".phonova.speech", __name__)

sys.modules[__name__ + ".analyzer"] = _analyzer
sys.modules[__name__ + ".config"] = _config
sys.modules[__name__ + ".backends"] = _backends
sys.modules[__name__ + ".transcripts"] = _transcripts
sys.modules[__name__ + ".speech"] = _speech

SpeechAnalyzer = _analyzer.SpeechAnalyzer
SpeechAnalyzerSettings = _config.SpeechAnalyzerSettings

__all__ = ["SpeechAnalyzer", "SpeechAnalyzerSettings"]
