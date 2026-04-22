from __future__ import annotations

import copy
import os
import sys
import importlib.util
import itertools
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENWILLIS_SRC = REPO_ROOT / "openwillis" / "openwillis-speech" / "src"
SPEECH_ROOT = OPENWILLIS_SRC / "openwillis" / "speech"
CHARACTERISTICS_UTIL_PATH = SPEECH_ROOT / "util" / "characteristics_util.py"
PAUSE_MODULE_PATH = SPEECH_ROOT / "util" / "speech" / "pause.py"
LEXICAL_MODULE_PATH = SPEECH_ROOT / "util" / "speech" / "lexical.py"
COHERENCE_MODULE_PATH = SPEECH_ROOT / "util" / "speech" / "coherence.py"
SPEECH_ATTRIBUTE_PATH = SPEECH_ROOT / "speech_attribute.py"

LANGUAGE_TO_PIPELINE = {
    "en": "en",
    "ua": "uk",
    "uk": "uk",
    "ukr": "uk",
    "eng": "en",
}

_PACKAGE_COUNTER = itertools.count()


@dataclass(frozen=True)
class SpeechRunResult:
    words: pd.DataFrame
    turns: pd.DataFrame
    summary: pd.DataFrame


def _force_cpu_runtime_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


def _ensure_openwillis_src() -> None:
    if str(OPENWILLIS_SRC) not in sys.path:
        sys.path.insert(0, str(OPENWILLIS_SRC))


def _build_package_hierarchy(package_name: str):
    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = [str(OPENWILLIS_SRC)]

    speech_pkg = types.ModuleType(f"{package_name}.speech")
    speech_pkg.__path__ = [str(SPEECH_ROOT)]

    util_pkg = types.ModuleType(f"{package_name}.speech.util")
    util_pkg.__path__ = [str(SPEECH_ROOT / "util")]

    util_speech_pkg = types.ModuleType(f"{package_name}.speech.util.speech")
    util_speech_pkg.__path__ = [str(SPEECH_ROOT / "util" / "speech")]

    root_pkg.speech = speech_pkg
    speech_pkg.util = util_pkg
    util_pkg.speech = util_speech_pkg

    sys.modules[package_name] = root_pkg
    sys.modules[f"{package_name}.speech"] = speech_pkg
    sys.modules[f"{package_name}.speech.util"] = util_pkg
    sys.modules[f"{package_name}.speech.util.speech"] = util_speech_pkg
    return speech_pkg, util_pkg


def _load_module(module_name: str, source_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _speech_characteristics():
    _force_cpu_runtime_env()
    _ensure_openwillis_src()
    package_name = f"local_openwillis_speech_runtime_{next(_PACKAGE_COUNTER)}"
    _, util_pkg = _build_package_hierarchy(package_name)

    pause_module = _load_module(f"{package_name}.speech.util.speech.pause", PAUSE_MODULE_PATH)
    lexical_module = _load_module(f"{package_name}.speech.util.speech.lexical", LEXICAL_MODULE_PATH)
    coherence_module = _load_module(f"{package_name}.speech.util.speech.coherence", COHERENCE_MODULE_PATH)
    util_speech_pkg = sys.modules[f"{package_name}.speech.util.speech"]
    util_speech_pkg.pause = pause_module
    util_speech_pkg.lexical = lexical_module
    util_speech_pkg.coherence = coherence_module

    cutil_module = _load_module(f"{package_name}.speech.util.characteristics_util", CHARACTERISTICS_UTIL_PATH)
    util_pkg.characteristics_util = cutil_module

    speech_attribute_module = _load_module(f"{package_name}.speech.speech_attribute", SPEECH_ATTRIBUTE_PATH)
    return speech_attribute_module.speech_characteristics


@lru_cache(maxsize=1)
def _normalize_whisper_turn_mode():
    def normalize(whisper_turn_mode):
        if whisper_turn_mode is None:
            return "auto"
        turn_mode = str(whisper_turn_mode).strip().lower()
        if turn_mode not in {"auto", "speaker", "segment"}:
            raise ValueError("Invalid whisper_turn_mode. Please use 'auto', 'speaker', or 'segment'")
        return turn_mode

    return normalize


def _normalize_pipeline_language(language: str) -> str:
    key = str(language).strip().lower()
    return LANGUAGE_TO_PIPELINE.get(key, key[:2])


def run_speech_characteristics(
    json_conf: dict,
    *,
    language: str,
    speaker_label: str = "participant",
    option: str = "coherence",
    feature_groups: Iterable[str] | None = None,
    whisper_turn_mode: str = "speaker",
) -> SpeechRunResult:
    payload = copy.deepcopy(json_conf)
    _normalize_whisper_turn_mode()(whisper_turn_mode)
    run = _speech_characteristics()
    result = run(
        payload,
        language=_normalize_pipeline_language(language),
        speaker_label=speaker_label,
        option=option,
        feature_groups=None if feature_groups is None else list(feature_groups),
        whisper_turn_mode=whisper_turn_mode,
    )
    if not isinstance(result, (list, tuple)) or len(result) != 3:
        result_len = len(result) if isinstance(result, (list, tuple)) else "n/a"
        raise AssertionError(f"Unexpected speech_characteristics result: {type(result)!r}, len={result_len}")

    words, turns, summary = result
    return SpeechRunResult(
        words=words.copy(deep=True),
        turns=turns.copy(deep=True),
        summary=summary.copy(deep=True),
    )
