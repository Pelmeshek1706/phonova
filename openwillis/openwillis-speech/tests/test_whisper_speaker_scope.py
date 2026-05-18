import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def load_local_speech_attribute():
    package_name = "local_openwillis_speech_test"
    module_name = f"{package_name}.speech.speech_attribute"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "openwillis"
        / "speech"
        / "speech_attribute.py"
    )

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    speech_pkg = types.ModuleType(f"{package_name}.speech")
    speech_pkg.__path__ = []
    util_pkg = types.ModuleType(f"{package_name}.speech.util")
    util_pkg.__path__ = []
    cutil_module = types.ModuleType(f"{package_name}.speech.util.characteristics_util")

    util_pkg.characteristics_util = cutil_module
    speech_pkg.util = util_pkg
    root_pkg.speech = speech_pkg

    sys.modules[package_name] = root_pkg
    sys.modules[f"{package_name}.speech"] = speech_pkg
    sys.modules[f"{package_name}.speech.util"] = util_pkg
    sys.modules[f"{package_name}.speech.util.characteristics_util"] = cutil_module

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_process_transcript_keeps_speaker_scope_for_whisper_speaker_mode(monkeypatch):
    speech_attribute = load_local_speech_attribute()
    captured = {}

    monkeypatch.setattr(speech_attribute, "common_summary_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        speech_attribute,
        "filter_whisper",
        lambda json_conf, measures, whisper_turn_mode="auto": (["word"], pd.DataFrame({"utterance": [1]})),
    )

    def fake_process_language_feature(*args, **kwargs):
        captured["speaker_filter_label"] = kwargs["speaker_filter_label"]
        captured["coherence_speaker_label"] = kwargs["coherence_speaker_label"]
        return args[0]

    monkeypatch.setattr(
        speech_attribute.cutil,
        "process_language_feature",
        fake_process_language_feature,
        raising=False,
    )

    df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    out = speech_attribute.process_transcript(
        df_list=df_list,
        json_conf={"segments": [{"speaker": "participant", "words": [{"start": 0.0, "end": 0.1, "word": "hi"}]}]},
        measures={},
        min_turn_length=1,
        min_coherence_turn_length=1,
        speaker_label="participant",
        source="whisper",
        language="en",
        option="coherence",
        whisper_turn_mode="speaker",
    )

    assert out is df_list
    assert captured == {
        "speaker_filter_label": "participant",
        "coherence_speaker_label": "participant",
    }


def test_process_transcript_keeps_speaker_scope_for_whisper_auto_mode(monkeypatch):
    speech_attribute = load_local_speech_attribute()
    captured = {}

    monkeypatch.setattr(speech_attribute, "common_summary_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        speech_attribute,
        "filter_whisper",
        lambda json_conf, measures, whisper_turn_mode="auto": (["word"], pd.DataFrame({"utterance": [1]})),
    )

    def fake_process_language_feature(*args, **kwargs):
        captured["speaker_filter_label"] = kwargs["speaker_filter_label"]
        captured["coherence_speaker_label"] = kwargs["coherence_speaker_label"]
        return args[0]

    monkeypatch.setattr(
        speech_attribute.cutil,
        "process_language_feature",
        fake_process_language_feature,
        raising=False,
    )

    df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    speech_attribute.process_transcript(
        df_list=df_list,
        json_conf={"segments": [{"speaker": "participant", "words": [{"start": 0.0, "end": 0.1, "word": "hi"}]}]},
        measures={},
        min_turn_length=1,
        min_coherence_turn_length=1,
        speaker_label="participant",
        source="whisper",
        language="en",
        option="coherence",
        whisper_turn_mode="auto",
    )

    assert captured == {
        "speaker_filter_label": "participant",
        "coherence_speaker_label": "participant",
    }
