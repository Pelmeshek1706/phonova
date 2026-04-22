import pandas as pd

from tests.helpers.module_loaders import load_local_speech_attribute


def test_process_transcript_keeps_speaker_scope_for_whisper_input(monkeypatch):
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

    speech_attribute.process_transcript(
        df_list=[pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        json_conf={"segments": [{"speaker": "participant", "words": [{"word": "hello"}]}]},
        measures={},
        min_turn_length=1,
        min_coherence_turn_length=1,
        speaker_label="participant",
        source="whisper",
        language="en",
        option="coherence",
        whisper_turn_mode="speaker",
    )

    assert captured == {
        "speaker_filter_label": "participant",
        "coherence_speaker_label": "participant",
    }
