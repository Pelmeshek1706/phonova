from tests.helpers.module_loaders import load_local_speech_attribute


def test_is_whisper_transcribe_detects_segment_payloads():
    speech_attribute = load_local_speech_attribute()

    assert speech_attribute.is_whisper_transcribe({"segments": [{"words": [{"word": "hello"}]}]})
    assert not speech_attribute.is_whisper_transcribe({"segments": []})
    assert not speech_attribute.is_whisper_transcribe({"results": {"items": []}})
