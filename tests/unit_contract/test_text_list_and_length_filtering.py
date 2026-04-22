from tests.helpers.module_loaders import load_local_characteristics_util, load_local_speech_attribute


def test_filter_json_transcribe_skips_words_without_span_metadata():
    cutil = load_local_characteristics_util()
    speech_attribute = load_local_speech_attribute()
    measures = speech_attribute.get_config(speech_attribute.__file__, "text.json")

    item_data = cutil.create_index_column(
        [
            {"speaker": "participant"},
            {"speaker": "participant", "words": [{"word": "hello"}]},
            {"speaker": "participant", "words": [{"start": 0.0, "word": "partial"}]},
            {"speaker": "participant", "words": [{"end": 0.2, "word": "partial"}]},
        ],
        measures,
    )

    assert cutil.filter_json_transcribe(item_data, measures) == []


def test_filter_whisper_handles_empty_segments_without_crashing():
    speech_attribute = load_local_speech_attribute()
    measures = speech_attribute.get_config(speech_attribute.__file__, "text.json")

    filtered_words, utterances = speech_attribute.filter_whisper(
        {"segments": []},
        measures,
        whisper_turn_mode="segment",
    )

    assert filtered_words == []
    assert utterances.empty
