from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def test_words_turns_summary_relationships_hold(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="coherence",
        feature_groups=None,
        whisper_turn_mode="speaker",
    )

    summary = result.summary.iloc[0]
    mock_summary = mock.summary.iloc[0]

    assert len(result.words) == int(mock_summary["speech_length_words"])
    assert len(result.turns) == int(mock_summary["num_turns"])
    assert float(result.turns["turn_length_words"].sum()) == float(mock_summary["speech_length_words"])
    assert abs(float(result.turns["turn_length_minutes"].sum()) - float(mock_summary["speech_length_minutes"])) < 1e-6
    assert float(result.words["pre_word_pause"].isna().sum()) == float(mock_summary["num_turns"])
    assert float(result.turns["pre_turn_pause"].isna().sum()) >= 1.0
    assert float(result.turns["interrupt_flag"].fillna(False).astype(bool).sum()) == float(mock_summary["num_interrupts"])
    assert abs(float(summary["speech_percentage"]) - float(mock_summary["speech_percentage"])) < 1e-6
