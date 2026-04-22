from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def test_lexical_diversity_matches_mock(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="simple",
        feature_groups=("sentiment", "first_person"),
        whisper_turn_mode="speaker",
    )

    mattr_columns = ["mattr_5", "mattr_10", "mattr_25", "mattr_50", "mattr_100"]
    assert_frame_matches_mock(
        result.summary.loc[:, mattr_columns],
        mock.summary.loc[:, mattr_columns],
        context=f"{language}/{case_id}/summary",
    )
    assert_frame_matches_mock(
        result.turns.loc[:, mattr_columns],
        mock.turns.loc[:, mattr_columns],
        context=f"{language}/{case_id}/turns",
    )

    summary_values = [result.summary.iloc[0][column] for column in mattr_columns]
    assert summary_values == sorted(summary_values, reverse=True)
