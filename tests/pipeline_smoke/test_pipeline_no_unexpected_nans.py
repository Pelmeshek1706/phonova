from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def _assert_nan_mask_matches(actual, expected, context):
    assert actual.isna().equals(expected.isna()), f"{context}: missing-value mask differs"


def test_pipeline_has_no_unexpected_nans(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="coherence",
        feature_groups=None,
        whisper_turn_mode="speaker",
    )

    _assert_nan_mask_matches(result.words, mock.words, f"{language}/{case_id}/words")
    _assert_nan_mask_matches(result.turns, mock.turns, f"{language}/{case_id}/turns")
    _assert_nan_mask_matches(result.summary, mock.summary, f"{language}/{case_id}/summary")
