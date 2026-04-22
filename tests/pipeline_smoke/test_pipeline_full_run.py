from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params
from tests.helpers.thresholds import MODEL_DF_ATOL, MODEL_DF_RTOL


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def test_pipeline_full_run_matches_mock(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="coherence",
        feature_groups=None,
        whisper_turn_mode="speaker",
    )

    assert_frame_matches_mock(
        result.words,
        mock.words,
        context=f"{language}/{case_id}/words",
        atol=MODEL_DF_ATOL,
        rtol=MODEL_DF_RTOL,
    )
    assert_frame_matches_mock(
        result.turns,
        mock.turns,
        context=f"{language}/{case_id}/turns",
        atol=MODEL_DF_ATOL,
        rtol=MODEL_DF_RTOL,
    )
    assert_frame_matches_mock(
        result.summary,
        mock.summary,
        context=f"{language}/{case_id}/summary",
        atol=MODEL_DF_ATOL,
        rtol=MODEL_DF_RTOL,
    )
