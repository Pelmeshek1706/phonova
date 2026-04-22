from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.feature_columns import COHERENCE_SUMMARY_COLUMNS
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params
from tests.helpers.thresholds import MODEL_DF_ATOL, MODEL_DF_RTOL


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def _assert_columns_missing(df, columns, context):
    for column in columns:
        assert column in df.columns, f"{context}: missing column {column}"
        assert df[column].isna().all(), f"{context}: expected NaN-only values in {column}"


def test_coherence_and_perplexity_matches_mock(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="coherence",
        feature_groups=("coherence",),
        whisper_turn_mode="speaker",
    )

    assert_frame_matches_mock(
        result.summary.loc[:, COHERENCE_SUMMARY_COLUMNS],
        mock.summary.loc[:, COHERENCE_SUMMARY_COLUMNS],
        context=f"{language}/{case_id}/summary",
        atol=MODEL_DF_ATOL,
        rtol=MODEL_DF_RTOL,
    )
    assert_frame_matches_mock(
        result.words.loc[:, [
            "word_coherence",
            "word_coherence_5",
            "word_coherence_10",
            "word_coherence_variability_2",
            "word_coherence_variability_3",
            "word_coherence_variability_4",
            "word_coherence_variability_5",
            "word_coherence_variability_6",
            "word_coherence_variability_7",
            "word_coherence_variability_8",
            "word_coherence_variability_9",
            "word_coherence_variability_10",
        ]],
        mock.words.loc[:, [
            "word_coherence",
            "word_coherence_5",
            "word_coherence_10",
            "word_coherence_variability_2",
            "word_coherence_variability_3",
            "word_coherence_variability_4",
            "word_coherence_variability_5",
            "word_coherence_variability_6",
            "word_coherence_variability_7",
            "word_coherence_variability_8",
            "word_coherence_variability_9",
            "word_coherence_variability_10",
        ]],
        context=f"{language}/{case_id}/words",
        atol=MODEL_DF_ATOL,
        rtol=MODEL_DF_RTOL,
    )

    _assert_columns_missing(
        result.summary,
        [
            "word_repeat_percentage",
            "first_person_percentage",
            "sentiment_pos",
            "first_order_sentence_tangeniality_mean",
            "semantic_perplexity_mean",
        ],
        context=f"{language}/{case_id}/summary",
    )

    _assert_columns_missing(
        result.turns,
        [
            "first_order_sentence_tangeniality",
            "second_order_sentence_tangeniality",
            "turn_to_turn_tangeniality",
            "turn_to_previous_speaker_turn_similarity",
            "semantic_perplexity",
            "semantic_perplexity_5",
            "semantic_perplexity_11",
            "semantic_perplexity_15",
        ],
        context=f"{language}/{case_id}/turns",
    )
