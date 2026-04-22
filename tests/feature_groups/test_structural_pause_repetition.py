from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.feature_columns import PAUSE_REPETITION_SUMMARY_COLUMNS
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def _assert_columns_missing(df, columns, context):
    for column in columns:
        assert column in df.columns, f"{context}: missing column {column}"
        assert df[column].isna().all(), f"{context}: expected NaN-only values in {column}"


def test_structural_pause_repetition_matches_mock(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="simple",
        feature_groups=("pause", "repetition"),
        whisper_turn_mode="speaker",
    )

    assert_frame_matches_mock(
        result.summary.loc[:, PAUSE_REPETITION_SUMMARY_COLUMNS],
        mock.summary.loc[:, PAUSE_REPETITION_SUMMARY_COLUMNS],
        context=f"{language}/{case_id}/summary",
    )
    assert_frame_matches_mock(
        result.turns.loc[:, [
            "pre_turn_pause",
            "turn_length_minutes",
            "turn_length_words",
            "words_per_min",
            "syllables_per_min",
            "speech_percentage",
            "mean_pause_length",
            "pause_variability",
            "word_repeat_percentage",
            "phrase_repeat_percentage",
        ]],
        mock.turns.loc[:, [
            "pre_turn_pause",
            "turn_length_minutes",
            "turn_length_words",
            "words_per_min",
            "syllables_per_min",
            "speech_percentage",
            "mean_pause_length",
            "pause_variability",
            "word_repeat_percentage",
            "phrase_repeat_percentage",
        ]],
        context=f"{language}/{case_id}/turns",
    )
    assert_frame_matches_mock(
        result.words.loc[:, ["pre_word_pause", "num_syllables"]],
        mock.words.loc[:, ["pre_word_pause", "num_syllables"]],
        context=f"{language}/{case_id}/words",
    )

    _assert_columns_missing(
        result.summary,
        [
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_overall",
            "word_coherence_mean",
            "first_person_percentage",
            "first_person_sentiment_positive",
            "semantic_perplexity_mean",
        ],
        context=f"{language}/{case_id}/summary",
    )
