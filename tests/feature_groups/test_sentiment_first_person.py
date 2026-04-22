from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.feature_columns import SENTIMENT_FIRST_PERSON_SUMMARY_COLUMNS
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))


def _assert_columns_missing(df, columns, context):
    for column in columns:
        assert column in df.columns, f"{context}: missing column {column}"
        assert df[column].isna().all(), f"{context}: expected NaN-only values in {column}"


def test_sentiment_and_first_person_matches_mock(language, case_id):
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="simple",
        feature_groups=("sentiment", "first_person"),
        whisper_turn_mode="speaker",
    )

    assert_frame_matches_mock(
        result.summary.loc[:, SENTIMENT_FIRST_PERSON_SUMMARY_COLUMNS],
        mock.summary.loc[:, SENTIMENT_FIRST_PERSON_SUMMARY_COLUMNS],
        context=f"{language}/{case_id}/summary",
    )
    assert_frame_matches_mock(
        result.turns.loc[:, [
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_overall",
            "sentiment_vader_pos",
            "sentiment_vader_neg",
            "sentiment_vader_neu",
            "sentiment_vader_overall",
            "mattr_5",
            "mattr_10",
            "mattr_25",
            "mattr_50",
            "mattr_100",
            "first_person_percentage",
            "first_person_sentiment_positive",
            "first_person_sentiment_negative",
            "first_person_sentiment_positive_vader",
            "first_person_sentiment_negative_vader",
        ]],
        mock.turns.loc[:, [
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_overall",
            "sentiment_vader_pos",
            "sentiment_vader_neg",
            "sentiment_vader_neu",
            "sentiment_vader_overall",
            "mattr_5",
            "mattr_10",
            "mattr_25",
            "mattr_50",
            "mattr_100",
            "first_person_percentage",
            "first_person_sentiment_positive",
            "first_person_sentiment_negative",
            "first_person_sentiment_positive_vader",
            "first_person_sentiment_negative_vader",
        ]],
        context=f"{language}/{case_id}/turns",
    )
    assert_frame_matches_mock(
        result.words.loc[:, ["part_of_speech", "first_person", "verb_tense"]],
        mock.words.loc[:, ["part_of_speech", "first_person", "verb_tense"]],
        context=f"{language}/{case_id}/words",
    )

    _assert_columns_missing(
        result.summary,
        [
            "word_coherence_mean",
            "word_repeat_percentage",
            "semantic_perplexity_mean",
        ],
        context=f"{language}/{case_id}/summary",
    )
