from tests.helpers.data_loading import load_raw_json
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_case_params


FEATURE_SPECS = [
    (
        "pause_repetition",
        ("pause", "repetition"),
        "simple",
        {
            "words": ("pre_word_pause", "num_syllables"),
            "turns": (
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
            ),
            "summary": (
                "file_length",
                "speech_length_minutes",
                "speech_percentage",
                "speaker_percentage",
                "num_turns",
                "speech_length_words",
                "words_per_min",
                "syllables_per_min",
                "mean_pre_word_pause",
                "mean_pause_variability",
                "mean_turn_length_minutes",
                "mean_turn_length_words",
                "mean_pre_turn_pause",
                "num_one_word_turns",
                "word_repeat_percentage",
                "phrase_repeat_percentage",
            ),
        },
        {
            "words": ("word_coherence", "sentiment_pos", "first_person"),
            "turns": ("sentiment_pos", "mattr_5", "semantic_perplexity", "turn_to_turn_tangeniality"),
            "summary": ("sentiment_pos", "word_coherence_mean", "first_person_percentage", "semantic_perplexity_mean"),
        },
    ),
    (
        "sentiment_first_person",
        ("sentiment", "first_person"),
        "simple",
        {
            "words": ("part_of_speech", "first_person", "verb_tense"),
            "turns": (
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
            ),
            "summary": (
                "file_length",
                "speaker_percentage",
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
                "prop_verb_past",
                "prop_function_words",
                "first_person_sentiment_positive",
                "first_person_sentiment_negative",
                "first_person_sentiment_overall",
                "first_person_sentiment_positive_vader",
                "first_person_sentiment_negative_vader",
                "first_person_sentiment_overall_vader",
            ),
        },
        {
            "words": ("word_coherence", "word_repeat_percentage", "semantic_perplexity"),
            "turns": ("word_repeat_percentage", "semantic_perplexity", "turn_to_turn_tangeniality"),
            "summary": ("word_coherence_mean", "semantic_perplexity_mean", "speech_length_minutes", "num_turns"),
        },
    ),
    (
        "coherence",
        ("coherence",),
        "coherence",
        {
            "words": (
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
            ),
            "turns": (),
            "summary": (
                "file_length",
                "speaker_percentage",
                "word_coherence_mean",
                "word_coherence_var",
                "word_coherence_5_mean",
                "word_coherence_5_var",
                "word_coherence_10_mean",
                "word_coherence_10_var",
            ),
        },
        {
            "words": ("sentiment_pos", "first_person", "word_repeat_percentage"),
            "turns": (
                "sentiment_pos",
                "first_person_percentage",
                "word_repeat_percentage",
                "first_order_sentence_tangeniality",
                "second_order_sentence_tangeniality",
                "turn_to_turn_tangeniality",
                "turn_to_previous_speaker_turn_similarity",
                "semantic_perplexity",
                "semantic_perplexity_5",
                "semantic_perplexity_11",
                "semantic_perplexity_15",
            ),
            "summary": (
                "sentiment_pos",
                "first_person_percentage",
                "word_repeat_percentage",
                "speech_percentage",
                "num_turns",
                "turn_to_turn_tangeniality_mean",
                "turn_to_turn_tangeniality_var",
                "turn_to_turn_tangeniality_slope",
                "turn_to_previous_speaker_turn_similarity_mean",
                "turn_to_previous_speaker_turn_similarity_var",
                "turn_to_previous_speaker_turn_similarity_slope",
                "semantic_perplexity_mean",
                "semantic_perplexity_var",
                "semantic_perplexity_5_mean",
                "semantic_perplexity_5_var",
                "semantic_perplexity_11_mean",
                "semantic_perplexity_11_var",
                "semantic_perplexity_15_mean",
                "semantic_perplexity_15_var",
            ),
        },
    ),
]


def pytest_generate_tests(metafunc):
    if {"language", "case_id"}.issubset(metafunc.fixturenames):
        metafunc.parametrize(("language", "case_id"), iter_stage_case_params(metafunc.config))
    if "feature_spec" in metafunc.fixturenames:
        metafunc.parametrize("feature_spec", FEATURE_SPECS, ids=[item[0] for item in FEATURE_SPECS])


def _assert_any_values(df, columns, context):
    for column in columns:
        assert column in df.columns, f"{context}: missing column {column}"
        assert df[column].notna().any(), f"{context}: expected at least one populated value in {column}"


def _assert_all_missing(df, columns, context):
    for column in columns:
        if column not in df.columns:
            continue
        assert df[column].isna().all(), f"{context}: expected NaN-only values in {column}"


def test_pipeline_feature_group_selection_smoke(language, case_id, feature_spec):
    name, feature_groups, option, populated_columns, empty_columns = feature_spec
    raw = load_raw_json(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option=option,
        feature_groups=feature_groups,
        whisper_turn_mode="speaker",
    )

    _assert_any_values(result.words, populated_columns["words"], f"{language}/{case_id}/{name}/words")
    _assert_any_values(result.turns, populated_columns["turns"], f"{language}/{case_id}/{name}/turns")
    _assert_any_values(result.summary, populated_columns["summary"], f"{language}/{case_id}/{name}/summary")

    _assert_all_missing(result.words, empty_columns["words"], f"{language}/{case_id}/{name}/words")
    _assert_all_missing(result.turns, empty_columns["turns"], f"{language}/{case_id}/{name}/turns")
    _assert_all_missing(result.summary, empty_columns["summary"], f"{language}/{case_id}/{name}/summary")
