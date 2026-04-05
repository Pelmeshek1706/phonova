from tests.helpers.feature_group_integrity import (
    assert_coherence_and_perplexity_group,
    assert_cross_level_relationships,
    assert_lexical_diversity_group,
    assert_sentiment_and_first_person_group,
    assert_structural_group,
)


def test_ukrainian_structural_pause_and_repetition_integrity():
    assert_structural_group("ukr")


def test_ukrainian_sentiment_and_first_person_integrity():
    assert_sentiment_and_first_person_group("ukr")


def test_ukrainian_lexical_diversity_integrity():
    assert_lexical_diversity_group("ukr")


def test_ukrainian_coherence_and_perplexity_integrity():
    assert_coherence_and_perplexity_group("ukr")


def test_ukrainian_cross_level_relationships():
    assert_cross_level_relationships("ukr")
