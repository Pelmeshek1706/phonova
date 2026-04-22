import pytest

from tests.helpers.module_loaders import load_local_lexical_module


def test_normalize_lang_accepts_ukrainian_aliases():
    lexical = load_local_lexical_module()

    assert lexical._normalize_lang("ua") == "ua"
    assert lexical._normalize_lang("uk") == "uk"
    assert lexical._normalize_lang("en") == "en"


def test_sentiment_values_and_repetition_helpers_are_deterministic():
    lexical = load_local_lexical_module()

    assert lexical._sentiment_values({"neg": 0.2, "neu": 0.3, "pos": 0.5, "compound": 0.1}) == [0.2, 0.3, 0.5, 0.1]
    assert lexical.calculate_repetitions(["hello", "hello", "world"], ["same", "same", "different"]) == (
        pytest.approx(33.333333333333336),
        pytest.approx(33.333333333333336),
    )
