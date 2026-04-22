from tests.helpers.runtime_checks import check_nltk_resources


def test_nltk_resources_are_available():
    assert check_nltk_resources() == ["punkt", "averaged_perceptron_tagger"]
