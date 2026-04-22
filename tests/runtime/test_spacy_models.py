from tests.helpers.runtime_checks import check_spacy_models


def test_spacy_models_load_on_cpu_runtime():
    loaded = check_spacy_models()

    assert set(loaded) == {"en_core_web_sm", "uk_core_news_sm"}
    assert all(loaded.values())
