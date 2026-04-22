from tests.helpers.runtime_checks import check_package_versions


def test_runtime_package_versions_import_cleanly():
    probes = check_package_versions()
    names = {probe.name for probe in probes}

    assert {
        "openwillis_speech_module",
        "huggingface_hub",
        "nltk",
        "sentencepiece",
        "sentence_transformers",
        "spacy",
        "torch",
        "transformers",
    } <= names
    assert all(probe.value for probe in probes)
