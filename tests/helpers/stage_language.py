from __future__ import annotations

from tests.helpers.data_loading import discover_paired_case_ids, normalize_language


def requested_language(config):
    raw_language = config.getoption("language")
    return None if raw_language is None else normalize_language(raw_language)


def iter_stage_languages(config, default_languages=("en", "ua")):
    requested = requested_language(config)
    if requested is not None:
        return [requested]
    return [normalize_language(language) for language in default_languages]


def iter_stage_case_params(config, default_languages=("en", "ua")):
    params = []
    for language in iter_stage_languages(config, default_languages):
        for case_id in discover_paired_case_ids(language):
            params.append((language, case_id))
    return params
