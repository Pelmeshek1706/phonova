from pathlib import Path

import pytest

from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.stage_language import iter_stage_languages


def pytest_generate_tests(metafunc):
    if "language" in metafunc.fixturenames:
        metafunc.parametrize("language", iter_stage_languages(metafunc.config))


def test_pipeline_rejects_invalid_whisper_turn_mode(language):
    with pytest.raises(ValueError):
        run_speech_characteristics(
            {},
            language=language,
            option="coherence",
            feature_groups=None,
            whisper_turn_mode="merge",
        )


def test_pipeline_segment_mode_runs_without_segment_snapshot(language):
    case_id = 300
    raw = load_raw_json(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="coherence",
        feature_groups=None,
        whisper_turn_mode="segment",
    )

    segment_mock_dir = Path(__file__).resolve().parents[1] / "data" / "mock" / language
    segment_snapshot_exists = any(segment_mock_dir.glob(f"*segment*_{case_id}.csv"))

    if segment_snapshot_exists:
        mock = load_mock_csv_triplet(language, case_id)
        assert len(result.words) == len(mock.words)
    else:
        assert len(result.words) > 0
        assert len(result.turns) > 0
        assert len(result.summary) == 1
