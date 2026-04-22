from __future__ import annotations

from functools import lru_cache

import pandas as pd

from tests.helpers.data_loading import load_mock_csv_triplet, load_raw_json
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.pipeline_runner import run_speech_characteristics
from tests.helpers.thresholds import MODEL_DF_ATOL, MODEL_DF_RTOL


@lru_cache(maxsize=None)
def load_validated_summary(language: str, case_id: int) -> pd.DataFrame:
    raw = load_raw_json(language, case_id)
    mock = load_mock_csv_triplet(language, case_id)
    result = run_speech_characteristics(
        raw,
        language=language,
        option="coherence",
        feature_groups=None,
        whisper_turn_mode="speaker",
    )

    assert_frame_matches_mock(
        result.summary,
        mock.summary,
        context=f"{language}/{case_id}/validated-summary",
        atol=MODEL_DF_ATOL,
        rtol=MODEL_DF_RTOL,
    )
    return result.summary.copy(deep=True)
