import math

import pandas as pd
import pytest

from tests.helpers.cross_language_outputs import load_validated_summary
from tests.helpers.data_loading import discover_cross_language_case_ids
from tests.helpers.feature_columns import (
    MODERATE_CROSS_LANGUAGE_SUMMARY_COLUMNS,
    STRONG_CROSS_LANGUAGE_SUMMARY_COLUMNS,
)
from tests.helpers.thresholds import (
    FIRST_PERSON_SENTIMENT_TOLERANCE,
    MODERATE_FLOAT_TOLERANCE,
    STRONG_FLOAT_TOLERANCE,
)


FIRST_PERSON_COLUMNS = {
    "first_person_sentiment_positive",
    "first_person_sentiment_negative",
    "first_person_sentiment_overall",
}


def _numeric_value(row, column):
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _assert_columns_within_threshold(en_summary, ua_summary, columns, *, context, default_tolerance):
    en_row = en_summary.iloc[0]
    ua_row = ua_summary.iloc[0]
    for column in columns:
        assert column in en_summary.columns
        assert column in ua_summary.columns
        en_value = _numeric_value(en_row, column)
        ua_value = _numeric_value(ua_row, column)
        if en_value is None or ua_value is None:
            assert en_value is None and ua_value is None, f"{context}/{column}: missing-value mismatch"
            continue
        tolerance = FIRST_PERSON_SENTIMENT_TOLERANCE if column in FIRST_PERSON_COLUMNS else default_tolerance
        assert math.isclose(en_value, ua_value, rel_tol=0.0, abs_tol=tolerance), (
            f"{context}/{column}: EN={en_value!r} UA={ua_value!r} exceeds abs tolerance {tolerance}"
        )


@pytest.mark.parametrize("case_id", discover_cross_language_case_ids())
def test_strong_cross_language_features_stay_within_threshold(case_id):
    en_summary = load_validated_summary("en", case_id)
    ua_summary = load_validated_summary("ua", case_id)

    _assert_columns_within_threshold(
        en_summary,
        ua_summary,
        STRONG_CROSS_LANGUAGE_SUMMARY_COLUMNS,
        context=f"cross-language/{case_id}/strong",
        default_tolerance=STRONG_FLOAT_TOLERANCE,
    )


@pytest.mark.parametrize("case_id", discover_cross_language_case_ids())
def test_moderate_cross_language_features_stay_within_threshold(case_id):
    en_summary = load_validated_summary("en", case_id)
    ua_summary = load_validated_summary("ua", case_id)

    _assert_columns_within_threshold(
        en_summary,
        ua_summary,
        MODERATE_CROSS_LANGUAGE_SUMMARY_COLUMNS,
        context=f"cross-language/{case_id}/moderate",
        default_tolerance=MODERATE_FLOAT_TOLERANCE,
    )
