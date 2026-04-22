import pytest

from tests.helpers.cross_language_outputs import load_validated_summary
from tests.helpers.data_loading import discover_cross_language_case_ids
from tests.helpers.feature_columns import LANGUAGE_DEPENDENT_SUMMARY_COLUMNS


@pytest.mark.parametrize("case_id", discover_cross_language_case_ids())
def test_language_dependent_features_are_not_strict_equality_contracts(case_id):
    en_summary = load_validated_summary("en", case_id)
    ua_summary = load_validated_summary("ua", case_id)

    for column in LANGUAGE_DEPENDENT_SUMMARY_COLUMNS:
        assert column in en_summary.columns
        assert column in ua_summary.columns

    differing_columns = [
        column
        for column in LANGUAGE_DEPENDENT_SUMMARY_COLUMNS
        if en_summary.iloc[0][column] != ua_summary.iloc[0][column]
    ]
    assert differing_columns, f"cross-language/{case_id}: language-dependent columns unexpectedly all match exactly"
