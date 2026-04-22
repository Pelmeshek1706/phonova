import pytest

from tests.helpers.cross_language_outputs import load_validated_summary
from tests.helpers.data_loading import discover_cross_language_case_ids
from tests.helpers.feature_columns import REPORT_ONLY_CROSS_LANGUAGE_SUMMARY_COLUMNS


@pytest.mark.parametrize("case_id", discover_cross_language_case_ids())
def test_cross_language_report_only_columns_are_present_without_pairwise_equality(case_id):
    en_summary = load_validated_summary("en", case_id)
    ua_summary = load_validated_summary("ua", case_id)

    assert list(en_summary.columns) == list(ua_summary.columns)
    for column in REPORT_ONLY_CROSS_LANGUAGE_SUMMARY_COLUMNS:
        assert column in en_summary.columns
        assert column in ua_summary.columns
        assert en_summary[column].isna().equals(ua_summary[column].isna()), (
            f"cross-language/{case_id}/{column}: missing-value mask differs"
        )
