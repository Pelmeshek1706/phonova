import pytest

from tests.helpers.cross_language_outputs import load_validated_summary
from tests.helpers.data_loading import discover_cross_language_case_ids
from tests.helpers.dataframe_compare import assert_frame_matches_mock
from tests.helpers.feature_columns import STRICT_SUMMARY_INVARIANTS
from tests.helpers.thresholds import STRICT_FLOAT_TOLERANCE


@pytest.mark.parametrize("case_id", discover_cross_language_case_ids())
def test_strict_structural_invariants_match_for_paired_cases(case_id):
    en_summary = load_validated_summary("en", case_id)
    ua_summary = load_validated_summary("ua", case_id)

    assert_frame_matches_mock(
        en_summary.loc[:, STRICT_SUMMARY_INVARIANTS],
        ua_summary.loc[:, STRICT_SUMMARY_INVARIANTS],
        context=f"cross-language/{case_id}/strict",
        atol=STRICT_FLOAT_TOLERANCE,
    )
