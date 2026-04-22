import pandas as pd

from tests.helpers.module_loaders import load_local_pause_module


def test_pause_series_non_negative_clamps_negative_values():
    pause = load_local_pause_module()

    values = pause._pause_series_non_negative(pd.Series([-1.5, 0.0, 2.5, None]))

    assert values.tolist()[:3] == [0.0, 0.0, 2.5]
    assert values.isna().iloc[3]


def test_bounded_speech_percentage_limits_range():
    pause = load_local_pause_module()

    assert pause._bounded_speech_percentage(30.0, 1.0) == 50.0
    assert pause._bounded_speech_percentage(120.0, 1.0) == 0.0
    assert pause._bounded_speech_percentage(-30.0, 1.0) == 100.0
