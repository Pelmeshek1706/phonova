import pandas as pd
import numpy as np

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


def test_get_pause_feature_turn_assigns_positional_values(monkeypatch):
    pause = load_local_pause_module()

    turn_df = pd.DataFrame(index=[10, 11])
    df_diff = pd.DataFrame(
        {
            "old_index": [1, 4],
            "pause": [np.nan, -0.4],
        }
    )
    turn_index = [(1, 1), (4, 4)]
    measures = {
        "old_index": "old_index",
        "pause": "pause",
        "turn_pause": "pre_turn_pause",
        "interrupt_flag": "interrupt_flag",
    }

    def fake_to_numeric(values, errors="raise"):
        return pd.Series([np.nan, -0.4], index=[100, 101])

    monkeypatch.setattr(pause.pd, "to_numeric", fake_to_numeric)
    monkeypatch.setattr(
        pause,
        "calculate_pause_features_for_turn",
        lambda df_diff, turn_df, turn_list, turn_index, time_index, measures, language: turn_df,
    )

    result = pause.get_pause_feature_turn(
        turn_df,
        df_diff,
        ["a", "b"],
        turn_index,
        ("start", "end"),
        measures,
        "en",
    )

    assert result[measures["turn_pause"]].isna().iloc[0]
    assert result[measures["turn_pause"]].iloc[1] == 0.0
    assert bool(result[measures["interrupt_flag"]].iloc[0]) is False
    assert bool(result[measures["interrupt_flag"]].iloc[1]) is True
