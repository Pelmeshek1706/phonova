from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tests.helpers.thresholds import DEFAULT_DF_ATOL, DEFAULT_DF_RTOL


def normalize_missing_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _coerce_numeric(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    except Exception:
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _format_context(context: str) -> str:
    return f"{context}: " if context else ""


def _assert_value_equal(actual: Any, expected: Any, *, context: str, column: str, row_index: int, atol: float, rtol: float) -> None:
    prefix = _format_context(context)
    if _is_missing(expected):
        if not _is_missing(actual):
            raise AssertionError(f"{prefix}{column}[{row_index}] expected missing value, got {actual!r}")
        return

    if _is_missing(actual):
        raise AssertionError(f"{prefix}{column}[{row_index}] expected {expected!r}, got missing value")

    actual_numeric = _coerce_numeric(actual)
    expected_numeric = _coerce_numeric(expected)
    if actual_numeric is not None and expected_numeric is not None:
        if not math.isclose(actual_numeric, expected_numeric, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(
                f"{prefix}{column}[{row_index}] numeric mismatch: actual={actual_numeric!r} expected={expected_numeric!r} "
                f"(atol={atol}, rtol={rtol})"
            )
        return

    if normalize_missing_value(actual) != normalize_missing_value(expected):
        raise AssertionError(
            f"{prefix}{column}[{row_index}] value mismatch: actual={actual!r} expected={expected!r}"
        )


def assert_frame_matches_mock(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    context: str = "",
    atol: float = DEFAULT_DF_ATOL,
    rtol: float = DEFAULT_DF_RTOL,
) -> None:
    prefix = _format_context(context)
    actual_columns = list(actual.columns)
    expected_columns = list(expected.columns)
    if actual_columns != expected_columns:
        raise AssertionError(
            f"{prefix}column mismatch:\nactual  = {actual_columns}\nexpected= {expected_columns}"
        )

    if len(actual) != len(expected):
        raise AssertionError(f"{prefix}row count mismatch: actual={len(actual)} expected={len(expected)}")

    for column in actual_columns:
        for row_index, (actual_value, expected_value) in enumerate(zip(actual[column], expected[column])):
            _assert_value_equal(
                actual_value,
                expected_value,
                context=context,
                column=column,
                row_index=row_index,
                atol=atol,
                rtol=rtol,
            )

    actual_nans = actual.isna()
    expected_nans = expected.isna()
    if not actual_nans.equals(expected_nans):
        diff = []
        mismatch = np.argwhere(actual_nans.to_numpy() != expected_nans.to_numpy())
        for row_index, col_index in mismatch[:10]:
            diff.append(
                f"{actual_columns[col_index]}[{row_index}] actual_missing={bool(actual_nans.iat[row_index, col_index])} "
                f"expected_missing={bool(expected_nans.iat[row_index, col_index])}"
            )
        raise AssertionError(f"{prefix}missing-value mismatch:\n" + "\n".join(diff))
