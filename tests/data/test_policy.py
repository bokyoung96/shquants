import pandas as pd
from pandas.api.types import is_integer_dtype

from backtesting.data.policy import expand_monthly_frame


def test_expand_monthly_frame_keeps_missing_gap() -> None:
    frame = pd.DataFrame(
        {
            "005930": [1.0, 3.0],
        },
        index=pd.to_datetime(["2024-03-31", "2024-05-31"]),
    )

    expanded = expand_monthly_frame(
        frame,
        calendar=pd.date_range("2024-03-01", "2024-05-31", freq="D"),
        validity="month_only",
    )

    assert expanded.loc["2024-04-15", "005930"] != expanded.loc["2024-03-15", "005930"]
    assert pd.isna(expanded.loc["2024-04-15", "005930"])


def test_expand_monthly_frame_preserves_numeric_dtype() -> None:
    frame = pd.DataFrame(
        {
            "005930": pd.Series([1.0, 3.0], dtype="float64"),
        },
        index=pd.to_datetime(["2024-03-31", "2024-05-31"]),
    )

    expanded = expand_monthly_frame(
        frame,
        calendar=pd.date_range("2024-03-01", "2024-05-31", freq="D"),
        validity="month_only",
    )

    assert expanded["005930"].dtype == frame["005930"].dtype


def test_expand_monthly_frame_preserves_integer_data_with_missing_days() -> None:
    frame = pd.DataFrame(
        {
            "005930": [1, 3],
        },
        index=pd.to_datetime(["2024-03-31", "2024-05-31"]),
    )

    expanded = expand_monthly_frame(
        frame,
        calendar=pd.date_range("2024-03-01", "2024-05-31", freq="D"),
        validity="month_only",
    )

    assert is_integer_dtype(expanded["005930"].dtype)
    assert str(expanded["005930"].dtype) == "Int64"
    assert expanded.loc["2024-03-15", "005930"] == 1
    assert pd.isna(expanded.loc["2024-04-15", "005930"])
