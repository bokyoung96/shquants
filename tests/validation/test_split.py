import pandas as pd
import pytest

from backtesting.validation import SplitConfig, split_frame


def test_split_frame_returns_is_and_oos_slices() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2, 3, 4, 5]},
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        ),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-02"),
        is_end=pd.Timestamp("2024-01-03"),
        oos_start=pd.Timestamp("2024-01-04"),
        oos_end=pd.Timestamp("2024-01-05"),
    )

    result = split_frame(frame, config)

    assert list(result.is_frame.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert list(result.oos_frame.index) == list(pd.to_datetime(["2024-01-04", "2024-01-05"]))


def test_split_frame_rejects_touching_boundaries() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-02"),
        oos_start=pd.Timestamp("2024-01-02"),
        oos_end=pd.Timestamp("2024-01-02"),
    )

    with pytest.raises(ValueError, match="is_end must be < oos_start"):
        split_frame(frame, config)


def test_split_frame_rejects_invalid_is_window() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-02"),
        is_end=pd.Timestamp("2024-01-01"),
        oos_start=pd.Timestamp("2024-01-03"),
        oos_end=pd.Timestamp("2024-01-04"),
    )

    with pytest.raises(ValueError, match="is_start must be <= is_end"):
        split_frame(frame, config)


def test_split_frame_rejects_invalid_oos_window() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-01"),
        oos_start=pd.Timestamp("2024-01-04"),
        oos_end=pd.Timestamp("2024-01-03"),
    )

    with pytest.raises(ValueError, match="oos_start must be <= oos_end"):
        split_frame(frame, config)


def test_split_frame_rejects_unsorted_index() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2]},
        index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-01"),
        oos_start=pd.Timestamp("2024-01-02"),
        oos_end=pd.Timestamp("2024-01-02"),
    )

    with pytest.raises(ValueError, match="frame.index must be monotonic increasing"):
        split_frame(frame, config)


def test_split_frame_rejects_is_window_without_overlap() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-10"),
        is_end=pd.Timestamp("2024-01-11"),
        oos_start=pd.Timestamp("2024-01-12"),
        oos_end=pd.Timestamp("2024-01-13"),
    )

    with pytest.raises(ValueError, match="IS window must be within frame bounds"):
        split_frame(frame, config)


def test_split_frame_rejects_oos_window_without_overlap() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-01"),
        oos_start=pd.Timestamp("2024-01-10"),
        oos_end=pd.Timestamp("2024-01-11"),
    )

    with pytest.raises(ValueError, match="OOS window must be within frame bounds"):
        split_frame(frame, config)


def test_split_frame_rejects_partial_is_window_overlap() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2, 3]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-03"),
        oos_start=pd.Timestamp("2024-01-04"),
        oos_end=pd.Timestamp("2024-01-04"),
    )

    with pytest.raises(ValueError, match="IS window must be within frame bounds"):
        split_frame(frame, config)


def test_split_frame_rejects_partial_oos_window_overlap() -> None:
    frame = pd.DataFrame(
        {"signal": [1, 2, 3]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    config = SplitConfig(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-01"),
        oos_start=pd.Timestamp("2024-01-02"),
        oos_end=pd.Timestamp("2024-01-05"),
    )

    with pytest.raises(ValueError, match="OOS window must be within frame bounds"):
        split_frame(frame, config)
